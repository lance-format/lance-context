use super::*;
use axum::{routing::post, Router};
use tokio::sync::Notify;

async fn serve(app: Router) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (endpoint, server)
}

#[tokio::test]
async fn publication_waits_for_every_worker_and_rejects_partial_success() {
    for first_status in [
        StatusCode::SERVICE_UNAVAILABLE,
        StatusCode::OK,
        StatusCode::NO_CONTENT,
    ] {
        let first = Router::new().route(
            "/api/v1/events/table/flush-local",
            post(move || async move { first_status }),
        );
        let entered = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let second = Router::new().route(
            "/api/v1/events/table/flush-local",
            post({
                let entered = entered.clone();
                let release = release.clone();
                move || {
                    let entered = entered.clone();
                    let release = release.clone();
                    async move {
                        entered.notify_one();
                        release.notified().await;
                        StatusCode::NO_CONTENT
                    }
                }
            }),
        );
        let (first_url, first_server) = serve(first).await;
        let (second_url, second_server) = serve(second).await;
        let pending = tokio::spawn(async move {
            flush_workers(
                &crate::state::worker_http_client().unwrap(),
                &[first_url, second_url],
                "events",
                "table",
            )
            .await
        });
        entered.notified().await;
        assert!(
            !pending.is_finished(),
            "the slow worker has not published yet"
        );
        release.notify_one();
        assert_eq!(
            pending.await.unwrap().is_ok(),
            first_status == StatusCode::NO_CONTENT
        );
        first_server.abort();
        second_server.abort();
    }
}

#[tokio::test]
async fn missing_or_unreachable_workers_cannot_acknowledge_publication() {
    let client = crate::state::worker_http_client().unwrap();
    assert!(matches!(
        flush_workers(&client, &[], "events", "table").await,
        Err(MasterError::Unavailable(_))
    ));
    let (endpoint, server) = serve(Router::new()).await;
    assert!(matches!(
        flush_workers(&client, &[endpoint], "events", "table").await,
        Err(MasterError::Unavailable(_))
    ));
    server.abort();
}

#[tokio::test]
async fn publication_rejects_redirects_instead_of_skipping_a_worker() {
    let app = Router::new()
        .route(
            "/api/v1/events/table/flush-local",
            post(|| async { axum::response::Redirect::temporary("/success") }),
        )
        .route("/success", post(|| async { StatusCode::NO_CONTENT }));
    let (endpoint, server) = serve(app).await;
    assert!(matches!(
        flush_workers(
            &crate::state::worker_http_client().unwrap(),
            &[endpoint],
            "events",
            "table"
        )
        .await,
        Err(MasterError::Unavailable(_))
    ));
    server.abort();
}

#[tokio::test]
#[ignore = "requires ETCD_TEST_ENDPOINTS"]
async fn registered_publication_route_validates_scope_and_bounds_worker_wait() {
    use axum::{body::Body, http::Request};
    use clap::Parser;
    use tower::ServiceExt;

    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let worker = Router::new().route(
        "/api/v1/events/table/flush-local",
        post({
            let entered = entered.clone();
            let release = release.clone();
            move || {
                let entered = entered.clone();
                let release = release.clone();
                async move {
                    entered.notify_one();
                    release.notified().await;
                    StatusCode::NO_CONTENT
                }
            }
        }),
    );
    let (endpoint, worker_server) = serve(worker).await;
    let directory = tempfile::tempdir().unwrap();
    let config = crate::config::MasterConfig::parse_from([
        "test",
        "--data-dir",
        directory.path().to_str().unwrap(),
        "--managed-store-kind",
        "events",
        "--worker-endpoints",
        &endpoint,
        "--table-flush-timeout-secs",
        "20",
        "--etcd-prefix",
        &format!("/publication-test/{}", lance_context_core::generate_id()),
        "--etcd-endpoints",
        &std::env::var("ETCD_TEST_ENDPOINTS").unwrap(),
    ]);
    let state = MasterState::new(config).await.unwrap();
    state
        .managed_registry
        .as_ref()
        .unwrap()
        .upsert("table", "unused-table-uri")
        .await
        .unwrap();
    let app = crate::routes::api_router().with_state(state.clone());
    for (path, expected) in [
        ("/stores/other/table/flush", StatusCode::BAD_REQUEST),
        ("/stores/events/missing/flush", StatusCode::NOT_FOUND),
        ("/stores/events/bad%2Fname/flush", StatusCode::BAD_REQUEST),
    ] {
        let response = app
            .clone()
            .oneshot(Request::post(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), expected);
    }
    for success in [true, false] {
        let pending = tokio::spawn(
            app.clone().oneshot(
                Request::post("/stores/events/table/flush")
                    .body(Body::empty())
                    .unwrap(),
            ),
        );
        entered.notified().await;
        assert!(!pending.is_finished());
        if success {
            release.notify_one();
            assert_eq!(
                pending.await.unwrap().unwrap().status(),
                StatusCode::NO_CONTENT
            );
        } else {
            tokio::time::pause();
            tokio::time::advance(Duration::from_secs(21)).await;
            assert_eq!(
                pending.await.unwrap().unwrap().status(),
                StatusCode::SERVICE_UNAVAILABLE
            );
            tokio::time::resume();
        }
    }
    // Publication is synchronous and must not create maintenance tasks.
    let response = app
        .oneshot(Request::get("/tasks").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let tasks: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(tasks["total"], 0);
    worker_server.abort();
}

#[tokio::test]
#[ignore = "requires ETCD_TEST_ENDPOINTS"]
async fn master_publication_makes_both_writers_readable_without_merging() {
    use arrow_array::{RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use axum::{body::Body, http::Request};
    use clap::Parser;
    use futures::TryStreamExt;
    use lance_context_core::{ListSource, StorageBase, StorageBaseOptions};
    use tower::ServiceExt;

    async fn keys(reader: &StorageBase) -> Vec<String> {
        reader
            .read_consistent(ListSource::All, |dataset, snapshots| async move {
                let scanner = StorageBase::lsm_scanner_for_dataset(
                    &dataset,
                    "id",
                    ListSource::All,
                    snapshots,
                );
                let batches: Vec<RecordBatch> =
                    scanner.try_into_stream().await?.try_collect().await?;
                let mut keys = Vec::new();
                for batch in batches {
                    let ids = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .unwrap();
                    keys.extend(ids.iter().map(|id| id.unwrap().to_owned()));
                }
                keys.sort();
                Ok(keys)
            })
            .await
            .unwrap()
    }

    let directory = tempfile::tempdir().unwrap();
    let table = directory.path().join("table");
    let uri = table.to_str().unwrap();
    let key =
        Field::new("id", DataType::Utf8, false).with_metadata(std::collections::HashMap::from([(
            "lance-schema:unenforced-primary-key".into(),
            "true".into(),
        )]));
    let schema = Arc::new(Schema::new(vec![key]));
    let mut options = StorageBaseOptions::new(schema.clone(), "id");
    options.seal_on_put = false;
    let mut endpoints = Vec::new();
    let mut servers = Vec::new();
    for id in ["first", "second"] {
        options.shard_id = Some(id.into());
        let writer = Arc::new(StorageBase::open(uri, options.clone(), true).await.unwrap());
        writer
            .put(vec![RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(StringArray::from(vec![id]))],
            )
            .unwrap()])
            .await
            .unwrap();
        let worker = Router::new().route(
            "/api/v1/events/table/flush-local",
            post(move || {
                let writer = writer.clone();
                async move {
                    writer.flush().await.unwrap();
                    StatusCode::NO_CONTENT
                }
            }),
        );
        let (endpoint, server) = serve(worker).await;
        endpoints.push(endpoint);
        servers.push(server);
    }
    let dataset = StorageBase::load_with_options(uri, None, None)
        .await
        .unwrap();
    let reader = StorageBase::from_dataset_for_read(dataset, options.clone()).unwrap();
    assert!(keys(&reader).await.is_empty());

    let config = crate::config::MasterConfig::parse_from([
        "test",
        "--data-dir",
        directory.path().join("master").to_str().unwrap(),
        "--managed-store-kind",
        "events",
        "--worker-endpoints",
        &endpoints.join(","),
        "--etcd-prefix",
        &format!("/publication-test/{}", lance_context_core::generate_id()),
        "--etcd-endpoints",
        &std::env::var("ETCD_TEST_ENDPOINTS").unwrap(),
    ]);
    let state = MasterState::new(config).await.unwrap();
    state
        .managed_registry
        .as_ref()
        .unwrap()
        .upsert("table", uri)
        .await
        .unwrap();
    let app = crate::routes::api_router().with_state(state);
    // Retrying publication through a master is safe after an uncertain response.
    for _ in 0..2 {
        let response = app
            .clone()
            .oneshot(
                Request::post("/stores/events/table/flush")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NO_CONTENT);
        assert_eq!(keys(&reader).await, ["first", "second"]);
    }
    for server in servers {
        server.abort();
        let _ = server.await;
    }
    let dataset = StorageBase::load_with_options(uri, None, None)
        .await
        .unwrap();
    assert_eq!(dataset.count_rows(None).await.unwrap(), 0);
    let resumed = StorageBase::from_dataset_for_read(dataset, options).unwrap();
    assert_eq!(keys(&resumed).await, ["first", "second"]);
}
