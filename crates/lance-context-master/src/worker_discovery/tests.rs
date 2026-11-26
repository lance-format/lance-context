use super::*;
use axum::{
    extract::Query,
    http::HeaderMap,
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

struct Server(tokio::task::JoinHandle<()>);
impl Drop for Server {
    fn drop(&mut self) {
        self.0.abort();
    }
}

async fn serve(app: Router) -> (String, Server) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    (
        url,
        Server(tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap()
        })),
    )
}

fn slice(url: &str, uid: &str, ready: bool) -> Value {
    let url = reqwest::Url::parse(url).unwrap();
    json!({
        "addressType": "IPv4",
        "ports": [{"name": "http", "port": url.port().unwrap(), "protocol": "TCP"}],
        "endpoints": [{
            "addresses": [url.host_str().unwrap()],
            "conditions": {"ready": ready, "terminating": !ready},
            "targetRef": {"uid": uid}
        }]
    })
}

fn page(items: Vec<Value>, revision: &str, next: &str) -> Value {
    json!({"metadata":{"resourceVersion":revision,"continue":next},"items":items})
}

fn client(url: String, token: &tempfile::NamedTempFile) -> KubernetesWorkers {
    std::fs::write(token.path(), "test-token").unwrap();
    KubernetesWorkers {
        client: Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap(),
        url,
        token_file: token.path().to_path_buf(),
        service: "workers".into(),
        port_name: "http".into(),
        statefulset_url: None,
    }
}

#[tokio::test]
async fn statefulset_replacement_gap_cannot_acknowledge_a_smaller_worker_list() {
    let flushed = Arc::new(AtomicUsize::new(0));
    let (worker, _worker_server) = serve(Router::new().route(
        "/api/v1/events/table/flush-local",
        post({
            let flushed = flushed.clone();
            move || {
                let flushed = flushed.clone();
                async move {
                    flushed.fetch_add(1, Ordering::SeqCst);
                    axum::http::StatusCode::NO_CONTENT
                }
            }
        }),
    ))
    .await;
    let named = |name: &str| {
        let mut value = slice(&worker, name, true);
        value["endpoints"][0]["targetRef"]["name"] = json!(name);
        value
    };
    let members = Arc::new(Mutex::new(vec![named("workers-5")]));
    let (url, _api) = serve(
        Router::new()
            .route(
                "/",
                get({
                    let members = members.clone();
                    move || {
                        let members = members.clone();
                        async move { Json(page(members.lock().unwrap().clone(), "10", "")) }
                    }
                }),
            )
            .route(
                "/statefulset",
                get(|| async {
                    Json(json!({
                        "metadata":{"name":"workers"},
                        "spec":{"replicas":2,"ordinals":{"start":5}}
                    }))
                }),
            ),
    )
    .await;
    let token = tempfile::NamedTempFile::new().unwrap();
    let mut discovery = client(url.clone(), &token);
    discovery.statefulset_url = Some(format!("{url}/statefulset"));
    let http = crate::state::worker_http_client().unwrap();
    for list in [
        vec![named("workers-5")],
        vec![named("workers-5"), named("unrelated-6")],
    ] {
        *members.lock().unwrap() = list;
        let result =
            crate::publication::flush_discovered(&http, &discovery, "events", "table").await;
        assert!(
            matches!(result, Err(crate::error::MasterError::Unavailable(message))
            if message.contains("membership is incomplete"))
        );
        assert_eq!(flushed.load(Ordering::SeqCst), 0);
    }
    *members.lock().unwrap() = vec![named("workers-5"), named("workers-6")];
    crate::publication::flush_discovered(&http, &discovery, "events", "table")
        .await
        .unwrap();
    assert_eq!(flushed.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn adding_a_worker_during_publication_requires_a_new_complete_barrier() {
    let second_flushed = Arc::new(AtomicUsize::new(0));
    let (second, _second_server) = serve(Router::new().route(
        "/api/v1/events/table/flush-local",
        post({
            let flushed = second_flushed.clone();
            move || {
                let flushed = flushed.clone();
                async move {
                    flushed.fetch_add(1, Ordering::SeqCst);
                    axum::http::StatusCode::NO_CONTENT
                }
            }
        }),
    ))
    .await;
    let members = Arc::new(Mutex::new(Vec::new()));
    let (first, _first_server) = serve(Router::new().route(
        "/api/v1/events/table/flush-local",
        post({
            let members = members.clone();
            let second = second.clone();
            move || {
                let members = members.clone();
                let second = second.clone();
                async move {
                    let mut list = members.lock().unwrap();
                    if list.len() == 1 {
                        list.push(slice(&second, "second", false));
                    }
                    axum::http::StatusCode::NO_CONTENT
                }
            }
        }),
    ))
    .await;
    members.lock().unwrap().push(slice(&first, "first", true));
    let (url, _api) = serve(Router::new().route(
        "/",
        get(move || {
            let members = members.clone();
            async move { Json(page(members.lock().unwrap().clone(), "10", "")) }
        }),
    ))
    .await;
    let token = tempfile::NamedTempFile::new().unwrap();
    let discovery = client(url, &token);
    let http = crate::state::worker_http_client().unwrap();
    let result = crate::publication::flush_discovered(&http, &discovery, "events", "table").await;
    assert!(matches!(result,
        Err(crate::error::MasterError::Unavailable(message)) if message.contains("membership changed")));
    assert_eq!(second_flushed.load(Ordering::SeqCst), 0);
    crate::publication::flush_discovered(&http, &discovery, "events", "table")
        .await
        .unwrap();
    assert_eq!(
        second_flushed.load(Ordering::SeqCst),
        1,
        "a NotReady/terminating writer must also publish"
    );
}

#[tokio::test]
async fn discovery_failure_or_empty_membership_cannot_acknowledge_publication() {
    for (status, body) in [
        (axum::http::StatusCode::FORBIDDEN, json!({})),
        (axum::http::StatusCode::OK, page(vec![], "10", "")),
        (
            axum::http::StatusCode::OK,
            json!({"metadata":{"resourceVersion":"10"},"items":[{"addressType":"IPv4","ports":[],"endpoints":[{"addresses":["127.0.0.1"]}]}]}),
        ),
    ] {
        let (url, _server) = serve(Router::new().route(
            "/",
            get(move || {
                let body = body.clone();
                async move { (status, Json(body)) }
            }),
        ))
        .await;
        let token = tempfile::NamedTempFile::new().unwrap();
        let discovery = client(url, &token);
        assert!(matches!(
            crate::publication::flush_discovered(
                &crate::state::worker_http_client().unwrap(),
                &discovery,
                "events",
                "table"
            )
            .await,
            Err(crate::error::MasterError::Unavailable(_))
        ));
    }
}

#[tokio::test]
async fn missing_or_empty_worker_identity_cannot_acknowledge_publication() {
    let flushed = Arc::new(AtomicUsize::new(0));
    let (worker, _worker_server) = serve(Router::new().route(
        "/api/v1/events/table/flush-local",
        post({
            let flushed = flushed.clone();
            move || {
                let flushed = flushed.clone();
                async move {
                    flushed.fetch_add(1, Ordering::SeqCst);
                    axum::http::StatusCode::NO_CONTENT
                }
            }
        }),
    ))
    .await;
    for target_ref in [
        None,
        Some(Value::Null),
        Some(json!({})),
        Some(json!({"uid": null})),
        Some(json!({"uid": ""})),
        Some(json!({"uid": " \t"})),
    ] {
        let mut endpoint_slice = slice(&worker, "original", true);
        let endpoint = endpoint_slice["endpoints"][0].as_object_mut().unwrap();
        if let Some(target_ref) = target_ref {
            endpoint.insert("targetRef".into(), target_ref);
        } else {
            endpoint.remove("targetRef");
        }
        let (url, _api) = serve(Router::new().route(
            "/",
            get(move || {
                let body = page(vec![endpoint_slice.clone()], "10", "");
                async move { Json(body) }
            }),
        ))
        .await;
        let token = tempfile::NamedTempFile::new().unwrap();
        let discovery = client(url, &token);
        let result = crate::publication::flush_discovered(
            &crate::state::worker_http_client().unwrap(),
            &discovery,
            "events",
            "table",
        )
        .await;
        assert!(matches!(result,
            Err(crate::error::MasterError::Unavailable(message)) if message.contains("Pod UID")));
        assert_eq!(
            flushed.load(Ordering::SeqCst),
            0,
            "invalid identity must reject the complete membership before contacting workers"
        );
    }
}

#[tokio::test]
async fn pagination_keeps_every_worker_and_rotated_tokens_are_used() {
    let required_token = Arc::new(Mutex::new("Bearer test-token".to_string()));
    let expected = required_token.clone();
    let (url, _server) = serve(Router::new().route(
        "/",
        get(
            move |headers: HeaderMap, Query(query): Query<HashMap<String, String>>| {
                let required_token = required_token.clone();
                async move {
                    assert_eq!(
                        headers.get("authorization").unwrap().to_str().unwrap(),
                        *required_token.lock().unwrap()
                    );
                    assert_eq!(
                        query.get("labelSelector").unwrap(),
                        "kubernetes.io/service-name=workers"
                    );
                    assert!(!query.contains_key("resourceVersion"));
                    Json(if query.get("continue").is_some_and(|s| s == "next") {
                        page(
                            vec![json!({
                                "addressType": "IPv6",
                                "ports": [{"name": "http", "port": 3000}],
                                "endpoints": [{"addresses": ["::1"], "conditions": {"ready": false},
                                    "targetRef": {"uid": "ipv6-worker"}}]
                            })],
                            "10",
                            "",
                        )
                    } else {
                        page(
                            vec![slice("http://127.0.0.1:3000", "first", true)],
                            "10",
                            "next",
                        )
                    })
                }
            },
        ),
    ))
    .await;
    let token = tempfile::NamedTempFile::new().unwrap();
    let discovery = client(url, &token);
    for rotated in [false, true] {
        if rotated {
            std::fs::write(token.path(), "rotated-token").unwrap();
            *expected.lock().unwrap() = "Bearer rotated-token".into();
        }
        assert_eq!(
            discovery.snapshot().await.unwrap().endpoints(),
            ["http://127.0.0.1:3000", "http://[::1]:3000"]
        );
    }
}

#[tokio::test]
async fn a_changed_page_revision_is_not_a_complete_membership() {
    let (url, _server) = serve(Router::new().route(
        "/",
        get(|Query(query): Query<HashMap<String, String>>| async move {
            Json(if query.get("continue").is_some_and(|s| s == "next") {
                page(
                    vec![slice("http://127.0.0.1:3001", "second", true)],
                    "11",
                    "",
                )
            } else {
                page(
                    vec![slice("http://127.0.0.1:3000", "first", true)],
                    "10",
                    "next",
                )
            })
        }),
    ))
    .await;
    let token = tempfile::NamedTempFile::new().unwrap();
    let discovery = client(url, &token);
    assert!(discovery
        .snapshot()
        .await
        .unwrap_err()
        .contains("changed revision"));
}

#[tokio::test]
async fn replacement_during_worker_retry_is_not_the_same_writer() {
    let uid = Arc::new(Mutex::new("old".to_string()));
    let calls = Arc::new(AtomicUsize::new(0));
    let (worker, _worker) = serve(Router::new().route(
        "/api/v1/events/table/flush-local",
        post({
            let uid = uid.clone();
            let calls = calls.clone();
            move || {
                let uid = uid.clone();
                let calls = calls.clone();
                async move {
                    *uid.lock().unwrap() = "new".into();
                    if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                        axum::http::StatusCode::SERVICE_UNAVAILABLE
                    } else {
                        axum::http::StatusCode::NO_CONTENT
                    }
                }
            }
        }),
    ))
    .await;
    let (url, _api) = serve(Router::new().route(
        "/",
        get(move || {
            let uid = uid.clone();
            let worker = worker.clone();
            async move {
                Json(page(
                    vec![slice(&worker, &uid.lock().unwrap(), true)],
                    "10",
                    "",
                ))
            }
        }),
    ))
    .await;
    let token = tempfile::NamedTempFile::new().unwrap();
    let discovery = client(url, &token);
    let result = crate::publication::flush_discovered(
        &crate::state::worker_http_client().unwrap(),
        &discovery,
        "events",
        "table",
    )
    .await;
    assert!(matches!(result,
        Err(crate::error::MasterError::Unavailable(message)) if message.contains("membership changed")));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
#[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
async fn registered_merge_uses_discovery_without_static_workers() {
    use crate::{config::MasterConfig, state::MasterState, task_store::TaskKinds};
    use clap::Parser;
    use lance_context_api::TaskKind;

    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let (worker, _worker_server) = serve(Router::new().route(
        "/api/v1/events/table/merge-wal",
        post(move || {
            let observed = observed.clone();
            async move {
                observed.fetch_add(1, Ordering::SeqCst);
                Json(json!({"reclaimed": 3}))
            }
        }),
    ))
    .await;
    let (url, _api) = serve(Router::new().route(
        "/",
        get(move || {
            let worker = worker.clone();
            async move { Json(page(vec![slice(&worker, "worker", true)], "10", "")) }
        }),
    ))
    .await;
    let token = tempfile::NamedTempFile::new().unwrap();
    let directory = tempfile::tempdir().unwrap();
    let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
    assert!(endpoint.starts_with("http://127.0.0.1:"));
    let cfg = MasterConfig::parse_from([
        "test",
        "--managed-store-kind",
        "events",
        "--data-dir",
        directory.path().to_str().unwrap(),
        "--etcd-endpoints",
        &endpoint,
        "--etcd-prefix",
        &format!(
            "/worker-discovery-tests/{}",
            lance_context_core::generate_id()
        ),
    ]);
    let mut state = MasterState::new(cfg).await.unwrap();
    // Inject the fake API instead of requiring a Kubernetes service-account mount.
    let mutable = Arc::get_mut(&mut state).unwrap();
    mutable.config.worker_discovery.managed_worker_service = Some("workers".into());
    mutable.managed_workers = Some(client(url, &token));
    assert!(state.config.worker_endpoints.is_empty());
    state
        .task_store
        .enqueue(TaskKind::MergeWal, "events:table", vec![])
        .await
        .unwrap();
    let claim = state
        .task_store
        .claim_next_of_kinds(TaskKinds::MERGE_WAL)
        .await
        .unwrap()
        .unwrap();
    let result = crate::merge_execution::run_merge_wal(&state, &claim)
        .await
        .unwrap();
    assert!(result.contains("3"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
