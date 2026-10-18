use axum::{
    body::{to_bytes, Body},
    http::{Request, StatusCode},
    middleware,
    routing::get,
    Router,
};
use lance_context_metrics::{http_metrics_layer, install_recorder, metrics_router};
use std::{process::Command, sync::Arc};
use tokio::sync::Notify;
use tower::ServiceExt;

#[tokio::test]
async fn middleware_records_inflight_demand_and_read_failures_without_payloads() {
    // Each opt-in setting gets a fresh process, without mutating global env
    // alongside other tests or resetting the production OnceLock.
    if std::env::var("RECOVERY_SIGNALS_TEST_CHILD").is_err() {
        for enabled in ["true", "false"] {
            assert!(Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "middleware_records_inflight_demand_and_read_failures_without_payloads"
                ])
                .env("RECOVERY_SIGNALS_TEST_CHILD", "1")
                .env("RECOVERY_SIGNALS_ENABLED", enabled)
                .status()
                .unwrap()
                .success());
        }
        return;
    }
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let app = Router::new()
        .route(
            "/api/v1/rollouts/{name}/records/{id}",
            get({
                let entered = entered.clone();
                let release = release.clone();
                move || {
                    let entered = entered.clone();
                    let release = release.clone();
                    async move {
                        entered.notify_one();
                        release.notified().await;
                        StatusCode::SERVICE_UNAVAILABLE
                    }
                }
            }),
        )
        .layer(middleware::from_fn(http_metrics_layer))
        .merge(metrics_router(install_recorder()));
    let mut task = tokio::spawn(
        app.clone().oneshot(
            Request::builder()
                .uri("/api/v1/rollouts/mai3_bigclimb_t1/records/secret-record-id?secret=query")
                .body(Body::from("secret-payload"))
                .unwrap(),
        ),
    );
    tokio::time::timeout(std::time::Duration::from_secs(5), entered.notified())
        .await
        .unwrap();
    for finished in [false, true] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/v1/internal/recovery-signals?nonce=canary")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.headers()["cache-control"], "no-store");
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        assert!(!String::from_utf8_lossy(&bytes).contains("secret"));
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        if std::env::var("RECOVERY_SIGNALS_ENABLED").unwrap() == "true" {
            assert_eq!(status, StatusCode::OK);
            assert_eq!(v["nonce"], "canary");
            assert_eq!(v["tables"].as_array().unwrap().len(), 1);
            assert_eq!(v["tables"][0]["name"], "mai3_bigclimb_t1");
            assert_eq!(
                v["tables"][0]["read_error_status"],
                if finished {
                    serde_json::json!(503)
                } else {
                    serde_json::Value::Null
                }
            );
        } else {
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        }
        if !finished {
            release.notify_one();
            // Wait until the middleware has observed the finished response.
            let response = tokio::time::timeout(std::time::Duration::from_secs(5), &mut task)
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        }
    }
    let invalid = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/internal/recovery-signals?nonce=bad%20nonce")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(invalid.status(), StatusCode::BAD_REQUEST);
}
