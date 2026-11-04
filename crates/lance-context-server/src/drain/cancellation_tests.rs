use super::*;
use axum::{
    body::Body,
    extract::Extension,
    middleware,
    routing::{get, post},
};
use tokio::sync::RwLock;
use tower::ServiceExt;

fn shield(app: Router, drain: Arc<Drain>) -> Router {
    app.layer(middleware::from_fn_with_state(drain, track_handler))
}
fn read_route(lock: Arc<RwLock<()>>, entered: Arc<Notify>) -> Router {
    Router::new().route(
        "/read",
        get(move |Extension(cancel): Extension<ClientCancellation>| {
            let (lock, entered) = (lock.clone(), entered.clone());
            async move {
                // Real integration must place this AFTER any protected lazy open.
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => StatusCode::REQUEST_TIMEOUT,
                    response = async {
                        let _guard = lock.read_owned().await;
                        entered.notify_one();
                        std::future::pending::<()>().await;
                        StatusCode::OK
                    } => response,
                }
            }
        }),
    )
}
async fn bounded(f: impl Future<Output = ()>) {
    tokio::time::timeout(Duration::from_secs(2), f)
        .await
        .unwrap();
}
#[tokio::test]
async fn opted_in_read_cancellation_releases_commit_lock_and_joins() {
    let lock = Arc::new(RwLock::new(()));
    let entered = Arc::new(Notify::new());
    let drain = Arc::new(Drain::default());
    let request = tokio::spawn(
        shield(read_route(lock.clone(), entered.clone()), drain.clone())
            .oneshot(Request::get("/read").body(Body::empty()).unwrap()),
    );
    bounded(entered.notified()).await;
    assert!(lock.try_write().is_err());
    request.abort();
    assert!(request.await.unwrap_err().is_cancelled());
    drain.close_admission();
    bounded(drain.joined()).await;
    assert!(lock.try_write().is_ok());
}
#[tokio::test]
async fn cancellation_during_protected_setup_waits_for_setup_then_skips_read() {
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let queried = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let drain = Arc::new(Drain::default());
    let app = Router::new().route(
        "/read",
        get({
            let (entered, release, queried) = (entered.clone(), release.clone(), queried.clone());
            move |Extension(cancel): Extension<ClientCancellation>| {
                let (entered, release, queried) =
                    (entered.clone(), release.clone(), queried.clone());
                async move {
                    entered.notify_one();
                    release.notified().await;
                    tokio::select! {
                        biased;
                        _ = cancel.cancelled() => StatusCode::REQUEST_TIMEOUT,
                        response = async {
                            queried.store(true, std::sync::atomic::Ordering::SeqCst);
                            StatusCode::OK
                        } => response,
                    }
                }
            }
        }),
    );
    let request = tokio::spawn(
        shield(app, drain.clone()).oneshot(Request::get("/read").body(Body::empty()).unwrap()),
    );
    bounded(entered.notified()).await;
    request.abort();
    assert!(request.await.unwrap_err().is_cancelled());
    drain.close_admission();
    assert!(
        tokio::time::timeout(Duration::from_millis(100), drain.joined())
            .await
            .is_err()
    );
    release.notify_one();
    bounded(drain.joined()).await;
    assert!(!queried.load(std::sync::atomic::Ordering::SeqCst));
}
#[tokio::test]
async fn accepted_post_remains_joined_and_commits_after_client_cancellation() {
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let committed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let drain = Arc::new(Drain::default());
    let app = Router::new().route(
        "/write",
        post({
            let (entered, release, committed) =
                (entered.clone(), release.clone(), committed.clone());
            move || {
                let (entered, release, committed) =
                    (entered.clone(), release.clone(), committed.clone());
                async move {
                    entered.notify_one();
                    release.notified().await;
                    committed.store(true, std::sync::atomic::Ordering::SeqCst);
                    StatusCode::OK
                }
            }
        }),
    );
    let request = tokio::spawn(
        shield(app, drain.clone()).oneshot(Request::post("/write").body(Body::empty()).unwrap()),
    );
    bounded(entered.notified()).await;
    request.abort();
    assert!(request.await.unwrap_err().is_cancelled());
    drain.close_admission();
    assert!(
        tokio::time::timeout(Duration::from_millis(100), drain.joined())
            .await
            .is_err()
    );
    assert!(!committed.load(std::sync::atomic::Ordering::SeqCst));
    release.notify_one();
    bounded(drain.joined()).await;
    assert!(committed.load(std::sync::atomic::Ordering::SeqCst));
}
#[tokio::test]
async fn connected_read_is_not_cancelled_by_closing_admission() {
    let drain = Arc::new(Drain::default());
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let app = Router::new().route(
        "/read",
        get({
            let (entered, release) = (entered.clone(), release.clone());
            move |Extension(cancel): Extension<ClientCancellation>| {
                let (entered, release) = (entered.clone(), release.clone());
                async move {
                    entered.notify_one();
                    tokio::select! {
                        _ = cancel.cancelled() => StatusCode::REQUEST_TIMEOUT,
                        _ = release.notified() => StatusCode::OK,
                    }
                }
            }
        }),
    );
    let request = tokio::spawn(
        shield(app, drain.clone()).oneshot(Request::get("/read").body(Body::empty()).unwrap()),
    );
    bounded(entered.notified()).await;
    drain.close_admission();
    assert!(
        tokio::time::timeout(Duration::from_millis(100), drain.joined())
            .await
            .is_err()
    );
    release.notify_one();
    assert_eq!(request.await.unwrap().unwrap().status(), StatusCode::OK);
    bounded(drain.joined()).await;
}
#[tokio::test]
async fn panic_still_returns_500_and_releases_drain_guard() {
    let drain = Arc::new(Drain::default());
    let app = Router::new().route(
        "/panic",
        get(|| async {
            panic!("injected panic");
            #[allow(unreachable_code)]
            StatusCode::OK
        }),
    );
    let response = shield(app, drain.clone())
        .oneshot(Request::get("/panic").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    drain.close_admission();
    bounded(drain.joined()).await;
}
#[tokio::test]
async fn closed_admission_still_rejects_new_requests() {
    let drain = Arc::new(Drain::default());
    drain.close_admission();
    let app = Router::new().route("/read", get(|| async { StatusCode::OK }));
    let response = shield(app, drain.clone())
        .oneshot(Request::get("/read").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    bounded(drain.joined()).await;
}
#[tokio::test]
async fn actual_http_disconnect_cancels_opted_in_read() {
    use tokio::io::AsyncWriteExt;
    let lock = Arc::new(RwLock::new(()));
    let entered = Arc::new(Notify::new());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (stop, stopping) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(serve(
        listener,
        read_route(lock.clone(), entered.clone()),
        Duration::from_millis(10),
        async {
            let _ = stopping.await;
        },
    ));
    let mut client = TcpStream::connect(address).await.unwrap();
    client
        .write_all(b"GET /read HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .await
        .unwrap();
    bounded(entered.notified()).await;
    let client = client.into_std().unwrap();
    client.shutdown(Shutdown::Both).unwrap();
    drop(client);
    bounded(async {
        let _writer = lock.write().await;
    })
    .await;
    stop.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(2), server)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}
