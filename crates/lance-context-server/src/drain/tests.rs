use super::*;
use axum::{
    body::{Body, Bytes},
    middleware,
    routing::{get, post},
};
use std::{
    convert::Infallible,
    sync::atomic::{AtomicBool, Ordering},
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    sync::oneshot,
    task::JoinHandle,
};
use tower::ServiceExt;

async fn start(
    app: Router,
    idle: Duration,
) -> (SocketAddr, oneshot::Sender<()>, JoinHandle<io::Result<()>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel();
    let server = tokio::spawn(serve(listener, app, idle, async {
        let _ = rx.await;
    }));
    (addr, tx, server)
}

async fn request(addr: SocketAddr, path: &str) -> TcpStream {
    let socket = tokio::net::TcpSocket::new_v4().unwrap();
    // Keep the receiver's window small enough to exercise real backpressure
    // without allocating a large response in either process.
    socket.set_recv_buffer_size(4096).unwrap();
    let mut stream = socket.connect(addr).await.unwrap();
    stream
        .write_all(
            format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
                .as_bytes(),
        )
        .await
        .unwrap();
    stream
}

#[tokio::test]
async fn disconnect_does_not_cancel_accepted_mutation_or_skip_shutdown_join() {
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let committed = Arc::new(AtomicBool::new(false));
    let app = Router::new().route(
        "/",
        post({
            let (entered, release, committed) =
                (entered.clone(), release.clone(), committed.clone());
            move || {
                let (entered, release, committed) =
                    (entered.clone(), release.clone(), committed.clone());
                async move {
                    entered.notify_one();
                    release.notified().await;
                    committed.store(true, Ordering::SeqCst);
                    StatusCode::OK
                }
            }
        }),
    );
    let (addr, shutdown, mut server) = start(app, Duration::from_millis(30)).await;
    let mut client = TcpStream::connect(addr).await.unwrap();
    client
        .write_all(b"POST / HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n")
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(2), entered.notified())
        .await
        .unwrap();
    drop(client);
    shutdown.send(()).unwrap();
    assert!(
        tokio::time::timeout(Duration::from_millis(150), &mut server)
            .await
            .is_err()
    );
    assert!(!committed.load(Ordering::SeqCst));
    release.notify_one();
    tokio::time::timeout(Duration::from_secs(2), server)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(committed.load(Ordering::SeqCst));
}

#[tokio::test]
async fn closed_admission_rejects_without_running_handler() {
    let drain = Arc::new(Drain::default());
    let app = Router::new()
        .route(
            "/",
            get(|| async {
                panic!("must not run");
                #[allow(unreachable_code)]
                StatusCode::OK
            }),
        )
        .layer(middleware::from_fn_with_state(drain.clone(), track_handler));
    drain.close_admission();
    let response = app
        .oneshot(Request::get("/").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    drain.joined().await;
}

#[tokio::test]
async fn panicking_and_failed_handlers_release_all_join_waiters() {
    let drain = Arc::new(Drain::default());
    let app = Router::new()
        .route(
            "/panic",
            get(|| async {
                panic!("injected handler panic");
                #[allow(unreachable_code)]
                StatusCode::OK
            }),
        )
        .route("/error", get(|| async { StatusCode::BAD_REQUEST }))
        .layer(middleware::from_fn_with_state(drain.clone(), track_handler));
    for (path, status) in [
        ("/panic", StatusCode::INTERNAL_SERVER_ERROR),
        ("/error", StatusCode::BAD_REQUEST),
    ] {
        let response = app
            .clone()
            .oneshot(Request::get(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), status);
    }
    // Both observers must wake, including a waiter registered just before drop.
    let guard = drain.admit(RequestInfo::default()).unwrap();
    drain.close_admission();
    let a = {
        let d = drain.clone();
        tokio::spawn(async move { d.joined().await })
    };
    let b = {
        let d = drain.clone();
        tokio::spawn(async move { d.joined().await })
    };
    tokio::task::yield_now().await;
    drop(guard);
    tokio::time::timeout(Duration::from_secs(2), async {
        a.await.unwrap();
        b.await.unwrap();
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn inventory_keeps_disconnected_mutation_and_distinguishes_targets() {
    let drain = Arc::new(Drain::default());
    let entered = Arc::new(tokio::sync::Semaphore::new(0));
    let release = Arc::new(tokio::sync::Semaphore::new(0));
    let handler = {
        let (entered, release) = (entered.clone(), release.clone());
        move |axum::extract::Path(name): axum::extract::Path<String>| {
            let (entered, release) = (entered.clone(), release.clone());
            async move {
                // Metadata extraction must preserve the router's decoded params.
                assert!(["table-one", "table-two"].contains(&name.as_str()));
                entered.add_permits(1);
                release.acquire().await.unwrap().forget();
                StatusCode::NO_CONTENT
            }
        }
    };
    let app = Router::new()
        .route("/api/v1/internal/merge-wal/{name}", post(handler.clone()))
        .route("/api/v1/rollouts/{name}", axum::routing::delete(handler))
        .layer(middleware::from_fn_with_state(drain.clone(), track_handler));
    let merge = tokio::spawn(
        app.clone().oneshot(
            Request::post("/api/v1/internal/merge-wal/table%2Done?token=secret-query")
                .header("authorization", "secret-header")
                .body(Body::from("secret-body"))
                .unwrap(),
        ),
    );
    let deletion = tokio::spawn(
        app.oneshot(
            Request::delete("/api/v1/rollouts/table-two")
                .body(Body::empty())
                .unwrap(),
        ),
    );
    tokio::time::timeout(Duration::from_secs(2), entered.acquire_many(2))
        .await
        .unwrap()
        .unwrap()
        .forget();
    merge.abort();
    assert!(merge.await.unwrap_err().is_cancelled());
    drain.close_admission();
    {
        let admission = drain.admission.lock().unwrap();
        assert_eq!(admission.active.len(), 2);
        let values: Vec<_> = admission.active.values().collect();
        assert!(values.iter().any(|r| r.method == "POST"
            && r.target.as_deref() == Some("table-one")
            && r.route.as_deref() == Some("/api/v1/internal/merge-wal/{name}")));
        assert!(values
            .iter()
            .any(|r| r.method == "DELETE" && r.target.as_deref() == Some("table-two")));
        assert!(!format!("{values:?}").contains("secret"));
    }
    // The detached mutation is still part of the join even without its caller.
    assert!(
        tokio::time::timeout(Duration::from_millis(20), drain.joined())
            .await
            .is_err()
    );
    release.add_permits(2);
    deletion.await.unwrap().unwrap();
    tokio::time::timeout(Duration::from_secs(2), drain.joined())
        .await
        .unwrap();
    assert!(drain.admission.lock().unwrap().active.is_empty());
}

#[tokio::test]
async fn inventory_does_not_collect_record_ids_or_invalid_store_identifiers() {
    let drain = Arc::new(Drain::default());
    let app = Router::new()
        .route(
            "/api/v1/rollouts/{name}/records/{id}/blob",
            get({
                let drain = drain.clone();
                move || {
                    let drain = drain.clone();
                    async move {
                        let admission = drain.admission.lock().unwrap();
                        let info = admission.active.values().next().unwrap();
                        assert_eq!(info.method, "GET");
                        assert_eq!(
                            info.route.as_deref(),
                            Some("/api/v1/rollouts/{name}/records/{id}/blob")
                        );
                        assert!(info.target.is_none());
                        assert!(!format!("{info:?}").contains("private-record"));
                        StatusCode::OK
                    }
                }
            }),
        )
        .layer(middleware::from_fn_with_state(drain.clone(), track_handler));
    for name in ["bad%2Fname", "%FF"] {
        let response = app
            .clone()
            .oneshot(
                Request::get(format!(
                    "/api/v1/rollouts/{name}/records/private-record/blob?credential=secret"
                ))
                .body(Body::empty())
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
    drain.joined().await;
}

#[test]
fn shutdown_log_cap_does_not_discard_unreported_handlers() {
    // Tracing's callsite interest cache is process-wide. Other drain tests emit
    // these same events without a subscriber, so capture logs in an isolated
    // test process instead of racing their callsite registration.
    const ISOLATED: &str = "LANCE_CONTEXT_DRAIN_INVENTORY_LOG_TEST";
    if std::env::var_os(ISOLATED).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "drain::tests::shutdown_log_cap_does_not_discard_unreported_handlers",
                "--nocapture",
            ])
            .env(ISOLATED, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "isolated log test failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }
    #[derive(Clone)]
    struct Log(Arc<Mutex<Vec<u8>>>);
    impl io::Write for Log {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let log = Log(Arc::default());
    let sink = log.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .without_time()
        .with_writer(move || sink.clone())
        .finish();
    let drain = Arc::new(Drain::default());
    let handlers: Vec<_> = (0..130)
        .map(|_| drain.admit(RequestInfo::default()).unwrap())
        .collect();
    tracing::subscriber::with_default(subscriber, || drain.close_admission());
    let output = String::from_utf8(log.0.lock().unwrap().clone()).unwrap();
    assert_eq!(
        output
            .matches("HTTP handler outstanding at shutdown")
            .count(),
        128
    );
    assert!(output.contains("active_http_handlers=130"));
    assert!(output.contains("inventory_truncated=true"));
    assert_eq!(drain.admission.lock().unwrap().active.len(), 130);
    assert!(drain.admit(RequestInfo::default()).is_none());
    drop(handlers);
    assert!(drain.admission.lock().unwrap().active.is_empty());
}

#[tokio::test]
async fn stalled_response_is_retired_only_after_all_handlers_join() {
    let release = Arc::new(Notify::new());
    let entered = Arc::new(Notify::new());
    let produced = Arc::new(Notify::new());
    struct Dropped(Arc<AtomicBool>);
    impl Drop for Dropped {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    let dropped = Arc::new(AtomicBool::new(false));
    let app = Router::new()
        .route(
            "/download",
            get({
                let produced = produced.clone();
                let dropped = dropped.clone();
                move || {
                    let produced = produced.clone();
                    let held = Dropped(dropped.clone());
                    async move {
                        Body::from_stream(futures::stream::unfold((0, held), move |(i, held)| {
                            let produced = produced.clone();
                            async move {
                                if i == 1024 {
                                    None
                                } else {
                                    produced.notify_one();
                                    Some((
                                        Ok::<_, Infallible>(Bytes::from(vec![b'x'; 64 * 1024])),
                                        (i + 1, held),
                                    ))
                                }
                            }
                        }))
                    }
                }
            }),
        )
        .route(
            "/write",
            get({
                let (release, entered) = (release.clone(), entered.clone());
                move || {
                    let (release, entered) = (release.clone(), entered.clone());
                    async move {
                        entered.notify_one();
                        release.notified().await;
                        StatusCode::OK
                    }
                }
            }),
        );
    let (addr, shutdown, mut server) = start(app, Duration::from_millis(50)).await;
    let download = request(addr, "/download").await;
    tokio::time::timeout(Duration::from_secs(2), produced.notified())
        .await
        .unwrap();
    let _write = request(addr, "/write").await;
    tokio::time::timeout(Duration::from_secs(2), entered.notified())
        .await
        .unwrap();
    shutdown.send(()).unwrap();
    assert!(
        tokio::time::timeout(Duration::from_millis(250), &mut server)
            .await
            .is_err()
    );
    // The response must not have been cancelled while another admitted storage
    // future is still running, even after several response-idle intervals.
    assert!(!dropped.load(Ordering::SeqCst));
    release.notify_one();
    tokio::time::timeout(Duration::from_secs(5), server)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    // Server completed while the non-reading client still owns its connection.
    assert!(dropped.load(Ordering::SeqCst));
    drop(download);
}

#[tokio::test]
async fn progressing_download_survives_beyond_total_idle_duration() {
    let app = Router::new().route(
        "/",
        get(|| async {
            Body::from_stream(futures::stream::unfold(0, |i| async move {
                if i == 16 {
                    None
                } else {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    Some((Ok::<_, Infallible>(Bytes::from_static(b"progress")), i + 1))
                }
            }))
        }),
    );
    let idle = Duration::from_millis(100);
    let (addr, shutdown, server) = start(app, idle).await;
    let mut client = request(addr, "/").await;
    let mut first = [0; 1];
    client.read_exact(&mut first).await.unwrap();
    let since = Instant::now();
    shutdown.send(()).unwrap();
    let mut rest = Vec::new();
    tokio::time::timeout(Duration::from_secs(5), client.read_to_end(&mut rest))
        .await
        .unwrap()
        .unwrap();
    assert!(since.elapsed() > idle * 2);
    assert_eq!(
        rest.windows(b"progress".len())
            .filter(|w| *w == b"progress")
            .count(),
        16
    );
    server.await.unwrap().unwrap();
}

#[test]
fn response_idle_timeout_must_be_positive() {
    use clap::Parser;
    assert!(crate::config::ServerConfig::try_parse_from([
        "server",
        "--http-shutdown-response-idle-secs",
        "0"
    ])
    .is_err());
}

#[tokio::test]
async fn resumed_socket_writes_reset_idle_even_when_download_exceeds_it() {
    let idle = Duration::from_secs(1);
    let drain = Arc::new(Drain::default());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mut client = TcpStream::connect(listener.local_addr().unwrap())
        .await
        .unwrap();
    let mut listener = DrainListener::new(listener, drain.clone());
    let (mut output, _) = listener.accept().await;
    let connection = output.connection.clone();
    let total = 32 * 1024 * 1024;
    let sender = tokio::spawn(async move {
        let chunk = [b'x'; 64 * 1024];
        for _ in 0..total / chunk.len() {
            output.write_all(&chunk).await?;
        }
        output.shutdown().await
    });
    // Unlike the incremental-body test, force a real socket Pending first.
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if connection.blocked_since.lock().unwrap().is_some() {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    drain.close_admission();
    let retirement = tokio::spawn(async move { drain.retire_idle_responses(idle).await });
    let started = Instant::now();
    let received = tokio::time::timeout(Duration::from_secs(30), async {
        let mut buffer = [0; 64 * 1024];
        let mut received = 0;
        loop {
            let n = client.read(&mut buffer).await.unwrap();
            if n == 0 {
                break;
            }
            assert!(buffer[..n].iter().all(|b| *b == b'x'));
            received += n;
            tokio::time::sleep(Duration::from_millis(8)).await;
        }
        received
    })
    .await
    .unwrap();
    retirement.abort();
    let _ = retirement.await;
    sender.await.unwrap().unwrap();
    assert_eq!(received, total);
    assert!(started.elapsed() > idle * 2);
}
