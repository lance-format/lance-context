//! Bound stalled uploads without imposing a deadline on merge or query execution.
use axum::{
    body::{Body, HttpBody},
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use http_body_util::BodyExt;
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};
use tower_http::timeout::{TimeoutBody, TimeoutError};

pub(crate) async fn guard_request_body(
    State(idle): State<Duration>,
    request: Request,
    next: Next,
) -> Response {
    if request.body().is_end_stream() {
        return next.run(request).await;
    }
    let expired = Arc::new(AtomicBool::new(false));
    let body_expired = expired.clone();
    let method = request.method().clone();
    let path = request.uri().path().to_owned();
    let (parts, body) = request.into_parts();
    // TimeoutBody starts its timer when the consumer requests a frame, and
    // disarms it after delivery. Handler backpressure between frames therefore
    // does not count as upload idleness.
    let body = TimeoutBody::new(idle, body).map_err(move |error| {
        if error.is::<TimeoutError>() {
            body_expired.store(true, Ordering::Relaxed);
        }
        error
    });
    let response = next.run(Request::from_parts(parts, Body::new(body))).await;
    if expired.load(Ordering::Relaxed) {
        metrics::counter!("http_request_body_timeouts_total").increment(1);
        tracing::warn!(%method, %path, idle_seconds = idle.as_secs(), "request body idle timeout");
        // Extractors normally classify body errors as 400; an upload that made
        // no progress is a request timeout, which the client may retry.
        (StatusCode::REQUEST_TIMEOUT, "request body idle timeout").into_response()
    } else {
        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Bytes, middleware, routing::post, Router};
    use futures::stream;
    use std::convert::Infallible;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tower::ServiceExt;

    fn app(idle: Duration) -> Router {
        Router::new()
            .route("/", post(|body: Bytes| async move { body }))
            .layer(middleware::from_fn_with_state(idle, guard_request_body))
    }

    #[tokio::test(start_paused = true)]
    async fn progressing_upload_may_exceed_total_idle_duration() {
        let body = Body::from_stream(stream::unfold(0, |i| async move {
            if i == 5 {
                None
            } else {
                tokio::time::sleep(Duration::from_secs(60)).await;
                Some((Ok::<_, Infallible>(Bytes::from_static(b"x")), i + 1))
            }
        }));
        let before = tokio::time::Instant::now();
        let response = app(Duration::from_secs(120))
            .oneshot(Request::post("/").body(body).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(before.elapsed() >= Duration::from_secs(300));
        assert_eq!(
            axum::body::to_bytes(response.into_body(), 100)
                .await
                .unwrap(),
            "xxxxx"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn completed_body_does_not_time_out_a_slow_handler() {
        let router = Router::new()
            .route(
                "/",
                post(|body: Bytes| async move {
                    tokio::time::sleep(Duration::from_secs(600)).await;
                    body
                }),
            )
            .layer(middleware::from_fn_with_state(
                Duration::from_secs(120),
                guard_request_body,
            ));
        let response = router
            .oneshot(Request::post("/").body(Body::from("complete")).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test(start_paused = true)]
    async fn consumer_backpressure_is_not_upload_idleness() {
        let router = Router::new()
            .route(
                "/",
                post(|request: Request| async move {
                    let mut body = request.into_body();
                    let first = body.frame().await.unwrap().unwrap();
                    tokio::time::sleep(Duration::from_secs(600)).await;
                    let second = body.frame().await.unwrap().unwrap();
                    assert_eq!(first.data_ref().unwrap(), "a");
                    assert_eq!(second.data_ref().unwrap(), "b");
                    StatusCode::OK
                }),
            )
            .layer(middleware::from_fn_with_state(
                Duration::from_secs(120),
                guard_request_body,
            ));
        let body = Body::from_stream(stream::iter([
            Ok::<_, Infallible>(Bytes::from_static(b"a")),
            Ok(Bytes::from_static(b"b")),
        ]));
        let response = router
            .oneshot(Request::post("/").body(body).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn other_body_errors_keep_the_extractor_status() {
        let body = Body::from_stream(stream::once(async {
            Err::<Bytes, _>(std::io::Error::other("upload connection lost"))
        }));
        let response = app(Duration::from_secs(120))
            .oneshot(Request::post("/").body(body).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn partial_upload_times_out_and_graceful_shutdown_finishes() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let admitted = Arc::new(tokio::sync::Notify::new());
        let notify = admitted.clone();
        let router = app(Duration::from_millis(200)).layer(middleware::from_fn(
            move |request: Request, next: Next| {
                let notify = notify.clone();
                async move {
                    notify.notify_one();
                    next.run(request).await
                }
            },
        ));
        let (stop, stopped) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(async { stopped.await.unwrap() })
                .await
                .unwrap();
        });
        let mut client = tokio::net::TcpStream::connect(address).await.unwrap();
        client
            .write_all(b"POST / HTTP/1.1\r\nHost: localhost\r\nContent-Length: 100\r\n\r\nx")
            .await
            .unwrap();
        admitted.notified().await;
        stop.send(()).unwrap();
        let mut response = String::new();
        tokio::time::timeout(Duration::from_secs(5), client.read_to_string(&mut response))
            .await
            .unwrap()
            .unwrap();
        assert!(response.starts_with("HTTP/1.1 408"), "{response}");
        tokio::time::timeout(Duration::from_secs(5), server)
            .await
            .unwrap()
            .unwrap();
    }
}
