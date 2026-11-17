//! Fault qualification through the actual Azure SDK over loopback HTTP.
//! These tests do not substitute for real Azure consistency/load qualification.
use super::*;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

async fn request(socket: &mut TcpStream) -> String {
    let mut bytes = Vec::new();
    let header_end = loop {
        let mut chunk = [0; 1024];
        let count = tokio::time::timeout(Duration::from_secs(5), socket.read(&mut chunk))
            .await
            .unwrap()
            .unwrap();
        assert!(count > 0 && bytes.len() < 16384);
        bytes.extend_from_slice(&chunk[..count]);
        if let Some(offset) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
            break offset + 4;
        }
    };
    let headers = String::from_utf8(bytes[..header_end].to_vec()).unwrap();
    let length: usize = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse().unwrap())
        })
        .unwrap_or(0);
    assert!(length <= 4096);
    while bytes.len() < header_end + length {
        let mut chunk = [0; 128];
        let count = tokio::time::timeout(Duration::from_secs(5), socket.read(&mut chunk))
            .await
            .unwrap()
            .unwrap();
        assert!(count > 0);
        bytes.extend_from_slice(&chunk[..count]);
    }
    String::from_utf8(bytes).unwrap()
}

async fn accept(listener: &TcpListener) -> TcpStream {
    tokio::time::timeout(Duration::from_secs(5), listener.accept())
        .await
        .unwrap()
        .unwrap()
        .0
}

async fn respond(socket: &mut TcpStream, status: &str, body: &str) {
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nContent-Type: application/xml\r\nConnection: close\r\nx-ms-request-id: wal-lifetime-test\r\n\r\n{body}",
        body.len()
    );
    socket.write_all(response.as_bytes()).await.unwrap();
}

async fn azure_store(endpoint: String) -> Arc<ObjectStore> {
    let azure = object_store::azure::MicrosoftAzureBuilder::new()
        .with_account("test")
        .with_container_name("wal")
        .with_endpoint(endpoint)
        .with_skip_signature(true)
        .with_allow_http(true)
        .with_retry(object_store::RetryConfig {
            max_retries: 0,
            ..Default::default()
        })
        .build()
        .unwrap();
    // Use the remote-style directory path, never the local filesystem shortcut.
    let (wrapper, _) = ObjectStore::from_uri("memory:///").await.unwrap();
    let mut wrapper = wrapper.as_ref().clone();
    wrapper.inner = Arc::new(azure);
    Arc::new(wrapper)
}

#[tokio::test]
async fn ambiguous_azure_put_does_not_start_read_and_attempts_cleanup() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let mut socket = accept(&listener).await;
        let put = request(&mut socket).await;
        assert!(put.starts_with("PUT "));
        assert!(put.to_ascii_lowercase().contains("if-none-match: *"));
        let key = put
            .lines()
            .next()
            .unwrap()
            .split_whitespace()
            .nth(1)
            .unwrap()
            .to_owned();
        assert!(key.contains("_wal_readers_v1/"));
        assert_eq!(put.split_once("\r\n\r\n").unwrap().1.len(), 32);
        // The backend accepted the bytes, but the client never receives an ACK.
        socket.shutdown().await.unwrap();
        drop(socket);
        let mut cleanup = accept(&listener).await;
        let delete = request(&mut cleanup).await;
        // The pinned SDK uses native DELETE for this one registration.
        // Cleanup must target exactly the registration whose PUT was uncertain.
        assert_eq!(
            delete.lines().next().unwrap(),
            format!("DELETE {key} HTTP/1.1")
        );
        assert!(delete.split_once("\r\n\r\n").unwrap().1.is_empty());
        // Cleanup is best effort. A failed cleanup still cannot authorize the
        // rejected read; the abandoned registration must expire separately.
        respond(
            &mut cleanup,
            "500 Internal Server Error",
            "<Error><Code>InternalError</Code><Message>injected cleanup failure</Message></Error>",
        )
        .await;
    });
    let store = azure_store(endpoint).await;
    let read_started = AtomicBool::new(false);
    let result = protect(store, &Path::from("table"), Policy::default(), async {
        read_started.store(true, Ordering::SeqCst);
        Ok(())
    })
    .await;
    assert!(result.is_err());
    assert!(!read_started.load(Ordering::SeqCst));
    server.await.unwrap();
}

#[tokio::test]
async fn azure_empty_page_then_failed_continuation_never_authorizes_deletion() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let mut first = accept(&listener).await;
        let first_request = request(&mut first).await;
        let first_line = first_request.lines().next().unwrap();
        assert!(first_line.starts_with("GET "));
        assert!(
            first_line.contains("prefix=table%2F_wal_readers_v1"),
            "{first_line}"
        );
        assert!(!first_line.contains("marker="));
        respond(
            &mut first,
            "200 OK",
            "<EnumerationResults><Blobs/><NextMarker>page-2</NextMarker></EnumerationResults>",
        )
        .await;
        let mut second = accept(&listener).await;
        let second_request = request(&mut second).await;
        let second_line = second_request.lines().next().unwrap();
        assert!(second_line.starts_with("GET "));
        assert!(
            second_line.contains("prefix=table%2F_wal_readers_v1"),
            "{second_line}"
        );
        assert!(second_line.contains("marker=page-2"), "{second_line}");
        respond(&mut second, "500 Internal Server Error", "<Error><Code>InternalError</Code><Message>injected continuation failure</Message></Error>").await;
        // Any generation listing/deletion would be a third request. Retain the
        // listener until reclaim finishes and assert none was attempted below.
        listener
    });
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        reclaim(
            azure_store(endpoint).await,
            Path::from("table"),
            vec![Path::from("table/_mem_wal/shard/gen_0")],
        ),
    )
    .await
    .unwrap();
    assert!(result.is_err());
    let listener = server.await.unwrap();
    assert!(
        tokio::time::timeout(Duration::from_millis(100), listener.accept())
            .await
            .is_err()
    );
}
