//! Real Azure SDK + HTTP body deadlines, with a loopback storage responder.
//! This isolates polling starvation; it does not attribute cloud failures.
use super::*;
use object_store::{azure::MicrosoftAzureBuilder, ClientOptions, RetryConfig};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};

const BODY: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<EnumerationResults ServiceEndpoint="http://localhost/" ContainerName="container">
<Blobs><Blob><Name>owned/data/a.lance</Name><Properties>
<Last-Modified>Mon, 02 Nov 2026 00:00:00 GMT</Last-Modified><Etag>test</Etag>
<Content-Length>6</Content-Length><Content-Type>application/octet-stream</Content-Type>
<BlobType>BlockBlob</BlobType>
</Properties></Blob></Blobs><NextMarker /></EnumerationResults>"#;

async fn mechanism(background: bool) -> Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let endpoint = format!("http://{}", listener.local_addr()?);
    let (headers_sent, headers_received) = oneshot::channel();
    let (release, body_allowed) = oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await?;
        let mut request = Vec::new();
        while !request.ends_with(b"\r\n\r\n") {
            ensure!(request.len() < 16 << 10, "oversized loopback request");
            request.push(socket.read_u8().await?);
        }
        ensure!(request.starts_with(b"GET "), "not a read-only LIST");
        let headers = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/xml\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            BODY.len()
        );
        socket.write_all(headers.as_bytes()).await?;
        headers_sent.send(()).unwrap();
        body_allowed.await?;
        socket.write_all(BODY.as_bytes()).await?;
        socket.shutdown().await?;
        Ok::<_, anyhow::Error>(tokio::time::Instant::now())
    });
    let timeout = Duration::from_secs(2);
    let store = MicrosoftAzureBuilder::new()
        .with_account("account")
        .with_container_name("container")
        .with_endpoint(endpoint)
        .with_skip_signature(true)
        .with_client_options(
            ClientOptions::new()
                .with_allow_http(true)
                .with_timeout(timeout),
        )
        .with_retry(RetryConfig {
            max_retries: 0,
            ..Default::default()
        })
        .build()?;
    let mut listing = if background {
        super::super::listing::drive(store.list(None))
    } else {
        store.list(None)
    };
    let started = tokio::time::Instant::now();
    let (sender, mut progress) = progress(false);
    {
        let next = next_listing(
            &mut listing,
            &mut progress,
            started + Duration::from_secs(10),
        );
        tokio::pin!(next);
        tokio::select! {
            result = &mut next => panic!("LIST completed before body delivery: {result:?}"),
            result = headers_received => result?,
        }
        // Continue driving the real client after the HTTP headers. The body is
        // withheld until after production next_listing yields to backpressure.
        tokio::select! {
            result = &mut next => panic!("LIST completed without response body: {result:?}"),
            _ = tokio::time::sleep(Duration::from_millis(100)) => {},
        }
        sender.send_modify(|p| p.backpressure = Some(ApplyBackpressure::Staging));
        assert!(next.await.is_none());
    }
    release.send(()).unwrap();
    let delivered = server.await??;
    ensure!(
        delivered - started < timeout / 2,
        "loopback response arrived too late"
    );
    tokio::time::sleep_until(started + timeout + Duration::from_secs(1)).await;
    // Reacquire the same stream after simulating a busy publisher. No
    // timeout/retry configuration changes between the two cases. The candidate
    // uses the exact wrapper installed by Collector, not a separate test driver.
    let outcome = next_listing(
        &mut listing,
        &mut progress,
        tokio::time::Instant::now() + Duration::from_secs(1),
    )
    .await
    .context("completed response was hidden by backpressure")?;
    if background {
        let object = outcome?.context("listing lost its original object")?;
        assert_eq!(object.location, Path::from("owned/data/a.lance"));
        assert_eq!(object.size, 6);
    } else {
        let error = format!("{:#}", anyhow::Error::new(outcome.unwrap_err()));
        assert!(error.contains("Error getting list response body"));
        assert!(
            error.contains("timed out"),
            "unexpected loopback error chain: {error}"
        );
    }
    println!(
        "{}",
        serde_json::json!({"background":background,
        "body_delivered_before_deadline":true,"request_retries":0,
        "response_body_timeout":!background})
    );
    Ok(())
}

#[tokio::test]
async fn original_http_list_response_survives_independent_polling() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(15), mechanism(true)).await?
}

#[tokio::test]
async fn unpolled_http_list_expires_after_body_already_arrived() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(15), mechanism(false)).await?
}
