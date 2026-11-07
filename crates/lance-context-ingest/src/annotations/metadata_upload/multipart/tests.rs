//! Exercise the real Azure SDK with an injected HTTP transport. Inspect the
//! original block IDs and Bytes buffers, including reordered part completion.
use super::*;
use futures::TryStreamExt;
use object_store::{
    azure::MicrosoftAzureBuilder,
    client::{
        HttpClient, HttpConnector, HttpError, HttpErrorKind, HttpRequest, HttpResponse,
        HttpResponseBody, HttpService,
    },
    Attribute, ClientOptions, RetryConfig,
};
use std::sync::Mutex;

#[derive(Clone, Copy, Debug)]
enum Fault {
    Status(u16),
    Timeout,
    Pending,
}

#[derive(Debug)]
struct Call {
    block: Option<String>,
    bytes: Vec<Bytes>,
    content_type: Option<String>,
    tags: Option<String>,
}

#[derive(Debug)]
struct TransportState {
    part_faults: Vec<Fault>,
    completion_faults: Mutex<Vec<Fault>>,
    calls: Mutex<Vec<Call>>,
    active: Arc<AtomicU64>,
}

#[derive(Debug, Clone)]
struct Transport(Arc<TransportState>);

impl HttpConnector for Transport {
    fn connect(&self, _: &ClientOptions) -> object_store::Result<HttpClient> {
        Ok(HttpClient::new(self.clone()))
    }
}

struct Active(Arc<AtomicU64>);
impl Drop for Active {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

#[async_trait::async_trait]
impl HttpService for Transport {
    async fn call(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
        self.0.active.fetch_add(1, Ordering::SeqCst);
        let _active = Active(self.0.active.clone());
        assert_eq!(request.method().as_str(), "PUT");
        let url = url::Url::parse(&request.uri().to_string()).unwrap();
        let block = url
            .query_pairs()
            .find(|(k, _)| k == "blockid")
            .map(|(_, v)| v.into_owned());
        let comp = url.query_pairs().find(|(k, _)| k == "comp").unwrap().1;
        assert_eq!(
            comp,
            if block.is_some() {
                "block"
            } else {
                "blocklist"
            }
        );
        let content_type = request
            .headers()
            .get("x-ms-blob-content-type")
            .map(|v| v.to_str().unwrap().to_owned());
        let tags = request
            .headers()
            .get("x-ms-tags")
            .map(|v| v.to_str().unwrap().to_owned());
        let bytes = HttpResponseBody::new(request.into_body())
            .bytes_stream()
            .try_collect()
            .await?;
        let fault = {
            let mut calls = self.0.calls.lock().unwrap();
            let fault = if block.is_some() {
                self.0
                    .part_faults
                    .get(calls.iter().filter(|c| c.block == block).count())
                    .copied()
            } else {
                self.0
                    .completion_faults
                    .lock()
                    .unwrap()
                    .get(calls.iter().filter(|c| c.block.is_none()).count())
                    .copied()
            };
            calls.push(Call {
                block,
                bytes,
                content_type,
                tags,
            });
            fault
        };
        let status = match fault {
            Some(Fault::Timeout) => {
                return Err(HttpError::new(
                    HttpErrorKind::Timeout,
                    std::io::Error::other("injected timeout"),
                ))
            }
            Some(Fault::Pending) => return futures::future::pending().await,
            Some(Fault::Status(code)) => code,
            None => 201,
        };
        let mut response =
            HttpResponse::new(HttpResponseBody::from("injected response".to_owned()));
        *response.status_mut() = status.try_into().unwrap();
        response
            .headers_mut()
            .insert("etag", "\"test-etag\"".parse().unwrap());
        Ok(response)
    }
}

fn fixture(
    part_faults: Vec<Fault>,
    completion_fault: Option<Fault>,
    sdk_retries: usize,
) -> Result<(Scope, UploadStore, Arc<TransportState>)> {
    let transport = Arc::new(TransportState {
        part_faults,
        completion_faults: Mutex::new(completion_fault.into_iter().collect()),
        calls: Mutex::new(Vec::new()),
        active: Arc::new(AtomicU64::new(0)),
    });
    let inner = MicrosoftAzureBuilder::new()
        .with_account("account")
        .with_container_name("container")
        .with_skip_signature(true)
        .with_retry(RetryConfig {
            max_retries: sdk_retries,
            ..Default::default()
        })
        .with_http_connector(Transport(transport.clone()))
        .build()?;
    let (lifetime, stopped) = watch::channel(());
    let enabled = Arc::new(AtomicBool::new(true));
    let counters = Arc::new(Counters::default());
    let scope = Scope {
        lifetime: Some(lifetime),
        enabled: enabled.clone(),
        counters: counters.clone(),
    };
    let store = UploadStore {
        inner: Arc::new(inner),
        base: Path::from("table"),
        stopped,
        enabled,
        counters,
    };
    Ok((scope, store, transport))
}

async fn upload(store: &UploadStore) -> Result<Box<dyn MultipartUpload>> {
    Ok(store
        .put_multipart_opts(
            &Path::from("table/body/data/column.lance"),
            PutMultipartOptions::default(),
        )
        .await?)
}

#[tokio::test]
async fn multipart_throttle_preserves_part_ids_shared_bytes_and_completion_order() -> Result<()> {
    let (scope, store, transport) = fixture(vec![Fault::Status(429), Fault::Status(503)], None, 0)?;
    let opts = PutMultipartOptions {
        attributes: [(
            Attribute::ContentType,
            object_store::AttributeValue::from("application/test"),
        )]
        .into_iter()
        .collect(),
        ..Default::default()
    };
    let mut upload = store
        .put_multipart_opts(&Path::from("table/body/data/column.lance"), opts)
        .await?;
    let original = [
        Bytes::from(vec![1; 8192]),
        Bytes::from(vec![2; 4096]),
        Bytes::from(vec![3; 1024]),
    ];
    let first = upload.put_part(original[..2].iter().cloned().collect());
    let second = upload.put_part(original[2].clone().into());
    tokio::time::timeout(Duration::from_secs(8), async {
        futures::try_join!(second, first)
    })
    .await??;
    upload.complete().await?;
    let calls = transport.calls.lock().unwrap();
    assert_eq!(calls.len(), 7);
    let first_calls: Vec<_> = calls
        .iter()
        .filter(|c| c.block.is_some() && c.bytes.len() == 2)
        .collect();
    let second_calls: Vec<_> = calls
        .iter()
        .filter(|c| c.block.is_some() && c.bytes.len() == 1)
        .collect();
    for (parts, bytes) in [
        (&first_calls, &original[..2]),
        (&second_calls, &original[2..]),
    ] {
        assert_eq!(parts.len(), 3);
        for call in parts {
            assert_eq!(call.block, parts[0].block);
            assert_eq!(call.bytes, bytes);
            for (actual, original) in call.bytes.iter().zip(bytes) {
                assert_eq!(actual.as_ptr(), original.as_ptr());
            }
        }
    }
    let complete = calls.last().unwrap();
    assert!(complete.block.is_none());
    assert_eq!(complete.content_type.as_deref(), Some("application/test"));
    let xml: Vec<u8> = complete
        .bytes
        .iter()
        .flat_map(|b| b.iter().copied())
        .collect();
    let xml = String::from_utf8(xml)?;
    let first_id = first_calls[0].block.as_ref().unwrap();
    let second_id = second_calls[0].block.as_ref().unwrap();
    assert_ne!(first_id, second_id);
    assert!(xml.find(first_id).unwrap() < xml.find(second_id).unwrap());
    // Preserve the pinned SDK's completion response-loss fix as well.
    assert_eq!(xml.matches("<Latest>").count(), 2);
    assert_eq!(scope.observation().waiting_part_retries, 0);
    assert_eq!(scope.observation().part_retry_decisions, 4);
    assert_eq!(scope.observation().put_resubmissions, 0);
    Ok(())
}

#[tokio::test]
async fn multipart_throttle_starts_only_after_sdk_retries_are_exhausted() -> Result<()> {
    let (scope, store, transport) = fixture(vec![Fault::Status(503); 3], None, 2)?;
    let mut upload = upload(&store).await?;
    tokio::time::timeout(
        Duration::from_secs(5),
        upload.put_part(Bytes::from_static(b"x").into()),
    )
    .await??;
    assert_eq!(transport.calls.lock().unwrap().len(), 4);
    assert_eq!(scope.observation().part_retry_decisions, 1);
    Ok(())
}

#[tokio::test]
async fn multipart_throttle_leaves_other_errors_to_sdk() -> Result<()> {
    for fault in [Fault::Status(403), Fault::Status(500), Fault::Timeout] {
        let (scope, store, transport) = fixture(vec![fault], None, 0)?;
        let mut upload = upload(&store).await?;
        assert!(upload
            .put_part(Bytes::from_static(b"x").into())
            .await
            .is_err());
        assert_eq!(transport.calls.lock().unwrap().len(), 1);
        assert_eq!(scope.observation().part_retry_decisions, 0);
    }
    for fault in [Fault::Status(403), Fault::Status(500), Fault::Timeout] {
        let (scope, store, transport) = fixture(vec![], Some(fault), 0)?;
        let mut upload = upload(&store).await?;
        upload.put_part(Bytes::from_static(b"x").into()).await?;
        assert!(upload.complete().await.is_err());
        assert_eq!(transport.calls.lock().unwrap().len(), 2);
        assert_eq!(scope.observation().part_retry_decisions, 0);
        assert_eq!(scope.observation().completion_retry_decisions, 0);
    }
    Ok(())
}

#[tokio::test]
async fn completion_throttle_retains_original_blocks_bytes_and_options() -> Result<()> {
    let (scope, store, transport) = fixture(vec![], None, 0)?;
    *transport.completion_faults.lock().unwrap() = vec![Fault::Status(503), Fault::Status(429)];
    let mut tags = object_store::TagSet::default();
    tags.push("purpose", "completion-test");
    let opts = PutMultipartOptions {
        attributes: [(
            Attribute::ContentType,
            object_store::AttributeValue::from("application/test"),
        )]
        .into_iter()
        .collect(),
        tags,
        ..Default::default()
    };
    let mut upload = store
        .put_multipart_opts(&Path::from("table/body/data/column.lance"), opts)
        .await?;
    let first = upload.put_part(Bytes::from_static(b"first").into());
    let second = upload.put_part(Bytes::from_static(b"second").into());
    futures::try_join!(second, first)?;
    tokio::time::timeout(Duration::from_secs(8), upload.complete()).await??;
    let calls = transport.calls.lock().unwrap();
    assert_eq!(calls.len(), 5);
    let parts: Vec<_> = calls.iter().filter(|c| c.block.is_some()).collect();
    let completions: Vec<_> = calls.iter().filter(|c| c.block.is_none()).collect();
    assert_eq!(parts.len(), 2);
    assert_eq!(completions.len(), 3);
    let xml = String::from_utf8(
        completions[0]
            .bytes
            .iter()
            .flat_map(|b| b.iter().copied())
            .collect(),
    )?;
    assert_eq!(xml.matches("<Latest>").count(), 2);
    // Transport completed the second part first; the SDK preserves upload order.
    assert!(
        xml.find(parts[1].block.as_ref().unwrap()).unwrap()
            < xml.find(parts[0].block.as_ref().unwrap()).unwrap()
    );
    for call in &completions {
        assert_eq!(call.content_type.as_deref(), Some("application/test"));
        assert_eq!(call.tags.as_deref(), Some("purpose=completion-test"));
        assert_eq!(call.bytes, completions[0].bytes);
        for (actual, original) in call.bytes.iter().zip(&completions[0].bytes) {
            assert_eq!(actual.as_ptr(), original.as_ptr());
        }
    }
    assert_eq!(scope.observation().waiting_completion_retries, 0);
    assert_eq!(scope.observation().completion_retry_decisions, 2);
    assert_eq!(scope.observation().part_retry_decisions, 0);
    Ok(())
}

#[tokio::test]
async fn completion_throttle_follows_sdk_retries_and_cancels_backoff() -> Result<()> {
    let (scope, store, transport) = fixture(vec![], None, 2)?;
    *transport.completion_faults.lock().unwrap() = vec![Fault::Status(503); 3];
    let mut upload = upload(&store).await?;
    upload.put_part(Bytes::from_static(b"x").into()).await?;
    tokio::time::timeout(Duration::from_secs(5), upload.complete()).await??;
    assert_eq!(transport.calls.lock().unwrap().len(), 5);
    assert_eq!(scope.observation().completion_retry_decisions, 1);

    let (mut scope, store, transport) = fixture(vec![], Some(Fault::Status(503)), 0)?;
    let mut upload = self::upload(&store).await?;
    upload.put_part(Bytes::from_static(b"x").into()).await?;
    let mut completion = upload.complete();
    assert!(futures::poll!(&mut completion).is_pending());
    assert_eq!(scope.observation().waiting_completion_retries, 1);
    scope.cancel();
    assert!(tokio::time::timeout(Duration::from_secs(1), completion)
        .await?
        .is_err());
    assert_eq!(scope.observation().waiting_completion_retries, 0);
    assert_eq!(scope.observation().completion_retry_decisions, 0);
    assert_eq!(transport.calls.lock().unwrap().len(), 2);
    Ok(())
}

#[tokio::test]
async fn completion_throttle_is_disabled_during_owner_connection() -> Result<()> {
    let (scope, store, transport) = fixture(vec![], Some(Fault::Status(503)), 0)?;
    scope.enabled.store(false, Ordering::Release);
    let mut upload = upload(&store).await?;
    upload.put_part(Bytes::from_static(b"x").into()).await?;
    assert!(upload.complete().await.is_err());
    assert_eq!(scope.observation().completion_retry_decisions, 0);
    assert_eq!(transport.calls.lock().unwrap().len(), 2);
    Ok(())
}

#[tokio::test]
async fn multipart_throttle_owner_stop_cancels_request_or_backoff() -> Result<()> {
    for fault in [Fault::Status(503), Fault::Pending] {
        let (mut scope, store, transport) = fixture(vec![fault], None, 0)?;
        let mut upload = upload(&store).await?;
        let mut part = upload.put_part(Bytes::from_static(b"x").into());
        assert!(futures::poll!(&mut part).is_pending());
        assert_eq!(transport.calls.lock().unwrap().len(), 1);
        assert_eq!(
            scope.observation().waiting_part_retries,
            u64::from(matches!(fault, Fault::Status(_)))
        );
        scope.cancel();
        assert!(tokio::time::timeout(Duration::from_secs(1), part)
            .await?
            .is_err());
        assert_eq!(transport.active.load(Ordering::SeqCst), 0);
        assert_eq!(scope.observation().waiting_part_retries, 0);
        assert_eq!(scope.observation().part_retry_decisions, 0);
        assert!(upload.complete().await.is_err());
        assert!(upload
            .put_part(Bytes::from_static(b"later").into())
            .await
            .is_err());
        assert_eq!(transport.calls.lock().unwrap().len(), 1);
        upload.abort().await?;
    }
    Ok(())
}

#[tokio::test]
async fn multipart_throttle_scope_drop_cancels_completion() -> Result<()> {
    let (scope, store, transport) = fixture(vec![], Some(Fault::Pending), 0)?;
    let mut upload = upload(&store).await?;
    upload.put_part(Bytes::from_static(b"x").into()).await?;
    let mut completion = upload.complete();
    assert!(futures::poll!(&mut completion).is_pending());
    assert_eq!(transport.active.load(Ordering::SeqCst), 1);
    drop(scope);
    assert!(tokio::time::timeout(Duration::from_secs(1), completion)
        .await?
        .is_err());
    assert_eq!(transport.active.load(Ordering::SeqCst), 0);
    assert_eq!(transport.calls.lock().unwrap().len(), 2);
    Ok(())
}

#[tokio::test]
async fn multipart_throttle_is_disabled_during_owner_connection() -> Result<()> {
    let (scope, store, transport) = fixture(vec![Fault::Status(503)], None, 0)?;
    scope.enabled.store(false, Ordering::Release);
    let mut upload = upload(&store).await?;
    assert!(upload
        .put_part(Bytes::from_static(b"x").into())
        .await
        .is_err());
    assert_eq!(transport.calls.lock().unwrap().len(), 1);
    assert_eq!(scope.observation().part_retry_decisions, 0);
    Ok(())
}
