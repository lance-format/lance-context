use super::*;
use lance_io::object_store::StorageOptionsAccessor;
use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    time::Duration,
};
use tokio::sync::Notify;

#[derive(Clone, Copy, Debug)]
enum Failure {
    Io(ErrorKind),
    Invalid,
    External,
    Pending,
}

#[derive(Debug, Default)]
struct Provider {
    failure: Mutex<Option<Failure>>,
    calls: AtomicUsize,
    forced: AtomicUsize,
    cancelled: AtomicUsize,
    failed: Notify,
    options: HashMap<String, String>,
}

#[async_trait::async_trait]
impl StorageOptionsProvider for Provider {
    async fn fetch_storage_options(&self) -> lance::Result<Option<HashMap<String, String>>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let failure = *self.failure.lock().unwrap();
        if let Some(failure) = failure {
            self.failed.notify_one();
            return Err(match failure {
                Failure::Io(kind) => {
                    std::io::Error::new(kind, "injected credential failure").into()
                }
                Failure::Invalid => lance::Error::invalid_input("injected binding failure"),
                Failure::External => lance::Error::external(Box::new(std::io::Error::new(
                    ErrorKind::TimedOut,
                    "external caller failure",
                ))),
                Failure::Pending => {
                    struct Cancelled<'a>(&'a AtomicUsize);
                    impl Drop for Cancelled<'_> {
                        fn drop(&mut self) {
                            self.0.fetch_add(1, Ordering::SeqCst);
                        }
                    }
                    let _cancelled = Cancelled(&self.cancelled);
                    std::future::pending().await
                }
            });
        }
        Ok(Some(self.options.clone()))
    }

    async fn force_fetch_storage_options(&self) -> lance::Result<Option<HashMap<String, String>>> {
        self.forced.fetch_add(1, Ordering::SeqCst);
        self.fetch_storage_options().await
    }

    fn provider_id(&self) -> String {
        "renewal-test-origin".into()
    }
}

#[tokio::test]
async fn permanent_unknown_and_external_errors_do_not_start_renewal() {
    for failure in [
        Failure::Io(ErrorKind::PermissionDenied),
        Failure::Io(ErrorKind::InvalidInput),
        Failure::Io(ErrorKind::NotFound),
        Failure::Io(ErrorKind::Other),
        Failure::Io(ErrorKind::Interrupted),
        Failure::Invalid,
        Failure::External,
    ] {
        let provider = Arc::new(Provider::default());
        *provider.failure.lock().unwrap() = Some(failure);
        let renewal = RenewingStorageOptionsProvider::new(provider.clone());
        assert_eq!(renewal.provider_id(), provider.provider_id());
        let result = tokio::time::timeout(
            Duration::from_secs(1),
            renewal.force_fetch_storage_options(),
        )
        .await
        .expect("nontransient failure must return without a renewal loop");
        assert!(result.is_err());
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert_eq!(provider.forced.load(Ordering::SeqCst), 1);
        let observation = renewal.observation();
        assert_eq!(observation.provider_rounds, 1);
        assert_eq!(observation.retry_rounds, 0);
        assert_eq!(observation.retry_decisions, 0);
        assert_eq!(observation.failed_calls, 1);
        assert_eq!(observation.active_calls, 0);
        assert_eq!(observation.cancelled_calls, 0);
    }
}

#[tokio::test]
async fn cancelling_renewal_releases_accessor_and_stops_fetches() {
    for failure in [Failure::Io(ErrorKind::TimedOut), Failure::Pending] {
        let provider = Arc::new(Provider::default());
        *provider.failure.lock().unwrap() = Some(failure);
        let renewal = Arc::new(RenewingStorageOptionsProvider::new(provider.clone()));
        let accessor = Arc::new(StorageOptionsAccessor::with_provider(renewal.clone()));
        let task = tokio::spawn({
            let accessor = accessor.clone();
            async move { accessor.get_storage_options().await }
        });
        tokio::time::timeout(Duration::from_secs(1), provider.failed.notified())
            .await
            .unwrap();
        assert!(!task.is_finished());
        tokio::time::sleep(Duration::from_millis(10)).await;
        let observation = renewal.observation();
        assert_eq!(observation.active_calls, 1);
        assert_eq!(
            observation.backing_off,
            u64::from(!matches!(failure, Failure::Pending))
        );
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert_eq!(
            provider.cancelled.load(Ordering::SeqCst),
            usize::from(matches!(failure, Failure::Pending))
        );
        let observation = renewal.observation();
        assert_eq!(observation.active_calls, 0);
        assert_eq!(observation.backing_off, 0);
        assert_eq!(observation.cancelled_calls, 1);
        assert_eq!(observation.failed_calls, 0);
        if matches!(failure, Failure::Pending) {
            assert!(observation.provider_elapsed_micros >= 1_000);
        } else {
            assert!(observation.backoff_elapsed_micros >= 1_000);
        }
        let before = provider.calls.load(Ordering::SeqCst);
        *provider.failure.lock().unwrap() = None;
        tokio::time::timeout(Duration::from_secs(1), accessor.get_storage_options())
            .await
            .unwrap()
            .unwrap();
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert_eq!(provider.calls.load(Ordering::SeqCst), before + 1);
    }
}

#[tokio::test]
async fn forced_renewal_keeps_force_semantics_until_same_future_succeeds() {
    let provider = Arc::new(Provider::default());
    *provider.failure.lock().unwrap() = Some(Failure::Io(ErrorKind::WouldBlock));
    let renewal = RenewingStorageOptionsProvider::new(provider.clone());
    let task = tokio::spawn({
        let renewal = renewal.clone();
        async move { renewal.force_fetch_storage_options().await }
    });
    tokio::time::timeout(Duration::from_secs(1), provider.failed.notified())
        .await
        .unwrap();
    assert!(!task.is_finished());
    *provider.failure.lock().unwrap() = None;
    tokio::time::timeout(Duration::from_secs(2), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    assert_eq!(provider.forced.load(Ordering::SeqCst), 2);
    let observation = renewal.observation();
    assert_eq!(observation.provider_rounds, 2);
    assert_eq!(observation.retry_decisions, 1);
    assert_eq!(observation.retry_rounds, 1);
    assert_eq!(observation.recovered_calls, 1);
    assert_eq!(observation.failed_calls, 0);
    assert_eq!(observation.cancelled_calls, 0);
    assert_eq!(observation.active_calls, 0);
    assert_eq!(observation.backing_off, 0);
    assert!(observation.backoff_elapsed_micros >= 90_000);
}

#[cfg(feature = "http")]
#[tokio::test]
async fn native_azure_waits_for_credentials_before_head_or_put() -> anyhow::Result<()> {
    use axum::{body::Body, http::Request, response::IntoResponse, routing::any, Router};
    use lance_io::object_store::{providers::ObjectStoreRegistry, ObjectStore, ObjectStoreParams};
    use object_store::{path::Path, ObjectStoreExt};

    let requests = Arc::new(AtomicUsize::new(0));
    let app = Router::new().fallback(any({
        let requests = requests.clone();
        move |request: Request<Body>| {
            requests.fetch_add(1, Ordering::SeqCst);
            async move {
                (
                    if request.method() == "PUT" {
                        axum::http::StatusCode::CREATED
                    } else {
                        axum::http::StatusCode::OK
                    },
                    [
                        ("etag", "\"test-etag\""),
                        ("last-modified", "Wed, 01 Jan 2025 00:00:00 GMT"),
                        ("content-length", "0"),
                    ],
                )
                    .into_response()
            }
        }
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let endpoint = format!("http://{}", listener.local_addr()?);
    let server = tokio::spawn(async move { axum::serve(listener, app).await });
    let provider = Arc::new(Provider {
        options: HashMap::from([
            ("azure_storage_account_name".into(), "account".into()),
            ("azure_storage_endpoint".into(), endpoint),
            (
                "azure_storage_sas_token".into(),
                "?sv=2022-11-02&sp=rw&sig=unit-test".into(),
            ),
            ("allow_http".into(), "true".into()),
            ("expires_at_millis".into(), "0".into()),
        ]),
        ..Default::default()
    });
    let (store, _) = ObjectStore::from_uri_and_params_uncached(
        Arc::new(ObjectStoreRegistry::default()),
        "az://container@account.blob.core.windows.net",
        &ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(
                StorageOptionsAccessor::with_initial_and_provider(
                    provider.options.clone(),
                    Arc::new(RenewingStorageOptionsProvider::new(provider.clone())),
                ),
            )),
            ..Default::default()
        },
    )
    .await?;
    assert_eq!(requests.load(Ordering::SeqCst), 0);
    for put in [false, true] {
        let before = requests.load(Ordering::SeqCst);
        *provider.failure.lock().unwrap() = Some(Failure::Io(ErrorKind::WouldBlock));
        let operation = tokio::spawn({
            let store = store.clone();
            async move {
                let path = Path::from("same-operation");
                if put {
                    store
                        .inner
                        .put(&path, bytes::Bytes::from_static(b"body").into())
                        .await
                        .map(|_| ())
                } else {
                    store.inner.head(&path).await.map(|_| ())
                }
            }
        });
        tokio::time::timeout(Duration::from_secs(2), provider.failed.notified()).await?;
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(!operation.is_finished());
        assert_eq!(requests.load(Ordering::SeqCst), before);
        *provider.failure.lock().unwrap() = None;
        tokio::time::timeout(Duration::from_secs(3), operation).await???;
        assert_eq!(requests.load(Ordering::SeqCst), before + 1);
    }
    server.abort();
    assert!(server.await.unwrap_err().is_cancelled());
    Ok(())
}
