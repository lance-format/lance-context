use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

fn policy() -> Policy {
    Policy {
        heartbeat: Duration::from_millis(20),
        operation_timeout: Duration::from_millis(100),
        max_silence: Duration::from_millis(300),
        stale_after: Duration::from_millis(700),
        poll: Duration::from_millis(10),
    }
}

#[derive(Clone, Copy, Debug)]
enum Fault {
    Healthy,
    PanicRenewal,
    TimeoutOnce,
    ErrorOnce,
    AlwaysTimeout,
    LateRetryAcknowledgement,
    ExpiredFirstFailure,
    InitialFailure,
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn protected_read_keeps_renewing_while_its_own_poll_is_busy() {
    let (_dir, store, root, faulty) = fixture(Fault::Healthy).await;
    let ticks = Arc::new(AtomicUsize::new(0));
    let (ready, started) = tokio::sync::oneshot::channel();
    let timer = tokio::spawn({
        let ticks = ticks.clone();
        async move {
            ready.send(()).unwrap();
            loop {
                tokio::time::sleep(Duration::from_millis(10)).await;
                ticks.fetch_add(1, Ordering::SeqCst);
            }
        }
    });
    started.await.unwrap();
    let read_faulty = faulty.clone();
    let read_ticks = ticks.clone();
    let result = tokio::spawn(async move {
        protect(store, &root, policy(), async {
            read_faulty.renewed.notified().await;
            read_ticks.store(0, Ordering::SeqCst);
            // Occupy an actual runtime worker, not just the test's caller.
            std::thread::sleep(Duration::from_millis(450));
            Ok(42)
        })
        .await
    })
    .await
    .unwrap();
    timer.abort();
    assert!(timer.await.unwrap_err().is_cancelled());
    assert!(
        ticks.load(Ordering::SeqCst) >= 10,
        "the other worker stalled"
    );
    assert_eq!(result.unwrap(), 42);
    assert!(faulty.renewals.load(Ordering::SeqCst) > 1);
}

#[derive(Debug)]
struct FaultyRenewal {
    inner: Arc<dyn object_store::ObjectStore>,
    fault: Fault,
    renewals: AtomicUsize,
    attempted: tokio::sync::Notify,
    renewed: tokio::sync::Notify,
}

impl std::fmt::Display for FaultyRenewal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FaultyRenewal")
    }
}

#[async_trait::async_trait]
impl object_store::ObjectStore for FaultyRenewal {
    async fn put_opts(
        &self,
        path: &Path,
        payload: object_store::PutPayload,
        options: PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        let renewal = matches!(&options.mode, PutMode::Overwrite);
        if renewal {
            let attempt = self.renewals.fetch_add(1, Ordering::SeqCst) + 1;
            self.attempted.notify_one();
            match (self.fault, attempt) {
                (Fault::PanicRenewal, _) => panic!("injected renewal panic"),
                (Fault::TimeoutOnce | Fault::LateRetryAcknowledgement, 1)
                | (Fault::AlwaysTimeout, _) => std::future::pending::<()>().await,
                (Fault::ErrorOnce, 1) => {
                    return Err(object_store::Error::Generic {
                        store: "injected renewal",
                        source: "transient PUT failure".into(),
                    });
                }
                (Fault::ExpiredFirstFailure, 1) => {
                    std::thread::sleep(policy().max_silence);
                    return Err(object_store::Error::Generic {
                        store: "injected expired renewal",
                        source: "failure observed after the reader deadline".into(),
                    });
                }
                (Fault::LateRetryAcknowledgement, 2) => {
                    // The runtime resumes with an apparently successful PUT
                    // after the old acknowledgement's deadline has elapsed.
                    std::thread::sleep(policy().max_silence);
                }
                _ => {}
            }
        } else if matches!(self.fault, Fault::InitialFailure) {
            return Err(object_store::Error::Generic {
                store: "injected registration",
                source: "initial PUT failure".into(),
            });
        }
        let result = self.inner.put_opts(path, payload, options).await?;
        if renewal {
            self.renewed.notify_one();
        }
        Ok(result)
    }

    async fn put_multipart_opts(
        &self,
        path: &Path,
        options: object_store::PutMultipartOptions,
    ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
        self.inner.put_multipart_opts(path, options).await
    }

    async fn get_opts(
        &self,
        path: &Path,
        options: GetOptions,
    ) -> object_store::Result<object_store::GetResult> {
        self.inner.get_opts(path, options).await
    }

    fn delete_stream(
        &self,
        paths: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        self.inner.delete_stream(paths)
    }

    fn list(
        &self,
        prefix: Option<&Path>,
    ) -> BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&Path>,
    ) -> object_store::Result<object_store::ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: object_store::CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
}

async fn fixture(
    fault: Fault,
) -> (
    tempfile::TempDir,
    Arc<ObjectStore>,
    Path,
    Arc<FaultyRenewal>,
) {
    let dir = tempfile::TempDir::new().unwrap();
    let (store, root) = ObjectStore::from_uri(dir.path().to_str().unwrap())
        .await
        .unwrap();
    let faulty = Arc::new(FaultyRenewal {
        inner: store.inner.clone(),
        fault,
        renewals: AtomicUsize::new(0),
        attempted: tokio::sync::Notify::new(),
        renewed: tokio::sync::Notify::new(),
    });
    let mut wrapped = store.as_ref().clone();
    wrapped.inner = faulty.clone();
    (dir, Arc::new(wrapped), root, faulty)
}

#[tokio::test]
async fn one_timed_out_renewal_can_recover_before_the_original_deadline() {
    let (_dir, store, root, faulty) = fixture(Fault::TimeoutOnce).await;
    let result = protect(store, &root, policy(), async {
        faulty.renewed.notified().await;
        Ok(7)
    })
    .await;
    assert_eq!(result.unwrap(), 7);
    assert_eq!(faulty.renewals.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn one_transient_storage_error_can_recover_without_restarting_the_read() {
    let (_dir, store, root, faulty) = fixture(Fault::ErrorOnce).await;
    let reads = AtomicUsize::new(0);
    let result = protect(store, &root, policy(), async {
        reads.fetch_add(1, Ordering::SeqCst);
        faulty.renewed.notified().await;
        Ok(9)
    })
    .await;
    assert_eq!(result.unwrap(), 9);
    assert_eq!(reads.load(Ordering::SeqCst), 1);
    assert_eq!(faulty.renewals.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn repeated_renewal_timeouts_fail_with_only_one_retry() {
    let (_dir, store, root, faulty) = fixture(Fault::AlwaysTimeout).await;
    // Test the bounded retry limit independently of lease expiry. The outer
    // two-second bound still requires failure well before this lease expires.
    let policy = Policy {
        max_silence: Duration::from_secs(10),
        stale_after: Duration::from_secs(20),
        ..policy()
    };
    let result = tokio::time::timeout(
        Duration::from_secs(2),
        protect(store, &root, policy, async {
            std::future::pending::<lance::Result<()>>().await
        }),
    )
    .await
    .unwrap();
    assert!(result.unwrap_err().to_string().contains("timed out"));
    assert_eq!(faulty.renewals.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn a_late_retry_acknowledgement_cannot_resurrect_an_expired_read() {
    let (_dir, store, root, faulty) = fixture(Fault::LateRetryAcknowledgement).await;
    let result = protect(store, &root, policy(), async {
        faulty.renewed.notified().await;
        Ok(11)
    })
    .await;
    assert!(result.unwrap_err().to_string().contains("expired"));
    assert_eq!(faulty.renewals.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn initial_registration_failure_never_starts_the_read_or_retries() {
    let (_dir, store, root, faulty) = fixture(Fault::InitialFailure).await;
    let result = protect(store, &root, policy(), async {
        panic!("a read without an acknowledged registration must not start");
        #[allow(unreachable_code)]
        Ok(())
    })
    .await;
    assert!(result.is_err());
    assert_eq!(faulty.renewals.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn expired_first_failure_never_starts_a_retry() {
    let (_dir, store, root, faulty) = fixture(Fault::ExpiredFirstFailure).await;
    let result = protect(store, &root, policy(), async {
        std::future::pending::<lance::Result<()>>().await
    })
    .await;
    assert!(result.unwrap_err().to_string().contains("expired"));
    assert_eq!(faulty.renewals.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn cancelling_during_failed_renewal_releases_the_registration() {
    let (_dir, store, root, faulty) = fixture(Fault::AlwaysTimeout).await;
    let reading = {
        let (store, root) = (store.clone(), root.clone());
        tokio::spawn(async move {
            protect(store, &root, policy(), async {
                std::future::pending::<lance::Result<()>>().await
            })
            .await
        })
    };
    faulty.attempted.notified().await;
    let captured = captured_readers(&store, &root).await.unwrap();
    assert_eq!(captured.len(), 1);
    reading.abort();
    assert!(reading.await.unwrap_err().is_cancelled());
    tokio::time::timeout(
        Duration::from_millis(500),
        wait_for_readers(store, captured, policy()),
    )
    .await
    .unwrap()
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn renewal_starts_before_a_non_yielding_first_read_poll() {
    let (_dir, store, root, faulty) = fixture(Fault::Healthy).await;
    let result = tokio::spawn(async move {
        protect(store, &root, policy(), async {
            std::thread::sleep(Duration::from_millis(450));
            Ok(17)
        })
        .await
    })
    .await
    .unwrap();
    assert_eq!(result.unwrap(), 17);
    assert!(faulty.renewals.load(Ordering::SeqCst) > 1);
}

#[tokio::test]
async fn a_panicked_renewal_fails_the_protected_read() {
    let (_dir, store, root, _faulty) = fixture(Fault::PanicRenewal).await;
    let result = tokio::time::timeout(
        Duration::from_secs(2),
        protect(store, &root, policy(), async {
            std::future::pending::<lance::Result<()>>().await
        }),
    )
    .await
    .unwrap();
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("renewal task failed"));
}

#[tokio::test]
async fn completed_and_cancelled_reads_stop_renewing_before_cleanup() {
    for cancel in [false, true] {
        let (_dir, store, root, faulty) = fixture(Fault::Healthy).await;
        let release = Arc::new(tokio::sync::Notify::new());
        let reading = {
            let (store, root, release) = (store.clone(), root.clone(), release.clone());
            tokio::spawn(async move {
                protect(store, &root, policy(), async {
                    release.notified().await;
                    Ok(23)
                })
                .await
            })
        };
        faulty.renewed.notified().await;
        assert_eq!(captured_readers(&store, &root).await.unwrap().len(), 1);
        if cancel {
            reading.abort();
            assert!(reading.await.unwrap_err().is_cancelled());
        } else {
            release.notify_one();
            assert_eq!(reading.await.unwrap().unwrap(), 23);
        }
        tokio::time::timeout(Duration::from_millis(500), async {
            while !captured_readers(&store, &root).await.unwrap().is_empty() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        let renewals = faulty.renewals.load(Ordering::SeqCst);
        tokio::time::sleep(policy().heartbeat * 5).await;
        assert_eq!(faulty.renewals.load(Ordering::SeqCst), renewals);
        assert!(captured_readers(&store, &root).await.unwrap().is_empty());
    }
}
