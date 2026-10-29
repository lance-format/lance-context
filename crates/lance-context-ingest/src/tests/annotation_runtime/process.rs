use super::*;

// Launched by the ordinary parent test, then SIGKILLed with its three real WAL
// replicas and annotation runtime. This role is not an independent CI test.
#[tokio::test]
#[ignore = "subprocess role, launched by annotation_runtime_recovers_after_sigkill"]
async fn annotation_process_role() -> Result<()> {
    let root =
        std::path::PathBuf::from(std::env::var("LANCE_CONTEXT_TEST_ANNOTATION_PROCESS_ROOT")?);
    let c = network::Cluster::open_signed(&root, annotation_publication::one_shard()).await?;
    let storage = process_storage(&root).await?;
    let limits = process_limits();
    annotation_publication::seed_contract(&c, &storage, Service::contract(&binding(), &limits)?)
        .await?;
    let (service, runtime) =
        owner_with_wait(&c, &storage, &limits, Duration::from_secs(60)).await?;
    let (_stop, task) = run_owner(runtime);
    let producer = producer(&service, 1).await?;
    for sequence in 1..=2 {
        let body = wire(&limits, &["T"], 1, Some("published"), 1, Some(0.9))?;
        let request = request(&service, &producer, sequence, body).await?;
        let result = service.submit(&request, Duration::from_secs(30)).await?;
        if sequence == 2 {
            drained(&c, &service, result.index).await?;
        }
    }
    let body = wire(&limits, &["T"], 1, Some("acknowledged-suffix"), 0, None)?;
    std::fs::write(root.join("request.arrow"), &body)?;
    let request = request(&service, &producer, 3, body).await?;
    let result = service.submit(&request, Duration::from_secs(30)).await?;
    assert!(service.observation().await.0.materialized < result.index);
    assert_eq!(
        service
            .get("T", Duration::from_secs(30))
            .await?
            .unwrap()
            .status
            .as_deref(),
        Some("acknowledged-suffix")
    );
    std::fs::write(root.join("ack.json"), serde_json::to_vec(&result)?)?;
    std::fs::write(root.join("ready"), b"ready")?;
    task.await??;
    bail!("annotation process unexpectedly stopped before SIGKILL");
}

fn process_limits() -> Limits {
    Limits {
        retained_bytes: 256 << 10,
        retained_entries: 128,
        value_bytes: 4096,
        readers: 16,
    }
}

async fn process_storage(root: &std::path::Path) -> Result<Storage> {
    Storage::resolve(
        root.join("turns.lance").to_str().unwrap().into(),
        Default::default(),
        Arc::new(lance::session::Session::default()),
    )
    .await
}

struct Child(std::process::Child);
impl Drop for Child {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[tokio::test]
async fn annotation_runtime_recovers_after_sigkill() -> Result<()> {
    use std::os::unix::process::ExitStatusExt;
    let root = tempfile::tempdir()?;
    let log = std::fs::File::create(root.path().join("child.log"))?;
    let mut child = Child(
        std::process::Command::new(std::env::current_exe()?)
            .args([
                "--exact",
                "tests::annotation_runtime::process::annotation_process_role",
                "--ignored",
                "--nocapture",
            ])
            .env("LANCE_CONTEXT_TEST_ANNOTATION_PROCESS_ROOT", root.path())
            .stdout(log.try_clone()?)
            .stderr(log)
            .spawn()?,
    );
    tokio::time::timeout(Duration::from_secs(60), async {
        while !root.path().join("ready").exists() {
            if let Some(status) = child.0.try_wait()? {
                bail!(
                    "annotation child exited {status}: {}",
                    std::fs::read_to_string(root.path().join("child.log"))?
                );
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        Ok::<_, anyhow::Error>(())
    })
    .await??;
    child.0.kill()?;
    assert_eq!(child.0.wait()?.signal(), Some(9));
    let expected: crate::annotations::Completed =
        serde_json::from_slice(&std::fs::read(root.path().join("ack.json"))?)?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let storage = process_storage(root.path()).await?;
    let (service, runtime) = owner(&c, &storage, &process_limits()).await?;
    let (stop, task) = run_owner(runtime);
    let producer = producer(&service, 1).await?;
    let body = Bytes::from(std::fs::read(root.path().join("request.arrow"))?);
    let request = request(&service, &producer, 3, body).await?;
    assert_eq!(
        service.submit(&request, Duration::from_secs(30)).await?,
        expected
    );
    let current = service.get("T", Duration::from_secs(30)).await?.unwrap();
    assert_eq!(current.status.as_deref(), Some("acknowledged-suffix"));
    assert_eq!(current.score, Some(0.9));
    drained(&c, &service, expected.index).await?;
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
