use super::*;

const PROCESS_ROOT: &str = "LANCE_CONTEXT_TEST_SINGLE_BODY_ROOT";

// The parent kills this process, including its three real disk-backed WAL
// replicas, after an acknowledged suffix is visible but not materialized.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "subprocess role, launched by full_body_rf3_recovers_acked_suffix_after_sigkill"]
async fn full_body_process_role() -> Result<()> {
    let root = std::path::PathBuf::from(std::env::var(PROCESS_ROOT)?);
    let input = bodies()?;
    let spec = TurnIngestionSpec::new(
        input[0].schema(),
        vec!["session_id".into(), "id".into()],
        "session_id",
    )?;
    let c = network::Cluster::open_signed(&root, annotation_publication::one_shard()).await?;
    let storage = fixture::storage(&root, "full-body.lance").await?;
    crate::bootstrap::provision_body_shard(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding(),
        storage.clone(),
        Identity {
            owner: Uuid::new_v4(),
            acquisition: Uuid::new_v4(),
            manifest: Uuid::new_v4(),
        },
        &fixture::limits(),
        Shard { bucket: 0 },
        &spec,
    )
    .await?;
    let (service, runtime) = owner_with_wait(&c, &storage, &spec, Duration::from_secs(60)).await?;
    let (_stop, task) = run_owner(runtime);
    let lane = producer(&service, 1).await?;
    // Two records trigger publication. The repeated create has different
    // payload values and must preserve the first complete body.
    for (i, body) in input.iter().take(2).enumerate() {
        let body = fixture::encoded(single_body::batch(
            &spec,
            &fixture::limits(),
            Shard { bucket: 0 },
            body,
        )?)?;
        let append = request(&service, &lane, i as u64 + 1, body).await?;
        let result = service.submit(&append, DEADLINE).await?;
        assert_eq!(result.result, MutationResult::Applied);
        if i == 1 {
            drained(&c, &service, result.index).await?;
        }
    }
    let selected = service.selected_full_body(DEADLINE).await?;
    let base = annotation_publication::read(&c, &storage, Some(selected.manifest.version)).await?;
    assert_eq!(base.count_rows(None).await?, 2);
    let body = fixture::encoded(single_body::batch(
        &spec,
        &fixture::limits(),
        Shard { bucket: 0 },
        &input[2],
    )?)?;
    std::fs::write(root.join("request.arrow"), &body)?;
    let append = request(&service, &lane, 3, body).await?;
    let result = service.submit(&append, DEADLINE).await?;
    assert_eq!(result.result, MutationResult::Applied);
    assert!(service.observation().await.0.materialized < result.index);
    let key = spec.row_key(&input[2], 2, 4096)?;
    assert_eq!(
        service
            .get_full_body(&key, DEADLINE)
            .await?
            .unwrap()
            .batch(),
        &input[2].slice(2, 1)
    );
    std::fs::write(root.join("ack.json"), serde_json::to_vec(&result)?)?;
    std::fs::write(root.join("ready"), b"ready")?;
    task.await??;
    bail!("single-body child stopped before SIGKILL")
}

struct Child(std::process::Child);

impl Drop for Child {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn full_body_rf3_recovers_acked_suffix_after_sigkill() -> Result<()> {
    use std::os::unix::process::ExitStatusExt;
    let root = tempfile::tempdir()?;
    let log = std::fs::File::create(root.path().join("child.log"))?;
    let mut child = Child(
        std::process::Command::new(std::env::current_exe()?)
            .args([
                "--exact",
                "tests::generic_turns::single_body::process::full_body_process_role",
                "--ignored",
                "--nocapture",
            ])
            .env(PROCESS_ROOT, root.path())
            .stdout(log.try_clone()?)
            .stderr(log)
            .spawn()?,
    );
    let ready = tokio::time::timeout(Duration::from_secs(60), async {
        while !root.path().join("ready").exists() {
            if let Some(status) = child.0.try_wait()? {
                bail!("single-body child exited {status}");
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        Ok::<_, anyhow::Error>(())
    })
    .await;
    match ready {
        Ok(Ok(())) => {}
        error => bail!(
            "single-body child did not reach the crash boundary: {error:?}; {}",
            std::fs::read_to_string(root.path().join("child.log"))?
        ),
    }
    child.0.kill()?;
    assert_eq!(child.0.wait()?.signal(), Some(9));
    let expected: crate::annotations::Completed =
        serde_json::from_slice(&std::fs::read(root.path().join("ack.json"))?)?;
    let input = bodies()?;
    let spec = TurnIngestionSpec::new(
        input[0].schema(),
        vec!["session_id".into(), "id".into()],
        "session_id",
    )?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let storage = fixture::storage(root.path(), "full-body.lance").await?;
    let (service, runtime) = owner(&c, &storage, &spec).await?;
    let (stop, task) = run_owner(runtime);
    // No new appends or client retries before these reads and full drain.
    for (batch, row) in [(0, 0), (2, 2)] {
        let key = spec.row_key(&input[batch], row, 4096)?;
        assert_eq!(
            service
                .get_full_body(&key, DEADLINE)
                .await?
                .unwrap()
                .batch(),
            &input[batch].slice(row, 1)
        );
    }
    drained(&c, &service, expected.index).await?;
    assert_eq!(service.retained_body_frames(), (0, 0));
    let selected = service.selected_full_body(DEADLINE).await?;
    let durable =
        annotation_publication::read(&c, &storage, Some(selected.manifest.version)).await?;
    assert_eq!(durable.count_rows(None).await?, 4);
    let lane = producer(&service, 1).await?;
    let body = Bytes::from(std::fs::read(root.path().join("request.arrow"))?);
    let append = request(&service, &lane, 3, body).await?;
    assert_eq!(service.submit(&append, DEADLINE).await?, expected);
    let selected = service.selected_full_body(DEADLINE).await?;
    let repeated =
        annotation_publication::read(&c, &storage, Some(selected.manifest.version)).await?;
    assert_eq!(repeated.manifest().fragments, durable.manifest().fragments);
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
