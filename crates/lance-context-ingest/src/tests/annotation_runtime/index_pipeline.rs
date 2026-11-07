use super::automatic_reclamation::deletion_faults::fault_store;
use super::*;
use lance::index::DatasetIndexExt;

async fn owner(
    c: &network::Cluster,
    storage: &Storage,
    status_index: bool,
) -> Result<(Service, Runtime)> {
    let state = annotation_publication::state(c).await?;
    let (service, runtime) = Runtime::connect(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding(),
        storage.clone(),
        Configuration {
            limits: Limits {
                retained_bytes: 256 << 10,
                retained_entries: 128,
                value_bytes: 4096,
                readers: 16,
            },
            upload_budget: MergeMemoryBudget::new(8 << 20),
            apply_budget: MergeMemoryBudget::new(8 << 20),
            publication_budget: MergeMemoryBudget::new(1024 << 20),
            publication: Options {
                max_source_bytes: 1 << 20,
                min_source_bytes: 1,
                max_records: 2,
                max_batch_wait: Duration::from_millis(20),
                retained_manifest_versions: 2,
                turn_id_index: (!status_index)
                    .then_some(crate::materializer::TurnIdIndex { min_rows: 1 }),
                status_index: status_index.then_some(crate::materializer::StatusIndex {
                    min_rows: 1,
                    ..Default::default()
                }),
                ..Default::default()
            },
        },
        Acquisition {
            owner: Uuid::new_v4(),
            request: Uuid::new_v4(),
            expected_epoch: state.authority.epoch,
        },
    )
    .await?;
    Ok((
        service,
        runtime.with_exclusive_reclamation(crate::annotations::runtime::reclamation::Options {
            max_objects: 128,
            max_bytes: 16 << 10,
            delete_parallelism: 2,
            interval: Duration::from_millis(10),
        })?,
    ))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn slow_id_index_allows_publication_and_gc_then_selects_unchanged_id_columns() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(90), paused(false, false)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn restart_during_id_index_preserves_acknowledged_updates_and_rebuilds_index() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(90), paused(true, false)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn slow_status_index_allows_publication_without_restoring_old_values() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(90), paused(false, true)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn restart_during_status_index_preserves_updates_and_catches_up() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(90), paused(true, true)).await?
}

async fn paused(restart: bool, status_index: bool) -> Result<()> {
    let root = tempfile::tempdir()?;
    let directory = root.path().join("turns.lance");
    let faults = Arc::new(fault_store::Faults::default());
    let storage = fault_store::storage(&directory, faults.clone()).await?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let limits = Limits {
        retained_bytes: 256 << 10,
        retained_entries: 128,
        value_bytes: 4096,
        readers: 16,
    };
    annotation_publication::seed_contract(&c, &storage, Service::contract(&binding(), &limits)?)
        .await?;
    let (entered, resume) = faults.pause_index_write();
    let (service, runtime) = owner(&c, &storage, status_index).await?;
    let (stop, task) = run_owner(runtime);
    let producer = producer(&service, 1).await?;
    let initial = request(
        &service,
        &producer,
        1,
        wire(&limits, &["T"], 1, Some("reviewed"), 0, None)?,
    )
    .await?;
    let first = service.submit(&initial, Duration::from_secs(10)).await?;
    let index_file = tokio::time::timeout(Duration::from_secs(10), entered).await??;
    // The serial status path already retains its maintenance plan here. Verify
    // the initial data cut independently, then require the next updates to drain.
    let checkpoint = annotation_publication::state(&c).await?.progress.checkpoint;
    assert!(checkpoint.get(&10).copied().unwrap_or(0) >= first.index);
    if !status_index {
        drained(&c, &service, first.index).await?;
    }
    let local_index_file = std::path::Path::new("/").join(index_file.as_ref());
    assert!(
        local_index_file.exists(),
        "pause follows a real native file upload"
    );
    let source_version = annotation_publication::read(&c, &storage, None)
        .await?
        .version_id();
    let orphan = directory
        .join("data")
        .join(format!("{}.lance", Uuid::new_v4()));
    std::fs::write(&orphan, b"unreferenced data file")?;
    let mut last = None;
    for sequence in 2..=7 {
        let status = if status_index && sequence % 2 == 1 {
            "accepted"
        } else {
            "reviewed"
        };
        let update = request(
            &service,
            &producer,
            sequence,
            wire(
                &limits,
                &["T"],
                u8::from(status_index),
                status_index.then_some(status),
                1,
                Some(sequence as f32),
            )?,
        )
        .await?;
        let result = service.submit(&update, Duration::from_secs(10)).await?;
        let value = service.get("T", Duration::from_secs(10)).await?.unwrap();
        assert_eq!(value.status.as_deref(), Some(status));
        assert_eq!(value.score, Some(sequence as f32));
        drained(&c, &service, result.index)
            .await
            .context("acknowledged update must materialize while native index I/O is paused")?;
        assert_eq!(service.source_observation().records, 0);
        assert!(local_index_file.exists(), "GC removed a building index");
        last = Some((update, result));
    }
    tokio::time::timeout(Duration::from_secs(10), async {
        while orphan.exists() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;
    let state = annotation_publication::state(&c).await?;
    assert!(
        state.progress.pending.is_none(),
        "native I/O must not hold a publication plan"
    );
    assert!(!state
        .manifests
        .as_ref()
        .unwrap()
        .versions
        .contains_key(&source_version));
    assert!(
        local_index_file.exists(),
        "retired build source still needs its index output"
    );
    let (service, stop, task) = if restart {
        stop.send(()).unwrap();
        task.await??;
        drop(service);
        // The stopped owner's native writer cannot select its output. Dropping
        // the paused response completes that original job with an error; its
        // unselected files may be collected by the successor.
        drop(resume);
        let (service, runtime) = owner(&c, &storage, status_index).await?;
        let (stop, task) = run_owner(runtime);
        (service, stop, task)
    } else {
        resume.send(()).unwrap();
        (service, stop, task)
    };
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let base = annotation_publication::read(&c, &storage, None).await?;
            let indices = base.load_indices().await?;
            let ready = if status_index {
                let covered: std::collections::BTreeSet<_> = indices
                    .iter()
                    .filter(|i| i.name == "lance_context_annotation_status")
                    .flat_map(|i| i.fragment_bitmap.as_ref().unwrap().iter())
                    .map(u64::from)
                    .collect();
                !covered.is_empty() && base.fragments().iter().all(|f| covered.contains(&f.id))
            } else {
                !indices.is_empty()
            };
            if ready
                && annotation_publication::state(&c)
                    .await?
                    .progress
                    .pending
                    .is_none()
            {
                break Ok::<_, anyhow::Error>(());
            }
            ensure!(!task.is_finished(), "index owner stopped");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await??;
    let value = service.get("T", Duration::from_secs(10)).await?.unwrap();
    assert_eq!(
        value.status.as_deref(),
        Some(if status_index { "accepted" } else { "reviewed" })
    );
    assert_eq!(value.score, Some(7.0));
    if status_index {
        let base = annotation_publication::read(&c, &storage, None).await?;
        assert_eq!(
            base.count_rows(Some("status = 'accepted'".into())).await?,
            1
        );
        assert_eq!(
            base.count_rows(Some("status = 'reviewed'".into())).await?,
            0
        );
    }
    let (last, result) = last.unwrap();
    assert_eq!(
        service.submit(&last, Duration::from_secs(10)).await?,
        result
    );
    if restart || status_index {
        tokio::time::timeout(Duration::from_secs(10), async {
            while local_index_file.exists() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await?;
    } else {
        assert!(
            local_index_file.exists(),
            "selected index file was collected"
        );
    }
    assert_eq!(service.reclamation_observation().delete_failures, 0);
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
