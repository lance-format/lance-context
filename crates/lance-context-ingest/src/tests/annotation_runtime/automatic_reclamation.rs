use super::*;
use futures::TryStreamExt;
use std::collections::BTreeSet;

fn files(path: &std::path::Path) -> Result<BTreeSet<std::path::PathBuf>> {
    std::fs::read_dir(path.join("data"))?
        .map(|entry| Ok(entry?.path()))
        .collect()
}

fn metadata_files(path: &std::path::Path) -> Result<BTreeSet<std::path::PathBuf>> {
    let mut files = BTreeSet::new();
    for directory in ["_versions", "_transactions"] {
        for entry in std::fs::read_dir(path.join(directory))? {
            let path = entry?.path();
            if path.file_name().unwrap() != "latest_version_hint.json" {
                files.insert(path);
            }
        }
    }
    Ok(files)
}

async fn owned(
    c: &network::Cluster,
    storage: &Storage,
    limits: &Limits,
) -> Result<(Service, Runtime)> {
    let state = annotation_publication::state(c).await?;
    let (service, runtime) = Runtime::connect(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding(),
        storage.clone(),
        Configuration {
            limits: limits.clone(),
            upload_budget: MergeMemoryBudget::new(8 << 20),
            apply_budget: MergeMemoryBudget::new(8 << 20),
            publication_budget: MergeMemoryBudget::new(8 << 20),
            publication: Options {
                max_source_bytes: 1 << 20,
                min_source_bytes: 1 << 20,
                max_records: 2,
                max_batch_wait: Duration::from_millis(20),
                retained_manifest_versions: 2,
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
            max_objects: 2,
            max_bytes: 16 << 10,
            delete_parallelism: 2,
            interval: Duration::from_millis(10),
        })?,
    ))
}

#[tokio::test]
async fn annotation_owner_collects_abandoned_preparation_and_continuously_drains_data() -> Result<()>
{
    let temporary = tempfile::Builder::new()
        .prefix("annotation-auto-gc-")
        .tempdir()?;
    let root = if let Ok(parent) = std::env::var("INGEST_TEST_ARTIFACTS") {
        tempfile::Builder::new()
            .prefix("annotation-auto-gc-")
            .tempdir_in(parent)?
            .keep()
    } else {
        temporary.path().to_owned()
    };
    let c = network::Cluster::open_signed(&root, annotation_publication::one_shard()).await?;
    let path = root.join("turns.lance");
    let storage = Storage::resolve(
        path.to_str().unwrap().into(),
        Default::default(),
        Arc::new(lance::session::Session::default()),
    )
    .await?;
    let limits = Limits {
        retained_bytes: 256 << 10,
        retained_entries: 128,
        value_bytes: 4096,
        readers: 16,
    };
    annotation_publication::seed_contract(&c, &storage, Service::contract(&binding(), &limits)?)
        .await?;
    let original = files(&path)?;
    let (service, driver) = connect(&c, &storage, &limits).await?;
    let (stop, task) = run(driver);
    let producer = producer(&service, 1).await?;
    let first = request(
        &service,
        &producer,
        1,
        wire(&limits, &["T"], 1, Some("acknowledged"), 0, None)?,
    )
    .await?;
    let first_result = service.submit(&first, Duration::from_secs(30)).await?;
    let frozen = service.freeze(Uuid::new_v4()).await?.unwrap();
    let prepared = frozen
        .prepare(&MergeMemoryBudget::new(8 << 20), 8192)
        .await?;
    // These are actual Lance uploaded files. No preparation descriptor reached
    // the durable catalog, so recovery must replay WAL and produce new files.
    let abandoned: Vec<_> = files(&path)?.difference(&original).cloned().collect();
    assert_eq!(abandoned.len(), 1);
    stop.send(()).unwrap();
    task.await??;
    drop((prepared, frozen, service));

    let (service, runtime) = owned(&c, &storage, &limits).await?;
    let (stop, task) = run_owner(runtime);
    assert_eq!(
        service.submit(&first, Duration::from_secs(30)).await?,
        first_result
    );
    let mut last = None;
    for sequence in 2..=8 {
        let status = format!("reviewed-{sequence}");
        let update = request(
            &service,
            &producer,
            sequence,
            wire(&limits, &["T"], 1, Some(&status), 1, Some(sequence as f32))?,
        )
        .await?;
        let completed = service.submit(&update, Duration::from_secs(30)).await?;
        let value = service.get("T", Duration::from_secs(30)).await?.unwrap();
        assert_eq!(value.status.as_deref(), Some(status.as_str()));
        assert_eq!(value.score, Some(sequence as f32));
        drained(&c, &service, completed.index).await?;
        tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                if files(&path)?.len() == 3 && metadata_files(&path)?.len() == 4 {
                    return Ok::<_, anyhow::Error>(());
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await??;
        last = Some((update, completed));
    }
    assert!(abandoned.iter().all(|p| !p.exists()));
    assert!(original.iter().all(|p| p.exists()));
    let observed = service.reclamation_observation();
    assert!(observed.passes > 1 && observed.deleted >= 7 && observed.deleted_bytes > 0);
    let (last, completed) = last.unwrap();
    assert_eq!(
        service.submit(&last, Duration::from_secs(30)).await?,
        completed
    );
    stop.send(()).unwrap();
    task.await??;
    drop(service);

    let (service, runtime) = owned(&c, &storage, &limits).await?;
    let (stop, task) = run_owner(runtime);
    assert_eq!(
        service.submit(&last, Duration::from_secs(30)).await?,
        completed
    );
    let value = service.get("T", Duration::from_secs(30)).await?.unwrap();
    assert_eq!(value.status.as_deref(), Some("reviewed-8"));
    assert_eq!(value.score, Some(8.0));
    let latest = annotation_publication::read(&c, &storage, None).await?;
    assert!(latest.read_transaction().await?.is_some());
    let bootstrap = annotation_publication::read(&c, &storage, Some(1)).await?;
    assert!(bootstrap.read_transaction().await?.is_some());
    let batches = latest
        .scan()
        .try_into_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 2);
    for batch in batches {
        let bodies = batch
            .column_by_name("body")
            .unwrap()
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .unwrap();
        for body in bodies.iter().flatten() {
            assert_eq!(body.len(), 256 << 10);
            assert!(body
                .iter()
                .enumerate()
                .all(|(i, b)| *b == ((i * 31 + i / 17) % 251) as u8));
        }
    }
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}

#[tokio::test]
async fn annotation_owner_refuses_collection_when_a_native_tag_exists() -> Result<()> {
    use lance_context_api::annotation::{AnnotationFieldUpdate::Set, TurnAnnotationPatch};
    use lance_context_core::annotation_staging::{AnnotationStaging, Scope};
    let root = tempfile::tempdir()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let path = root.path().join("turns.lance");
    let storage = Storage::resolve(
        path.to_str().unwrap().into(),
        Default::default(),
        Arc::new(lance::session::Session::default()),
    )
    .await?;
    let limits = Limits {
        retained_bytes: 256 << 10,
        retained_entries: 128,
        value_bytes: 4096,
        readers: 16,
    };
    annotation_publication::seed_contract(&c, &storage, Service::contract(&binding(), &limits)?)
        .await?;
    let base = annotation_publication::read(&c, &storage, None).await?;
    base.tags().create("retained", base.version_id()).await?;
    let staging = AnnotationStaging::new(
        base,
        Scope {
            table_instance: binding().incarnation.to_string(),
            source_group: 10,
            shard: 0,
        },
        0,
        limits.clone(),
    )?;
    staging
        .apply(
            0,
            1,
            "T",
            &TurnAnnotationPatch {
                score: Set(0.9),
                ..Default::default()
            },
        )
        .await?;
    let frozen = staging.freeze(Uuid::new_v4()).await?.unwrap();
    let prepared = frozen
        .prepare(&MergeMemoryBudget::new(8 << 20), 8192)
        .await?;
    let before = files(&path)?;
    assert_eq!(before.len(), 3);
    drop((prepared, frozen, staging));
    let (service, runtime) = owned(&c, &storage, &limits).await?;
    let error = tokio::time::timeout(Duration::from_secs(30), runtime.run(std::future::pending()))
        .await?
        .unwrap_err();
    assert!(format!("{error:#}").contains("without tags or branches"));
    assert_eq!(service.reclamation_observation().deleted, 0);
    assert_eq!(files(&path)?, before);
    c.shutdown().await
}

mod deletion_faults;
