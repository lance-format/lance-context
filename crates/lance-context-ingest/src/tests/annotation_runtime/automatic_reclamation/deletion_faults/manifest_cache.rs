//! Real orphan pages exercise metadata reuse and cold-cache fallback.
use super::*;
use lance_context_api::annotation::{AnnotationFieldUpdate::Set, TurnAnnotationPatch};
use lance_context_core::annotation_staging::{AnnotationStaging, Scope};
use std::sync::atomic::Ordering;

#[tokio::test]
async fn annotation_collector_reuses_manifests_across_orphan_pages() -> Result<()> {
    orphan_pages(1 << 20, false, 1).await
}

#[tokio::test]
async fn annotation_collector_reads_manifests_with_disabled_cache() -> Result<()> {
    orphan_pages(0, false, 2).await
}

#[tokio::test]
async fn annotation_collector_reads_manifests_after_cache_clear() -> Result<()> {
    orphan_pages(1 << 20, true, 2).await
}

#[tokio::test]
async fn annotation_collector_does_not_exceed_the_session_cache_capacity() -> Result<()> {
    orphan_pages(1, false, 2).await
}

async fn orphan_pages(cache_bytes: usize, clear: bool, expected_reads: usize) -> Result<()> {
    let root = root()?;
    let path = root.path().join("turns.lance");
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let faults = Arc::new(Faults::default());
    let storage = fault_store::storage_with_cache(&path, faults.clone(), cache_bytes).await?;
    annotation_publication::seed_contract(&c, &storage, Service::contract(&binding(), &limits())?)
        .await?;
    let original = files(&path)?;
    for score in [0.1, 0.9] {
        // These preparations never enter the source log, so publication cannot
        // change the retained manifest set between our two collector pages.
        let staging = AnnotationStaging::new(
            annotation_publication::read(&c, &storage, None).await?,
            Scope {
                table_instance: binding().incarnation.to_string(),
                source_group: 10,
                shard: 0,
            },
            0,
            limits(),
        )?;
        staging
            .apply(
                0,
                1,
                "T",
                &TurnAnnotationPatch {
                    score: Set(score),
                    ..Default::default()
                },
            )
            .await?;
        let frozen = staging.freeze(Uuid::new_v4()).await?.unwrap();
        let prepared = frozen
            .prepare(&MergeMemoryBudget::new(8 << 20), 8192)
            .await?;
        drop((prepared, frozen, staging));
    }
    let abandoned: Vec<_> = files(&path)?.difference(&original).cloned().collect();
    assert_eq!(abandoned.len(), 2);
    let (service, runtime) = owned(&c, &storage, &limits()).await?;
    let (entered, reached) = oneshot::channel();
    let (resume, resumed) = oneshot::channel();
    faults.pause_listing(&path, Some(&abandoned[0]), entered, resumed)?;
    faults.manifest_reads.store(0, Ordering::Relaxed);
    let (stop, task) = run_owner(whole_page(runtime)?);
    tokio::time::timeout(Duration::from_secs(30), reached).await??;
    wait_files(&path, original.len() + 1).await?;
    assert!(!abandoned[0].exists());
    assert!(abandoned[1].exists());
    let first_page_reads = faults.manifest_reads.load(Ordering::Relaxed);
    let cache = storage.session.file_metadata_cache();
    assert_eq!(cache.capacity_bytes(), Some(cache_bytes));
    assert!(cache.size_bytes().await <= cache_bytes);
    if clear {
        storage.session.file_metadata_cache().clear().await;
    }
    resume.send(()).unwrap();
    wait_files(&path, original.len()).await?;
    stop.send(()).unwrap();
    task.await??;
    let all_page_reads = faults.manifest_reads.load(Ordering::Relaxed);
    assert_eq!(files(&path)?, original);
    assert_eq!(service.reclamation_observation().deleted, 2);
    let base = annotation_publication::read(&c, &storage, None).await?;
    assert_eq!(base.count_rows(None).await?, 2);
    assert!(base.read_transaction().await?.is_some());
    c.shutdown().await?;
    eprintln!(
        "cache_bytes={cache_bytes} clear={clear}: first page GETs={first_page_reads}, total GETs={all_page_reads}"
    );
    assert_eq!(first_page_reads, 1);
    assert_eq!(all_page_reads, expected_reads);
    Ok(())
}

#[tokio::test]
async fn annotation_collector_protects_a_retired_base_until_its_reader_releases() -> Result<()> {
    let root = root()?;
    let path = root.path().join("turns.lance");
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let storage = fault_store::storage(&path, Arc::new(Faults::default())).await?;
    annotation_publication::seed_contract(&c, &storage, Service::contract(&binding(), &limits())?)
        .await?;
    let original = files(&path)?;
    let (service, runtime) = owned(&c, &storage, &limits()).await?;
    let (stop, task) = run_owner(whole_page(runtime)?);
    let producer = producer(&service, 1).await?;
    let first = request(
        &service,
        &producer,
        1,
        wire(&limits(), &["T"], 1, Some("acknowledged"), 0, None)?,
    )
    .await?;
    let first_result = service.submit(&first, Duration::from_secs(30)).await?;
    drained(&c, &service, first_result.index).await?;
    let held = service.capture_read_for_test("T").await?;
    let old_columns: Vec<_> = files(&path)?.difference(&original).cloned().collect();
    assert_eq!(old_columns.len(), 1);
    for sequence in 2..=4 {
        let update = request(
            &service,
            &producer,
            sequence,
            wire(&limits(), &["T"], 0, None, 1, Some(sequence as f32))?,
        )
        .await?;
        let result = service.submit(&update, Duration::from_secs(30)).await?;
        drained(&c, &service, result.index).await?;
        // Wait for physical collection to leave exactly the current narrow
        // file, the held old narrow file, and the two original body files.
        wait_files(&path, original.len() + 2).await?;
        // The first iteration already has that file count before the next GC
        // pass. Wait for retirement explicitly rather than racing that pass.
        tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                let state = annotation_publication::state(&c).await?;
                if !state
                    .manifests
                    .unwrap()
                    .versions
                    .contains_key(&held.base_version)
                {
                    return Ok::<_, anyhow::Error>(());
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await??;
        assert!(old_columns[0].exists());
        let old = held.resolve().await?.unwrap();
        assert_eq!(old.status.as_deref(), Some("acknowledged"));
        assert_eq!(old.score, None);
        assert_eq!(
            service
                .get("T", Duration::from_secs(30))
                .await?
                .unwrap()
                .score,
            Some(sequence as f32)
        );
    }
    drop(held);
    wait_files(&path, original.len() + 1).await?;
    assert!(!old_columns[0].exists());
    stop.send(()).unwrap();
    task.await??;
    cold_check(&c, &storage, Some(4.0)).await?;
    c.shutdown().await
}
