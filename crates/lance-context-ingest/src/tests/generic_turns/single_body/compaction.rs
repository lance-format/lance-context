use super::*;
use crate::materializer::SmallFileCompaction;
use futures::TryStreamExt;
use std::collections::BTreeMap;

fn compact_binding() -> TableBinding {
    TableBinding {
        max_record_bytes: 256 << 10,
        ..binding()
    }
}

async fn compact_owner(
    c: &network::Cluster,
    storage: &Storage,
    spec: &TurnIngestionSpec,
    enabled: bool,
) -> Result<(Service, Runtime)> {
    let view = annotation_publication::state(c).await?;
    Runtime::connect_body_shard(
        c.client.clone(),
        annotation_publication::one_shard(),
        compact_binding(),
        storage.clone(),
        Configuration {
            limits: fixture::limits(),
            upload_budget: MergeMemoryBudget::new(8 << 20),
            apply_budget: MergeMemoryBudget::new(8 << 20),
            publication_budget: MergeMemoryBudget::new(8 << 20),
            publication: Options {
                max_source_bytes: 1 << 20,
                min_source_bytes: 1 << 20,
                max_records: 2,
                max_batch_wait: Duration::from_millis(20),
                compaction: enabled.then_some(SmallFileCompaction {
                    target_file_bytes: 256 << 10,
                    min_fragments: 2,
                    ..Default::default()
                }),
                ..Default::default()
            },
        },
        Acquisition {
            owner: Uuid::new_v4(),
            request: Uuid::new_v4(),
            expected_epoch: view.authority.epoch,
        },
        Shard { bucket: 0 },
        spec,
    )
    .await
}

async fn row_ids(base: &lance::Dataset) -> Result<BTreeMap<(String, String), u64>> {
    let mut scan = base.scan();
    scan.project(&["session_id", "id"])?.with_row_id();
    let mut stream = scan.try_into_stream().await?;
    let mut found = BTreeMap::new();
    while let Some(batch) = stream.try_next().await? {
        let session = batch
            .column_by_name("session_id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let id = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let rows = batch
            .column_by_name("_rowid")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .unwrap();
        for row in 0..batch.num_rows() {
            assert!(found
                .insert(
                    (session.value(row).into(), id.value(row).into()),
                    rows.value(row)
                )
                .is_none());
        }
    }
    Ok(found)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn full_body_compaction_preserves_bodies_row_ids_and_request_results() -> Result<()> {
    let input = bodies()?;
    let spec = TurnIngestionSpec::new(
        input[0].schema(),
        vec!["session_id".into(), "id".into()],
        "session_id",
    )?;
    let root = tempfile::tempdir()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let storage = fixture::storage(root.path(), "full-body.lance").await?;
    crate::bootstrap::provision_body_shard(
        c.client.clone(),
        annotation_publication::one_shard(),
        compact_binding(),
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
    let (service, runtime) = compact_owner(&c, &storage, &spec, false).await?;
    let (stop, task) = run_owner(runtime);
    let lane = producer(&service, 1).await?;
    let mut last = None;
    for (sequence, index) in [0, 2, 3].into_iter().enumerate() {
        let bytes = fixture::encoded(single_body::batch(
            &spec,
            &fixture::limits(),
            Shard { bucket: 0 },
            &input[index],
        )?)?;
        let append = request(&service, &lane, sequence as u64 + 1, bytes).await?;
        let result = service.submit(&append, DEADLINE).await?;
        drained(&c, &service, result.index).await?;
        last = Some((append, result));
    }
    let (last, completed) = last.unwrap();
    let selected = service.selected_full_body(DEADLINE).await?;
    assert_eq!(selected.rows, Some(6));
    let old = annotation_publication::read(&c, &storage, Some(selected.manifest.version)).await?;
    assert_eq!(old.manifest().fragments.len(), 3);
    let before = service.observation_with_manifest().await.0.installed_base;
    assert_eq!(before.version, old.version_id());
    let before_indices = service.inspect_installed_indices().await?;
    assert_eq!(before_indices.version, before.version);
    assert_eq!(before_indices.source, before.source);
    assert_eq!(before_indices.fragments, 3);
    assert!(before_indices.indices.is_empty());
    let fragments = before.fragment_summary.unwrap();
    assert_eq!(fragments.fragments, 3);
    assert_eq!(fragments.known_size_fragments, 3);
    assert_eq!(fragments.known_physical_rows_fragments, 3);
    assert_eq!(fragments.known_physical_rows, Some(6));
    assert_eq!(fragments.small_fragments[0].fragments, 3);
    assert_eq!(
        fragments.small_fragments[0].bytes,
        fragments.known_size_bytes
    );
    let ids = row_ids(&old).await?;
    assert_eq!(ids.len(), 6);
    stop.send(()).unwrap();
    task.await??;
    drop(service);
    let (service, runtime) = compact_owner(&c, &storage, &spec, true).await?;
    let (stop, mut task) = run_owner(runtime);
    let rewritten = tokio::time::timeout(DEADLINE, async {
        loop {
            let selected = service.selected_full_body(DEADLINE).await?;
            let current =
                annotation_publication::read(&c, &storage, Some(selected.manifest.version)).await?;
            if current.manifest().fragments.len() == 1 {
                assert_eq!(selected.rows, Some(6));
                return Ok::<_, anyhow::Error>(current);
            }
            tokio::select! {
                stopped = &mut task => bail!("single-body compactor stopped: {stopped:?}"),
                _ = tokio::time::sleep(Duration::from_millis(10)) => {},
            }
        }
    })
    .await??;
    assert_eq!(row_ids(&rewritten).await?, ids);
    // The old exact read remains valid; compaction does not authorize file GC.
    assert_eq!(row_ids(&old).await?, ids);
    for (batch, row) in [(0, 0), (2, 2), (3, 0)] {
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
    assert_eq!(service.submit(&last, DEADLINE).await?, completed);
    drained(&c, &service, completed.index).await?;
    stop.send(()).unwrap();
    task.await??;
    assert!(service
        .observation()
        .await
        .0
        .installed_base
        .fragment_summary
        .is_none());
    let installed = service.observation_with_manifest().await.0.installed_base;
    assert_eq!(installed.rows, Some(6));
    assert_eq!(installed.version, rewritten.version_id());
    let after_indices = service.inspect_installed_indices().await?;
    assert_eq!(after_indices.version, installed.version);
    assert_eq!(after_indices.materialized, before_indices.materialized);
    assert_eq!(after_indices.fragments, 1);
    let fragments = installed.fragment_summary.unwrap();
    assert_eq!(fragments.fragments, 1);
    assert_eq!(fragments.known_size_fragments, 1);
    assert_eq!(fragments.known_physical_rows, Some(6));
    c.shutdown().await
}
