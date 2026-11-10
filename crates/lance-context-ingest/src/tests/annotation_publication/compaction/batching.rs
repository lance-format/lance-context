//! Maintenance byte batching must handle bodies larger than mutation requests.
use super::*;

const ROWS: usize = 64;
const BATCH_BYTES: usize = 1 << 20;

fn bodies() -> Vec<Vec<u8>> {
    (0..ROWS)
        .map(|row| {
            let len = if row % 4 == 3 {
                BATCH_BYTES + (64 << 10)
            } else {
                128 + row
            };
            let mut state = row as u64 + 1;
            (0..len)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    state as u8
                })
                .collect()
        })
        .collect()
}

async fn seed_skewed(storage: &Storage, bodies: &[Vec<u8>]) -> Result<Dataset> {
    let mut labels = ListBuilder::new(StringBuilder::new());
    for _ in bodies {
        labels.append(true);
    }
    let labels = labels.finish();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("body", DataType::LargeBinary, false),
        Field::new("labels", labels.data_type().clone(), false),
        Field::new("status", DataType::Utf8, true),
        Field::new("score", DataType::Float32, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from_iter_values(
                (0..ROWS).map(|row| format!("t{row:08}")),
            )),
            Arc::new(LargeBinaryArray::from_iter_values(bodies)),
            Arc::new(labels),
            Arc::new(StringArray::from(vec!["pending"; ROWS])),
            Arc::new(Float32Array::from(vec![None; ROWS])),
        ],
    )?;
    let mut base = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        &storage.uri,
        Some(WriteParams {
            commit_handler: Some(Arc::new(ConditionalPutCommitHandler)),
            enable_stable_row_ids: true,
            max_rows_per_file: 8,
            max_rows_per_group: 8,
            ..Default::default()
        }),
    )
    .await?;
    base.create_index_builder(
        &["id"],
        lance_index::IndexType::BTree,
        &lance_index::scalar::ScalarIndexParams::default(),
    )
    .name("annotation_point_id".into())
    .await?;
    assert_eq!(base.fragments().len(), 8);
    Ok(base)
}

fn maintenance_options() -> Options {
    Options {
        compaction: Some(materializer::SmallFileCompaction {
            target_file_bytes: 8 << 20,
            min_fragments: 2,
        }),
        ..options()
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn annotation_compaction_batches_small_rows_and_separates_oversized_bodies() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(90), exercise()).await?
}

async fn exercise() -> Result<()> {
    let root = if let Ok(parent) = std::env::var("INGEST_TEST_ARTIFACTS") {
        tempfile::Builder::new()
            .prefix("compaction-byte-batches-")
            .disable_cleanup(true)
            .tempdir_in(parent)?
    } else {
        tempfile::tempdir()?
    };
    let storage = fixture_storage(root.path()).await?;
    let limits = fixture_limits();
    let c = network::Cluster::open_signed(root.path(), one_shard()).await?;
    let bodies = bodies();
    let base = seed_skewed(&storage, &bodies).await?;
    let index_id = base.load_indices().await?[0].uuid;
    seed_base(&c, Service::contract(&fixture_binding(), &limits)?, &base).await?;
    let (service, owner) = owner(&c, &storage, &limits).await?;
    let (stop, task) = runtime::run_owner(owner);
    let p = producer(&service, 1).await?;
    // Update one turn in every fragment so maintenance must merge column files.
    let keys: Vec<_> = (0..ROWS)
        .step_by(8)
        .map(|row| format!("t{row:08}"))
        .collect();
    let key_refs: Vec<_> = keys.iter().map(String::as_str).collect();
    let request = request(
        &service,
        &p,
        1,
        wire(&limits, &key_refs, 1, Some("reviewed"), 0, None)?,
    )
    .await?;
    let result = service.submit(&request, Duration::from_secs(30)).await?;
    assert_eq!(result.result, MutationResult::Applied);
    runtime::drained(&c, &service, result.index).await?;
    stop.send(()).unwrap();
    task.await??;
    drop(service);
    let ingress = Service::validated_ingress(
        c.client.clone(),
        one_shard(),
        fixture_binding(),
        &limits,
        MergeMemoryBudget::new(32 << 20),
        &MergeMemoryBudget::new(16 << 20),
    )
    .await?;
    let publisher = materializer::Publisher::acquire(
        ingress,
        storage.clone(),
        maintenance_options(),
        MergeMemoryBudget::new(64 << 20),
        acquire(&c).await?,
    )
    .await?;
    let before = state(&c).await?;
    let version = before.manifests.as_ref().unwrap().head;
    let base = materializer::compaction::read_compaction_base(&publisher, version).await?;
    assert!(base.manifest().fragments.iter().all(|f| f.files.len() == 2));
    let mut scan = base.scan();
    scan.io_buffer_size(BATCH_BYTES as u64).scan_in_order(true);
    let mut stream = scan.try_into_stream().await?;
    let mut rows = 0;
    let mut small_shared_batch = false;
    while let Some(batch) = stream.try_next().await? {
        rows += batch.num_rows();
        let body = batch["body"]
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .unwrap();
        let oversized = (0..batch.num_rows())
            .filter(|&row| body.value(row).len() > BATCH_BYTES)
            .count();
        assert!(
            oversized <= 1,
            "maintenance combined oversized bodies in one output batch"
        );
        small_shared_batch |= oversized == 0 && batch.num_rows() > 1;
    }
    assert_eq!(rows, ROWS);
    assert!(
        small_shared_batch,
        "small rows still require one batch per row"
    );
    drop((stream, scan, base, publisher));

    let (service, owner) = Runtime::connect(
        c.client.clone(),
        one_shard(),
        fixture_binding(),
        storage.clone(),
        Configuration {
            limits: limits.clone(),
            upload_budget: MergeMemoryBudget::new(32 << 20),
            apply_budget: MergeMemoryBudget::new(16 << 20),
            publication_budget: MergeMemoryBudget::new(64 << 20),
            publication: maintenance_options(),
        },
        acquire(&c).await?,
    )
    .await?;
    let (stop, task) = runtime::run_owner(owner);
    loop {
        let view = state(&c).await?;
        if view.progress.pending.is_none() && view.manifests.as_ref().unwrap().head > version {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(
        service.submit(&request, Duration::from_secs(30)).await?,
        result
    );
    stop.send(()).unwrap();
    task.await??;
    let selected = read(&c, &storage, None).await?;
    assert!(selected.fragments().len() < 8);
    let indices = selected.load_indices().await?;
    assert_ne!(indices[0].uuid, index_id);
    assert_eq!(indices[0].index_version, 1);
    assert!(selected.manifest().fragments.iter().all(|f| indices[0]
        .fragment_bitmap
        .as_ref()
        .unwrap()
        .contains(f.id as u32)));
    let mut scan = selected.scan();
    scan.batch_size(1);
    let mut stream = scan.try_into_stream().await?;
    let mut seen = std::collections::BTreeSet::new();
    while let Some(batch) = stream.try_next().await? {
        let ids = batch["id"].as_any().downcast_ref::<StringArray>().unwrap();
        let actual = batch["body"]
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .unwrap();
        let status = batch["status"]
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for row in 0..batch.num_rows() {
            let id: usize = ids.value(row)[1..].parse()?;
            assert!(id < ROWS && seen.insert(id));
            assert_eq!(actual.value(row), bodies[id]);
            assert_eq!(
                status.value(row),
                if id.is_multiple_of(8) {
                    "reviewed"
                } else {
                    "pending"
                }
            );
            assert!(batch["score"].is_null(row));
        }
    }
    assert_eq!(seen.len(), ROWS);
    c.shutdown().await?;
    Ok(())
}
