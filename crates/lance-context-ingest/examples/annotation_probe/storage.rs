use super::*;
use arrow_array::{
    new_null_array, types::Int8Type, ArrayRef, DictionaryArray, Int32Array, Int8Array,
    LargeBinaryArray, RecordBatchIterator, TimestampMicrosecondArray,
};
use futures::TryStreamExt;
use lance::dataset::{builder::DatasetBuilder, WriteParams};
use lance::index::DatasetIndexExt;
use lance_context_ingest::materializer::Storage;
use lance_context_wal::replication::{
    publication::lance::ManagedManifest,
    runtime::client::{Reply, Request},
};
use lance_io::object_store::{ObjectStoreParams, StorageOptionsAccessor};
use lance_table::io::commit::ConditionalPutCommitHandler;

fn batch(first: usize, rows: usize) -> Result<RecordBatch, arrow_schema::ArrowError> {
    let schema = Arc::new(lance_context_core::rollout_schema());
    let ids = (first..first + rows)
        .map(|n| format!("t{n:08}"))
        .collect::<Vec<_>>();
    let payload = vec![0x5a; 16 << 10];
    let columns = schema
        .fields()
        .iter()
        .map(|f| -> Result<ArrayRef, arrow_schema::ArrowError> {
            Ok(match f.name().as_str() {
                "id" => Arc::new(StringArray::from(ids.clone())),
                "rollout_id" => Arc::new(StringArray::from(vec!["session"; rows])),
                "problem_id" => Arc::new(StringArray::from(vec!["problem"; rows])),
                "sequence_order" => Arc::new(Int32Array::from_iter_values(0..rows as i32)),
                "role" => Arc::new(DictionaryArray::<Int8Type>::try_new(
                    Int8Array::from(vec![0; rows]),
                    Arc::new(StringArray::from(vec!["assistant"])),
                )?),
                "created_at" => Arc::new(TimestampMicrosecondArray::from(vec![1234; rows])),
                "content_type" => Arc::new(StringArray::from(vec!["binary"; rows])),
                "binary_payload" => Arc::new(LargeBinaryArray::from_iter_values(
                    (0..rows).map(|_| payload.as_slice()),
                )),
                "score" => Arc::new(Float32Array::from(vec![Some(0.5); rows])),
                _ => new_null_array(f.data_type(), rows),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    RecordBatch::try_new(schema, columns)
}
async fn storage(config: &Config) -> Result<Storage> {
    let table = &config.tables[0];
    Storage::resolve(
        table.uri.clone(),
        table.storage_options.clone(),
        Arc::default(),
    )
    .await
}
pub(super) async fn seed(config: &Config, w: &Workload) -> Result<()> {
    let store = storage(config).await?;
    let turns = w.turns;
    let batches = (0..turns)
        .step_by(256)
        .map(move |n| batch(n, (turns - n).min(256)));
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(batches, Arc::new(lance_context_core::rollout_schema())),
        &store.uri,
        Some(WriteParams {
            enable_stable_row_ids: true,
            commit_handler: Some(Arc::new(ConditionalPutCommitHandler)),
            store_params: Some(ObjectStoreParams {
                storage_options_accessor: Some(Arc::new(
                    StorageOptionsAccessor::with_static_options(store.options),
                )),
                ..Default::default()
            }),
            ..Default::default()
        }),
    )
    .await?;
    let index_started = Instant::now();
    if w.id_index {
        dataset
            .create_index_builder(
                &["id"],
                lance_index::IndexType::BTree,
                &lance_index::scalar::ScalarIndexParams::default(),
            )
            .name("annotation_point_id".to_string())
            .await?;
    }
    let index_ms = index_started.elapsed().as_millis();
    let id_index_uuid = dataset
        .load_indices()
        .await?
        .iter()
        .find(|index| index.name == "annotation_point_id")
        .map(|index| index.uuid.to_string());
    println!(
        "{}",
        serde_json::json!({"source": {
        "version": dataset.version_id(), "transaction": dataset.read_transaction().await?.context("source transaction missing")?.uuid
    }, "rows": w.turns, "fragments": dataset.manifest().fragments.len(),
    "id_index_uuid": id_index_uuid, "index_build_ms": index_ms})
    );
    Ok(())
}
pub(super) async fn verify(config: &Config, w: &Workload) -> Result<()> {
    let (expected, through) = expected(w)?;
    ensure!(through > 0, "no acknowledged writes to verify");
    let store = storage(config).await?;
    let client = config.client()?;
    let table = &config.tables[0];
    let reader = ManagedManifest::reader(
        Arc::new(client.manifest_catalog()),
        store.base.clone(),
        &config.layout,
        table.binding.table,
        table.binding.incarnation,
    )?;
    let started = Instant::now();
    let dataset = loop {
        let dataset = DatasetBuilder::from_uri(&store.uri)
            .with_storage_options(store.options.clone())
            .with_session(store.session.clone())
            .with_commit_handler(Arc::new(reader.clone()))
            .load()
            .await?;
        let transaction = dataset
            .read_transaction()
            .await?
            .context("selected transaction missing")?;
        let cut = transaction
            .transaction_properties
            .as_ref()
            .and_then(|p| p.get("lance-context.annotation-cut"));
        let materialized = if let Some(cut) = cut {
            let (scope, cut): (lance_context_core::annotation_staging::Scope, u64) =
                serde_json::from_str(cut)?;
            ensure!(
                scope.table_instance == table.binding.incarnation.to_string()
                    && scope.source_group == config.layout.buckets[0]
                    && scope.shard == 0,
                "foreign materialized scope"
            );
            cut
        } else {
            0
        };
        if materialized >= through {
            let reply = client
                .request_routed(Request::Fetch {
                    group: config.layout.buckets[0],
                    index: through,
                })
                .await?;
            if matches!(reply.reply, Reply::Record(None)) {
                break dataset;
            }
        }
        ensure!(
            started.elapsed() < Duration::from_secs(180),
            "publication/source GC did not drain"
        );
        tokio::time::sleep(Duration::from_secs(1)).await;
    };
    let mut scan = dataset.scan();
    scan.project(&["id", "labels", "status", "score"])?;
    let mut stream = scan.try_into_stream().await?;
    let mut seen = vec![false; w.turns];
    while let Some(batch) = stream.try_next().await? {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .context("invalid ids")?;
        let labels = batch
            .column(1)
            .as_any()
            .downcast_ref::<ListArray>()
            .context("invalid labels")?;
        let statuses = batch
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .context("invalid statuses")?;
        let scores = batch
            .column(3)
            .as_any()
            .downcast_ref::<Float32Array>()
            .context("invalid scores")?;
        for row in 0..batch.num_rows() {
            let id: usize = ids
                .value(row)
                .strip_prefix('t')
                .context("invalid turn id")?
                .parse()?;
            ensure!(id < w.turns && !seen[id], "unknown/duplicate turn");
            seen[id] = true;
            ensure!(
                statuses.is_null(row) == expected[id].is_none()
                    && expected[id]
                        .as_deref()
                        .is_none_or(|v| statuses.value(row) == v)
                    && !scores.is_null(row)
                    && scores.value(row) == 0.5
                    && (labels.is_null(row) || labels.value_length(row) == 0),
                "final annotation mismatch at {id}"
            );
        }
    }
    ensure!(seen.into_iter().all(|v| v), "missing final rows");
    if w.id_index {
        let indices = dataset.load_indices().await?;
        let index = indices
            .iter()
            .find(|index| index.name == "annotation_point_id")
            .context("ID index disappeared after annotation publication")?;
        ensure!(
            w.expected_id_index.as_deref() == Some(index.uuid.to_string().as_str()),
            "ID index was replaced during annotation publication"
        );
        let coverage = index
            .fragment_bitmap
            .as_ref()
            .context("ID index fragment coverage missing")?;
        for fragment in dataset.manifest().fragments.iter() {
            ensure!(
                coverage.contains(u32::try_from(fragment.id)?),
                "ID index lost coverage of fragment {}",
                fragment.id
            );
        }
        let mut scan = dataset.scan();
        scan.project(&["labels", "status", "score"])?;
        scan.filter("id = 't00000000'")?;
        scan.limit(Some(2), None)?;
        scan.batch_size(128);
        let plan = scan.explain_plan(false).await?;
        ensure!(
            plan.contains("annotation_point_id"),
            "ID index not used: {plan}"
        );
        eprintln!("FINAL_ID_POINT_PLAN:\n{plan}");
    }
    println!(
        "{}",
        serde_json::json!({"verified_rows":w.turns,"through":through,
        "version":dataset.version_id(),"fragments":dataset.manifest().fragments.len(),"source_reclaimed":true,
        "verify_and_drain_ms":started.elapsed().as_millis()})
    );
    Ok(())
}
