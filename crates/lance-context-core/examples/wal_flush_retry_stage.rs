//! Bounded Azure qualification of retained-generation flush retries.
//! Only creates UUID fixtures under the fixed staging prefix; no production URI input.
use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance::dataset::mem_wal::write::{ShardWriter, ShardWriterConfig};
use lance_io::object_store::{ObjectStore, ObjectStoreRegistry};
use std::{collections::BTreeMap, sync::Arc, time::Duration};
use uuid::Uuid;

// Repository-only fault fixture, not a feature or hook in production binaries.
#[allow(dead_code)]
#[path = "../../../vendor/lance/src/dataset/mem_wal/test_util.rs"]
mod fault_store;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const PREFIX: &str = "abfss://snowflake-rollouts@falconphxdataprocessing.dfs.core.windows.net/rocketkeep-staging/flush-retry-48455";

fn batch(schema: Arc<Schema>, start: i32, end: i32) -> RecordBatch {
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from_iter_values(start..end)),
            Arc::new(StringArray::from_iter_values(
                (start..end).map(|id| format!("fixture-{id}")),
            )),
        ],
    )
    .unwrap()
}

fn collect(batch: &RecordBatch, rows: &mut BTreeMap<i32, String>) {
    let ids = batch
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let values = batch
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    for i in 0..batch.num_rows() {
        assert!(
            rows.insert(ids.value(i), values.value(i).to_owned())
                .is_none(),
            "duplicate fixture ID"
        );
    }
}

async fn qualify(case: &str) -> Result<()> {
    let uri = format!("{PREFIX}/stage_{}.lance", Uuid::new_v4().simple());
    let shard_id = Uuid::new_v4();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false).with_metadata(
            [(
                "lance-schema:unenforced-primary-key".to_owned(),
                "1".to_owned(),
            )]
            .into_iter()
            .collect(),
        ),
        Field::new("name", DataType::Utf8, true),
    ]));
    let (params, controls) = fault_store::observable_store_params();
    let (store, base) =
        ObjectStore::from_uri_and_params(Arc::new(ObjectStoreRegistry::default()), &uri, &params)
            .await?;
    let config = ShardWriterConfig {
        shard_id,
        shard_spec_id: 0,
        durable_write: true,
        max_wal_buffer_size: 1024 * 1024,
        max_wal_flush_interval: Some(Duration::from_millis(10)),
        max_memtable_size: 8 * 1024 * 1024,
        manifest_scan_batch_size: 2,
        store_params: Some(params.clone()),
        ..Default::default()
    };
    println!(
        "{}",
        serde_json::json!({"event":"start", "case":case, "uri":uri, "shard":shard_id})
    );
    let writer = ShardWriter::open(
        store.clone(),
        base.clone(),
        uri.clone(),
        config.clone(),
        schema.clone(),
        vec![],
    )
    .await?;
    writer.put(vec![batch(schema.clone(), 0, 3)]).await?;
    let boundary = writer.wal_stats().next_wal_entry_position - 1;
    assert!(boundary > 0);
    if case == "lost_ack" {
        controls.lose_shard_manifest_acks(1);
        let first = writer.force_seal_active().await?.wait().await;
        assert_eq!(
            controls.shard_manifest_lost_acks(),
            1,
            "fault must reach real Azure publication"
        );
        if let Err(error) = &first {
            assert!(error
                .to_string()
                .contains("injected shard manifest lost acknowledgement"));
        }
        // Preparation must be reused even if the public seal initially failed.
        let writes = controls.writes_under("_gen_1/");
        assert!(writes > 0);
        writer.force_seal_active().await?.wait().await?;
        assert_eq!(
            controls.writes_under("_gen_1/"),
            writes,
            "retry rewrote immutable files"
        );
        let manifest = writer.manifest().await?.unwrap();
        assert_eq!(manifest.sstables.len(), 1);
        assert_eq!(manifest.replay_after_wal_entry_position, boundary);
        println!(
            "{}",
            serde_json::json!({"event":"lost_ack_reconciled", "generation_writes":writes})
        );
    } else {
        controls.fail_generation_manifest_puts(1000);
        let error = writer
            .force_seal_active()
            .await?
            .wait()
            .await
            .expect_err("fault must surface");
        assert!(error
            .to_string()
            .contains("injected generation manifest publication failure"));
        assert!(controls.generation_manifest_failures() > 0);
        assert!(writer.memory().frozen_bytes() > 0);
        if case != "idle" {
            writer.put(vec![batch(schema.clone(), 3, 5)]).await?;
            assert!(writer.wal_stats().next_wal_entry_position - 1 > boundary);
        }
        controls.fail_generation_manifest_puts(0);
        if case == "later_seal" {
            writer.force_seal_active().await?.wait().await?;
        } else {
            // No seal/close/new write after healing: the idle timer must retry.
            tokio::time::timeout(Duration::from_secs(60), async {
                while writer.memory().frozen_bytes() != 0 {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            })
            .await?;
            let manifest = writer.manifest().await?.unwrap();
            assert_eq!(manifest.sstables.len(), 1);
            assert_eq!(
                manifest.replay_after_wal_entry_position, boundary,
                "idle retry used newer active WAL boundary"
            );
            assert_eq!(manifest.sstables[0].physical_rows, Some(3));
        }
        println!(
            "{}",
            serde_json::json!({"event":"publication_recovered", "injected_failures":controls.generation_manifest_failures()})
        );
    }
    // Discard the original in-memory union without graceful close. Reopen must
    // account for every acknowledged row, including a newer unsealed WAL tail.
    writer.abort().await?;
    drop(writer);
    let reopened = ShardWriter::open(store, base, uri.clone(), config, schema, vec![]).await?;
    let mut rows = BTreeMap::new();
    collect(&reopened.scan().await?.try_into_batch().await?, &mut rows);
    let manifest = reopened.manifest().await?.unwrap();
    for generation in &manifest.sstables {
        let generation_uri = format!("{uri}/_mem_wal/{shard_id}/{}", generation.path);
        let dataset = lance::Dataset::open(&generation_uri).await?;
        let mut scan = dataset.scan();
        scan.project(&["id", "name"])?;
        collect(&scan.try_into_batch().await?, &mut rows);
    }
    let count = if case == "idle" || case == "lost_ack" {
        3
    } else {
        5
    };
    assert_eq!(
        rows,
        (0..count).map(|id| (id, format!("fixture-{id}"))).collect()
    );
    reopened.close().await?;
    println!(
        "{}",
        serde_json::json!({"event":"pass", "case":case, "rows":count, "uri":uri, "shard":shard_id})
    );
    Ok(())
}

fn main() -> Result<()> {
    let case = std::env::args()
        .nth(1)
        .ok_or("expected idle, idle_tail, later_seal, or lost_ack")?;
    if std::env::args().nth(2).is_some()
        || !["idle", "idle_tail", "later_seal", "lost_ack"].contains(&case.as_str())
    {
        return Err("refusing unknown staging arguments".into());
    }
    if std::env::var("ETCD_ENDPOINTS").is_ok_and(|s| !s.is_empty()) {
        return Err("staging driver must not inherit etcd endpoints".into());
    }
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    rt.block_on(async { tokio::time::timeout(Duration::from_secs(180), qualify(&case)).await })??;
    Ok(())
}
