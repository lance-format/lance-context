use super::*;
use arrow_array::{RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lance::{dataset::WriteParams, index::DatasetIndexExt, Dataset};
use lance_index::{scalar::ScalarIndexParams, IndexType};

#[tokio::test]
async fn annotation_owner_index_cache_budget_preserves_native_query_results() -> Result<()> {
    for budget in [None, Some(0), Some(2 << 20)] {
        // Existing JSON may omit the field; explicit zero must remain distinct.
        let mut encoded = serde_json::json!({
            "owner_url": "http://127.0.0.1:8000",
            "apply_memory_bytes": 8 << 20,
        });
        if let Some(bytes) = budget {
            encoded["index_cache_bytes"] = serde_json::json!(bytes);
        }
        let options: config::Annotations = serde_json::from_value(encoded)?;
        let session = owner_session(&options);
        assert_eq!(
            serde_json::to_value(&options)?.get("index_cache_bytes"),
            budget.map(serde_json::Value::from).as_ref()
        );
        assert_eq!(
            session.file_metadata_cache().capacity_bytes(),
            Some(128 << 20)
        );

        let root = tempfile::tempdir()?;
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from_iter_values(
                (0..4096).map(|row| format!("turn-{row}")),
            ))],
        )?;
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            root.path().join("turns.lance").to_str().unwrap(),
            Some(WriteParams {
                session: Some(session.clone()),
                ..Default::default()
            }),
        )
        .await?;
        dataset
            .create_index_builder(&["id"], IndexType::BTree, &ScalarIndexParams::default())
            .name("turn_id".into())
            .await?;
        let mut scan = dataset.scan();
        scan.project(&["id"])?;
        scan.filter("id = 'turn-7'")?;
        let batches: Vec<RecordBatch> = scan.try_into_stream().await?.try_collect().await?;
        let ids = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
            })
            .collect::<Vec<_>>();
        assert_eq!(ids, vec![Some("turn-7")]);
        let stats = session.index_cache_stats().await;
        if budget == Some(0) {
            assert_eq!(stats.num_entries, 0);
            assert_eq!(stats.size_bytes, 0);
        } else {
            assert!(stats.num_entries > 0);
            assert!(stats.size_bytes > 0);
            assert!(stats.size_bytes <= budget.unwrap_or(lance::dataset::DEFAULT_INDEX_CACHE_SIZE));
        }
    }
    Ok(())
}
