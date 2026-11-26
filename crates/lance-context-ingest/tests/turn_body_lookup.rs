#![cfg(feature = "materializer")]

use anyhow::Result;
use arrow_array::{ArrayRef, BinaryArray, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance::{index::DatasetIndexExt, Dataset};
use lance_context_core::MergeMemoryBudget;
use lance_context_ingest::turn_ingestion::TurnIngestionSpec;
use lance_index::{
    scalar::{BuiltinIndexType, ScalarIndexParams},
    IndexType,
};
use std::sync::Arc;

fn batch(rows: &[(String, String)]) -> Result<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("session_id", DataType::Utf8, false),
        Field::new("id", DataType::Utf8, false),
        Field::new("payload", DataType::Binary, false),
    ]));
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|(session, _)| session.as_str()),
            )) as ArrayRef,
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|(_, id)| id.as_str()),
            )),
            Arc::new(BinaryArray::from_iter_values(
                rows.iter().map(|_| &[0xff, 0x00, 0x80][..]),
            )),
        ],
    )?)
}

#[tokio::test]
async fn full_body_keys_are_batched_scoped_and_work_before_and_after_zonemap() -> Result<()> {
    let root = tempfile::tempdir()?;
    let mut rows = (0..4997)
        .map(|id| (format!("session-{id}"), format!("id-{id}")))
        .collect::<Vec<_>>();
    rows.extend([
        ("a".into(), "b/c".into()),
        ("a/b".into(), "c".into()),
        ("q'中文".into(), "`\\\"".into()),
    ]);
    let input = batch(&rows)?;
    let spec = TurnIngestionSpec::new(
        input.schema(),
        vec!["session_id".into(), "id".into()],
        "session_id",
    )?;
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(input.clone())], input.schema()),
        root.path().to_str().unwrap(),
        None,
    )
    .await?;
    let keys = rows
        .iter()
        .map(|(session, id)| spec.business_key(&[session, id], 4096))
        .collect::<Result<Vec<_>>>()?;
    let requested = keys.iter().map(String::as_str).collect::<Vec<_>>();
    let workspace = MergeMemoryBudget::new(16 << 20);
    let mut old_base = None;
    for indexed in [false, true] {
        if indexed {
            dataset
                .create_index(
                    &["session_id"],
                    IndexType::ZoneMap,
                    Some("sessions".into()),
                    &ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap),
                    false,
                )
                .await?;
        }
        let base = Arc::new(dataset.clone());
        let found = spec
            .lookup_existing(base.clone(), &requested, &workspace)
            .await?
            .unwrap();
        assert!(found.matches(&base));
        assert_eq!(found.observation().requested_keys, 5000);
        assert_eq!(found.observation().scans_started, 1);
        assert_eq!(found.observation().projected_rows, 5000);
        if let Some(old) = &old_base {
            assert!(!found.matches(old));
        }
        for key in &keys {
            assert_eq!(found.get(key), Some(true));
        }
        let absent = spec.business_key(&["never", "seen"], 4096)?;
        assert_eq!(found.get(&absent), None);
        drop(found);
        let probe = spec
            .lookup_existing(base.clone(), &[&keys[0], &absent, &keys[0]], &workspace)
            .await?
            .unwrap();
        assert_eq!(probe.get(&keys[0]), Some(true));
        assert_eq!(probe.get(&absent), Some(false));
        drop(probe);
        let held = workspace.reserve(workspace.limit()).await;
        assert!(spec
            .lookup_existing(base.clone(), &[&keys[0]], &workspace)
            .await?
            .is_none());
        drop(held);
        old_base = Some(base);
    }
    Ok(())
}

#[tokio::test]
async fn composite_cross_product_candidates_do_not_create_false_hits() -> Result<()> {
    let root = tempfile::tempdir()?;
    let input = batch(&[("a".into(), "1".into()), ("b".into(), "2".into())])?;
    let spec = TurnIngestionSpec::new(
        input.schema(),
        vec!["session_id".into(), "id".into()],
        "session_id",
    )?;
    let dataset = Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(input.clone())], input.schema()),
            root.path().to_str().unwrap(),
            None,
        )
        .await?,
    );
    let first = spec.business_key(&["a", "2"], 4096)?;
    let second = spec.business_key(&["b", "1"], 4096)?;
    let result = spec
        .lookup_existing(
            dataset,
            &[&first, &second],
            &MergeMemoryBudget::new(1 << 20),
        )
        .await?
        .unwrap();
    assert_eq!(result.get(&first), Some(false));
    assert_eq!(result.get(&second), Some(false));
    assert_eq!(result.observation().scans_started, 1);
    assert_eq!(result.observation().projected_rows, 2);
    Ok(())
}
