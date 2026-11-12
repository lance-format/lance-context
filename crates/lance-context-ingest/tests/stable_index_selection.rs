#![cfg(feature = "materializer")]

use std::sync::Arc;

use anyhow::{ensure, Result};
use arrow_array::{ArrayRef, Int32Array, RecordBatch, RecordBatchIterator, UInt64Array};
use lance::{
    dataset::{
        optimize::{compact_files, CompactionOptions},
        rowids::load_row_id_sequence,
        WriteParams,
    },
    index::DatasetIndexExt,
    Dataset,
};
use lance_index::{metrics::NoOpMetricsCollector, scalar::ScalarIndexParams, IndexType};
use lance_table::rowids::segment::U64Segment;

// Exercise the real address-domain BTree path over compacted stable IDs. The
// oracle uses original values, not another row-ID conversion implementation.
#[tokio::test]
async fn indexed_selection_preserves_bitmap_ids_and_deletions() -> Result<()> {
    const ROWS: i32 = 4096;
    let root = tempfile::tempdir()?;
    let batch = RecordBatch::try_from_iter([
        (
            "i",
            Arc::new(Int32Array::from_iter_values(0..ROWS)) as ArrayRef,
        ),
        (
            "status",
            Arc::new(Int32Array::from_iter(
                (0..ROWS).map(|i| (i % 7 != 0).then_some(i % 3)),
            )) as ArrayRef,
        ),
    ])?;
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        root.path().to_str().unwrap(),
        Some(WriteParams {
            enable_stable_row_ids: true,
            max_rows_per_file: 1024,
            ..Default::default()
        }),
    )
    .await?;
    dataset.delete("i % 4 = 0").await?;
    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 8192,
            ..Default::default()
        },
        None,
    )
    .await?;
    let mut bitmap = false;
    for fragment in dataset.fragments().iter() {
        bitmap |= load_row_id_sequence(&dataset, fragment)
            .await?
            .segments()
            .iter()
            .any(|segment| matches!(segment, U64Segment::RangeWithBitmap { .. }));
    }
    ensure!(bitmap, "fixture did not produce compacted bitmap row IDs");
    dataset
        .create_index(
            &["status"],
            IndexType::BTree,
            Some("status_idx".into()),
            &ScalarIndexParams::default(),
            false,
        )
        .await?;
    let indices = dataset.load_indices().await?;
    let index = lance::index::scalar::open_scalar_index(
        &dataset,
        "status",
        &indices[0],
        &NoOpMetricsCollector,
    )
    .await?;
    ensure!(index.results_are_row_addresses());
    // The index still contains these rows. Their physical offsets must be
    // filtered, while survivors keep their original sequence positions.
    dataset.delete("i % 5 = 0").await?;
    for predicate in [
        "status = 1",
        "NOT (status = 1)",
        "status IS NULL",
        "status = 99",
    ] {
        let mut scan = dataset.scan();
        scan.project(&["i"])?;
        scan.with_row_id();
        scan.filter(predicate)?;
        let plan = scan.explain_plan(false).await?;
        ensure!(plan.contains("ScalarIndexQuery"), "index not used: {plan}");
        let batch = scan.try_into_batch().await?;
        let values = batch["i"].as_any().downcast_ref::<Int32Array>().unwrap();
        let ids = batch["_rowid"]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let mut actual = values
            .values()
            .iter()
            .copied()
            .zip(ids.values().iter().copied())
            .collect::<Vec<_>>();
        actual.sort_unstable();
        let expected = (0..ROWS)
            .filter(|i| i % 4 != 0 && i % 5 != 0)
            .filter(|i| match predicate {
                "status = 1" => i % 7 != 0 && i % 3 == 1,
                "NOT (status = 1)" => i % 7 != 0 && i % 3 != 1,
                "status IS NULL" => i % 7 == 0,
                "status = 99" => false,
                _ => unreachable!(),
            })
            .map(|i| (i, i as u64))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "{predicate}");
    }
    Ok(())
}
