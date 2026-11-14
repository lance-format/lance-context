// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Keep this a separate integration binary: pool and batch configuration is
//! process-wide. A wide-key sort near the pool limit exercises a different
//! reservation pattern from the many-spilled-runs test in scalar_index_spill.

use lance::dataset::WriteParams;
use lance::index::DatasetIndexExt;
use lance::Dataset;
use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
use lance_index::IndexType;

#[tokio::test]
async fn btree_sort_honors_configured_execution_batch() {
    unsafe {
        std::env::set_var("LANCE_MEM_POOL_SIZE", (128 * 1024 * 1024).to_string());
        std::env::set_var("LANCE_DEFAULT_BATCH_SIZE", "256");
        std::env::remove_var("LANCE_BYPASS_SPILLING");
    }

    // Approximately 80 MiB of keys fits in the pool, but default-sized sort
    // output batches can require two simultaneous ~64 MiB merge reservations.
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Utf8,
        false,
    )]));
    let batch_schema = schema.clone();
    let batches = (0..80).map(move |batch| {
        let keys = (0..256).map(|row| {
            let position = batch * 256 + row;
            format!("{:08}-{}", 20_480 - position, "k".repeat(4087))
        });
        RecordBatch::try_new(
            batch_schema.clone(),
            vec![Arc::new(StringArray::from_iter_values(keys))],
        )
    });
    let data = RecordBatchIterator::new(batches, schema);
    let mut dataset = Dataset::write(
        data,
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 10_240,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.count_rows(None).await.unwrap(), 20_480);
    assert!(dataset.get_fragments().len() > 1);

    dataset
        .create_index(
            &["value"],
            IndexType::BTree,
            None,
            &ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
            false,
        )
        .await
        .unwrap();
    assert_eq!(dataset.load_indices().await.unwrap().len(), 1);
}
