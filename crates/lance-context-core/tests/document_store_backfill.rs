//! Maintenance must preserve compression of large extracted text.
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, RecordBatchIterator};
use arrow_schema::{Field, Schema};
use futures::TryStreamExt;
use lance_context_core::{shredding::ShredPolicy, DocumentStore};
use serde_json::json;

async fn live_data_bytes(uri: &str) -> u64 {
    let dataset = lance::Dataset::open(uri).await.unwrap();
    let files: std::collections::BTreeSet<_> = dataset
        .manifest()
        .fragments
        .iter()
        .flat_map(|fragment| fragment.files.iter().map(|file| &file.path))
        .collect();
    files
        .into_iter()
        .map(|file| {
            std::fs::metadata(std::path::Path::new(uri).join("data").join(file))
                .unwrap()
                .len()
        })
        .sum()
}

#[tokio::test]
async fn large_text_and_binary_singletons_stay_compressed_and_readable() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("singletons.lance");
    let uri = uri.to_str().unwrap();
    let text = "compressible document ".repeat(64 * 1024);
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(arrow_array::StringArray::from(vec![text.as_str()])),
        Arc::new(arrow_array::LargeStringArray::from(vec![text.as_str()])),
        Arc::new(arrow_array::BinaryArray::from(vec![text.as_bytes()])),
        Arc::new(arrow_array::LargeBinaryArray::from(vec![text.as_bytes()])),
    ];
    let schema = Arc::new(Schema::new(
        arrays
            .iter()
            .enumerate()
            .map(|(index, array)| Field::new(format!("c{index}"), array.data_type().clone(), true))
            .collect::<Vec<_>>(),
    ));
    let batch = RecordBatch::try_new(schema.clone(), arrays).unwrap();
    lance::Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], schema),
        uri,
        None,
    )
    .await
    .unwrap();
    let bytes = live_data_bytes(uri).await;
    assert!(bytes < text.len() as u64, "singletons stored {bytes} bytes");
    let reopened = lance::Dataset::open(uri).await.unwrap();
    let batches: Vec<RecordBatch> = reopened
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].columns(), batch.columns());
}

#[tokio::test]
async fn backfill_preserves_large_text_compression_and_values() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("documents.lance");
    let uri = uri.to_str().unwrap();
    let mut store = DocumentStore::create(uri, ShredPolicy::default(), None)
        .await
        .unwrap();
    // Each value exceeds both the read batch target and writer page buffer.
    // The two different values compress well; a one-row constant page must not
    // bypass compression and expand them into tens of megabytes on backfill.
    let bodies: Vec<_> = (0..2)
        .map(|row| format!("record {row} {}", "compressible body ".repeat(1_200_000)))
        .collect();
    let documents: Vec<_> = bodies.iter().map(|body| json!({"body": body})).collect();
    let ids = store.add(documents).await.unwrap();
    let before = live_data_bytes(uri).await;
    store.backfill().await.unwrap();
    let after = live_data_bytes(uri).await;
    assert!(
        after < before * 2,
        "backfill expanded compressed text from {before} to {after} bytes"
    );
    drop(store);
    let reopened = DocumentStore::open(uri, None).await.unwrap();
    for (id, body) in ids.iter().zip(bodies) {
        assert_eq!(reopened.get(id).await.unwrap(), Some(json!({"body": body})));
    }
}
