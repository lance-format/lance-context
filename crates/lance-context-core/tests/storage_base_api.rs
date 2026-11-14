use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use arrow_array::{LargeBinaryArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lance_context_core::{FlushOutcome, ListSource, StorageBase, StorageBaseOptions};
use tempfile::TempDir;

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("entry_key", DataType::Utf8, false).with_metadata(HashMap::from([(
            "lance-schema:unenforced-primary-key".to_string(),
            "true".to_string(),
        )])),
        Field::new("category", DataType::Utf8, false),
        Field::new("payload", DataType::LargeBinary, false),
    ]))
}

fn options() -> StorageBaseOptions {
    let mut options = StorageBaseOptions::new(schema(), "entry_key");
    options.shard_id = Some("writer".to_string());
    options
}

async fn read_batches(
    store: &StorageBase,
    filter: Option<&str>,
    columns: Option<&[&str]>,
) -> Vec<RecordBatch> {
    store
        .read_consistent(ListSource::All, |dataset, snapshots| async move {
            let mut scanner = StorageBase::lsm_scanner_for_dataset(
                &dataset,
                store.key_column(),
                ListSource::All,
                snapshots,
            );
            if let Some(columns) = columns {
                scanner = scanner.project(columns)?;
            }
            if let Some(filter) = filter {
                scanner = scanner.filter(filter)?;
            }
            Ok(scanner.try_into_stream().await?.try_collect().await?)
        })
        .await
        .unwrap()
}

fn batch(key: &str, category: &str, payload: &[u8]) -> RecordBatch {
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(StringArray::from(vec![key])),
            Arc::new(StringArray::from(vec![category])),
            Arc::new(LargeBinaryArray::from(vec![payload])),
        ],
    )
    .unwrap()
}

fn payloads(batches: &[RecordBatch]) -> BTreeMap<String, Vec<u8>> {
    let mut result = BTreeMap::new();
    for batch in batches {
        let keys = batch
            .column_by_name("entry_key")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let bytes = batch
            .column_by_name("payload")
            .unwrap()
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .unwrap();
        for row in 0..batch.num_rows() {
            assert!(result
                .insert(keys.value(row).to_string(), bytes.value(row).to_vec())
                .is_none());
        }
    }
    result
}

#[tokio::test]
async fn external_store_reads_custom_keys_and_binary_columns_across_wal_merge_and_reopen() {
    let directory = TempDir::new().unwrap();
    let uri = directory.path().to_str().unwrap();
    let mut options = options();
    let mut writer = StorageBase::open(uri, options.clone(), true).await.unwrap();
    options.shard_id = Some("reader".to_string());
    let mut reader = StorageBase::open(uri, options.clone(), false)
        .await
        .unwrap();

    // Warm a reader before another writer publishes to its WAL.
    assert!(read_batches(&reader, None, None).await.is_empty());
    let first = batch("first", "selected", b"\x00\xfffirst");
    let second = batch("second", "other", b"\x01\xfesecond");
    writer
        .put(vec![first.clone(), second.clone()])
        .await
        .unwrap();
    writer.put(vec![first, second]).await.unwrap();

    let projected = read_batches(&reader, Some("category = 'selected'"), Some(&["category"])).await;
    assert_eq!(
        projected.iter().map(RecordBatch::num_rows).sum::<usize>(),
        1
    );
    for batch in projected {
        assert_eq!(
            batch
                .schema()
                .fields()
                .iter()
                .map(|field| field.name().as_str())
                .collect::<Vec<_>>(),
            vec!["category", "entry_key"]
        );
    }
    let mut expected = BTreeMap::from([
        ("first".to_string(), b"\x00\xfffirst".to_vec()),
        ("second".to_string(), b"\x01\xfesecond".to_vec()),
    ]);
    assert_eq!(payloads(&read_batches(&reader, None, None).await), expected);

    writer.cleanup_own_shard().await.unwrap();
    writer.create_key_btree_index().await.unwrap();
    assert_eq!(writer.dataset().count_rows(None).await.unwrap(), 2);
    writer
        .put(vec![batch("third", "selected", b"\x02third")])
        .await
        .unwrap();
    expected.insert("third".to_string(), b"\x02third".to_vec());
    assert_eq!(payloads(&read_batches(&reader, None, None).await), expected);

    writer.close().await.unwrap();
    reader.close().await.unwrap();
    let mut reopened = StorageBase::open(uri, options, false).await.unwrap();
    assert_eq!(
        payloads(&read_batches(&reopened, None, None).await),
        expected
    );
    reopened.close().await.unwrap();
}

#[tokio::test]
async fn deferred_writes_become_readable_after_explicit_publication() {
    let directory = TempDir::new().unwrap();
    let uri = directory.path().to_str().unwrap();
    let mut options = options();
    options.seal_on_put = false;
    let mut writer = StorageBase::open(uri, options.clone(), true).await.unwrap();
    options.shard_id = Some("reader".to_string());
    let mut reader = StorageBase::open(uri, options, false).await.unwrap();

    assert_eq!(
        writer.flush_with_outcome().await.unwrap(),
        FlushOutcome::Noop
    );
    writer
        .put(vec![batch("pending", "selected", b"\x00\xff")])
        .await
        .unwrap();
    assert_eq!(
        payloads(&read_batches(&reader, None, None).await),
        BTreeMap::new()
    );
    assert_eq!(
        writer.flush_with_outcome().await.unwrap(),
        FlushOutcome::Sealed
    );
    assert_eq!(
        payloads(&read_batches(&reader, None, None).await),
        BTreeMap::from([("pending".to_string(), b"\x00\xff".to_vec())])
    );
    writer.close().await.unwrap();
    reader.close().await.unwrap();
}
