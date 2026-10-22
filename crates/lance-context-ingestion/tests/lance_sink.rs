#![cfg(feature = "lance")]

use std::sync::Arc;

use arrow_array::{Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lance::dataset::{Dataset, WriteParams};
use lance_context_ingestion::lance_sink::{encode_records, stage, table_schema, LanceTableSink};
use lance_context_ingestion::{Binding, Consumer, Entry, Journal, Sink, Transition};
use lance_file::version::LanceFileVersion;
use lance_table::io::commit::commit_handler_from_url;
use object_store::{memory::InMemory, path::Path};

fn binding(partition: u32) -> Binding {
    Binding {
        run: "isolated-run".into(),
        schema: "aligned-v2".into(),
        partition,
    }
}

async fn table(uri: &str) -> Dataset {
    let schema = Arc::new(table_schema(
        &Schema::new(vec![Field::new("id", DataType::Utf8, false)]),
        &binding(0).run,
        &binding(0).schema,
    ));
    let empty = RecordBatch::new_empty(schema.clone());
    Dataset::write(
        RecordBatchIterator::new(vec![Ok(empty)], schema),
        uri,
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await
    .unwrap()
}

async fn sink(dataset: Dataset) -> LanceTableSink {
    let handler = commit_handler_from_url(dataset.uri(), &None).await.unwrap();
    LanceTableSink::new(dataset, handler, 8 << 20).unwrap()
}

fn entry(dataset: &Dataset, sequence: u64, id: Option<&str>) -> Entry {
    let records = id
        .map(|id| {
            let batch = RecordBatch::try_new(
                Arc::new(Schema::from(dataset.schema())),
                vec![Arc::new(StringArray::from(vec![id]))],
            )
            .unwrap();
            encode_records(&[batch]).unwrap()
        })
        .unwrap_or_default();
    Entry {
        sequence,
        session: "session".into(),
        receipt: format!("receipt-{sequence}"),
        input_digest: format!("input-{sequence}"),
        transition: Transition {
            delta: vec![],
            records,
        },
    }
}

async fn ids(dataset: &Dataset) -> Vec<String> {
    let batches = dataset
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let mut ids = batches
        .iter()
        .flat_map(|batch| {
            let strings = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            (0..strings.len())
                .map(|i| strings.value(i).to_owned())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    ids.sort();
    ids
}

#[tokio::test]
async fn immutable_staging_and_atomic_multi_partition_publication() {
    let dir = tempfile::tempdir().unwrap();
    let dataset = table(dir.path().to_str().unwrap()).await;
    let original = dataset.version().version;
    let a = stage(
        &dataset,
        &binding(0),
        &[entry(&dataset, 1, Some("a")), entry(&dataset, 2, None)],
        8 << 20,
    )
    .await
    .unwrap();
    let b = stage(
        &dataset,
        &binding(1),
        &[entry(&dataset, 1, Some("b"))],
        8 << 20,
    )
    .await
    .unwrap();
    assert_eq!((a.rows(), a.first_sequence(), a.last_sequence()), (1, 1, 2));
    let unchanged = Dataset::open(dataset.uri()).await.unwrap();
    assert_eq!(unchanged.version().version, original);
    assert!(ids(&unchanged).await.is_empty());
    let mut sink = sink(dataset).await;
    assert_eq!(sink.commit_staged(vec![a, b]).await.unwrap(), 2);
    assert_eq!(sink.dataset().version().version, original + 1);
    assert_eq!(sink.covered_sequence(&binding(0)).await.unwrap(), 2);
    assert_eq!(sink.covered_sequence(&binding(1)).await.unwrap(), 1);
    assert_eq!(ids(sink.dataset()).await, vec!["a", "b"]);
    assert_eq!(sink.dataset().manifest().data_storage_format.version, "2.2");
}

#[tokio::test]
async fn reopen_and_regroup_retries_preserve_exact_output_and_empty_coverage() {
    let dir = tempfile::tempdir().unwrap();
    let dataset = table(dir.path().to_str().unwrap()).await;
    let entries = vec![
        entry(&dataset, 1, Some("a")),
        entry(&dataset, 2, None),
        entry(&dataset, 3, Some("c")),
        entry(&dataset, 4, None),
    ];
    let mut first = sink(dataset).await;
    first.apply(&binding(0), &entries[..2]).await.unwrap();
    let mut restarted = sink(Dataset::open(dir.path().to_str().unwrap()).await.unwrap()).await;
    restarted.apply(&binding(0), &entries).await.unwrap();
    let version = restarted.dataset().version().version;
    restarted.apply(&binding(0), &entries).await.unwrap();
    assert_eq!(restarted.dataset().version().version, version);
    assert_eq!(restarted.covered_sequence(&binding(0)).await.unwrap(), 4);
    assert_eq!(ids(restarted.dataset()).await, vec!["a", "c"]);
}

#[tokio::test]
async fn wal_batches_merge_independently_and_lost_consumer_cursor_does_not_duplicate() {
    let dir = tempfile::tempdir().unwrap();
    let dataset = table(dir.path().to_str().unwrap()).await;
    let journal = Journal::new(
        Arc::new(InMemory::new()),
        Path::from("test"),
        binding(0),
        1 << 20,
        100,
    )
    .unwrap();
    let mut writer = journal.acquire().await.unwrap();
    for sequence in 1..=5 {
        writer
            .append(vec![entry(
                &dataset,
                sequence,
                Some(&format!("id-{sequence}")),
            )])
            .await
            .unwrap();
    }
    let mut first = sink(dataset).await;
    let mut consumer = Consumer::open(journal.clone(), "table").await.unwrap();
    assert_eq!(consumer.consume(&mut first, 3, 8 << 20).await.unwrap(), 3);
    let version = first.dataset().version().version;
    // A different cursor starts at zero, modeling a table commit whose cursor ACK was lost.
    let mut retry = Consumer::open(journal, "lost-cursor").await.unwrap();
    let mut restarted = sink(Dataset::open(dir.path().to_str().unwrap()).await.unwrap()).await;
    assert_eq!(retry.consume(&mut restarted, 5, 8 << 20).await.unwrap(), 5);
    assert_eq!(restarted.dataset().version().version, version + 1);
    assert_eq!(restarted.covered_sequence(&binding(0)).await.unwrap(), 5);
    assert_eq!(ids(restarted.dataset()).await.len(), 5);
}

#[tokio::test]
async fn gaps_partial_overlaps_and_wrong_identity_cannot_advance_coverage() {
    let dir = tempfile::tempdir().unwrap();
    let dataset = table(dir.path().to_str().unwrap()).await;
    let entries = vec![entry(&dataset, 1, Some("a")), entry(&dataset, 2, Some("b"))];
    let overlap = stage(&dataset, &binding(0), &entries, 8 << 20)
        .await
        .unwrap();
    let gap = stage(
        &dataset,
        &binding(0),
        &[entry(&dataset, 3, Some("c"))],
        8 << 20,
    )
    .await
    .unwrap();
    let mut wrong = binding(0);
    wrong.run = "different-run".into();
    assert!(stage(&dataset, &wrong, &entries, 8 << 20).await.is_err());
    assert!(stage(
        &dataset,
        &binding(0),
        &[entries[1].clone(), entries[0].clone()],
        8 << 20
    )
    .await
    .is_err());
    let mut sink = sink(dataset).await;
    sink.apply(&binding(0), &entries[..1]).await.unwrap();
    let version = sink.dataset().version().version;
    assert!(sink.commit_staged(vec![overlap]).await.is_err());
    assert!(sink.commit_staged(vec![gap]).await.is_err());
    assert_eq!(sink.dataset().version().version, version);
    assert_eq!(sink.covered_sequence(&binding(0)).await.unwrap(), 1);
    assert_eq!(ids(sink.dataset()).await, vec!["a"]);
}

#[tokio::test]
async fn duplicate_staging_workers_cannot_insert_same_partition_twice() {
    let dir = tempfile::tempdir().unwrap();
    let dataset = table(dir.path().to_str().unwrap()).await;
    let entries = vec![entry(&dataset, 1, Some("a"))];
    let a = stage(&dataset, &binding(0), &entries, 8 << 20)
        .await
        .unwrap();
    let b = stage(&dataset, &binding(0), &entries, 8 << 20)
        .await
        .unwrap();
    let mut sink = sink(dataset).await;
    assert_eq!(sink.commit_staged(vec![a]).await.unwrap(), 1);
    let version = sink.dataset().version().version;
    assert_eq!(sink.commit_staged(vec![b]).await.unwrap(), 0);
    assert_eq!(sink.dataset().version().version, version);
    assert_eq!(ids(sink.dataset()).await, vec!["a"]);
}

#[tokio::test]
async fn schema_change_and_decoding_budget_fail_before_publication() {
    let dir = tempfile::tempdir().unwrap();
    let mut dataset = table(dir.path().to_str().unwrap()).await;
    let entries = vec![entry(&dataset, 1, Some("a"))];
    assert!(stage(&dataset, &binding(0), &entries, 1).await.is_err());
    let staged = stage(&dataset, &binding(0), &entries, 8 << 20)
        .await
        .unwrap();
    dataset
        .update_schema_metadata([("changed", "schema")])
        .await
        .unwrap();
    let mut sink = sink(dataset).await;
    assert!(sink.commit_staged(vec![staged]).await.is_err());
    assert_eq!(sink.covered_sequence(&binding(0)).await.unwrap(), 0);
    assert!(ids(sink.dataset()).await.is_empty());
}

#[derive(Debug)]
struct CommitThenFail {
    delegate: Arc<dyn lance_table::io::commit::CommitHandler>,
}

#[async_trait::async_trait]
impl lance_table::io::commit::CommitHandler for CommitThenFail {
    async fn commit(
        &self,
        manifest: &mut lance_table::format::Manifest,
        indices: Option<Vec<lance_table::format::IndexMetadata>>,
        base: &Path,
        store: &lance_io::object_store::ObjectStore,
        writer: lance_table::io::commit::ManifestWriter,
        scheme: lance_table::io::commit::ManifestNamingScheme,
        transaction: Option<lance_table::format::Transaction>,
    ) -> std::result::Result<
        lance_table::io::commit::ManifestLocation,
        lance_table::io::commit::CommitError,
    > {
        self.delegate
            .commit(manifest, indices, base, store, writer, scheme, transaction)
            .await?;
        Err(lance_table::io::commit::CommitError::OtherError(
            lance::Error::invalid_input("injected lost successful commit response"),
        ))
    }
}

#[tokio::test]
async fn lost_manifest_ack_poisons_sink_then_recovers_atomic_coverage() {
    let dir = tempfile::tempdir().unwrap();
    let dataset = table(dir.path().to_str().unwrap()).await;
    let entries = vec![entry(&dataset, 1, Some("a"))];
    let handler = Arc::new(CommitThenFail {
        delegate: commit_handler_from_url(dataset.uri(), &None).await.unwrap(),
    });
    let mut failed = LanceTableSink::new(dataset, handler, 8 << 20).unwrap();
    assert!(failed.apply(&binding(0), &entries).await.is_err());
    assert!(matches!(
        failed.apply(&binding(0), &entries).await,
        Err(lance_context_ingestion::Error::Fenced)
    ));
    let mut recovered = sink(Dataset::open(dir.path().to_str().unwrap()).await.unwrap()).await;
    assert_eq!(recovered.covered_sequence(&binding(0)).await.unwrap(), 1);
    recovered.apply(&binding(0), &entries).await.unwrap();
    assert_eq!(ids(recovered.dataset()).await, vec!["a"]);
}

#[tokio::test]
async fn physical_files_are_22_and_contain_zstd_frames_with_lossless_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let dataset = table(dir.path().to_str().unwrap()).await;
    let payload =
        "a repeated compressible string with unicode 你好 and escaped quotes \"\\\n".repeat(8000);
    // A constant-valued page uses Lance's scalar layout before compression selection.
    // Distinct values exercise the actual configurable Zstd path.
    let second = format!("{payload}second");
    let entries = vec![
        entry(&dataset, 1, Some(&payload)),
        entry(&dataset, 2, Some(&second)),
    ];
    let mut sink = sink(dataset).await;
    sink.apply(&binding(0), &entries).await.unwrap();
    assert_eq!(ids(sink.dataset()).await, vec![payload.clone(), second]);
    let mut zstd_frame_found = false;
    for fragment in sink.dataset().get_fragments() {
        for file in &fragment.metadata().files {
            assert_eq!((file.file_major_version, file.file_minor_version), (2, 2));
            let bytes = std::fs::read(dir.path().join("data").join(&file.path)).unwrap();
            zstd_frame_found |= bytes
                .windows(4)
                .any(|window| window == [0x28, 0xb5, 0x2f, 0xfd]);
            assert!(
                bytes.len() < payload.len() / 4,
                "large repeated value should be physically compressed"
            );
        }
    }
    assert!(
        zstd_frame_found,
        "actual file must contain a Zstandard frame, not merely schema annotations"
    );
}
