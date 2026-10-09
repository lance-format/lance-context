#![cfg(feature = "lance")]

use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lance::{dataset::WriteParams, Dataset};
use lance_context_ingestion::{
    local_lance::{log_schema, restore_checkpoint, AlignedCall, LocalLancePartition, Mutation},
    Binding,
};
use lance_file::version::LanceFileVersion;
use lance_table::io::commit::commit_handler_from_url;

fn binding() -> Binding {
    Binding {
        run: "test-run".into(),
        schema: "typed-state-v1".into(),
        partition: 7,
    }
}
fn records(values: &[&str]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "content",
            DataType::Utf8,
            false,
        )])),
        vec![Arc::new(StringArray::from(values.to_vec()))],
    )
    .unwrap()
}
async fn table(uri: &str) -> Dataset {
    let schema = Arc::new(log_schema(records(&[]).schema().as_ref(), &binding()));
    Dataset::write(
        RecordBatchIterator::new(vec![Ok(RecordBatch::new_empty(schema.clone()))], schema),
        uri,
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await
    .unwrap()
}
async fn open(path: &std::path::Path, dataset: Dataset) -> LocalLancePartition {
    let handler = commit_handler_from_url(dataset.uri(), &None).await.unwrap();
    LocalLancePartition::open(path, dataset, binding(), handler, 1 << 20, 16 << 20, 2)
        .await
        .unwrap()
}
fn call(sequence: u64, session: &str, value: Option<&[u8]>, output: &[&str]) -> AlignedCall {
    AlignedCall {
        sequence,
        session: session.into(),
        receipt: format!("call-{sequence}"),
        input_digest: format!("input-{sequence}"),
        mutations: vec![Mutation {
            key: b"node".to_vec(),
            value: value.map(<[u8]>::to_vec),
        }],
        records: records(output),
    }
}

#[tokio::test]
async fn batch_commit_restart_and_empty_disk_restore_preserve_binary_state_and_typed_output() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("wal.lance");
    let dataset = table(uri.to_str().unwrap()).await;
    let local = dir.path().join("local.redb");
    let mut writer = open(&local, dataset).await;
    let base = writer.dataset().version().version;
    let calls = [
        call(
            1,
            "session-a",
            Some(&[0, 255, 13]),
            &["α", "escaped\nmessage"],
        ),
        call(2, "session-b", Some(b"old"), &["other"]),
        call(3, "session-b", None, &[]),
    ];
    assert_eq!(writer.commit(&calls).await.unwrap(), 3);
    assert_eq!(writer.dataset().version().version, base + 1);
    assert_eq!(
        writer.get("session-a", b"node").unwrap(),
        Some(vec![0, 255, 13])
    );
    assert_eq!(writer.get("session-b", b"node").unwrap(), None);
    assert_eq!(
        writer.receipt("session-b", "call-3", "input-3").unwrap(),
        Some(3)
    );
    assert!(writer.receipt("session-b", "call-3", "changed").is_err());
    assert!(writer.commit(&calls).await.is_err());
    let mut scan = writer.dataset().scan();
    scan.filter("kind = 3")
        .unwrap()
        .project(&["record.content"])
        .unwrap();
    let rows = scan
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    assert_eq!(rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
    assert_eq!(
        writer.dataset().manifest().data_storage_format.version,
        "2.2"
    );
    drop(writer);
    let writer = open(&local, Dataset::open(uri.to_str().unwrap()).await.unwrap()).await;
    assert_eq!(writer.through_sequence(), 3);
    drop(writer);
    std::fs::remove_file(&local).unwrap();
    let writer = open(&local, Dataset::open(uri.to_str().unwrap()).await.unwrap()).await;
    assert_eq!(
        writer.get("session-a", b"node").unwrap(),
        Some(vec![0, 255, 13])
    );
    assert_eq!(writer.get("session-b", b"node").unwrap(), None);
    assert_eq!(
        writer.receipt("session-b", "call-3", "input-3").unwrap(),
        Some(3)
    );
}

#[tokio::test]
async fn stale_aligner_cannot_commit_or_ack_different_output_at_the_same_sequence() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("wal.lance");
    let dataset = table(uri.to_str().unwrap()).await;
    let mut a = open(&dir.path().join("a.redb"), dataset.clone()).await;
    let mut b = open(&dir.path().join("b.redb"), dataset).await;
    a.commit(&[call(1, "s", Some(b"winner"), &["winner"])])
        .await
        .unwrap();
    assert!(b
        .commit(&[call(1, "s", Some(b"loser"), &["loser"])])
        .await
        .is_err());
    assert!(b.get("s", b"node").is_err());
    drop(b);
    let recovered = open(
        &dir.path().join("b.redb"),
        Dataset::open(uri.to_str().unwrap()).await.unwrap(),
    )
    .await;
    assert_eq!(
        recovered.get("s", b"node").unwrap(),
        Some(b"winner".to_vec())
    );
    assert_eq!(
        recovered.receipt("s", "call-1", "input-1").unwrap(),
        Some(1)
    );
}

#[tokio::test]
async fn acknowledged_state_reads_are_local_and_session_keys_do_not_collide() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("wal.lance");
    let dataset = table(uri.to_str().unwrap()).await;
    let mut writer = open(&dir.path().join("cache.redb"), dataset).await;
    writer
        .commit(&[
            call(1, "a\0b", Some(b"one"), &["one"]),
            call(2, "a", Some(b"two"), &["two"]),
        ])
        .await
        .unwrap();
    std::fs::rename(&uri, dir.path().join("offline.lance")).unwrap();
    assert_eq!(writer.get("a\0b", b"node").unwrap(), Some(b"one".to_vec()));
    assert_eq!(writer.get("a", b"node").unwrap(), Some(b"two".to_vec()));
    assert_eq!(writer.receipt("a", "call-2", "input-2").unwrap(), Some(2));
}

#[tokio::test]
async fn missing_committed_file_fails_recovery_without_advancing_local_state() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("wal.lance");
    let dataset = table(uri.to_str().unwrap()).await;
    let mut writer = open(&dir.path().join("writer.redb"), dataset).await;
    writer
        .commit(&[call(1, "s", Some(b"v"), &["r"])])
        .await
        .unwrap();
    let dataset = Dataset::open(uri.to_str().unwrap()).await.unwrap();
    let file = dataset.get_fragments()[0].metadata().files[0].path.clone();
    let path = uri.join("data").join(file);
    let hidden = path.with_extension("hidden");
    std::fs::rename(&path, &hidden).unwrap();
    let handler = commit_handler_from_url(dataset.uri(), &None).await.unwrap();
    let local = dir.path().join("fresh.redb");
    assert!(
        LocalLancePartition::open(&local, dataset, binding(), handler, 1 << 20, 16 << 20, 2)
            .await
            .is_err()
    );
    std::fs::rename(hidden, path).unwrap();
    let recovered = open(&local, Dataset::open(uri.to_str().unwrap()).await.unwrap()).await;
    assert_eq!(recovered.through_sequence(), 1);
    assert_eq!(recovered.get("s", b"node").unwrap(), Some(b"v".to_vec()));
}

#[tokio::test]
async fn checkpoint_restores_then_reads_only_new_fragments_and_replays_deletes() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("wal.lance");
    let dataset = table(uri.to_str().unwrap()).await;
    let mut writer = open(&dir.path().join("writer.redb"), dataset).await;
    writer
        .commit(&[
            call(1, "a", Some(&[0, 255]), &["old"]),
            call(2, "b", Some(b"old"), &[]),
        ])
        .await
        .unwrap();
    let checkpoint = writer
        .checkpoint(dir.path().join("checkpoint.lance").to_str().unwrap())
        .await
        .unwrap();
    let old_files = writer
        .dataset()
        .get_fragments()
        .iter()
        .flat_map(|f| {
            f.metadata()
                .files
                .iter()
                .map(|f| f.path.clone())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    writer
        .commit(&[
            call(3, "b", None, &[]),
            call(4, "c", Some(b"new"), &["new"]),
        ])
        .await
        .unwrap();
    let wal = Dataset::open(uri.to_str().unwrap()).await.unwrap();
    let local = dir.path().join("recovered.redb");
    restore_checkpoint(&local, &checkpoint, &wal, &binding(), 1 << 20, 1)
        .await
        .unwrap();
    for name in old_files {
        let path = uri.join("data").join(name);
        std::fs::rename(&path, path.with_extension("hidden")).unwrap();
    }
    let recovered = open(&local, wal).await;
    assert_eq!(recovered.through_sequence(), 4);
    assert_eq!(recovered.get("a", b"node").unwrap(), Some(vec![0, 255]));
    assert_eq!(recovered.get("b", b"node").unwrap(), None);
    assert_eq!(recovered.get("c", b"node").unwrap(), Some(b"new".to_vec()));
    assert_eq!(
        recovered.receipt("a", "call-1", "input-1").unwrap(),
        Some(1)
    );
    assert_eq!(
        recovered.receipt("b", "call-3", "input-3").unwrap(),
        Some(3)
    );
    assert!(restore_checkpoint(
        &local,
        &checkpoint,
        recovered.dataset(),
        &binding(),
        1 << 20,
        1
    )
    .await
    .is_err());
}

#[tokio::test]
async fn published_checkpoint_is_discovered_automatically_without_old_data_files() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("wal.lance");
    let dataset = table(uri.to_str().unwrap()).await;
    let mut writer = open(&dir.path().join("writer.redb"), dataset).await;
    writer
        .commit(&[call(1, "s", Some(b"before"), &["old"])])
        .await
        .unwrap();
    let old_files = writer
        .dataset()
        .get_fragments()
        .iter()
        .flat_map(|f| {
            f.metadata()
                .files
                .iter()
                .map(|f| f.path.clone())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let checkpoint = writer
        .checkpoint(dir.path().join("checkpoint.lance").to_str().unwrap())
        .await
        .unwrap();
    writer.publish_checkpoint(&checkpoint).await.unwrap();
    writer
        .commit(&[call(2, "s", Some(b"after"), &["new"])])
        .await
        .unwrap();
    assert!(writer.publish_checkpoint(&checkpoint).await.is_err());
    for name in old_files {
        let path = uri.join("data").join(name);
        std::fs::rename(&path, path.with_extension("hidden")).unwrap();
    }
    let recovered = open(
        &dir.path().join("fresh.redb"),
        Dataset::open(uri.to_str().unwrap()).await.unwrap(),
    )
    .await;
    assert_eq!(recovered.through_sequence(), 2);
    assert_eq!(
        recovered.get("s", b"node").unwrap(),
        Some(b"after".to_vec())
    );
    assert_eq!(
        recovered.receipt("s", "call-1", "input-1").unwrap(),
        Some(1)
    );
    assert_eq!(
        recovered.scan("s", b"n", None, 1).unwrap(),
        vec![Mutation {
            key: b"node".to_vec(),
            value: Some(b"after".to_vec())
        }]
    );
    assert!(recovered
        .scan("s", b"n", Some(b"node"), 1)
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn oversized_state_cells_fail_before_publication_and_do_not_poison_valid_retry() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("wal.lance");
    let dataset = table(uri.to_str().unwrap()).await;
    let mut writer = open(&dir.path().join("writer.redb"), dataset).await;
    let version = writer.dataset().version().version;
    let huge = vec![1u8; (1 << 20) + 1];
    assert!(writer
        .commit(&[call(1, "s", Some(&huge), &[])])
        .await
        .is_err());
    assert_eq!(writer.through_sequence(), 0);
    assert_eq!(
        Dataset::open(uri.to_str().unwrap())
            .await
            .unwrap()
            .version()
            .version,
        version
    );
    writer
        .commit(&[call(1, "s", Some(b"small"), &[])])
        .await
        .unwrap();
    assert_eq!(writer.get("s", b"node").unwrap(), Some(b"small".to_vec()));
}

#[tokio::test]
async fn failed_checkpoint_restore_leaves_no_cache_and_retries_without_covered_wal() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("wal.lance");
    let dataset = table(uri.to_str().unwrap()).await;
    let mut writer = open(&dir.path().join("writer.redb"), dataset).await;
    writer
        .commit(&[call(1, "s", Some(b"v"), &["r"])])
        .await
        .unwrap();
    let checkpoint_uri = dir.path().join("checkpoint.lance");
    let checkpoint = writer
        .checkpoint(checkpoint_uri.to_str().unwrap())
        .await
        .unwrap();
    writer.publish_checkpoint(&checkpoint).await.unwrap();
    for fragment in writer.dataset().get_fragments() {
        for file in &fragment.metadata().files {
            let path = uri.join("data").join(&file.path);
            std::fs::rename(&path, path.with_extension("hidden")).unwrap();
        }
    }
    let file = checkpoint.get_fragments()[0].metadata().files[0]
        .path
        .clone();
    let path = checkpoint_uri.join("data").join(file);
    let hidden = path.with_extension("hidden");
    std::fs::rename(&path, &hidden).unwrap();
    let local = dir.path().join("recovered.redb");
    let wal = Dataset::open(uri.to_str().unwrap()).await.unwrap();
    let handler = commit_handler_from_url(wal.uri(), &None).await.unwrap();
    assert!(
        LocalLancePartition::open(&local, wal, binding(), handler, 1 << 20, 16 << 20, 2)
            .await
            .is_err()
    );
    assert!(!local.exists());
    std::fs::rename(hidden, path).unwrap();
    let recovered = open(&local, Dataset::open(uri.to_str().unwrap()).await.unwrap()).await;
    assert_eq!(recovered.through_sequence(), 1);
    assert_eq!(recovered.get("s", b"node").unwrap(), Some(b"v".to_vec()));
    assert_eq!(
        recovered.receipt("s", "call-1", "input-1").unwrap(),
        Some(1)
    );
}

#[tokio::test]
async fn checkpoint_byte_batches_and_empty_checkpoint_restore() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("wal.lance");
    let dataset = table(uri.to_str().unwrap()).await;
    let local = dir.path().join("writer.redb");
    let mut writer = open(&local, dataset).await;
    let empty = writer
        .checkpoint(dir.path().join("empty.lance").to_str().unwrap())
        .await
        .unwrap();
    writer.publish_checkpoint(&empty).await.unwrap();
    let restored = open(
        &dir.path().join("empty.redb"),
        Dataset::open(uri.to_str().unwrap()).await.unwrap(),
    )
    .await;
    assert_eq!(restored.through_sequence(), 0);
    drop(restored);
    // Individual calls fit, but their combined checkpoint exceeds the writer
    // byte budget. Checkpoint must split before reaching its row-count limit.
    let value = vec![7u8; 512 << 10];
    for sequence in 1..=3 {
        writer
            .commit(&[call(sequence, &format!("s{sequence}"), Some(&value), &[])])
            .await
            .unwrap();
    }
    drop(writer);
    let wal = Dataset::open(uri.to_str().unwrap()).await.unwrap();
    let handler = commit_handler_from_url(wal.uri(), &None).await.unwrap();
    let writer =
        LocalLancePartition::open(&local, wal, binding(), handler, 1 << 20, 768 << 10, 100)
            .await
            .unwrap();
    let checkpoint = writer
        .checkpoint(dir.path().join("bounded.lance").to_str().unwrap())
        .await
        .unwrap();
    let restored = dir.path().join("bounded.redb");
    restore_checkpoint(
        &restored,
        &checkpoint,
        writer.dataset(),
        &binding(),
        1 << 20,
        100,
    )
    .await
    .unwrap();
    let restored = open(
        &restored,
        Dataset::open(uri.to_str().unwrap()).await.unwrap(),
    )
    .await;
    for sequence in 1..=3 {
        assert_eq!(
            restored.get(&format!("s{sequence}"), b"node").unwrap(),
            Some(value.clone())
        );
    }
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
        base: &object_store::path::Path,
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
async fn lost_manifest_ack_recovers_both_local_state_and_receipts_without_new_version() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("wal.lance");
    let dataset = table(uri.to_str().unwrap()).await;
    let local = dir.path().join("writer.redb");
    let handler = Arc::new(CommitThenFail {
        delegate: commit_handler_from_url(dataset.uri(), &None).await.unwrap(),
    });
    let mut writer =
        LocalLancePartition::open(&local, dataset, binding(), handler, 1 << 20, 16 << 20, 2)
            .await
            .unwrap();
    let calls = [call(1, "s", Some(b"once"), &["once"])];
    assert!(writer.commit(&calls).await.is_err());
    assert!(writer.receipt("s", "call-1", "input-1").is_err());
    drop(writer);
    let committed = Dataset::open(uri.to_str().unwrap()).await.unwrap();
    let version = committed.version().version;
    let mut recovered = open(&local, committed).await;
    assert_eq!(recovered.through_sequence(), 1);
    assert_eq!(recovered.get("s", b"node").unwrap(), Some(b"once".to_vec()));
    assert_eq!(
        recovered.receipt("s", "call-1", "input-1").unwrap(),
        Some(1)
    );
    assert!(recovered.commit(&calls).await.is_err());
    assert_eq!(
        Dataset::open(uri.to_str().unwrap())
            .await
            .unwrap()
            .version()
            .version,
        version
    );
}

#[tokio::test]
async fn immutable_output_range_reads_only_its_payload_and_enforces_real_byte_budgets() {
    use lance_context_ingestion::local_lance_reader::{LogRange, OutputRange};
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("wal.lance");
    let dataset = table(uri.to_str().unwrap()).await;
    let mut writer = open(&dir.path().join("writer.redb"), dataset).await;
    writer
        .commit(&[call(1, "old", Some(b"old"), &["old"])])
        .await
        .unwrap();
    let base = writer.dataset().version().version;
    let files = writer
        .dataset()
        .get_fragments()
        .iter()
        .flat_map(|f| {
            f.metadata()
                .files
                .iter()
                .map(|f| f.path.clone())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    writer
        .commit(&[
            call(2, "s", Some(b"new-state"), &["new-output"]),
            call(3, "s", None, &[]),
        ])
        .await
        .unwrap();
    let version = writer.dataset().version().version;
    let range = LogRange {
        base_version: base,
        version,
        first_sequence: 2,
        last_sequence: 3,
    };
    for file in files {
        let path = uri.join("data").join(file);
        std::fs::rename(&path, path.with_extension("hidden")).unwrap();
    }
    let snapshot = writer.dataset().clone();
    let input = OutputRange::open(snapshot.clone(), &binding(), range.clone())
        .await
        .unwrap();
    assert!(input.physical_bytes > 0);
    assert!(input.read(1, 1 << 20, 2).await.is_err());
    let input = OutputRange::open(snapshot.clone(), &binding(), range.clone())
        .await
        .unwrap();
    assert!(input.read(1 << 20, 1, 2).await.is_err());
    let input = OutputRange::open(snapshot.clone(), &binding(), range.clone())
        .await
        .unwrap();
    let rows = input.read(1 << 20, 1 << 20, 2).await.unwrap();
    assert_eq!(rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    assert_eq!(
        rows[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0),
        "new-output"
    );
    let wrong = LogRange {
        first_sequence: 1,
        ..range.clone()
    };
    assert!(OutputRange::open(snapshot.clone(), &binding(), wrong)
        .await
        .is_err());
    writer
        .commit(&[call(4, "s", Some(b"later"), &["later"])])
        .await
        .unwrap();
    assert!(
        OutputRange::open(writer.dataset().clone(), &binding(), range.clone())
            .await
            .is_err()
    );
    assert_eq!(
        OutputRange::open(snapshot, &binding(), range)
            .await
            .unwrap()
            .read(1 << 20, 1 << 20, 2)
            .await
            .unwrap()
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>(),
        1
    );
    let before = writer.dataset().version().version;
    writer.commit(&[call(5, "s", None, &[])]).await.unwrap();
    let empty = LogRange {
        base_version: before,
        version: writer.dataset().version().version,
        first_sequence: 5,
        last_sequence: 5,
    };
    assert!(
        OutputRange::open(writer.dataset().clone(), &binding(), empty)
            .await
            .unwrap()
            .read(1 << 20, 1 << 20, 2)
            .await
            .unwrap()
            .is_empty()
    );
}

#[test]
fn binary_batch_reference_rejects_corruption_truncation_and_invalid_continuity() {
    use lance_context_ingestion::local_lance_reader::{BatchReference, LogRange};
    use sha2::{Digest, Sha256};
    let mut reference = BatchReference {
        binding: binding(),
        uri: "az://bucket/session-log.lance".into(),
        generation: 7,
        previous_generation: 6,
        range: LogRange {
            base_version: 11,
            version: 13,
            first_sequence: 9,
            last_sequence: 21,
        },
        physical_bytes: 105_000,
        decoded_byte_limit: 1 << 20,
        output_rows: 30,
        application_metadata: vec![0, 255, 13, 10],
    };
    let bytes = reference.encode().unwrap();
    assert_eq!(BatchReference::decode(&bytes).unwrap(), reference);
    for len in 0..bytes.len() {
        assert!(BatchReference::decode(&bytes[..len]).is_err());
    }
    for i in 0..bytes.len() {
        let mut changed = bytes.clone();
        changed[i] ^= 1;
        assert!(BatchReference::decode(&changed).is_err());
    }
    // Recompute checksum so framing/continuity validation is independently tested.
    let mut changed = bytes[..bytes.len() - 32].to_vec();
    changed[8..16].copy_from_slice(&9u64.to_le_bytes());
    changed.extend_from_slice(&Sha256::digest(&changed));
    assert!(BatchReference::decode(&changed).is_err());
    let mut trailing = bytes[..bytes.len() - 32].to_vec();
    trailing.push(0);
    trailing.extend_from_slice(&Sha256::digest(&trailing));
    assert!(BatchReference::decode(&trailing).is_err());
    reference.application_metadata = vec![0; 65537];
    assert!(reference.encode().is_err());
    reference.application_metadata.clear();
    reference.uri = "x".repeat(4097);
    assert!(reference.encode().is_err());
}

#[tokio::test]
async fn wide_shared_ipc_buffers_keep_small_calls_in_one_bounded_commit() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("wide.lance");
    let output_schema = Arc::new(Schema::new(
        (0..24)
            .map(|i| Field::new(format!("field{i}"), DataType::Utf8, false))
            .collect::<Vec<_>>(),
    ));
    let output = RecordBatch::try_new(
        output_schema.clone(),
        (0..24)
            .map(|_| {
                Arc::new(StringArray::from(vec!["small escaped 世界\nvalue"]))
                    as arrow_array::ArrayRef
            })
            .collect(),
    )
    .unwrap();
    let schema = Arc::new(log_schema(&output_schema, &binding()));
    let dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(RecordBatch::new_empty(schema.clone()))], schema),
        uri.to_str().unwrap(),
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let mut writer = open(&dir.path().join("wide.redb"), dataset).await;
    let calls = (1..=128)
        .map(|sequence| AlignedCall {
            sequence,
            session: "session".into(),
            receipt: format!("r{sequence}"),
            input_digest: format!("d{sequence}"),
            mutations: vec![],
            records: output.clone(),
        })
        .collect::<Vec<_>>();
    let charge = calls
        .iter()
        .map(|call| writer.batch_charge(call).unwrap())
        .sum::<usize>();
    assert!(
        charge <= 4 << 20,
        "shared allocations overcharged: {charge}"
    );
    let base = writer.dataset().version().version;
    writer.commit(&calls).await.unwrap();
    assert_eq!(writer.dataset().version().version, base + 1);
    assert_eq!(writer.through_sequence(), 128);
}

#[tokio::test]
async fn projected_output_avoids_unneeded_payload_and_preserves_read_limits() {
    use lance_context_ingestion::local_lance_reader::{
        LogRange, OutputRange, OutputReadPhase, OutputReadStage,
    };

    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("projected.lance");
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("content", DataType::Utf8, false),
        Field::new("source_metadata", DataType::Utf8, false),
        Field::new("nullable", DataType::Utf8, true),
    ]));
    let large = "unused recovery metadata".repeat(4096);
    let contents = (0..8).map(|i| format!("kept 世界 {i}")).collect::<Vec<_>>();
    let output = RecordBatch::try_new(
        output_schema.clone(),
        vec![
            Arc::new(StringArray::from(contents.clone())),
            Arc::new(StringArray::from(vec![large.as_str(); 8])),
            Arc::new(StringArray::from(vec![None::<&str>; 8])),
        ],
    )
    .unwrap();
    let schema = Arc::new(log_schema(&output_schema, &binding()));
    let dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(RecordBatch::new_empty(schema.clone()))], schema),
        uri.to_str().unwrap(),
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let base_version = dataset.version().version;
    let mut writer = open(&dir.path().join("projected.redb"), dataset).await;
    writer
        .commit(&[AlignedCall {
            sequence: 1,
            session: "s".into(),
            receipt: "receipt".into(),
            input_digest: "digest".into(),
            mutations: vec![],
            records: output,
        }])
        .await
        .unwrap();
    let snapshot = writer.dataset().clone();
    let range = LogRange {
        base_version,
        version: snapshot.version().version,
        first_sequence: 1,
        last_sequence: 1,
    };
    let expected_binding = binding();
    let open_range = || OutputRange::open(snapshot.clone(), &expected_binding, range.clone());
    assert!(open_range()
        .await
        .unwrap()
        .read(1 << 20, 64 << 10, 2)
        .await
        .is_err());
    let projected = open_range()
        .await
        .unwrap()
        .read_projected(1 << 20, 64 << 10, 2, &["content"])
        .await
        .unwrap();
    let stages = std::sync::Mutex::new(Vec::new());
    let observer = |stage, batch| stages.lock().unwrap().push((stage, batch));
    let observed = open_range()
        .await
        .unwrap()
        .read_projected_observed(1 << 20, 64 << 10, 2, &["content"], Some(&observer))
        .await
        .unwrap();
    assert_eq!(observed, projected);
    let mut expected_stages = vec![(OutputReadStage::StreamCreation, 0)];
    for batch in 0..observed.len() as u64 {
        expected_stages.push((OutputReadStage::RowIds, batch));
        expected_stages.push((OutputReadStage::TakeRows, batch));
    }
    expected_stages.push((OutputReadStage::RowIds, observed.len() as u64));
    expected_stages.push((OutputReadStage::Complete, observed.len() as u64));
    assert_eq!(*stages.lock().unwrap(), expected_stages);
    let phases = std::sync::Mutex::new(Vec::new());
    let phase_observer = |phase, batch| phases.lock().unwrap().push((phase, batch));
    for batch_rows in [1, 2, 1024] {
        phases.lock().unwrap().clear();
        let expected = open_range()
            .await
            .unwrap()
            .read_projected(1 << 20, 64 << 10, batch_rows, &["content", "nullable"])
            .await
            .unwrap();
        let detailed = open_range()
            .await
            .unwrap()
            .read_projected_phases(
                1 << 20,
                64 << 10,
                batch_rows,
                &["content", "nullable"],
                &phase_observer,
            )
            .await
            .unwrap();
        assert_eq!(detailed, expected);
        let actual: Vec<_> = detailed
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .map(|value| value.unwrap().to_owned())
            })
            .collect();
        assert_eq!(actual, contents);
        let mut expected_phases = vec![
            (OutputReadPhase::Read(OutputReadStage::StreamCreation), 0),
            (OutputReadPhase::PlanCreation, 0),
            (OutputReadPhase::ExecutionInitialization, 0),
        ];
        for batch in 0..detailed.len() as u64 {
            expected_phases.push((OutputReadPhase::Read(OutputReadStage::RowIds), batch));
            expected_phases.push((OutputReadPhase::Read(OutputReadStage::TakeRows), batch));
        }
        expected_phases.push((
            OutputReadPhase::Read(OutputReadStage::RowIds),
            detailed.len() as u64,
        ));
        expected_phases.push((
            OutputReadPhase::Read(OutputReadStage::Complete),
            detailed.len() as u64,
        ));
        assert_eq!(*phases.lock().unwrap(), expected_phases);
    }
    phases.lock().unwrap().clear();
    let phase_error = open_range()
        .await
        .unwrap()
        .read_projected_phases(1 << 20, 1, 2, &["content"], &phase_observer)
        .await
        .unwrap_err();
    assert!(!phases
        .lock()
        .unwrap()
        .iter()
        .any(|(phase, _)| { *phase == OutputReadPhase::Read(OutputReadStage::Complete) }));
    stages.lock().unwrap().clear();
    let observed_error = open_range()
        .await
        .unwrap()
        .read_projected_observed(1 << 20, 1, 2, &["content"], Some(&observer))
        .await
        .unwrap_err();
    let original_error = open_range()
        .await
        .unwrap()
        .read_projected(1 << 20, 1, 2, &["content"])
        .await
        .unwrap_err();
    assert_eq!(observed_error.to_string(), original_error.to_string());
    assert_eq!(phase_error.to_string(), original_error.to_string());
    assert!(!stages
        .lock()
        .unwrap()
        .iter()
        .any(|(stage, _)| *stage == OutputReadStage::Complete));
    let full = open_range()
        .await
        .unwrap()
        .read(1 << 20, 8 << 20, 2)
        .await
        .unwrap();
    assert_eq!(projected.len(), full.len());
    assert_eq!(
        projected.iter().map(RecordBatch::num_rows).sum::<usize>(),
        8
    );
    for (selected, complete) in projected.iter().zip(&full) {
        assert_eq!(selected, &complete.project(&[0]).unwrap());
    }
    let nullable = open_range()
        .await
        .unwrap()
        .read_projected(1 << 20, 64 << 10, 2, &["nullable"])
        .await
        .unwrap();
    for (selected, complete) in nullable.iter().zip(&full) {
        assert_eq!(selected, &complete.project(&[2]).unwrap());
    }
    for (physical, decoded, fields) in [
        (1, 64 << 10, vec!["content"]),
        (1 << 20, 1, vec!["content"]),
        (1 << 20, 64 << 10, vec![]),
        (1 << 20, 64 << 10, vec!["missing"]),
    ] {
        let expected_error = open_range()
            .await
            .unwrap()
            .read_projected(physical, decoded, 2, &fields)
            .await
            .unwrap_err();
        phases.lock().unwrap().clear();
        let detailed_error = open_range()
            .await
            .unwrap()
            .read_projected_phases(physical, decoded, 2, &fields, &phase_observer)
            .await
            .unwrap_err();
        assert_eq!(detailed_error.to_string(), expected_error.to_string());
        assert!(!phases
            .lock()
            .unwrap()
            .iter()
            .any(|(phase, _)| { *phase == OutputReadPhase::Read(OutputReadStage::Complete) }));
        if physical == 1 || fields.is_empty() || fields == ["missing"] {
            assert!(phases.lock().unwrap().is_empty());
        }
    }
    // A null record is corrupt, even when all selected child values may be null.
    let mut scan = snapshot.scan();
    scan.filter("kind = 3")
        .unwrap()
        .limit(Some(1), None)
        .unwrap();
    let batch = scan.try_into_batch().await.unwrap();
    let mut columns = batch.columns().to_vec();
    columns[8] = arrow_array::new_null_array(batch.schema().field(8).data_type(), 1);
    let corrupt = RecordBatch::try_new(batch.schema(), columns).unwrap();
    let corrupt = Dataset::write(
        RecordBatchIterator::new(vec![Ok(corrupt)], batch.schema()),
        uri.to_str().unwrap(),
        Some(WriteParams {
            mode: lance::dataset::WriteMode::Append,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let mut corrupt_range = range;
    corrupt_range.version = corrupt.version().version;
    let error = OutputRange::open(corrupt, &expected_binding, corrupt_range)
        .await
        .unwrap()
        .read_projected(1 << 20, 64 << 10, 2, &["nullable"])
        .await
        .unwrap_err();
    assert!(error.to_string().contains("null Lance log output row"));
}
