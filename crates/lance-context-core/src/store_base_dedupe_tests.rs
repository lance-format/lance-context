use super::dedupe_merge_batches;
use arrow_array::{LargeBinaryArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use std::{collections::HashMap, sync::Arc};

fn batch(rows: &[(&str, &[u8])]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("payload", DataType::LargeBinary, false),
        ])),
        vec![
            Arc::new(StringArray::from_iter_values(rows.iter().map(|row| row.0))),
            Arc::new(LargeBinaryArray::from_iter_values(
                rows.iter().map(|row| row.1),
            )),
        ],
    )
    .unwrap()
}

fn values(batches: &[RecordBatch]) -> HashMap<String, Vec<u8>> {
    let mut result = HashMap::new();
    for batch in batches {
        let keys = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let payloads = batch
            .column(1)
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .unwrap();
        for row in 0..batch.num_rows() {
            assert!(result
                .insert(keys.value(row).into(), payloads.value(row).to_vec())
                .is_none());
        }
    }
    result
}

#[test]
fn unique_wal_blobs_reuse_arrays_and_preserve_requested_schema() {
    let blob = vec![7; 1024 * 1024];
    let original = batch(&[("first", &blob), ("second", &blob), ("third", &blob)]);
    let target = Arc::new(
        original
            .schema()
            .as_ref()
            .clone()
            .with_metadata(HashMap::from([(
                "latest_metadata".into(),
                "retained".into(),
            )])),
    );
    let result =
        dedupe_merge_batches(vec![(1, vec![original.clone()])], "id", target.clone()).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].schema(), target);
    assert_eq!(result[0].num_rows(), original.num_rows());
    for (before, after) in original.columns().iter().zip(result[0].columns()) {
        assert!(
            Arc::ptr_eq(before, after),
            "unique WAL payload must not be reallocated"
        );
    }
    assert_eq!(values(&result), values(&[original]));
}

#[test]
fn duplicate_wal_keys_keep_newest_generation_batch_and_row() {
    let old = batch(&[
        ("changed", b"old"),
        ("old_only", b"retained"),
        ("within", b"obsolete"),
    ]);
    let newer_first = batch(&[
        ("changed", b"middle"),
        ("within", b"first"),
        ("within", b"last"),
    ]);
    let newer_last = batch(&[("changed", b"newest"), ("new_only", b"new")]);
    let result = dedupe_merge_batches(
        // Deliberately reversed generation input; precedence is the generation ID.
        vec![
            (2, vec![newer_first, newer_last.clone()]),
            (1, vec![old.clone()]),
        ],
        "id",
        old.schema(),
    )
    .unwrap();
    assert_eq!(
        values(&result),
        HashMap::from([
            ("changed".into(), b"newest".to_vec()),
            ("within".into(), b"last".to_vec()),
            ("old_only".into(), b"retained".to_vec()),
            ("new_only".into(), b"new".to_vec()),
        ])
    );
    assert!(Arc::ptr_eq(result[0].column(1), newer_last.column(1)));
    // Partial survivors still use compact output arrays, not a slice retaining
    // an entire discarded generation's blob buffers.
    assert!(!Arc::ptr_eq(
        result.last().unwrap().column(1),
        old.column(1)
    ));
}

#[test]
fn fully_shadowed_batches_do_not_retain_their_payloads() {
    let old = batch(&[("one", &[1; 1024])]);
    let new = batch(&[("one", b"replacement")]);
    let result = dedupe_merge_batches(
        vec![(1, vec![old]), (2, vec![new.clone()])],
        "id",
        new.schema(),
    )
    .unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(values(&result), values(&[new]));
}
