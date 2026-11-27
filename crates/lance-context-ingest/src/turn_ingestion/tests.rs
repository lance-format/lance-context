use super::*;
use arrow_array::{Float64Array, LargeBinaryArray};
use arrow_schema::Field;
use std::{collections::HashMap, sync::Arc};
use uuid::Uuid;

#[cfg(feature = "materializer")]
mod readahead;

fn schema() -> SchemaRef {
    Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("session_id", DataType::Utf8, false),
            Field::new("id", DataType::Utf8, false),
            Field::new("parent_id", DataType::Utf8, true),
            Field::new("content", DataType::LargeBinary, false).with_metadata(HashMap::from([(
                "source-type".into(),
                "test-binary".into(),
            )])),
            // A user body's score is not necessarily the annotation score.
            Field::new("score", DataType::Float64, true),
        ],
        HashMap::from([("projector".into(), "test-v1".into())]),
    ))
}

fn spec() -> Result<TurnIngestionSpec> {
    TurnIngestionSpec::new(
        schema(),
        vec!["session_id".into(), "id".into()],
        "session_id",
    )
}

fn body() -> Result<RecordBatch> {
    Ok(RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(StringArray::from(vec!["a", "a", "b", "a"])),
            Arc::new(StringArray::from(vec!["x", "x", "x", "y"])),
            Arc::new(StringArray::from(vec![None, None, None, Some("x")])),
            Arc::new(LargeBinaryArray::from_iter_values([
                b"original".as_slice(),
                b"duplicate",
                b"other-session",
                b"child",
            ])),
            Arc::new(Float64Array::from(vec![
                Some(0.1),
                None,
                Some(0.3),
                Some(0.4),
            ])),
        ],
    )?)
}

#[tokio::test]
async fn key_scan_reports_native_work_for_early_hit() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("body.lance");
    let reader = arrow_array::RecordBatchIterator::new(vec![Ok(body()?)], schema());
    let base = Arc::new(lance::Dataset::write(reader, path.to_str().unwrap(), None).await?);
    let key = spec()?.business_key(&["a", "x"], 1024)?;
    let budget = lance_context_core::MergeMemoryBudget::new(1 << 20);
    let lookup = spec()?
        .lookup_existing(base, &[key.as_str()], &budget)
        .await?
        .expect("lookup workspace admitted");
    assert_eq!(lookup.get(&key), Some(true));
    let observed = lookup.observation();
    assert_eq!(observed.scans_started, 1);
    assert_eq!(observed.projected_rows, 2);
    assert_eq!(observed.history_hits, 1);
    assert_eq!(observed.history_misses, 0);
    // Both duplicate rows arrive in one batch. Finding the requested key
    // drops the scan before polling EOF, but its actual work still counts.
    assert!(observed.lance_bytes_read.is_some_and(|bytes| bytes > 0));
    assert!(observed.lance_requests.is_some_and(|requests| requests > 0));
    assert_eq!(observed.fragments_scanned, Some(1));
    assert_eq!(observed.rows_scanned, Some(4));
    assert!(observed.task_wait_time_ns.is_some());
    Ok(())
}

#[tokio::test]
async fn key_scan_reports_native_work_for_missing_composite_key() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("body.lance");
    let reader = arrow_array::RecordBatchIterator::new(vec![Ok(body()?)], schema());
    let base = Arc::new(lance::Dataset::write(reader, path.to_str().unwrap(), None).await?);
    // Both column values exist, but this tuple does not. The miss exhausts the
    // native stream, so a zero-row result must still expose its scan work.
    let key = spec()?.business_key(&["b", "y"], 1024)?;
    let budget = lance_context_core::MergeMemoryBudget::new(1 << 20);
    let lookup = spec()?
        .lookup_existing(base, &[key.as_str()], &budget)
        .await?
        .expect("lookup workspace admitted");
    assert_eq!(lookup.get(&key), Some(false));
    let observed = lookup.observation();
    assert_eq!(observed.history_hits, 0);
    assert_eq!(observed.history_misses, 1);
    assert_eq!(observed.scans_started, 1);
    assert_eq!(observed.projected_rows, 0);
    assert_eq!(observed.fragments_scanned, Some(1));
    assert!(observed.ranges_scanned.is_some_and(|count| count > 0));
    assert_eq!(observed.rows_scanned, Some(4));
    assert!(observed.task_wait_time_ns.is_some());
    Ok(())
}

#[tokio::test]
async fn key_scan_uses_existing_zone_maps_and_reads_uncovered_fragments() -> Result<()> {
    use lance::index::DatasetIndexExt;
    use lance_index::{
        scalar::{BuiltinIndexType, ScalarIndexParams},
        IndexType,
    };

    let directory = tempfile::tempdir()?;
    let path = directory.path().join("body.lance");
    let reader = arrow_array::RecordBatchIterator::new(vec![Ok(body()?)], schema());
    let mut base = lance::Dataset::write(reader, path.to_str().unwrap(), None).await?;
    let spec = spec()?;
    let budget = lance_context_core::MergeMemoryBudget::new(1 << 20);
    let in_range = spec.business_key(&["b", "y"], 1024)?;
    let new_key = spec.business_key(&["z", "y"], 1024)?;
    let missing = spec.business_key(&["zz", "y"], 1024)?;

    let unindexed = spec
        .lookup_existing(Arc::new(base.clone()), &[&new_key, &missing], &budget)
        .await?
        .unwrap();
    assert_eq!(unindexed.observation().rows_scanned, Some(4));
    drop(unindexed);
    for column in ["session_id", "id"] {
        base.create_index(
            &[column],
            IndexType::ZoneMap,
            None,
            &ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap),
            false,
        )
        .await?;
    }
    let indexed = spec
        .lookup_existing(Arc::new(base.clone()), &[&new_key, &missing], &budget)
        .await?
        .unwrap();
    assert_eq!(indexed.get(&new_key), Some(false));
    assert_eq!(indexed.observation().rows_scanned, Some(0));
    assert_eq!(indexed.observation().fragments_scanned, Some(0));
    drop(indexed);

    // Independent column extrema cannot prove that an in-range tuple is absent.
    let in_range_lookup = spec
        .lookup_existing(Arc::new(base.clone()), &[&in_range], &budget)
        .await?
        .unwrap();
    assert_eq!(in_range_lookup.get(&in_range), Some(false));
    assert_eq!(in_range_lookup.observation().rows_scanned, Some(4));
    drop(in_range_lookup);

    let one = body()?.slice(0, 1);
    let mut columns = one.columns().to_vec();
    columns[0] = Arc::new(StringArray::from(vec!["z"]));
    columns[1] = Arc::new(StringArray::from(vec!["y"]));
    let appended = RecordBatch::try_new(schema(), columns)?;
    base.append(
        arrow_array::RecordBatchIterator::new(vec![Ok(appended)], schema()),
        None,
    )
    .await?;
    let uncovered = spec
        .lookup_existing(Arc::new(base), &[&new_key, &missing], &budget)
        .await?
        .unwrap();
    assert_eq!(uncovered.get(&new_key), Some(true));
    assert_eq!(uncovered.get(&missing), Some(false));
    assert_eq!(uncovered.observation().rows_scanned, Some(1));
    assert_eq!(uncovered.observation().fragments_scanned, Some(1));
    Ok(())
}

fn layout() -> Layout {
    Layout {
        cluster: Uuid::from_u128(1),
        nodes: (1..=3).map(|id| (id, format!("wal-{id}:7800"))).collect(),
        groups: [(1, vec![1, 2, 3])].into(),
        buckets: vec![1; 64],
    }
}

#[test]
fn preparation_preserves_body_buffers_duplicates_and_order() -> Result<()> {
    let original = body()?;
    let prepared = spec()?.prepare(original.clone(), 1024)?;
    assert_eq!(prepared.body(), &original);
    assert!(Arc::ptr_eq(&prepared.body().schema(), &original.schema()));
    for (before, after) in original.columns().iter().zip(prepared.body().columns()) {
        assert!(Arc::ptr_eq(before, after), "body arrays must not be copied");
    }
    assert_eq!(
        prepared.keys(),
        &[
            r#"["a","x"]"#,
            r#"["a","x"]"#,
            r#"["b","x"]"#,
            r#"["a","y"]"#
        ]
    );
    assert_eq!(
        prepared.body().num_rows(),
        4,
        "ordered apply owns first-create decisions"
    );
    assert!(prepared.identity_bytes() >= prepared.keys().iter().map(String::len).sum());
    let (returned, keys) = prepared.into_parts();
    assert_eq!(returned, original);
    assert_eq!(keys.len(), 4);
    Ok(())
}

#[test]
fn equal_business_keys_and_sessions_route_together() -> Result<()> {
    let prepared = spec()?.prepare(body()?, 1024)?;
    let layout = layout();
    layout.validate()?;
    let a = layout.session_bucket("tenant", "table", "a")?;
    for row in [0, 1, 3] {
        assert_eq!(prepared.bucket(row, &layout, "tenant", "table")?, a);
    }
    assert_eq!(
        prepared.bucket(2, &layout, "tenant", "table")?,
        layout.session_bucket("tenant", "table", "b")?
    );
    assert!(prepared.bucket(4, &layout, "tenant", "table").is_err());
    Ok(())
}

#[test]
fn tuple_identity_has_no_delimiter_or_escape_aliases() -> Result<()> {
    let spec = spec()?;
    let tuples = [
        ["a:b", "c"],
        ["a", "b:c"],
        ["a\0b", "c"],
        ["a", "b\0c"],
        ["a\"", "b\\c"],
        ["a\\\"", "b\\c"],
        ["会话", "事件"],
        ["事件", "会话"],
    ];
    let mut keys = HashSet::new();
    for tuple in tuples {
        let key = spec.business_key(&tuple, 1024)?;
        assert!(keys.insert(key.clone()));
        assert_eq!(serde_json::from_str::<Vec<String>>(&key)?, tuple);
    }
    assert!(spec.business_key(&["a"], 1024).is_err());
    assert!(spec.business_key(&["a", ""], 1024).is_err());
    let key = spec.business_key(&["a\n", "b"], 1024)?;
    assert!(spec.business_key(&["a\n", "b"], key.len() - 1).is_err());
    assert_eq!(spec.business_key(&["a\n", "b"], key.len())?, key);
    Ok(())
}

#[test]
fn fingerprint_binds_schema_metadata_key_order_and_routing() -> Result<()> {
    let spec = spec()?;
    let fingerprint = spec.fingerprint()?;
    assert_eq!(fingerprint, spec.clone().fingerprint()?);
    for (key, route) in [
        (vec!["id".into(), "session_id".into()], "session_id"),
        (vec!["session_id".into(), "id".into()], "id"),
        (vec!["id".into()], "id"),
    ] {
        assert_ne!(
            fingerprint,
            TurnIngestionSpec::new(schema(), key, route)?.fingerprint()?
        );
    }
    let changed = Arc::new(Schema::new_with_metadata(
        schema().fields().clone(),
        HashMap::from([("projector".into(), "test-v2".into())]),
    ));
    assert_ne!(
        fingerprint,
        TurnIngestionSpec::new(changed.clone(), spec.primary_key().to_vec(), "session_id")?
            .fingerprint()?
    );
    assert!(spec.prepare(RecordBatch::new_empty(changed), 1024).is_err());
    Ok(())
}

#[test]
fn invalid_or_ambiguous_identity_definitions_are_rejected() -> Result<()> {
    for (key, route) in [
        (vec![], "session_id"),
        (vec!["id".into(), "id".into()], "id"),
        (vec!["absent".into()], "absent"),
        (vec!["parent_id".into()], "parent_id"),
        (vec!["score".into()], "score"),
        (vec!["id".into()], "session_id"),
    ] {
        assert!(TurnIngestionSpec::new(schema(), key, route).is_err());
    }
    let mut fields = schema().fields().to_vec();
    fields.push(fields[0].clone());
    assert!(
        TurnIngestionSpec::new(Arc::new(Schema::new(fields)), vec!["id".into()], "id").is_err()
    );
    let reserved = Arc::new(Schema::new_with_metadata(
        schema().fields().clone(),
        HashMap::from([(SPEC_KEY.into(), "untrusted".into())]),
    ));
    assert!(TurnIngestionSpec::new(reserved, vec!["id".into()], "id").is_err());
    Ok(())
}

#[test]
fn empty_input_is_preserved_and_empty_keys_are_rejected() -> Result<()> {
    let spec = spec()?;
    let prepared = spec.prepare(RecordBatch::new_empty(schema()), 1024)?;
    assert_eq!(prepared.body().num_rows(), 0);
    assert!(prepared.keys().is_empty());
    let original = body()?;
    let mut columns = original.columns().to_vec();
    columns[1] = Arc::new(StringArray::from(vec!["x", "", "x", "y"]));
    assert!(spec
        .prepare(RecordBatch::try_new(schema(), columns)?, 1024)
        .is_err());
    Ok(())
}
