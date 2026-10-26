use super::*;
use arrow_array::{BinaryArray, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance::dataset::{CommitBuilder, WriteMode, WriteParams};
use lance::index::DatasetIndexExt;
use lance_index::{scalar::ScalarIndexParams, IndexType};

fn batch(ids: &[&str], payload: &[u8]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("payload", DataType::Binary, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(ids.to_vec())),
            Arc::new(BinaryArray::from(vec![payload; ids.len()])),
        ],
    )
    .unwrap()
}

async fn fixture(index: Option<IndexType>) -> (tempfile::TempDir, Arc<Dataset>) {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("base.lance");
    let initial = batch(&["covered", "keep"], &vec![7; 256 * 1024]);
    let reader = RecordBatchIterator::new(vec![Ok(initial.clone())], initial.schema());
    let mut ds = Dataset::write(reader, uri.to_str().unwrap(), None)
        .await
        .unwrap();
    if let Some(kind) = index {
        ds.create_index_builder(&["id"], kind, &ScalarIndexParams::default())
            .await
            .unwrap();
    }
    // Every indexed case includes an uncovered fragment, as on busy production
    // tables where merges append after an index was built.
    let appended = batch(&["uncovered"], &vec![9; 256 * 1024]);
    let reader = RecordBatchIterator::new(vec![Ok(appended.clone())], appended.schema());
    let ds = Dataset::write(
        reader,
        uri.to_str().unwrap(),
        Some(WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    (dir, Arc::new(ds))
}

#[tokio::test]
async fn scan_plan_excludes_old_payload_with_no_index_zonemap_and_forced_scan() {
    for (index, force_scan) in [
        (None, false),
        (Some(IndexType::ZoneMap), false),
        (Some(IndexType::BTree), true),
    ] {
        let (_dir, snapshot) = fixture(index).await;
        let incoming = batch(&["covered", "uncovered", "absent"], b"new value");
        let (job, source) = job_and_source(snapshot, &[incoming], "id", force_scan)
            .await
            .unwrap();
        let plan = job
            .explain_plan(Some(source[0].schema().as_ref()), true)
            .await
            .unwrap();
        let scans: Vec<_> = plan
            .lines()
            .filter(|line| line.contains("LanceRead:"))
            .collect();
        assert!(!scans.is_empty(), "{plan}");
        assert!(
            scans.iter().all(|line| line.contains("projection=[id]")),
            "{plan}"
        );
        assert!(scans.iter().all(|line| !line.contains("payload")), "{plan}");
    }
}

#[tokio::test]
async fn indexed_and_scan_deletes_are_uncommitted_and_handle_uncovered_fragments() {
    for (index, force_scan) in [
        (None, false),
        (Some(IndexType::ZoneMap), false),
        (Some(IndexType::BTree), false),
        (Some(IndexType::BTree), true),
    ] {
        let (_dir, snapshot) = fixture(index).await;
        let incoming = batch(&["covered", "uncovered", "absent"], b"new value");
        let prepared = prepare(snapshot.clone(), &[incoming], "id", force_scan)
            .await
            .unwrap();
        let unchanged = Dataset::open(snapshot.uri()).await.unwrap();
        assert_eq!(unchanged.version().version, snapshot.version().version);
        assert_eq!(unchanged.count_rows(None).await.unwrap(), 3);
        let after = CommitBuilder::new(snapshot)
            .execute(prepared.transaction)
            .await
            .unwrap();
        let actual = after.scan().try_into_batch().await.unwrap();
        assert_eq!(actual.num_rows(), 1);
        assert_eq!(
            actual
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(0),
            "keep"
        );
        assert_eq!(
            actual
                .column(1)
                .as_any()
                .downcast_ref::<BinaryArray>()
                .unwrap()
                .value(0),
            vec![7; 256 * 1024]
        );
    }
}

#[tokio::test]
async fn indexed_lookup_keeps_key_only_source_and_scan_source_shares_buffers() {
    let incoming = batch(&["covered"], &vec![3; 256 * 1024]);
    let (_dir, snapshot) = fixture(Some(IndexType::BTree)).await;
    let (_, source) = job_and_source(
        snapshot.clone(),
        std::slice::from_ref(&incoming),
        "id",
        false,
    )
    .await
    .unwrap();
    assert_eq!(source[0].schema().fields().len(), 1);
    assert!(Arc::ptr_eq(source[0].column(0), incoming.column(0)));
    let (_, source) = job_and_source(snapshot, std::slice::from_ref(&incoming), "id", true)
        .await
        .unwrap();
    assert_eq!(source[0].schema(), incoming.schema());
    assert!(Arc::ptr_eq(source[0].column(1), incoming.column(1)));
}
