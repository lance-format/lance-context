use super::*;
use serde_json::json;

#[tokio::test]
async fn automatically_shreds_large_text_and_reopens_without_duplication() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("documents.lance");
    let uri = uri.to_str().unwrap();
    let mut store = DocumentStore::create(uri, ShredPolicy::default(), None)
        .await
        .unwrap();
    let text = "searchable document text ".repeat(100_000);
    let original =
        json!({"body": text, "nested": {"title": "hello"}, "null": null, "array": [{"keep": 1}]});
    let ids = store.add(vec![original.clone()]).await.unwrap();
    assert!(store
        .layout()
        .fields
        .iter()
        .any(|field| field.path == vec!["body"]));
    let batches: Vec<RecordBatch> = store
        .dataset
        .scan()
        .project(&[RESIDUAL])
        .unwrap()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    let residual = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<LargeStringArray>()
        .unwrap()
        .value(0);
    assert!(!residual.contains("searchable"));
    assert!(residual.len() < 100);
    drop(store);
    let reopened = DocumentStore::open(uri, None).await.unwrap();
    assert_eq!(reopened.get(&ids[0]).await.unwrap(), Some(original));
    assert_eq!(
        reopened.get_path(&ids[0], &["body".into()]).await.unwrap(),
        Some(json!(text))
    );
    assert_eq!(
        reopened.get_path(&ids[0], &["null".into()]).await.unwrap(),
        Some(Value::Null)
    );
    assert_eq!(
        reopened
            .get_path(&ids[0], &["missing".into()])
            .await
            .unwrap(),
        None
    );
    assert_eq!(reopened.get("absent").await.unwrap(), None);
}

#[tokio::test]
async fn evolved_layout_reads_old_residuals_mixed_types_and_snapshots() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("evolve.lance");
    let uri = uri.to_str().unwrap();
    let policy = ShredPolicy {
        min_frequency: 1.0,
        ..Default::default()
    };
    let mut store = DocumentStore::create(uri, policy, None).await.unwrap();
    let old = json!({"common": true, "later": "rare"});
    let old_ids = store
        .add(vec![old.clone(), json!({"common": true})])
        .await
        .unwrap();
    assert_eq!(store.layout().fields.len(), 1);
    let old_version = store.version();
    let later = json!({"common": false, "later": "now common"});
    let new_ids = store.add(vec![later.clone()]).await.unwrap();
    assert_eq!(store.layout().fields.len(), 2);
    assert_eq!(
        store
            .get_path(&old_ids[0], &["later".into()])
            .await
            .unwrap(),
        Some(json!("rare"))
    );
    let mixed = json!({"common": "different type", "later": null});
    let mixed_ids = store.add(vec![mixed.clone()]).await.unwrap();
    assert_eq!(store.get(&old_ids[0]).await.unwrap(), Some(old.clone()));
    assert_eq!(store.get(&new_ids[0]).await.unwrap(), Some(later));
    assert_eq!(store.get(&mixed_ids[0]).await.unwrap(), Some(mixed));
    store.checkout(old_version).await.unwrap();
    assert_eq!(store.layout().fields.len(), 1);
    assert_eq!(store.get(&old_ids[0]).await.unwrap(), Some(old));
    assert_eq!(store.get(&new_ids[0]).await.unwrap(), None);
    assert!(store.add(vec![json!({})]).await.is_err());
    store.refresh().await.unwrap();
    assert_eq!(store.layout().fields.len(), 2);
    assert!(store.get(&new_ids[0]).await.unwrap().is_some());
}

#[tokio::test]
async fn exact_numeric_values_and_arbitrary_json_roots_survive_storage() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("types.lance");
    let mut store = DocumentStore::create(uri.to_str().unwrap(), ShredPolicy::default(), None)
        .await
        .unwrap();
    let documents = vec![
        json!({"signed": i64::MIN, "unsigned": u64::MAX, "float": 1.0, "negative_zero": -0.0, "": {"a/b": ""}}),
    ];
    let ids = store.add(documents.clone()).await.unwrap();
    assert_eq!(
        store.get(&ids[0]).await.unwrap(),
        Some(documents[0].clone())
    );
    for document in [
        json!([
            1,
            null,
            {},
            51.248178375505404_f64,
            2.0030397744267762e-253_f64
        ]),
        Value::Null,
        json!(42),
        json!("root"),
        json!({}),
    ] {
        let ids = store.add(vec![document.clone()]).await.unwrap();
        assert_eq!(store.get(&ids[0]).await.unwrap(), Some(document));
    }
    let version = store.version();
    assert!(store.add(vec![]).await.unwrap().is_empty());
    assert_eq!(store.version(), version);
}

#[tokio::test]
async fn rejects_unknown_format_and_corrupt_field_mapping() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("metadata.lance");
    let uri = uri.to_str().unwrap();
    let mut store = DocumentStore::create(uri, ShredPolicy::default(), None)
        .await
        .unwrap();
    store.add(vec![json!({"x": "text"})]).await.unwrap();
    store
        .dataset
        .update_schema_metadata([(FORMAT_KEY, "99")])
        .await
        .unwrap();
    assert!(DocumentStore::open(uri, None).await.is_err());
    store
        .dataset
        .update_schema_metadata([(FORMAT_KEY, "1")])
        .await
        .unwrap();
    // A type declaration cannot silently reinterpret an existing physical column.
    let mut schema = Schema::from(store.dataset.schema());
    let mut fields = schema.fields().to_vec();
    fields[2] = Arc::new(fields[2].as_ref().clone().with_metadata(HashMap::new()));
    schema = Schema::new(fields).with_metadata(schema.metadata().clone());
    // Validate the rejection through a real foreign dataset with missing metadata.
    let other = dir.path().join("foreign.lance");
    Dataset::write(
        RecordBatchIterator::new(
            Vec::<std::result::Result<RecordBatch, arrow_schema::ArrowError>>::new(),
            Arc::new(schema),
        ),
        other.to_str().unwrap(),
        None,
    )
    .await
    .unwrap();
    assert!(DocumentStore::open(other.to_str().unwrap(), None)
        .await
        .is_err());
}

#[tokio::test]
async fn rejects_unreadable_nesting_before_committing() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("depth.lance");
    let mut store = DocumentStore::create(uri.to_str().unwrap(), ShredPolicy::default(), None)
        .await
        .unwrap();
    let version = store.version();
    let mut document = Value::Null;
    for _ in 0..128 {
        document = Value::Array(vec![document]);
    }
    assert!(store.add(vec![document]).await.is_err());
    assert_eq!(store.version(), version);
}
