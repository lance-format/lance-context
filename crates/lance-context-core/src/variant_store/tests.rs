use super::*;
use serde_json::json;

#[tokio::test]
async fn automatically_migrates_exhausted_layout_and_preserves_every_version() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("evolve.lance");
    let uri = uri.to_str().unwrap();
    let mut store = VariantDocumentStore::create(uri, ShredPolicy::default(), None)
        .await
        .unwrap();
    let numeric: serde_json::Map<String, Value> =
        (0..32).map(|i| (format!("field{i}"), json!(i))).collect();
    let old = Value::Object(numeric);
    let first = store.add(vec![old.clone()]).await.unwrap()[0].clone();
    let version = store.version();
    let new = json!({"body":"new text ".repeat(8192),"field0":"now text","tags":["fallback"]});
    let second = store.add(vec![new.clone()]).await.unwrap()[0].clone();
    assert!(store
        .layout()
        .fields()
        .iter()
        .any(|field| field.path == ["body"]));
    assert!(store
        .layout()
        .fields()
        .iter()
        .any(|field| field.path == ["field0"]
            && field.data_type == crate::shredding::ShredType::String));
    assert_eq!(store.get(&first).await.unwrap(), Some(old.clone()));
    assert_eq!(store.get(&second).await.unwrap(), Some(new.clone()));
    assert_eq!(
        store.get_path(&first, &["field0".into()]).await.unwrap(),
        Some(json!(0))
    );
    assert_eq!(
        store.get_path(&second, &["tags".into()]).await.unwrap(),
        Some(json!(["fallback"]))
    );
    let final_version = store.version();
    // No intermediate layout can expose a partial append or lose old records.
    for revision in version..=final_version {
        store.checkout(revision).await.unwrap();
        assert_eq!(store.get(&first).await.unwrap(), Some(old.clone()));
        assert!(store
            .get(&second)
            .await
            .unwrap()
            .is_none_or(|value| value == new));
    }
    store.checkout(version).await.unwrap();
    assert!(store.get(&second).await.unwrap().is_none());
    assert!(store.add(vec![json!(1)]).await.is_err());
    store.refresh().await.unwrap();
    let reopened = VariantDocumentStore::open(uri, None).await.unwrap();
    assert_eq!(reopened.get(&second).await.unwrap(), Some(new));
    assert!(reopened
        .dataset
        .unindexed_fragments(ID_INDEX)
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn projected_paths_match_whole_value_for_mixed_shapes_and_literal_keys() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("paths.lance");
    let mut store =
        VariantDocumentStore::create(uri.to_str().unwrap(), ShredPolicy::default(), None)
            .await
            .unwrap();
    let documents = vec![
        json!({"body":"large text","a.b":{"":"quoted","nested":{"v":7}},"null":null,"tags":["a"]}),
        json!({"body":8,"a.b":{"nested":false},"null":null}),
        json!({"a.b":"scalar"}),
        json!({}),
        json!(null),
        json!([1, null]),
        json!({"quote\"":"literal","integer":u64::MAX,"negative":i64::MIN}),
    ];
    store.add(vec![documents[0].clone()]).await.unwrap();
    assert!(store
        .layout()
        .fields()
        .iter()
        .any(|field| field.path == ["a.b", ""]));
    let ids = store.add(documents.clone()).await.unwrap();
    let paths = vec![
        vec!["body"],
        vec!["a.b"],
        vec!["a.b", ""],
        vec!["a.b", "nested", "v"],
        vec!["a.b", "missing"],
        vec!["null"],
        vec!["tags"],
        vec!["quote\""],
        vec!["integer"],
    ];
    for (id, document) in ids.iter().zip(&documents) {
        assert_eq!(store.get(id).await.unwrap(), Some(document.clone()));
        for path in &paths {
            let expected = path
                .iter()
                .try_fold(document, |value, key| value.as_object()?.get(*key))
                .cloned();
            let actual = store
                .get_path(id, &path.iter().map(|s| s.to_string()).collect::<Vec<_>>())
                .await
                .unwrap();
            assert_eq!(actual, expected, "path {path:?} in {document}");
        }
    }
    assert_eq!(store.get("missing").await.unwrap(), None);
}

#[tokio::test]
async fn stale_layout_rewrite_never_discards_a_later_append() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("stale.lance");
    let uri = uri.to_str().unwrap();
    let mut writer = VariantDocumentStore::create(uri, ShredPolicy::default(), None)
        .await
        .unwrap();
    writer.add(vec![json!({"a":1})]).await.unwrap();
    let mut stale = VariantDocumentStore::open(uri, None).await.unwrap();
    let kept = writer.add(vec![json!({"a":2})]).await.unwrap()[0].clone();
    assert!(stale.add(vec![json!({"new":"migrate"})]).await.is_err());
    let reopened = VariantDocumentStore::open(uri, None).await.unwrap();
    assert_eq!(reopened.get(&kept).await.unwrap(), Some(json!({"a":2})));
}
