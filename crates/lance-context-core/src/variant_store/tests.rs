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

#[tokio::test]
async fn text_search_indexes_variant_children_and_covers_residuals_without_copies() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("search.lance");
    let mut store = VariantDocumentStore::create(
        uri.to_str().unwrap(),
        ShredPolicy {
            max_columns: 1,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    let first = store
        .add(vec![json!({"body":"otter indexed"})])
        .await
        .unwrap()[0]
        .clone();
    store.prepare_text_search().await.unwrap();
    let added = store
        .add(vec![
            json!({"body":"otter both","tags":["otter"]}),
            json!({"tags":["otter residual"]}),
            json!({"keyonly":987654}),
            json!("rootword"),
        ])
        .await
        .unwrap();
    let before = store.version();
    store.prepare_text_search().await.unwrap();
    assert_eq!(
        store.version(),
        before,
        "preparation of unchanged layout must be a no-op"
    );
    let hits = store.search("OTTER", 10).await.unwrap();
    assert_eq!(hits.len(), 3);
    let ids = hits
        .iter()
        .map(|hit| hit.id.as_str())
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(ids.len(), 3);
    assert!(ids.contains(first.as_str()));
    assert!(ids.contains(added[0].as_str()));
    assert_eq!(hits.last().unwrap().id, added[1]);
    assert_eq!(hits.last().unwrap().score, 0.0);
    assert_eq!(store.search("rootword", 10).await.unwrap()[0].id, added[3]);
    for query in ["keyonly", "987654", "ott", "unmatchedneedle"] {
        assert!(store.search(query, 10).await.unwrap().is_empty());
    }
    assert_eq!(
        store.dataset.schema().fields.len(),
        2,
        "only ID plus the standard Variant column"
    );
    let version = store.version();
    store.checkout(version).await.unwrap();
    assert_eq!(store.search("otter", 10).await.unwrap().len(), 3);
}

#[tokio::test]
async fn variant_equality_preserves_missing_null_and_residual_containers() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("filter.lance");
    let mut store =
        VariantDocumentStore::create(uri.to_str().unwrap(), ShredPolicy::default(), None)
            .await
            .unwrap();
    let ids = store
        .add(vec![
            json!({"v":null}),
            json!({}),
            json!({"v":1}),
            json!({"v":[1,null]}),
        ])
        .await
        .unwrap();
    assert_eq!(
        store
            .filter_eq(&["v".into()], &Value::Null, 10)
            .await
            .unwrap(),
        vec![ids[0].clone()]
    );
    assert_eq!(
        store.filter_eq(&["v".into()], &json!(1), 10).await.unwrap(),
        vec![ids[2].clone()]
    );
    assert_eq!(
        store
            .filter_eq(&["v".into()], &json!([1, null]), 10)
            .await
            .unwrap(),
        vec![ids[3].clone()]
    );
}

#[tokio::test]
async fn nested_and_literal_text_paths_are_searchable() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("nested_search.lance");
    let mut store =
        VariantDocumentStore::create(uri.to_str().unwrap(), ShredPolicy::default(), None)
            .await
            .unwrap();
    let record = json!({"nested":{"body":"indexedotter"}, "a.b":{"":"literalotter"}, "back`tick":"tickotter", "quote\"":"quoteotter"});
    let id = store.add(vec![record.clone()]).await.unwrap()[0].clone();
    assert_eq!(store.layout().fields().len(), 4);
    for query in ["indexedotter", "literalotter", "tickotter", "quoteotter"] {
        let hits = store.search(query, 10).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, id);
    }
    assert_eq!(
        store
            .get_path(&id, &["a.b".into(), "".into()])
            .await
            .unwrap(),
        Some(json!("literalotter"))
    );
    assert_eq!(store.get(&id).await.unwrap(), Some(record));
}

#[tokio::test]
async fn text_search_survives_layout_replacement_and_pinned_snapshots() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("replace_index.lance");
    let mut store = VariantDocumentStore::create(
        uri.to_str().unwrap(),
        ShredPolicy {
            max_columns: 1,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    let old = store
        .add(vec![json!({"body":"originalotter"})])
        .await
        .unwrap()[0]
        .clone();
    assert_eq!(store.search("originalotter", 10).await.unwrap()[0].id, old);
    let version = store.version();
    let new = store
        .add(vec![json!({"later":"recentotter"})])
        .await
        .unwrap()[0]
        .clone();
    assert_eq!(store.layout().fields()[0].path, ["later"]);
    assert_eq!(store.search("originalotter", 10).await.unwrap()[0].id, old);
    assert_eq!(store.search("recentotter", 10).await.unwrap()[0].id, new);
    store.checkout(version).await.unwrap();
    assert_eq!(store.search("originalotter", 10).await.unwrap()[0].id, old);
    assert!(store.search("recentotter", 10).await.unwrap().is_empty());
    store.refresh().await.unwrap();
    assert_eq!(store.search("recentotter", 10).await.unwrap()[0].id, new);
}

#[tokio::test]
async fn sparse_large_text_stays_compressed_across_migration_and_append() {
    fn retained_bytes(path: &std::path::Path) -> u64 {
        std::fs::read_dir(path)
            .unwrap()
            .map(|entry| {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    retained_bytes(&path)
                } else {
                    path.metadata().unwrap().len()
                }
            })
            .sum()
    }
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("sparse.lance");
    let mut store = VariantDocumentStore::create(
        uri.to_str().unwrap(),
        ShredPolicy {
            max_columns: 1,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    let ids = store
        .add((0..1000).map(|i| json!({"n":i})).collect())
        .await
        .unwrap();
    let first = store.version();
    let body = "late text ".repeat(400_000);
    let text_id = store.add(vec![json!({"body":body})]).await.unwrap()[0].clone();
    assert_eq!(store.get(&ids[0]).await.unwrap(), Some(json!({"n":0})));
    assert_eq!(
        store.get(&text_id).await.unwrap(),
        Some(json!({"body":body}))
    );
    let bytes = retained_bytes(&uri);
    assert!(
        bytes < 1024 * 1024,
        "sparse migrated text stored raw: {bytes} bytes"
    );
    let mut documents = vec![json!({}); 1000];
    documents.push(json!({"body":body}));
    let appended = store.add(documents).await.unwrap();
    assert_eq!(
        store.get(appended.last().unwrap()).await.unwrap(),
        Some(json!({"body":body}))
    );
    let bytes = retained_bytes(&uri);
    assert!(
        bytes < 1024 * 1024,
        "sparse appended text stored raw: {bytes} bytes"
    );
    store.checkout(first).await.unwrap();
    assert!(store.get(&text_id).await.unwrap().is_none());
    store.refresh().await.unwrap();
    let reopened = VariantDocumentStore::open(uri.to_str().unwrap(), None)
        .await
        .unwrap();
    assert_eq!(
        reopened.get(&text_id).await.unwrap(),
        Some(json!({"body":body}))
    );
}
