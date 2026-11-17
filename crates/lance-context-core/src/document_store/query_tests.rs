use super::*;
use serde_json::json;

#[tokio::test]
async fn logical_filters_include_old_residuals_and_distinguish_null_from_missing() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("filter.lance");
    let policy = ShredPolicy {
        min_frequency: 1.0,
        ..Default::default()
    };
    let mut store = DocumentStore::create(uri.to_str().unwrap(), policy, None)
        .await
        .unwrap();
    let ids = store
        .add(vec![json!({"x": "match"}), json!({})])
        .await
        .unwrap();
    assert!(store.layout().fields.is_empty());
    let newer = store.add(vec![json!({"x": "match"})]).await.unwrap();
    let nulls = store
        .add(vec![json!({"x": null}), json!({}), json!({"x": [1]})])
        .await
        .unwrap();
    let path = vec!["x".into()];
    let mut matches = store.filter_eq(&path, &json!("match"), 100).await.unwrap();
    matches.sort();
    let mut expected = vec![ids[0].clone(), newer[0].clone()];
    expected.sort();
    assert_eq!(matches, expected);
    assert_eq!(
        store.filter_eq(&path, &Value::Null, 100).await.unwrap(),
        vec![nulls[0].clone()]
    );
    assert_eq!(
        store.filter_eq(&path, &json!([1]), 100).await.unwrap(),
        vec![nulls[2].clone()]
    );
    assert_eq!(
        store
            .filter_eq(&path, &json!("match"), 1)
            .await
            .unwrap()
            .len(),
        1
    );
    assert!(store
        .filter_eq(&path, &Value::Null, 0)
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn search_automatically_backfills_historical_text_and_reads_new_tail() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("search.lance");
    let uri = uri.to_str().unwrap();
    let policy = ShredPolicy {
        min_frequency: 1.0,
        ..Default::default()
    };
    let mut store = DocumentStore::create(uri, policy, None).await.unwrap();
    let old = json!({"body": "otter swims upstream"});
    let ids = store
        .add(vec![old.clone(), json!({"other": 1})])
        .await
        .unwrap();
    assert!(store.layout().fields.is_empty());
    let recent = store
        .add(vec![json!({"body": "otter swims downstream"})])
        .await
        .unwrap();
    let before_search = store.version();
    let mut matches: Vec<_> = store
        .search("OTTER", 10)
        .await
        .unwrap()
        .into_iter()
        .map(|hit| hit.id)
        .collect();
    matches.sort();
    let mut expected = vec![ids[0].clone(), recent[0].clone()];
    expected.sort();
    assert_eq!(matches, expected);
    assert_eq!(store.get(&ids[0]).await.unwrap(), Some(old.clone()));
    let prepared_version = store.version();
    assert!(prepared_version > before_search);
    assert_eq!(store.search("otter", 1).await.unwrap().len(), 1);
    assert_eq!(store.version(), prepared_version);
    let tail = store
        .add(vec![json!({"body": "otter loves water"})])
        .await
        .unwrap();
    assert!(store
        .search("otter", 10)
        .await
        .unwrap()
        .iter()
        .any(|hit| hit.id == tail[0]));
    let another_path = store
        .add(vec![
            json!({"body": "nothing", "new_text": "otter in a new column"}),
        ])
        .await
        .unwrap();
    assert!(store
        .search("otter", 10)
        .await
        .unwrap()
        .iter()
        .any(|hit| hit.id == another_path[0]));
    let mut reopened = DocumentStore::open(uri, None).await.unwrap();
    assert_eq!(reopened.search("otter", 10).await.unwrap().len(), 4);
    reopened.checkout(prepared_version).await.unwrap();
    assert_eq!(reopened.search("otter", 10).await.unwrap().len(), 2);
    reopened.checkout(before_search).await.unwrap();
    assert_eq!(reopened.get(&ids[0]).await.unwrap(), Some(old));
    assert!(reopened.search("otter", 10).await.is_err());
}

#[tokio::test]
async fn stale_backfill_cannot_overwrite_a_newer_append() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("conflict.lance");
    let uri = uri.to_str().unwrap();
    let mut writer = DocumentStore::create(uri, ShredPolicy::default(), None)
        .await
        .unwrap();
    writer.add(vec![json!({"text": "first"})]).await.unwrap();
    let mut stale = DocumentStore::open(uri, None).await.unwrap();
    let ids = writer.add(vec![json!({"text": "second"})]).await.unwrap();
    assert!(stale.backfill().await.is_err());
    let current = DocumentStore::open(uri, None).await.unwrap();
    assert_eq!(
        current.get(&ids[0]).await.unwrap(),
        Some(json!({"text": "second"}))
    );
}
#[tokio::test]
async fn projected_residual_filters_preserve_containers_ancestors_and_batch_boundaries() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("projected.lance");
    let mut store = DocumentStore::create(uri.to_str().unwrap(), ShredPolicy::default(), None)
        .await
        .unwrap();
    // Select a scalar ancestor before its later object-shaped values arrive.
    let first = store
        .add(vec![
            json!({"node": "scalar", "body": "large unrelated text"}),
        ])
        .await
        .unwrap();
    let documents: Vec<_> = (0..1030)
        .map(|i| {
            json!({
                "body": "unrelated payload ".repeat(64),
                "node": {"child": i, "sibling": true},
                "tags": [i], "nullable": null,
            })
        })
        .collect();
    let ids = store.add(documents.clone()).await.unwrap();
    let last = documents.last().unwrap();
    assert_eq!(
        store
            .filter_eq(&["tags".into()], &json!([1029]), 10)
            .await
            .unwrap(),
        vec![ids[1029].clone()]
    );
    assert_eq!(
        store
            .filter_eq(&["node".into()], &last["node"], 10)
            .await
            .unwrap(),
        vec![ids[1029].clone()]
    );
    assert_eq!(
        store.filter_eq(&[], last, 10).await.unwrap(),
        vec![ids[1029].clone()]
    );
    assert_eq!(
        store
            .filter_eq(&["node".into()], &json!("scalar"), 10)
            .await
            .unwrap(),
        first
    );
    assert_eq!(
        store
            .filter_eq(&["nullable".into()], &Value::Null, 2000)
            .await
            .unwrap()
            .len(),
        ids.len()
    );
    assert!(store
        .filter_eq(&["absent".into()], &Value::Null, 2000)
        .await
        .unwrap()
        .is_empty());
    let version = store.version();
    store.backfill().await.unwrap();
    for (id, expected) in ids.iter().zip(&documents).step_by(127) {
        assert_eq!(store.get(id).await.unwrap().as_ref(), Some(expected));
    }
    assert_eq!(store.get(&ids[1029]).await.unwrap().as_ref(), Some(last));
    store.checkout(version).await.unwrap();
    assert_eq!(store.get(&ids[1029]).await.unwrap().as_ref(), Some(last));
}
