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

fn current_data_files(store: &DocumentStore) -> Vec<String> {
    let mut files: Vec<_> = store
        .dataset
        .manifest()
        .fragments
        .iter()
        .flat_map(|fragment| fragment.files.iter().map(|file| file.path.clone()))
        .collect();
    files.sort();
    files
}

#[tokio::test]
async fn text_preparation_preserves_files_when_no_historical_text_needs_promotion() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("no-rewrite.lance");
    let uri = uri.to_str().unwrap();
    let mut store = DocumentStore::create(uri, ShredPolicy::default(), None)
        .await
        .unwrap();
    let first = store
        .add(vec![json!({"body": "otter original"})])
        .await
        .unwrap();
    let files = current_data_files(&store);
    store.prepare_text_search().await.unwrap();
    assert_eq!(
        current_data_files(&store),
        files,
        "fresh extracted text must not be rewritten"
    );
    let prepared = store.version();
    store.prepare_text_search().await.unwrap();
    assert_eq!(store.version(), prepared);

    let second = store
        .add(vec![json!({"body": "otter later", "counter": 42})])
        .await
        .unwrap();
    let files = current_data_files(&store);
    let version = store.version();
    store.prepare_text_search().await.unwrap();
    assert_eq!(
        store.version(),
        version,
        "numeric growth requires no text maintenance"
    );
    assert_eq!(current_data_files(&store), files);
    let mut store = DocumentStore::open(uri, None).await.unwrap();
    assert_eq!(store.search("otter", 10).await.unwrap().len(), 2);
    assert_eq!(store.version(), version);
    store.checkout(prepared).await.unwrap();
    assert_eq!(store.search("otter", 10).await.unwrap()[0].id, first[0]);
    assert!(store.get(&second[0]).await.unwrap().is_none());
}

#[tokio::test]
async fn legacy_backfill_marker_survives_numeric_growth_and_new_text_still_backfills() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("legacy.lance");
    let uri = uri.to_str().unwrap();
    let mut store = DocumentStore::create(uri, ShredPolicy::default(), None)
        .await
        .unwrap();
    store.add(vec![json!({"body": "otter"})]).await.unwrap();
    store.backfill().await.unwrap();
    store.prepare_text_search().await.unwrap();
    // Reproduce metadata written by the previous release.
    store
        .dataset
        .update_schema_metadata([("lance-context:shred-backfilled-text-columns", None::<&str>)])
        .await
        .unwrap();
    store
        .add(vec![json!({"body": "otter", "counter": 42})])
        .await
        .unwrap();
    let files = current_data_files(&store);
    let version = store.version();
    store.prepare_text_search().await.unwrap();
    assert_eq!(current_data_files(&store), files);
    assert_eq!(store.version(), version);

    // A rare text path remains residual in old rows and is promoted later.
    let mut docs = vec![json!({"body": "otter", "late": "historical platypus"})];
    docs.extend((0..3).map(|_| json!({"body": "otter"})));
    let old = store.add(docs).await.unwrap();
    assert!(!store
        .layout
        .fields
        .iter()
        .any(|field| field.path == ["late"]));
    store
        .add(vec![json!({"body": "otter", "late": "new platypus"})])
        .await
        .unwrap();
    let before = current_data_files(&store);
    assert_eq!(store.search("platypus", 10).await.unwrap().len(), 2);
    assert_ne!(current_data_files(&store), before);
    assert_eq!(
        store.get_path(&old[0], &["late".into()]).await.unwrap(),
        Some(json!("historical platypus"))
    );
}

#[tokio::test]
async fn search_covers_exhausted_layout_and_type_changes_without_duplicate_text() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("coverage.lance");
    let uri = uri.to_str().unwrap();
    let mut store = DocumentStore::create(uri, ShredPolicy::default(), None)
        .await
        .unwrap();
    let numeric: serde_json::Map<String, Value> =
        (0..32).map(|i| (format!("field{i}"), json!(i))).collect();
    store.add(vec![Value::Object(numeric)]).await.unwrap();
    let document = json!({"body": "LATEOTTER ".repeat(8192), "field0": "typedrift"});
    let id = store.add(vec![document.clone()]).await.unwrap()[0].clone();
    assert_eq!(store.layout.fields.len(), 32);
    assert!(!store
        .layout
        .fields
        .iter()
        .any(|field| field.data_type == ShredType::String));
    let version = store.version();
    for query in ["lateotter", "TYPEDRIFT", "missing lateotter"] {
        let hits = store.search(query, 10).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, id);
        assert_eq!(hits[0].score, 0.0);
    }
    assert_eq!(
        store.version(),
        version,
        "residual search must not rewrite documents"
    );
    let mut reopened = DocumentStore::open(uri, None).await.unwrap();
    assert_eq!(reopened.get(&id).await.unwrap(), Some(document));
    reopened.checkout(version).await.unwrap();
    assert_eq!(reopened.search("typedrift", 1).await.unwrap()[0].id, id);
}

#[tokio::test]
async fn residual_search_matches_string_values_not_keys_or_json_escape_syntax() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("tokens.lance");
    let mut store = DocumentStore::create(
        uri.to_str().unwrap(),
        ShredPolicy {
            max_columns: 1,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    store.add(vec![json!({"slot": 1})]).await.unwrap();
    let ids = store
        .add(vec![
            json!({"keyonly": 987654, "body": "MiXeD\nquoted\t世界"}),
            json!([{"inner": "arrayotter"}]),
            json!("roototter"),
            json!({"body": "not a substringmatch"}),
        ])
        .await
        .unwrap();
    assert_eq!(store.search("MIXED", 10).await.unwrap()[0].id, ids[0]);
    assert_eq!(store.search("世界", 10).await.unwrap()[0].id, ids[0]);
    assert_eq!(store.search("arrayotter", 10).await.unwrap()[0].id, ids[1]);
    assert_eq!(store.search("roototter", 10).await.unwrap()[0].id, ids[2]);
    for query in ["keyonly", "987654", "nquoted", "substring", "!!!"] {
        assert!(
            store.search(query, 10).await.unwrap().is_empty(),
            "false positive for {query}"
        );
    }
    assert!(store.search("mixed", 0).await.unwrap().is_empty());
}

#[tokio::test]
async fn indexed_and_residual_search_deduplicates_and_preserves_native_ranking() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("mixed-search.lance");
    let mut store = DocumentStore::create(
        uri.to_str().unwrap(),
        ShredPolicy {
            max_columns: 1,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    let indexed = store
        .add(vec![json!({"body": "otter indexed"})])
        .await
        .unwrap();
    store.prepare_text_search().await.unwrap();
    let tail = store
        .add(vec![
            json!({"body": "otter", "extra": "otter"}),
            json!({"extra": "otter residual"}),
        ])
        .await
        .unwrap();
    let hits = store.search("OTTER", 10).await.unwrap();
    assert_eq!(hits.len(), 3);
    let unique: std::collections::HashSet<_> = hits.iter().map(|hit| hit.id.as_str()).collect();
    assert_eq!(unique.len(), 3);
    assert!(unique.contains(indexed[0].as_str()));
    assert!(unique.contains(tail[0].as_str()));
    assert_eq!(hits.last().unwrap().id, tail[1]);
    assert_eq!(hits.last().unwrap().score, 0.0);
    assert_eq!(store.search("otter", 1).await.unwrap().len(), 1);
}
