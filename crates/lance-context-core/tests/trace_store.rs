use std::collections::BTreeSet;

use lance_context_api::MAX_BATCH_GET_IDS;
use lance_context_core::{
    trace_schema, GenericStore, GenericStoreOptions, TraceRecord, TraceStore,
};
use serde_json::json;
use tempfile::TempDir;

fn turn(id: &str, session: &str, turn_id: i64, content: Option<&str>) -> TraceRecord {
    TraceRecord {
        id: id.into(),
        session_id: session.into(),
        turn_id,
        role: "assistant".into(),
        content: content.map(str::to_owned),
        content_type: "text/plain".into(),
        source: Some("model_calls".into()),
        metadata: Some(json!({"parsed":true})),
    }
}

#[tokio::test]
async fn late_turns_variants_and_replays_survive_merge_and_restart() {
    let dir = TempDir::new().unwrap();
    let uri = dir
        .path()
        .join("trace.lance")
        .to_string_lossy()
        .into_owned();
    let backend = GenericStore::open(
        &uri,
        trace_schema(),
        GenericStoreOptions {
            seal_on_add: true,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let store = TraceStore::new(backend).unwrap();
    let later = turn("later", "s", 2, Some("C"));
    let early = turn("early", "s", 0, Some("A"));
    let variant = turn("branch", "s", 2, Some("different C"));
    let other = turn("other-session", "other", 0, Some("A"));
    store.add(std::slice::from_ref(&later)).await.unwrap();
    store
        .add(&[early.clone(), variant.clone(), other.clone()])
        .await
        .unwrap();
    // Replay both before and after sealing: identity, not arrival order, decides
    // which logical rows survive. Equal positions are allowed across branches.
    store.add(&[later.clone(), early.clone()]).await.unwrap();
    let ids = vec![
        "branch".into(),
        "missing".into(),
        "early".into(),
        "later".into(),
        "branch".into(),
    ];
    assert_eq!(
        store.existing_ids(&ids).await.unwrap(),
        ["branch", "early", "later"]
    );
    assert_eq!(
        store.get_many(&ids).await.unwrap(),
        [variant.clone(), early.clone(), later.clone()]
    );
    let mut backend = store.into_generic();
    backend.cleanup_wal().await.unwrap();
    backend.create_id_index().await.unwrap();
    backend.close().await.unwrap();
    let backend = GenericStore::open_existing(&uri, Default::default())
        .await
        .unwrap();
    let store = TraceStore::new(backend).unwrap();
    store.add(&[later.clone(), early.clone()]).await.unwrap();
    assert_eq!(store.get_many(&ids).await.unwrap(), [variant, early, later]);
    let rows = store.as_generic().list(None, None).await.unwrap();
    assert_eq!(rows.len(), 4);
    assert_eq!(
        rows.iter()
            .map(|r| r["id"].as_str().unwrap())
            .collect::<BTreeSet<_>>()
            .len(),
        4
    );
    assert_eq!(store.get_many(&[other.id.clone()]).await.unwrap(), [other]);
    store.into_generic().close().await.unwrap();
    assert_eq!(
        std::fs::read_dir(dir.path()).unwrap().count(),
        1,
        "trace creates one dataset"
    );
}

#[tokio::test]
async fn batch_probe_uses_generic_limits_projection_and_wal_visibility() {
    let dir = TempDir::new().unwrap();
    let backend = GenericStore::open(
        dir.path().join("trace.lance").to_str().unwrap(),
        trace_schema(),
        GenericStoreOptions::default(),
    )
    .await
    .unwrap();
    let store = TraceStore::new(backend).unwrap();
    let records: Vec<_> = (0..MAX_BATCH_GET_IDS)
        .map(|i| turn(&format!("id-{i}"), "s", i as i64, Some("payload")))
        .collect();
    let ids: Vec<_> = records.iter().map(|r| r.id.clone()).collect();
    assert_eq!(store.add(&records).await.unwrap().count, records.len());
    assert!(store.existing_ids(&ids).await.unwrap().is_empty());
    store.flush().await.unwrap();
    assert_eq!(store.existing_ids(&ids).await.unwrap(), ids);
    assert_eq!(store.get_many(&ids).await.unwrap(), records);
    let projected = store
        .as_generic()
        .get_many(&ids[..1], Some(&["id".into()]))
        .await
        .unwrap();
    assert_eq!(
        projected,
        vec![json!({"id":"id-0"}).as_object().unwrap().clone()]
    );
    assert!(store.existing_ids(&[]).await.unwrap().is_empty());
    assert!(store.get_many(&[]).await.unwrap().is_empty());
    let too_many = vec!["id-0".into(); MAX_BATCH_GET_IDS + 1];
    assert!(store.existing_ids(&too_many).await.is_err());
    assert!(store.get_many(&too_many).await.is_err());
    store.into_generic().close().await.unwrap();
}

#[tokio::test]
async fn invalid_batch_writes_nothing_and_mismatched_schema_is_rejected() {
    let dir = TempDir::new().unwrap();
    let backend = GenericStore::open(
        dir.path().join("trace.lance").to_str().unwrap(),
        trace_schema(),
        GenericStoreOptions {
            seal_on_add: true,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let store = TraceStore::new(backend).unwrap();
    let valid = turn("valid", "s", 0, None);
    let invalid = turn("invalid", "s", -1, None);
    assert!(store.add(&[valid.clone(), invalid]).await.is_err());
    assert!(store
        .existing_ids(&["valid".into()])
        .await
        .unwrap()
        .is_empty());
    let mut empty = turn("empty", "s", 1, Some(""));
    empty.metadata = Some(serde_json::Value::Null);
    store.add(&[valid.clone(), empty.clone()]).await.unwrap();
    assert_eq!(
        store
            .get_many(&["valid".into(), "empty".into()])
            .await
            .unwrap(),
        [valid, empty]
    );
    store.into_generic().close().await.unwrap();
    let wrong = lance_context_api::SchemaSpec::new(vec![(
        "id".into(),
        lance_context_api::ColumnSpec::required(lance_context_api::ColumnType::String {
            large: false,
        }),
    )]);
    let backend = GenericStore::open(
        dir.path().join("wrong.lance").to_str().unwrap(),
        wrong,
        Default::default(),
    )
    .await
    .unwrap();
    assert!(TraceStore::new(backend).is_err());
}
