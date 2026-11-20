//! The residual predicate must not fetch unrelated extracted text.
#![cfg(feature = "lance-io-metrics")]

use lance_context_core::{shredding::ShredPolicy, VariantDocumentStore};
use metrics_util::debugging::{DebugValue, DebuggingRecorder, Snapshotter};
use serde_json::json;

fn read_bytes(snapshotter: &Snapshotter) -> u64 {
    snapshotter
        .snapshot()
        .into_vec()
        .into_iter()
        .filter_map(|(key, _, _, value)| {
            if key.key().name() != "lance_object_store_request_bytes_total"
                || !key
                    .key()
                    .labels()
                    .any(|label| label.key() == "operation" && label.value() == "get")
            {
                return None;
            }
            if let DebugValue::Counter(value) = value {
                Some(value)
            } else {
                None
            }
        })
        .sum()
}

#[tokio::test]
async fn variant_projection_and_search_skip_sibling_text_despite_type_drift() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    metrics::set_global_recorder(recorder).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("documents.lance");
    let uri = uri.to_str().unwrap();
    let mut store = VariantDocumentStore::create(uri, ShredPolicy::default(), None)
        .await
        .unwrap();
    // High-entropy ASCII keeps compression from disguising accidental body reads.
    let mut state = 832_u64;
    let body: String = (0..4 * 1024 * 1024)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            char::from(b' ' + (state % 95) as u8)
        })
        .collect();
    let ids = store
        .add(vec![
            json!({"body": "seed", "sequence": 7}),
            json!({"body": "seed", "sequence": 8}),
            json!({"body": body, "sequence": true, "tags": ["wanted"]}),
        ])
        .await
        .unwrap();
    store.prepare_text_search().await.unwrap();
    drop(store);
    let mut store = VariantDocumentStore::open(uri, None).await.unwrap();
    let before = read_bytes(&snapshotter);
    assert_eq!(
        store
            .filter_eq(&["tags".into()], &json!(["wanted"]), 1)
            .await
            .unwrap(),
        vec![ids[2].clone()]
    );
    let bytes = read_bytes(&snapshotter) - before;
    assert!(bytes > 0, "storage metrics must be active");
    assert!(
        bytes < 1024 * 1024,
        "small residual predicate read {bytes} bytes, including unrelated text"
    );
    let before = read_bytes(&snapshotter);
    assert_eq!(
        store.get_path(&ids[2], &["tags".into()]).await.unwrap(),
        Some(json!(["wanted"]))
    );
    assert_eq!(
        store.get_path(&ids[2], &["absent".into()]).await.unwrap(),
        None
    );
    assert_eq!(
        store.get_path(&ids[2], &["sequence".into()]).await.unwrap(),
        Some(json!(true))
    );
    let bytes = read_bytes(&snapshotter) - before;
    assert!(
        bytes < 1024 * 1024,
        "residual point reads fetched {bytes} bytes of unrelated text"
    );
    // Load the index dictionary before measuring residual-scan I/O.
    assert!(store.search("absentneedle", 10).await.unwrap().is_empty());
    let before = read_bytes(&snapshotter);
    assert!(store.search("absentneedle", 10).await.unwrap().is_empty());
    let bytes = read_bytes(&snapshotter) - before;
    assert!(
        bytes < 1024 * 1024,
        "no-hit search read {bytes} bytes, including compatible sibling text"
    );
    let before = read_bytes(&snapshotter);
    assert_eq!(store.get(&ids[2]).await.unwrap().unwrap()["body"], body);
    let bytes = read_bytes(&snapshotter) - before;
    assert!(
        bytes > 1024 * 1024,
        "full read must actually fetch text: {bytes}"
    );
}
