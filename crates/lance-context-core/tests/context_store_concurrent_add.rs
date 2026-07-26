//! Does `ContextStore` lose data when two handles append concurrently?
//!
//! `ContextStore` is `#[derive(Clone)]`, so `add(&mut self)` does not prevent a
//! second concurrent writer — cloning the store yields a second mutable handle
//! onto the same dataset. Each `add` opens a fresh `mem_wal_writer` per
//! `(bot_id, session_id)` group, and opening claims the shard epoch, which
//! fences any other live writer of that shard. Unlike `RolloutStore::add` and
//! `DatagenStore::append`, this path has **no fence retry**.
//!
//! Shards are derived from the record's own `(bot_id, session_id)`
//! (`derive_region_id`), not from writer identity, so two concurrent writers
//! touching the same session necessarily target the same shard.
//!
//! This test exists to establish, empirically, whether that is a real data-loss
//! path or whether something upstream serializes it.

#![recursion_limit = "256"]

use lance_context_core::{ContextRecord, ContextStore, ContextStoreOptions};

fn rec(id: &str, session: &str) -> ContextRecord {
    ContextRecord {
        id: id.to_string(),
        external_id: None,
        run_id: "run".to_string(),
        bot_id: Some("bot".to_string()),
        session_id: Some(session.to_string()),
        tenant: None,
        source: None,
        created_at: chrono::Utc::now(),
        role: "user".to_string(),
        state_metadata: None,
        metadata: None,
        relationships: vec![],
        expires_at: None,
        retention_policy: None,
        lifecycle_status: "active".to_string(),
        retired_at: None,
        retired_reason: None,
        supersedes_id: None,
        superseded_by_id: None,
        content_type: "text/plain".to_string(),
        text_payload: Some("hello".to_string()),
        binary_payload: None,
        payload_uri: None,
        payload_size: None,
        payload_checksum: None,
        embedding: None,
    }
}

/// Two clones appending to the **same** `(bot_id, session_id)` — therefore the
/// same MemWAL shard — concurrently.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_clones_same_session_do_not_lose_rows() {
    let tmp = tempfile::tempdir().unwrap();
    let uri = tmp.path().to_string_lossy().to_string();

    // Deliberately NO serial warm-up: both writers race from a cold dataset, so
    // this also covers the one-time `initialize_mem_wal` (CreateIndex) conflict
    // on first write, not just steady-state shard-claim contention.
    let store = ContextStore::open_with_options(&uri, ContextStoreOptions::default())
        .await
        .unwrap();

    let n = 20;
    let mut a = store.clone();
    let mut b = store.clone();

    let (ra, rb) = tokio::join!(
        async move {
            let mut errs = Vec::new();
            for i in 0..n {
                if let Err(e) = a.add(&[rec(&format!("a-{i}"), "shared")]).await {
                    errs.push(format!("a-{i}: {e}"));
                }
            }
            errs
        },
        async move {
            let mut errs = Vec::new();
            for i in 0..n {
                if let Err(e) = b.add(&[rec(&format!("b-{i}"), "shared")]).await {
                    errs.push(format!("b-{i}: {e}"));
                }
            }
            errs
        }
    );

    let reader = ContextStore::open_with_options(&uri, ContextStoreOptions::default())
        .await
        .unwrap();
    let listed = reader.list(None, None).await.unwrap();
    let ids: std::collections::HashSet<String> = listed.iter().map(|r| r.id.clone()).collect();

    let missing: Vec<String> = (0..n)
        .flat_map(|i| [format!("a-{i}"), format!("b-{i}")])
        .filter(|id| !ids.contains(id))
        .collect();

    eprintln!(
        "errors_a={} errors_b={} listed={} missing={}",
        ra.len(),
        rb.len(),
        listed.len(),
        missing.len()
    );
    if !ra.is_empty() {
        eprintln!("first a error: {}", ra[0]);
    }
    if !rb.is_empty() {
        eprintln!("first b error: {}", rb[0]);
    }

    // An append that returned Ok must be readable. Silent loss is the failure
    // mode under test: a fenced writer whose `put` succeeded into a dead epoch.
    assert!(
        missing.is_empty(),
        "{} appends returned Ok but are not readable: {missing:?}",
        missing.len()
    );
    assert!(
        ra.is_empty() && rb.is_empty(),
        "concurrent appends to the same session errored: a={ra:?} b={rb:?}"
    );
}

/// Two clones appending to **different** sessions — therefore different shards.
/// Should be entirely contention-free; this isolates whether any failure above
/// is shard contention or something broader.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_clones_distinct_sessions_do_not_lose_rows() {
    let tmp = tempfile::tempdir().unwrap();
    let uri = tmp.path().to_string_lossy().to_string();

    // Deliberately NO serial warm-up: both writers race from a cold dataset, so
    // this also covers the one-time `initialize_mem_wal` (CreateIndex) conflict
    // on first write, not just steady-state shard-claim contention.
    let store = ContextStore::open_with_options(&uri, ContextStoreOptions::default())
        .await
        .unwrap();

    let n = 20;
    let mut a = store.clone();
    let mut b = store.clone();

    let (ra, rb) = tokio::join!(
        async move {
            let mut errs = Vec::new();
            for i in 0..n {
                if let Err(e) = a.add(&[rec(&format!("a-{i}"), "session-a")]).await {
                    errs.push(format!("a-{i}: {e}"));
                }
            }
            errs
        },
        async move {
            let mut errs = Vec::new();
            for i in 0..n {
                if let Err(e) = b.add(&[rec(&format!("b-{i}"), "session-b")]).await {
                    errs.push(format!("b-{i}: {e}"));
                }
            }
            errs
        }
    );

    let reader = ContextStore::open_with_options(&uri, ContextStoreOptions::default())
        .await
        .unwrap();
    let listed = reader.list(None, None).await.unwrap();
    let ids: std::collections::HashSet<String> = listed.iter().map(|r| r.id.clone()).collect();

    let missing: Vec<String> = (0..n)
        .flat_map(|i| [format!("a-{i}"), format!("b-{i}")])
        .filter(|id| !ids.contains(id))
        .collect();

    eprintln!(
        "distinct-shards: errors_a={} errors_b={} listed={} missing={}",
        ra.len(),
        rb.len(),
        listed.len(),
        missing.len()
    );

    assert!(
        missing.is_empty(),
        "{} appends to distinct sessions returned Ok but are not readable: {missing:?}",
        missing.len()
    );
    assert!(
        ra.is_empty() && rb.is_empty(),
        "concurrent appends to distinct sessions errored: a={ra:?} b={rb:?}"
    );
}
