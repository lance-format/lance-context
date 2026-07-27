//! Concurrency tests for WAL self-merge: a merge must never block or corrupt
//! concurrent appends.
//!
//! The pre-existing suite (`wal_merge_generation_cleanup.rs`) is explicitly
//! "one store, one shard, fully serial — no concurrency, no fence", so none of
//! the properties below had coverage. They matter because a merge used to
//! `claim_epoch`, which fences the shard's live writer; callers hid that window
//! by holding an exclusive lock across the whole merge, turning background
//! compaction into a stop-the-world pause on the write path (~17s observed on
//! abfss with 20 concurrent writers).
//!
//! The merge now reuses the shard's current epoch instead of claiming a new
//! one, so the writer is never fenced and callers can merge under a shared
//! lock. These tests pin the properties that makes safe:
//!
//! 1. appends succeed while a merge runs, and no row is lost;
//! 2. a generation sealed *during* a merge is not silently dropped by the drain;
//! 3. concurrent merges do not duplicate rows;
//! 4. an interrupted merge loses nothing (rows stay readable exactly once);
//! 5. `add` is not blocked for the merge's duration.

use std::sync::Arc;
use std::time::{Duration, Instant};

use lance_context_core::{RolloutRecord, RolloutStore, RolloutStoreOptions, ROLE_ASSISTANT};
use tokio::sync::RwLock;

/// Run one full merge exactly the way the server's sweepers do: seal + read the
/// generations under a **read** lock (so appends keep running), then take the
/// write lock only for the short commit. Returns generations reclaimed.
///
/// Every test drives merges through this helper so the lock discipline under
/// test is the same one production uses -- a test that merged under a single
/// exclusive lock would pass while the real stall persisted.
async fn merge_like_sweeper(store: &Arc<RwLock<RolloutStore>>) -> usize {
    let prepared = { store.read().await.prepare_cleanup_merge().await.unwrap() };
    match prepared {
        Some((manifest_store, manifest, prepared)) => store
            .write()
            .await
            .commit_prepared_merge(&manifest_store, &manifest, prepared)
            .await
            .unwrap(),
        None => 0,
    }
}

fn rec(id: &str) -> RolloutRecord {
    RolloutRecord {
        id: id.to_string(),
        rollout_id: "r".to_string(),
        problem_id: "p".to_string(),
        dataset: Some("d".to_string()),
        sequence_order: 0,
        role: ROLE_ASSISTANT.to_string(),
        created_at: chrono::Utc::now(),
        content: Some("x".to_string()),
        content_type: "text/plain".to_string(),
        model_input_string: None,
        model_output_string: None,
        rationale: None,
        problem_text: None,
        user_metadata: None,
        input_tokens: None,
        output_tokens: None,
        num_input_tokens: None,
        num_output_tokens: None,
        output_logprobs: None,
        input_logprobs: None,
        ref_logprobs: None,
        loss_mask: None,
        advantage: None,
        reward: None,
        raw_reward: None,
        grader_id: None,
        score: None,
        include_in_training: None,
        exclude_reason: None,
        policy_version: None,
        relationships: vec![],
        binary_payload: None,
        payload_size: None,
        payload_checksum: None,
        artifact_type: None,
        metadata: None,
    }
}

fn opts(shard: &str) -> RolloutStoreOptions {
    RolloutStoreOptions {
        shard_id: Some(shard.to_string()),
        // Never fire the count trigger implicitly; every test drives merges
        // explicitly so the interleaving under test is the one being asserted.
        merge_after_generations: None,
        ..Default::default()
    }
}

/// Read every row through the deduplicating LSM path and return sorted ids.
///
/// Asserting on `observe().row_count` would be wrong: it is a raw `count_rows()`
/// over the base table and does not de-duplicate, so it counts the physical
/// duplicates an interrupted merge can legitimately leave behind.
async fn read_ids(store: &Arc<RwLock<RolloutStore>>) -> Vec<String> {
    let mut ids: Vec<String> = store
        .read()
        .await
        .list(None, None)
        .await
        .unwrap()
        .iter()
        .map(|r| r.id.clone())
        .collect();
    ids.sort();
    ids
}

/// Appends must keep succeeding while a merge is in flight, and every row
/// written before, during, and after the merge must be readable exactly once.
///
/// This is the core regression: when the merge claimed the epoch it fenced the
/// live writer, and `add` only retries a fence **once** (un-looped), so appends
/// racing a merge could fail outright.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn appends_succeed_and_are_not_lost_while_merge_runs() {
    let tmp = tempfile::tempdir().unwrap();
    let uri = tmp.path().to_string_lossy().to_string();
    let store = Arc::new(RwLock::new(
        RolloutStore::open_with_options(&uri, opts("solo"))
            .await
            .unwrap(),
    ));

    // Build several sealed generations for the merge to chew through.
    for i in 0..8 {
        store
            .read()
            .await
            .add(&[rec(&format!("pre-{i}"))])
            .await
            .unwrap();
        store.read().await.flush().await.unwrap();
    }

    // Merge concurrently with a stream of appends. Both take `&self`.
    let merger = {
        let store = store.clone();
        tokio::spawn(async move { merge_like_sweeper(&store).await })
    };
    let appender = {
        let store = store.clone();
        tokio::spawn(async move {
            for i in 0..40 {
                // Must not error: a fenced writer would surface here.
                store
                    .read()
                    .await
                    .add(&[rec(&format!("during-{i}"))])
                    .await
                    .unwrap_or_else(|e| panic!("append {i} failed during merge: {e}"));
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
    };

    let reclaimed = merger.await.unwrap();
    appender.await.unwrap();
    assert!(reclaimed > 0, "merge should have reclaimed generations");

    // Seal whatever the appender left buffered, then read everything back.
    store.read().await.flush().await.unwrap();
    let ids = read_ids(&store).await;

    let mut expected: Vec<String> = (0..8)
        .map(|i| format!("pre-{i}"))
        .chain((0..40).map(|i| format!("during-{i}")))
        .collect();
    expected.sort();

    assert_eq!(
        ids, expected,
        "every row written before and during the merge must be readable exactly once"
    );
}

/// A generation sealed *while* a merge is running must survive the drain.
///
/// The drain is a relative edit (retain everything not in the merged set) and
/// `commit_update` re-runs it against a freshly-read manifest on every CAS
/// retry. An absolute `flushed_generations = []` would silently discard a
/// generation that was never merged into the base table — data loss that no
/// epoch value prevents.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn generation_sealed_during_merge_is_not_dropped() {
    let tmp = tempfile::tempdir().unwrap();
    let uri = tmp.path().to_string_lossy().to_string();
    let store = Arc::new(RwLock::new(
        RolloutStore::open_with_options(&uri, opts("solo"))
            .await
            .unwrap(),
    ));

    for i in 0..6 {
        store
            .read()
            .await
            .add(&[rec(&format!("old-{i}"))])
            .await
            .unwrap();
        store.read().await.flush().await.unwrap();
    }

    // Race a merge against a writer that keeps sealing brand-new generations.
    let merger = {
        let store = store.clone();
        tokio::spawn(async move { merge_like_sweeper(&store).await })
    };
    let sealer = {
        let store = store.clone();
        tokio::spawn(async move {
            for i in 0..10 {
                store
                    .read()
                    .await
                    .add(&[rec(&format!("new-{i}"))])
                    .await
                    .unwrap();
                // Seal immediately so these land as generations the merge did
                // not snapshot.
                store.read().await.flush().await.unwrap();
            }
        })
    };

    merger.await.unwrap();
    sealer.await.unwrap();

    store.read().await.flush().await.unwrap();
    let ids = read_ids(&store).await;

    for i in 0..6 {
        assert!(
            ids.contains(&format!("old-{i}")),
            "pre-merge row old-{i} vanished; the drain dropped a merged generation"
        );
    }
    for i in 0..10 {
        assert!(
            ids.contains(&format!("new-{i}")),
            "row new-{i}, sealed during the merge, was dropped by the drain"
        );
    }
    assert_eq!(ids.len(), 16, "no row may be lost or duplicated: {ids:?}");
}

/// Two merges racing must not append the same generations twice.
///
/// Merges are serialized by an internal mutex (not the store lock, which would
/// also exclude appends); the loser returns 0 rather than waiting.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_merges_do_not_duplicate_rows() {
    let tmp = tempfile::tempdir().unwrap();
    let uri = tmp.path().to_string_lossy().to_string();
    let store = Arc::new(RwLock::new(
        RolloutStore::open_with_options(&uri, opts("solo"))
            .await
            .unwrap(),
    ));

    for i in 0..10 {
        store
            .read()
            .await
            .add(&[rec(&format!("row-{i}"))])
            .await
            .unwrap();
        store.read().await.flush().await.unwrap();
    }

    // Fire several merges at once at the same pending set.
    let mut handles = Vec::new();
    for _ in 0..4 {
        let store = store.clone();
        handles.push(tokio::spawn(
            async move { merge_like_sweeper(&store).await },
        ));
    }
    for h in handles {
        // None may error; a loser simply reports 0.
        h.await.unwrap();
    }

    let ids = read_ids(&store).await;
    let mut deduped = ids.clone();
    deduped.dedup();
    assert_eq!(
        ids, deduped,
        "the read path must not surface duplicate ids after racing merges"
    );
    assert_eq!(ids.len(), 10, "all rows readable exactly once: {ids:?}");
}

/// A merge abandoned partway (the sweeper's timeout does exactly this) must not
/// lose data. Rows may be physically duplicated in the base table — the read
/// path de-dups by id — but nothing may disappear, and a later merge must still
/// converge.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn interrupted_merge_loses_nothing_and_next_merge_converges() {
    let tmp = tempfile::tempdir().unwrap();
    let uri = tmp.path().to_string_lossy().to_string();
    let store = Arc::new(RwLock::new(
        RolloutStore::open_with_options(&uri, opts("solo"))
            .await
            .unwrap(),
    ));

    for i in 0..12 {
        store
            .read()
            .await
            .add(&[rec(&format!("row-{i}"))])
            .await
            .unwrap();
        store.read().await.flush().await.unwrap();
    }

    // Cancel the merge mid-flight by dropping the future at a very short
    // timeout. Whether it lands before or after the drain is timing-dependent —
    // both outcomes must preserve every row.
    let _ = tokio::time::timeout(Duration::from_millis(5), merge_like_sweeper(&store)).await;

    let after_interrupt = read_ids(&store).await;
    let mut unique = after_interrupt.clone();
    unique.dedup();
    assert_eq!(
        unique.len(),
        12,
        "no row may be lost when a merge is interrupted: {after_interrupt:?}"
    );

    // A subsequent merge must still succeed and converge.
    merge_like_sweeper(&store).await;
    let after_retry = read_ids(&store).await;
    let mut unique_retry = after_retry.clone();
    unique_retry.dedup();
    assert_eq!(
        unique_retry.len(),
        12,
        "retry after an interrupted merge must converge to the same rows"
    );
    assert_eq!(
        after_retry, unique_retry,
        "read path must de-duplicate rows an interrupted merge left in the base table"
    );
}

/// The regression this whole change exists for: an append must not wait for a
/// merge to finish.
///
/// Asserts on *append latency during a merge*, not on the merge itself. The old
/// code held the store's write lock for the merge's full duration, so a
/// concurrent append blocked for exactly that long (~17s in production).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn append_is_not_blocked_for_the_duration_of_a_merge() {
    let tmp = tempfile::tempdir().unwrap();
    let uri = tmp.path().to_string_lossy().to_string();
    let store = Arc::new(RwLock::new(
        RolloutStore::open_with_options(&uri, opts("solo"))
            .await
            .unwrap(),
    ));

    // Enough generations that the merge takes materially longer than an append.
    for i in 0..25 {
        store
            .read()
            .await
            .add(&[rec(&format!("bulk-{i}"))])
            .await
            .unwrap();
        store.read().await.flush().await.unwrap();
    }

    let merge_start = Instant::now();
    let merger = {
        let store = store.clone();
        tokio::spawn(async move { merge_like_sweeper(&store).await })
    };

    // Give the merge a moment to get into its expensive read phase.
    tokio::time::sleep(Duration::from_millis(10)).await;

    let append_start = Instant::now();
    store.read().await.add(&[rec("racer")]).await.unwrap();
    let append_elapsed = append_start.elapsed();

    let reclaimed = merger.await.unwrap();
    let merge_elapsed = merge_start.elapsed();

    assert!(reclaimed > 0, "merge should have done real work");
    // The append must not have been serialized behind the merge. Compare
    // against the merge's own duration rather than a fixed threshold, so the
    // assertion holds on slow CI without becoming vacuous.
    assert!(
        append_elapsed * 2 < merge_elapsed || append_elapsed < Duration::from_millis(200),
        "append took {append_elapsed:?} while the merge took {merge_elapsed:?}; \
         the append appears to have been blocked by the merge"
    );

    store.read().await.flush().await.unwrap();
    let ids = read_ids(&store).await;
    assert!(
        ids.contains(&"racer".to_string()),
        "the row appended during the merge must be readable"
    );
    assert_eq!(ids.len(), 26, "all rows readable exactly once");
}
