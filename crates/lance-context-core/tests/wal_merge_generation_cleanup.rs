//! Regression test: WAL merge must delete the merged generation's blob dir.
//!
//! One store, one shard, fully serial — no concurrency, no fence. This isolates
//! a single property: `merge_own_shard` appends merged generations into the base
//! table and drains them from the shard manifest, and must ALSO delete the
//! `_mem_wal/{shard}/{gen}/` blob directory. Historically it did not, leaking one
//! zombie directory per merged generation (a pure storage leak: data was correct
//! but `_mem_wal/` grew without bound). After the fix, the count of on-disk
//! `_gen_` directories tracks the manifest's pending generations exactly.

use std::path::Path;

use lance_context_core::{RolloutRecord, RolloutStore, RolloutStoreOptions, ROLE_ASSISTANT};

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

fn count_gen_dirs_on_disk(dataset_dir: &Path) -> usize {
    let mem_wal = dataset_dir.join("_mem_wal");
    let mut count = 0;
    if let Ok(shards) = std::fs::read_dir(&mem_wal) {
        for shard in shards.flatten() {
            if let Ok(entries) = std::fs::read_dir(shard.path()) {
                for e in entries.flatten() {
                    let name = e.file_name().to_string_lossy().to_string();
                    if name.contains("_gen_") && e.path().is_dir() {
                        count += 1;
                    }
                }
            }
        }
    }
    count
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn serial_merge_deletes_merged_generation_dirs() {
    let tmp = tempfile::tempdir().unwrap();
    let uri = tmp.path().to_string_lossy().to_string();

    // Merge every 10 accumulated generations into the base table.
    let opts = RolloutStoreOptions {
        shard_id: Some("solo".to_string()),
        merge_after_generations: Some(10),
        ..Default::default()
    };

    let mut store = RolloutStore::open_with_options(&uri, opts.clone())
        .await
        .unwrap();

    // 30 serial single-row appends. Each append flushes one generation; every
    // 10th triggers a count-merge that folds the accumulated generations into
    // the base table and drains them from the manifest.
    let n = 30;
    for i in 0..n {
        store.add(&[rec(&format!("row-{i}"))]).await.unwrap();
    }

    let obs = store.observe().await.unwrap();
    let manifest_pending = obs.pending_wal_generations as usize;
    let row_count = obs.row_count;

    let on_disk = count_gen_dirs_on_disk(Path::new(&uri));

    eprintln!(
        "appends={n} row_count={row_count} \
         manifest_pending_generations={manifest_pending} \
         gen_dirs_on_disk={on_disk} leaked={}",
        on_disk.saturating_sub(manifest_pending)
    );

    // Data correctness: all rows present exactly once.
    assert_eq!(
        row_count as usize, n,
        "all rows must be readable exactly once"
    );

    // The fix: merged generations are drained from the manifest AND their blob
    // dirs are deleted, so on-disk gen dirs never exceed the manifest's pending
    // count. (Before the fix this was 30 dirs on disk vs 0 pending = 30 leaks.)
    assert_eq!(
        on_disk, manifest_pending,
        "on-disk generation dirs ({on_disk}) must match manifest pending \
         generations ({manifest_pending}); a surplus means merged generations \
         leaked their blob directories"
    );
}
