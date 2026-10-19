//! Process-exit tests for the legacy delete/append/drain commit boundaries.
//! These hooks are compiled only into the core unit-test executable.

use super::{
    tests::assistant_record, ListSource, RolloutFilters, RolloutStore, RolloutStoreOptions,
};
use std::{collections::BTreeMap, io::Write, path::PathBuf, time::Duration};

const ROOT_ENV: &str = "RK_MERGE_CRASH_TEST_ROOT";
const PHASE_ENV: &str = "RK_MERGE_CRASH_TEST_PHASE";
const OWNER: &str = "merge-crash-test-owner";

pub(crate) fn checkpoint(uri: &str, phase: &str) {
    let Ok(root) = std::env::var(ROOT_ENV) else {
        return;
    };
    let root = PathBuf::from(root);
    if std::env::var(PHASE_ENV).as_deref() != Ok(phase)
        || uri != root.join("table").to_string_lossy()
        || !root.join("armed").is_file()
    {
        return;
    }
    // Persist evidence of the exact reached boundary, then terminate without
    // dropping the store/runtime. The parent must join this process first.
    let mut receipt = std::fs::File::create(root.join("crash-phase")).unwrap();
    receipt.write_all(phase.as_bytes()).unwrap();
    receipt.sync_all().unwrap();
    std::process::exit(86);
}

fn options() -> RolloutStoreOptions {
    RolloutStoreOptions {
        shard_id: Some(OWNER.into()),
        merge_after_generations: Some(0),
        ..Default::default()
    }
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap()
}

#[test]
#[ignore = "owned subprocess for the merge crash-boundary tests"]
fn legacy_merge_crash_child() {
    let root = PathBuf::from(std::env::var(ROOT_ENV).unwrap());
    assert!(root.is_dir());
    let phase = std::env::var(PHASE_ENV).unwrap();
    assert!(matches!(phase.as_str(), "after_delete" | "after_append"));
    runtime().block_on(async {
        let uri = root.join("table").to_string_lossy().into_owned();
        let mut store = RolloutStore::open_with_options(&uri, options())
            .await
            .unwrap();
        let initial: Vec<_> = (0..20)
            .map(|i| assistant_record(&format!("row-{i:02}")))
            .collect();
        store.add(&initial).await.unwrap();
        store.flush().await.unwrap();
        assert!(store.cleanup_own_shard().await.unwrap() > 0);
        assert!(!crate::rollout_append::has_cutover(
            &store.base.dataset,
            store.base.write_shard
        ));

        let changed: Vec<_> = (0..10)
            .chain(20..25)
            .map(|i| {
                let mut record = assistant_record(&format!("row-{i:02}"));
                record.content = Some(format!("new-content-{i}"));
                record
            })
            .collect();
        store.add(&changed).await.unwrap();
        store.flush().await.unwrap();
        // Do not arm during the initial merge. Only the second merge, which
        // replaces existing keys as well as adding new ones, may crash.
        let mut receipt = std::fs::File::create(root.join("armed")).unwrap();
        receipt
            .write_all(b"15 replacement/new rows acknowledged and flushed")
            .unwrap();
        receipt.sync_all().unwrap();
        store.cleanup_own_shard().await.unwrap();
        panic!("the selected merge crash boundary was not reached");
    });
}

fn recover_after_crash(phase: &str, interrupted_base_count: usize) {
    let dir = tempfile::tempdir().unwrap();
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "rollout_store::crash_tests::legacy_merge_crash_child",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(ROOT_ENV, dir.path())
        .env(PHASE_ENV, phase)
        .spawn()
        .unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(90);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if std::time::Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("owned crash-test child exceeded its deadline");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    assert_eq!(status.code(), Some(86));
    assert_eq!(
        std::fs::read_to_string(dir.path().join("crash-phase")).unwrap(),
        phase
    );
    runtime().block_on(async {
        let uri = dir.path().join("table").to_string_lossy().into_owned();
        let mut store = RolloutStore::open_existing_with_options(&uri, options())
            .await
            .unwrap();
        // Inspect fragments explicitly: an LSM read could otherwise hide a
        // missing append or duplicate physical base rows behind WAL contents.
        let base = store
            .list_filtered_source(&RolloutFilters::default(), 100, 0, ListSource::Fragments)
            .await
            .unwrap();
        assert_eq!(base.records.len(), interrupted_base_count);
        assert!(
            store
                .own_shard_watermark()
                .await
                .unwrap()
                .unwrap()
                .pending_generations
                > 0
        );
        let expected: BTreeMap<_, _> = (0..25)
            .map(|i| {
                let id = format!("row-{i:02}");
                let content = if i < 10 || i >= 20 {
                    Some(format!("new-content-{i}"))
                } else {
                    assistant_record(&id).content
                };
                (id, content)
            })
            .collect();
        // No append after the joined crash. Recover, merge, close, and reopen
        // twice so neither an in-memory writer nor an LSM overlay can mask loss.
        for attempt in 0..2 {
            store.replay_and_flush_own_wal().await.unwrap();
            let reclaimed = store.cleanup_own_shard().await.unwrap();
            if attempt == 0 {
                assert!(reclaimed > 0);
            } else {
                assert_eq!(reclaimed, 0);
            }
            for source in [ListSource::All, ListSource::Fragments] {
                let rows = store
                    .list_filtered_source(&RolloutFilters::default(), 100, 0, source)
                    .await
                    .unwrap()
                    .records;
                assert_eq!(
                    rows.len(),
                    25,
                    "physical duplicate or missing row at {phase}"
                );
                let actual: BTreeMap<_, _> = rows.into_iter().map(|r| (r.id, r.content)).collect();
                assert_eq!(actual, expected);
            }
            assert_eq!(
                store
                    .own_shard_watermark()
                    .await
                    .unwrap()
                    .unwrap()
                    .pending_generations,
                0
            );
            store.close().await.unwrap();
            store = RolloutStore::open_existing_with_options(&uri, options())
                .await
                .unwrap();
        }
        store.close().await.unwrap();
    });
}

#[test]
fn legacy_merge_recovers_after_delete_commit_crash() {
    recover_after_crash("after_delete", 10);
}

#[test]
fn legacy_merge_recovers_after_append_before_drain_crash() {
    recover_after_crash("after_append", 25);
}
