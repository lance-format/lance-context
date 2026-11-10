use super::*;
use std::path::{Path, PathBuf};

fn files(root: &Path) -> Result<BTreeSet<PathBuf>> {
    let mut files = BTreeSet::new();
    for directory in std::fs::read_dir(root.join("_indices"))? {
        let directory = directory?.path();
        if !directory.is_dir() {
            continue;
        }
        for name in ["page_lookup.lance", "page_data.lance"] {
            let file = directory.join(name);
            if file.exists() {
                files.insert(file);
            }
        }
    }
    Ok(files)
}

async fn referenced(base: &Dataset, root: &Path) -> Result<BTreeSet<PathBuf>> {
    Ok(base
        .load_indices()
        .await?
        .iter()
        .flat_map(|i| {
            ["page_lookup.lance", "page_data.lance"]
                .map(|name| root.join("_indices").join(i.uuid.to_string()).join(name))
        })
        .collect())
}

fn collecting(runtime: service::runtime::Runtime) -> Result<service::runtime::Runtime> {
    runtime.with_exclusive_reclamation(service::runtime::reclamation::Options {
        max_objects: 2,
        max_bytes: 16 << 10,
        delete_parallelism: 2,
        interval: Duration::from_millis(10),
    })
}

async fn wait_files(root: &Path, expected: &BTreeSet<PathBuf>) -> Result<()> {
    tokio::time::timeout(Duration::from_secs(30), async {
        while &files(root)? != expected {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        Ok::<_, anyhow::Error>(())
    })
    .await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mixed_id_index_gc_preserves_a_retired_reader_then_collects_old_segments() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(90), retained_reader()).await?
}

async fn retained_reader() -> Result<()> {
    let root = artifact_root()?;
    let path = root.path().join("annotations.lance");
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = storage(root.path(), "annotations.lance").await?;
    let body = storage(root.path(), "body.lance").await?;
    seed_with_binding(&c, &narrow, &body, compaction::compact_binding()).await?;
    let (a, runtime) = indexed_owner_with_history(&c, &narrow, &body, true, 4).await?;
    let (stop, task) = annotation_runtime::run_owner(collecting(runtime)?);
    let creator = producer(&a, 1).await?;
    let patcher = producer(&a, 2).await?;
    let mut held = None;
    for row in 0..8 {
        let input = request(
            &a,
            &creator,
            row + 1,
            encoded(create(&[&format!("turn-{row}")], &["s"], &[row as u8])?)?,
        )
        .await?;
        let result = a.submit(&input, Duration::from_secs(10)).await?;
        annotation_runtime::drained(&c, &a, result.index).await?;
        let count = row as usize + 1;
        let base = fully_indexed(&c, &narrow, count, count / 4 + count % 4).await?;
        if row == 0 {
            let reader = a.capture_read_for_test("turn-0").await?;
            assert_eq!(reader.base_version, base.version_id());
            let old = referenced(&base, &path).await?;
            assert_eq!(old.len(), 2);
            assert!(old.iter().all(|p| p.exists()));
            held = Some((reader, old));
        }
    }
    let mut last = None;
    // Advance durable history past the last merge. Native indexes survive
    // narrow column updates, while obsolete segment roots can now retire.
    for sequence in 1..=4 {
        let input = request(
            &a,
            &patcher,
            sequence,
            encoded(mixed::wire::patches(
                &patch(&["turn-0"], Some("reviewed"), Some(sequence as f32))?,
                &limits(),
            )?)?,
        )
        .await?;
        let result = a.submit(&input, Duration::from_secs(10)).await?;
        annotation_runtime::drained(&c, &a, result.index).await?;
        last = Some((input, result));
    }
    let (held, old) = held.unwrap();
    let base = fully_indexed(&c, &narrow, 8, 2).await?;
    let live = referenced(&base, &path).await?;
    assert_eq!(live.len(), 4);
    assert!(live.is_disjoint(&old));
    let expected = live.union(&old).cloned().collect();
    wait_files(&path, &expected).await?;
    assert!(!annotation_publication::state(&c)
        .await?
        .manifests
        .unwrap()
        .versions
        .contains_key(&held.base_version));
    let original = held.resolve().await?.unwrap();
    assert_eq!(original.status.as_deref(), Some("pending"));
    assert_eq!(original.score, None);
    assert!(old.iter().all(|p| p.exists()));
    check(&a, "turn-0", "s", 0, "reviewed", Some(4.0)).await?;
    drop(held);
    wait_files(&path, &live).await?;
    assert!(old.iter().all(|p| !p.exists()));
    for row in 1..8 {
        check(&a, &format!("turn-{row}"), "s", row, "pending", None).await?;
    }
    stop.send(()).unwrap();
    task.await??;
    drop(a);
    c.shutdown().await?;

    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let (a, runtime) = indexed_owner_with_history(&c, &narrow, &body, true, 4).await?;
    let (stop, task) = annotation_runtime::run_owner(collecting(runtime)?);
    check(&a, "turn-0", "s", 0, "reviewed", Some(4.0)).await?;
    for row in 1..8 {
        check(&a, &format!("turn-{row}"), "s", row, "pending", None).await?;
    }
    let (input, result) = last.unwrap();
    assert_eq!(a.submit(&input, Duration::from_secs(10)).await?, result);
    wait_files(&path, &live).await?;
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mixed_id_index_gc_collects_abandoned_native_preparation() -> Result<()> {
    let root = artifact_root()?;
    let path = root.path().join("annotations.lance");
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = storage(root.path(), "annotations.lance").await?;
    let body = storage(root.path(), "body.lance").await?;
    let (last, last_result) = compaction::fragmented_with_slot(&c, &narrow, &body, 1).await?;
    let mut base = (*annotation_publication::read(&c, &narrow, None).await?).clone();
    let abandoned = base
        .create_index_builder(
            &["id"],
            lance_index::IndexType::BTree,
            &lance_index::scalar::ScalarIndexParams::default(),
        )
        .name("abandoned".into())
        .execute_uncommitted()
        .await?;
    let old = files(&path)?;
    assert_eq!(old.len(), 2);
    assert!(old
        .iter()
        .all(|p| p.parent().unwrap().ends_with(abandoned.uuid.to_string())));
    // This case inspects retained metadata to distinguish published pages from
    // the abandoned index. Keep this finite fixture inside its history window:
    // enumerating catalog versions does not register readers that pin them.
    // The test above separately exercises retirement with a four-version window.
    let history = 64;
    let (a, runtime) = indexed_owner_with_history(&c, &narrow, &body, true, history).await?;
    let (stop, task) = annotation_runtime::run_owner(collecting(runtime)?);
    let selected = fully_indexed(&c, &narrow, 3, 1).await?;
    let live = referenced(&selected, &path).await?;
    // Address-index compaction creates new native pages. The pre-compaction
    // pages remain live while their selected versions are in durable history;
    // they are distinct from the never-published index created above.
    let manifests = annotation_publication::state(&c).await?.manifests.unwrap();
    assert!(manifests.head - manifests.initial_version < history as u64);
    assert_eq!(
        manifests.history_boundary().retained_from,
        manifests.initial_version + 1
    );
    let mut protected = live.clone();
    for version in manifests
        .versions
        .keys()
        .filter(|version| **version != manifests.initial_version)
    {
        let retained = annotation_publication::read(&c, &narrow, Some(*version)).await?;
        protected.extend(referenced(&retained, &path).await?);
        assert_eq!(retained.count_rows(Some("id = 'A'".into())).await?, 1);
    }
    assert!(old.is_disjoint(&protected));
    wait_files(&path, &protected).await?;
    assert!(old.iter().all(|file| !file.exists()));
    for (id, value) in [("A", 1), ("B", 2), ("C", 3)] {
        check(&a, id, "s", value, "pending", None).await?;
    }
    assert_eq!(a.submit(&last, Duration::from_secs(10)).await?, last_result);
    let after = annotation_publication::state(&c).await?.manifests.unwrap();
    assert!(after.head - after.initial_version < history as u64);
    assert_eq!(after.history_boundary(), manifests.history_boundary());
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
