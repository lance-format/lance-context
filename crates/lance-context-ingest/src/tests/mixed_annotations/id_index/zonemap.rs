//! Turn-ID ZoneMap segments: build, merge, compaction withdrawal, rebuild and
//! restart, through the actual RF3 owner, publisher and native Lance files.
use super::*;

const ZONEMAP: crate::materializer::TurnIdIndexType = crate::materializer::TurnIdIndexType::Zonemap;
const NAME: &str = "lance_context_turn_id";

fn is_zonemap(index: &lance_table::format::IndexMetadata) -> bool {
    index
        .index_details
        .as_ref()
        .is_some_and(|d| d.type_url.ends_with("lance.table.ZoneMapIndexDetails"))
}

/// Segments, coverage and fragments of the currently selected narrow version.
struct Coverage {
    segments: usize,
    covered: BTreeSet<u32>,
    fragments: BTreeSet<u32>,
}

async fn coverage(c: &network::Cluster, narrow: &Storage) -> Result<(Arc<Dataset>, Coverage)> {
    let selected = annotation_publication::read(c, narrow, None).await?;
    let indices = selected.load_indices().await?;
    let turn: Vec<_> = indices.iter().filter(|i| i.name == NAME).collect();
    assert!(
        turn.iter().all(|i| is_zonemap(i)),
        "a ZoneMap table must hold only ZoneMap segments"
    );
    let covered = turn
        .iter()
        .flat_map(|i| i.fragment_bitmap.as_ref().unwrap().iter())
        .collect();
    let fragments = selected.fragments().iter().map(|f| f.id as u32).collect();
    Ok((
        selected,
        Coverage {
            segments: turn.len(),
            covered,
            fragments,
        },
    ))
}

async fn settled(c: &network::Cluster) -> Result<bool> {
    Ok(annotation_publication::state(c)
        .await?
        .progress
        .pending
        .is_none())
}

/// Wait until every live fragment is covered by a ZoneMap segment.
async fn fully_covered(c: &network::Cluster, narrow: &Storage, rows: usize) -> Result<Coverage> {
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let (selected, now) = coverage(c, narrow).await?;
            if selected.count_rows(None).await? == rows
                && now.segments > 0
                && now.covered == now.fragments
                && settled(c).await?
            {
                return Ok::<_, anyhow::Error>(now);
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn zonemap_id_index_build_merge_compaction_rebuild_and_restart() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(120), zonemap_load()).await?
}

async fn zonemap_load() -> Result<()> {
    let root = artifact_root()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = storage(root.path(), "annotations.lance").await?;
    let body = storage(root.path(), "body.lance").await?;
    seed_with_binding(&c, &narrow, &body, compaction::compact_binding()).await?;
    let (a, runtime) = typed_owner(&c, &narrow, &body, true, 64, ZONEMAP)
        .await
        .context("open ZoneMap-indexed owner")?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    let creator = producer(&a, 1).await?;
    let patcher = producer(&a, 2).await?;
    for row in 0..8 {
        let id = format!("turn-{row}");
        let input = request(
            &a,
            &creator,
            row + 1,
            encoded(create(&[&id], &["s"], &[row as u8])?)?,
        )
        .await?;
        a.submit(&input, Duration::from_secs(10)).await?;
        let patch = request(
            &a,
            &patcher,
            row + 1,
            encoded(mixed::wire::patches(
                &patch(&["turn-0"], Some("reviewed"), Some(row as f32 / 10.0))?,
                &limits(),
            )?)?,
        )
        .await?;
        let result = a.submit(&patch, Duration::from_secs(10)).await?;
        annotation_runtime::drained(&c, &a, result.index).await?;
        // Exact reads stay correct whether or not this fragment is covered yet.
        check(
            &a,
            &id,
            "s",
            row as u8,
            if row == 0 { "reviewed" } else { "pending" },
            (row == 0).then_some(0.0),
        )
        .await?;
    }
    let indexed = fully_covered(&c, &narrow, 8).await?;

    // Read the retained native transaction history: ZoneMap segments were
    // built, consolidated through the native merge, and compaction rewrote
    // covered fragments without remapping (withdrawing) their ZoneMap coverage.
    let state = annotation_publication::state(&c).await?;
    let manifests = state.manifests.as_ref().unwrap();
    let (mut builds, mut merges, mut withdrawing_rewrites) = (0, 0, 0);
    for version in manifests
        .versions
        .keys()
        .filter(|v| **v != manifests.initial_version)
    {
        let selected = annotation_publication::read(&c, &narrow, Some(*version)).await?;
        let Some(transaction) = selected.read_transaction().await? else {
            continue;
        };
        match transaction.operation {
            Operation::CreateIndex {
                new_indices,
                removed_indices,
            } if new_indices.iter().any(|i| i.name == NAME) => {
                assert!(new_indices.iter().all(is_zonemap));
                assert!(removed_indices.iter().all(is_zonemap));
                if removed_indices
                    .iter()
                    .any(|i| !i.fragment_bitmap.as_ref().unwrap().is_empty())
                {
                    merges += 1;
                } else {
                    builds += 1;
                }
            }
            Operation::Rewrite {
                rewritten_indices, ..
            } => {
                // The managed turn-ID ZoneMap is never in the remap set.
                let previous = annotation_publication::read(&c, &narrow, Some(version - 1)).await;
                if let Ok(previous) = previous {
                    let turn: Vec<_> = previous
                        .load_indices()
                        .await?
                        .iter()
                        .filter(|i| i.name == NAME)
                        .map(|i| i.uuid)
                        .collect();
                    assert!(rewritten_indices.iter().all(|r| !turn.contains(&r.old_id)));
                    if !turn.is_empty() {
                        withdrawing_rewrites += 1;
                    }
                }
            }
            _ => {}
        }
    }
    assert!(builds > 0, "no ZoneMap segment was built");
    eprintln!(
        "zonemap evidence: builds={builds} merges={merges} withdrawing_rewrites={withdrawing_rewrites} \
         final_segments={} final_fragments={}",
        indexed.segments,
        indexed.fragments.len()
    );

    // A repeated first-create keeps the first body and current annotations.
    let again = request(
        &a,
        &creator,
        9,
        encoded(create(&["turn-0"], &["other"], &[99])?)?,
    )
    .await?;
    let repeated = a.submit(&again, Duration::from_secs(10)).await?;
    annotation_runtime::drained(&c, &a, repeated.index).await?;
    check(&a, "turn-0", "s", 0, "reviewed", Some(0.7)).await?;
    for row in 1..8 {
        check(&a, &format!("turn-{row}"), "s", row, "pending", None).await?;
    }
    stop.send(()).unwrap();
    task.await??;
    drop(a);
    c.shutdown().await?;

    // Restart: the same selected version still answers exactly, and coverage
    // remains complete without retraining already covered fragments.
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let (a, runtime) = typed_owner(&c, &narrow, &body, true, 64, ZONEMAP)
        .await
        .context("reopen ZoneMap-indexed owner")?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    check(&a, "turn-0", "s", 0, "reviewed", Some(0.7)).await?;
    for row in 1..8 {
        check(&a, &format!("turn-{row}"), "s", row, "pending", None).await?;
    }
    assert_eq!(a.submit(&again, Duration::from_secs(10)).await?, repeated);
    fully_covered(&c, &narrow, 8).await?;
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn zonemap_and_btree_turn_segments_never_mix() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(90), async {
        let root = artifact_root()?;
        let c =
            network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
        let narrow = storage(root.path(), "annotations.lance").await?;
        let body = storage(root.path(), "body.lance").await?;
        seed_with_binding(&c, &narrow, &body, compaction::compact_binding()).await?;
        let (a, runtime) = typed_owner(&c, &narrow, &body, false, 64, ZONEMAP).await?;
        let (stop, task) = annotation_runtime::run_owner(runtime);
        let creator = producer(&a, 1).await?;
        let input = request(&a, &creator, 1, encoded(create(&["only"], &["s"], &[1])?)?).await?;
        let result = a.submit(&input, Duration::from_secs(10)).await?;
        annotation_runtime::drained(&c, &a, result.index).await?;
        fully_covered(&c, &narrow, 1).await?;
        stop.send(()).unwrap();
        task.await??;
        drop(a);
        // Reopening the same table with the BTree setting must fail closed in
        // maintenance rather than add a BTree segment beside ZoneMap ones.
        let (a, runtime) = typed_owner(
            &c,
            &narrow,
            &body,
            false,
            64,
            crate::materializer::TurnIdIndexType::Btree,
        )
        .await?;
        let (_stop, task) = annotation_runtime::run_owner(runtime);
        let input = request(&a, &creator, 2, encoded(create(&["next"], &["s"], &[2])?)?).await?;
        // The submit may fail too once the runtime stops; the runtime's own
        // error is the one that must name the type mismatch.
        let _ = a.submit(&input, Duration::from_secs(10)).await;
        let error = tokio::time::timeout(Duration::from_secs(30), task)
            .await??
            .unwrap_err();
        assert!(
            format!("{error:#}").contains("turn ID index type differs"),
            "{error:#}"
        );
        let (_, now) = coverage(&c, &narrow).await?;
        assert!(now.segments >= 1);
        drop(a);
        c.shutdown().await
    })
    .await?
}

/// Count rows for one ID with an exact native filter at a fixed version. This
/// is the existing lookup predicate; ZoneMap zones only prune candidates.
async fn exact_rows(dataset: &Dataset, id: &str) -> Result<usize> {
    Ok(dataset.count_rows(Some(format!("id = '{id}'"))).await?)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn zonemap_segments_merge_natively_without_compaction() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(90), async {
        let root = artifact_root()?;
        let c =
            network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
        let narrow = storage(root.path(), "annotations.lance").await?;
        let body = storage(root.path(), "body.lance").await?;
        seed_with_binding(&c, &narrow, &body, compaction::compact_binding()).await?;
        let (a, runtime) = typed_owner(&c, &narrow, &body, false, 64, ZONEMAP).await?;
        let (stop, task) = annotation_runtime::run_owner(runtime);
        let creator = producer(&a, 1).await?;
        for row in 0..8 {
            let id = format!("turn-{row}");
            let input = request(
                &a,
                &creator,
                row + 1,
                encoded(create(&[&id], &["s"], &[row as u8])?)?,
            )
            .await?;
            let result = a.submit(&input, Duration::from_secs(10)).await?;
            annotation_runtime::drained(&c, &a, result.index).await?;
            fully_covered(&c, &narrow, row as usize + 1).await?;
        }
        // Eight one-row segments consolidate 4 -> 1 twice: two segments remain.
        // Coverage completes before the last merge publishes; wait for it.
        tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                let (_, now) = coverage(&c, &narrow).await?;
                if now.segments == 2 && now.covered == now.fragments && settled(&c).await? {
                    return Ok::<_, anyhow::Error>(());
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .context("native ZoneMap merge did not consolidate to two segments")??;
        let state = annotation_publication::state(&c).await?;
        let manifests = state.manifests.as_ref().unwrap();
        let mut merges = 0;
        for version in manifests
            .versions
            .keys()
            .filter(|v| **v != manifests.initial_version)
        {
            let selected = annotation_publication::read(&c, &narrow, Some(*version)).await?;
            if let Some(transaction) = selected.read_transaction().await? {
                if let Operation::CreateIndex {
                    new_indices,
                    removed_indices,
                } = transaction.operation
                {
                    if new_indices.iter().any(|i| i.name == NAME) && removed_indices.len() == 4 {
                        assert!(removed_indices.iter().chain(&new_indices).all(is_zonemap));
                        merges += 1;
                    }
                }
            }
        }
        assert_eq!(merges, 2);
        let current = annotation_publication::read(&c, &narrow, None).await?;
        for row in 0..8 {
            assert_eq!(exact_rows(&current, &format!("turn-{row}")).await?, 1);
        }
        assert_eq!(exact_rows(&current, "absent").await?, 0);
        stop.send(()).unwrap();
        task.await??;
        c.shutdown().await
    })
    .await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn zonemap_withdrawn_coverage_still_answers_exactly() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(120), async {
        let root = artifact_root()?;
        let c =
            network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
        let narrow = storage(root.path(), "annotations.lance").await?;
        let body = storage(root.path(), "body.lance").await?;
        seed_with_binding(&c, &narrow, &body, compaction::compact_binding()).await?;
        let (a, runtime) = typed_owner(&c, &narrow, &body, true, 64, ZONEMAP).await?;
        let (stop, task) = annotation_runtime::run_owner(runtime);
        let creator = producer(&a, 1).await?;
        for row in 0..8 {
            let id = format!("turn-{row}");
            let input = request(
                &a,
                &creator,
                row + 1,
                encoded(create(&[&id], &["s"], &[row as u8])?)?,
            )
            .await?;
            let result = a.submit(&input, Duration::from_secs(10)).await?;
            annotation_runtime::drained(&c, &a, result.index).await?;
        }
        fully_covered(&c, &narrow, 8).await?;
        stop.send(()).unwrap();
        task.await??;
        // Find a retained version committed by a Rewrite while a turn-ID ZoneMap
        // existed, i.e. after compaction and before the coverage was rebuilt.
        let state = annotation_publication::state(&c).await?;
        let manifests = state.manifests.as_ref().unwrap();
        let mut checked = 0;
        for version in manifests
            .versions
            .keys()
            .filter(|v| **v != manifests.initial_version)
        {
            let selected = annotation_publication::read(&c, &narrow, Some(*version)).await?;
            let Some(transaction) = selected.read_transaction().await? else {
                continue;
            };
            if !matches!(transaction.operation, Operation::Rewrite { .. }) {
                continue;
            }
            let turn: Vec<_> = selected
                .load_indices()
                .await?
                .iter()
                .filter(|i| i.name == NAME)
                .cloned()
                .collect();
            if turn.is_empty() {
                continue;
            }
            let covered: BTreeSet<_> = turn
                .iter()
                .flat_map(|i| i.fragment_bitmap.as_ref().unwrap().iter())
                .collect();
            let uncovered: Vec<_> = selected
                .fragments()
                .iter()
                .map(|f| f.id as u32)
                .filter(|id| !covered.contains(id))
                .collect();
            assert!(
                !uncovered.is_empty(),
                "a rewrite must withdraw the rewritten fragments"
            );
            // Every row, covered or withdrawn, is still found exactly once.
            let rows = selected.count_rows(None).await?;
            for row in 0..rows {
                assert_eq!(
                    exact_rows(&selected, &format!("turn-{row}")).await?,
                    1,
                    "version {version}"
                );
            }
            assert_eq!(exact_rows(&selected, "absent").await?, 0);
            checked += 1;
        }
        assert!(checked > 0, "no withdrawn-coverage version was retained");
        drop(a);
        c.shutdown().await
    })
    .await?
}
