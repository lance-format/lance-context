use super::*;
use futures::TryStreamExt;
use lance::index::DatasetIndexExt;
use std::collections::BTreeSet;

const NAME: &str = "lance_context_annotation_status";
mod reclamation;
#[cfg(feature = "service")]
mod recovery;

async fn indexed_owner(
    c: &network::Cluster,
    narrow: &Storage,
    body: &Storage,
    history: usize,
) -> Result<(Service, service::runtime::Runtime)> {
    owner_with_indexes(c, narrow, body, history, true, true, false).await
}

async fn owner_with_indexes(
    c: &network::Cluster,
    narrow: &Storage,
    body: &Storage,
    history: usize,
    turn_ids: bool,
    status: bool,
    compact: bool,
) -> Result<(Service, service::runtime::Runtime)> {
    let state = annotation_publication::state(c).await?;
    service::runtime::Runtime::connect_mixed(
        c.client.clone(),
        annotation_publication::one_shard(),
        compaction::compact_binding(),
        narrow.clone(),
        body.clone(),
        service::runtime::Configuration {
            limits: limits(),
            upload_budget: MergeMemoryBudget::new(8 << 20),
            apply_budget: MergeMemoryBudget::new(8 << 20),
            publication_budget: MergeMemoryBudget::new(512 << 20),
            publication: Options {
                max_source_bytes: 1 << 20,
                min_source_bytes: 1 << 20,
                max_records: 4,
                max_batch_wait: Duration::from_millis(20),
                retained_manifest_versions: history,
                turn_id_index: turn_ids.then_some(crate::materializer::TurnIdIndex {
                    min_rows: 1,
                    ..Default::default()
                }),
                status_index: status.then(crate::materializer::StatusIndex::default),
                compaction: compact.then_some(crate::materializer::SmallFileCompaction {
                    target_file_bytes: 64 << 10,
                    min_fragments: 2,
                    ..Default::default()
                }),
                ..Default::default()
            },
        },
        Acquisition {
            owner: Uuid::new_v4(),
            request: Uuid::new_v4(),
            expected_epoch: state.authority.epoch,
        },
    )
    .await
}

async fn caught_up(c: &network::Cluster, narrow: &Storage, rows: usize) -> Result<Arc<Dataset>> {
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let base = annotation_publication::read(c, narrow, None).await?;
            let indices = base.load_indices().await?;
            let covered = |name| {
                indices
                    .iter()
                    .filter(|i| i.name == name)
                    .flat_map(|i| i.fragment_bitmap.as_ref().unwrap().iter())
                    .collect::<BTreeSet<_>>()
            };
            let status = covered(NAME);
            let id = covered("lance_context_turn_id");
            if base.count_rows(None).await? == rows
                && !status.is_empty()
                && base
                    .fragments()
                    .iter()
                    .all(|f| status.contains(&(f.id as u32)) && id.contains(&(f.id as u32)))
                && annotation_publication::state(c)
                    .await?
                    .progress
                    .pending
                    .is_none()
            {
                return Ok::<_, anyhow::Error>(base);
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?
}

async fn ids(base: &Dataset, predicate: &str, indexed: bool) -> Result<BTreeSet<String>> {
    let mut scan = base.scan();
    scan.project(&["id"])?;
    scan.filter(predicate)?;
    scan.use_scalar_index(indexed);
    let mut stream = scan.try_into_stream().await?;
    let mut result = BTreeSet::new();
    while let Some(batch) = stream.try_next().await? {
        let values = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for value in values.iter() {
            assert!(
                result.insert(value.unwrap().to_owned()),
                "duplicate indexed turn"
            );
        }
    }
    Ok(result)
}

async fn matches_scan(base: &Dataset, expected: Option<&str>) -> Result<()> {
    for (predicate, contains) in [
        ("status = 'pending'", expected == Some("pending")),
        ("status = 'reviewed'", expected == Some("reviewed")),
        ("status IS NULL", expected.is_none()),
        ("status IS NOT NULL", expected.is_some()),
        ("NOT (status = 'pending')", expected == Some("reviewed")),
    ] {
        let scanned = ids(base, predicate, false).await?;
        assert_eq!(scanned.contains("turn-0"), contains, "{predicate}");
        assert_eq!(ids(base, predicate, true).await?, scanned, "{predicate}");
    }
    let mut scan = base.scan();
    scan.project(&["id"])?;
    scan.filter("status = 'pending'")?;
    assert!(scan.explain_plan(false).await?.contains(NAME));
    Ok(())
}

fn status_patch(status: Option<&str>) -> Result<RecordBatch> {
    Ok(ipc::decode(
        crate::tests::annotations::wire(
            &limits(),
            &["turn-0"],
            if status.is_some() { 1 } else { 2 },
            status,
            0,
            None,
        )?,
        service::wire::schema(&limits()),
    )?
    .remove(0))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mixed_status_index_catches_idle_updates_merges_and_recovers_current_values() -> Result<()>
{
    tokio::time::timeout(Duration::from_secs(120), indexed_load()).await?
}

async fn indexed_load() -> Result<()> {
    let root = artifact_root()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = storage(root.path(), "annotations.lance").await?;
    let body = storage(root.path(), "body.lance").await?;
    seed_with_binding(&c, &narrow, &body, compaction::compact_binding()).await?;
    let (a, runtime) = indexed_owner(&c, &narrow, &body, 64).await?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    let creator = producer(&a, 1).await?;
    let patcher = producer(&a, 2).await?;
    let mut sequence = 0;
    let mut last = None;
    for row in 0..8 {
        let id = format!("turn-{row}");
        let input = request(
            &a,
            &creator,
            row + 1,
            encoded(create(&[&id], &["s"], &[row as u8])?)?,
        )
        .await?;
        let ack = a.submit(&input, Duration::from_secs(10)).await?;
        annotation_runtime::drained(&c, &a, ack.index).await?;
        caught_up(&c, &narrow, row as usize + 1).await?;
        // The first pass repeatedly invalidates the last index definition;
        // later passes withdraw only part of a merged segment's coverage.
        for status in [Some("reviewed"), None, Some("pending")] {
            sequence += 1;
            let input = request(
                &a,
                &patcher,
                sequence,
                encoded(mixed::wire::patches(&status_patch(status)?, &limits())?)?,
            )
            .await?;
            let result = a.submit(&input, Duration::from_secs(10)).await?;
            assert_eq!(
                a.get("turn-0", Duration::from_secs(10))
                    .await?
                    .unwrap()
                    .status
                    .as_deref(),
                status
            );
            annotation_runtime::drained(&c, &a, result.index).await?;
            // No subsequent write is needed to trigger a small catch-up batch.
            let base = caught_up(&c, &narrow, row as usize + 1).await?;
            matches_scan(&base, status).await?;
            last = Some((input, result));
        }
    }
    let base = caught_up(&c, &narrow, 8).await?;
    assert!(
        base.load_indices()
            .await?
            .iter()
            .filter(|i| i.name == NAME)
            .count()
            < 8
    );
    let state = annotation_publication::state(&c).await?;
    let manifests = state.manifests.as_ref().unwrap();
    let mut merges = 0;
    for version in manifests
        .versions
        .keys()
        .filter(|v| **v != manifests.initial_version)
    {
        let selected = annotation_publication::read(&c, &narrow, Some(*version)).await?;
        if let Some(Transaction {
            operation:
                Operation::CreateIndex {
                    new_indices,
                    removed_indices,
                },
            ..
        }) = selected.read_transaction().await?
        {
            if new_indices.iter().any(|i| i.name == NAME) && removed_indices.len() == 4 {
                merges += 1;
            }
        }
    }
    assert!(merges > 0, "status segments never consolidated");
    for row in 0..8 {
        check(&a, &format!("turn-{row}"), "s", row, "pending", None).await?;
    }
    stop.send(()).unwrap();
    task.await??;
    drop(a);
    c.shutdown().await?;

    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let (a, runtime) = indexed_owner(&c, &narrow, &body, 64).await?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    let base = caught_up(&c, &narrow, 8).await?;
    matches_scan(&base, Some("pending")).await?;
    let (input, result) = last.unwrap();
    assert_eq!(a.submit(&input, Duration::from_secs(10)).await?, result);
    for row in 0..8 {
        check(&a, &format!("turn-{row}"), "s", row, "pending", None).await?;
    }
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mixed_status_index_coexists_with_body_and_narrow_compaction() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(90), compacted_load()).await?
}

async fn compacted_load() -> Result<()> {
    let root = artifact_root()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = storage(root.path(), "annotations.lance").await?;
    let body = storage(root.path(), "body.lance").await?;
    seed_with_binding(&c, &narrow, &body, compaction::compact_binding()).await?;
    let (a, runtime) = owner_with_indexes(&c, &narrow, &body, 64, true, true, true).await?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    let creator = producer(&a, 1).await?;
    let patcher = producer(&a, 2).await?;
    for row in 0..8 {
        let input = request(
            &a,
            &creator,
            row + 1,
            encoded(create(&[&format!("turn-{row}")], &["s"], &[row as u8])?)?,
        )
        .await?;
        a.submit(&input, Duration::from_secs(10)).await?;
        let input = request(
            &a,
            &patcher,
            row + 1,
            encoded(mixed::wire::patches(
                &status_patch(Some("reviewed"))?,
                &limits(),
            )?)?,
        )
        .await?;
        let result = a.submit(&input, Duration::from_secs(10)).await?;
        annotation_runtime::drained(&c, &a, result.index).await?;
    }
    let base = tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let base = caught_up(&c, &narrow, 8).await?;
            let reference = BodySnapshot::read(&base).await?.unwrap();
            let selected_body =
                mixed::load_body(&body, &reference, &binding().incarnation.to_string()).await?;
            if base.fragments().len() < 8 && selected_body.fragments().len() < 8 {
                return Ok::<_, anyhow::Error>(base);
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;
    let base = base?;
    matches_scan(&base, Some("reviewed")).await?;
    for row in 0..8 {
        check(
            &a,
            &format!("turn-{row}"),
            "s",
            row,
            if row == 0 { "reviewed" } else { "pending" },
            None,
        )
        .await?;
    }
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
