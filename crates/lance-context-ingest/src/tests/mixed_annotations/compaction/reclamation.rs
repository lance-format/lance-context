use super::*;
use lance_context_core::annotation_staging::TurnBodySource;
use std::{collections::BTreeSet, path::PathBuf};

mod reference_cache;
mod takeover;

fn files(dataset: &Dataset, root: &std::path::Path) -> BTreeSet<PathBuf> {
    dataset
        .manifest()
        .fragments
        .iter()
        .flat_map(|f| f.referenced_lance_files())
        .map(|file| root.join("data").join(&file.path))
        .collect()
}

fn collecting(runtime: service::runtime::Runtime) -> Result<service::runtime::Runtime> {
    runtime.with_exclusive_reclamation(service::runtime::reclamation::Options {
        max_objects: 2,
        max_bytes: 16 << 10,
        delete_parallelism: 2,
        interval: Duration::from_millis(10),
    })
}

#[tokio::test]
async fn mixed_paired_collection_keeps_captured_bodies_then_releases_both_tables() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), exercise()).await?
}

async fn exercise() -> Result<()> {
    let root = artifact_root()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = storage(root.path(), "annotations.lance").await?;
    let body = storage(root.path(), "body.lance").await?;
    let (last, last_result) = fragmented(&c, &narrow, &body).await?;
    let original_narrow = annotation_publication::read(&c, &narrow, None).await?;
    let original_body = selected_body(&c, &narrow, &body).await?;
    let original_ids = ids(&original_body).await?;
    let narrow_files = files(&original_narrow, &root.path().join("annotations.lance"));
    let body_files = files(&original_body, &root.path().join("body.lance"));
    assert_eq!(body_files.len(), 3);
    let (a, runtime) = compact_owner(&c, &narrow, &body).await?;
    // Hold the real compatible read view before the owner can compact. Leave
    // its body unresolved so cached body arrays cannot hide premature deletion.
    let held = a.capture_turn_for_test("A").await?;
    let (stop, task) = annotation_runtime::run_owner(collecting(runtime)?);
    let p = producer(&a, 2).await?;
    let mut latest = None;
    for sequence in 1..=6 {
        let update = request(
            &a,
            &p,
            sequence,
            encoded(mixed::wire::patches(
                &patch(&["A"], Some("reviewed"), Some(sequence as f32))?,
                &limits(),
            )?)?,
        )
        .await?;
        let result = a.submit(&update, Duration::from_secs(10)).await?;
        annotation_runtime::drained(&c, &a, result.index).await?;
        latest = Some((update, result));
    }
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let state = annotation_publication::state(&c).await?;
            let manifests = state.manifests.as_ref().unwrap();
            if manifests.history_boundary().retained_from > original_narrow.version_id()
                && manifests.versions.len() <= 4
                && state.progress.pending.is_none()
                && annotation_publication::read(&c, &narrow, None)
                    .await?
                    .fragments()
                    .len()
                    == 1
                && selected_body(&c, &narrow, &body).await?.fragments().len() == 1
                && a.reclamation_observation().deleted > 0
            {
                for version in manifests
                    .versions
                    .keys()
                    .filter(|v| **v != manifests.initial_version)
                {
                    // All writes and both rewrites are drained and the history
                    // has reached its configured bound. Earlier enumeration
                    // raced legitimate retirement between these raw reads.
                    let narrow = annotation_publication::read(&c, &narrow, Some(*version))
                        .await
                        .with_context(|| format!("reading retained test version {version}"))?;
                    let reference = BodySnapshot::read(&narrow).await?.unwrap();
                    let body =
                        mixed::load_body(&body, &reference, &binding().incarnation.to_string())
                            .await?;
                    assert!(body_files.is_disjoint(&files(&body, &root.path().join("body.lance"))));
                }
                return Ok::<_, anyhow::Error>(());
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await??;
    let passes = a.reclamation_observation().passes;
    tokio::time::timeout(Duration::from_secs(30), async {
        while a.reclamation_observation().passes < passes + 40 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;
    assert!(body_files.iter().all(|p| p.exists()));
    assert!(narrow_files.iter().all(|p| p.exists()));
    let TurnBodySource::Materialized(old) = held.body_source() else {
        panic!("expected captured materialized body")
    };
    let cold_body = storage(root.path(), "body.lance").await?;
    let reference = BodySnapshot::read(old).await?.unwrap();
    let captured =
        mixed::load_body(&cold_body, &reference, &binding().incarnation.to_string()).await?;
    assert_eq!(ids(&captured).await?, original_ids);
    let mut scan = captured.scan();
    scan.project(&["binary_payload"])?;
    let rows = scan
        .try_into_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    assert_eq!(rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
    assert_eq!(
        held.annotations()
            .resolve()
            .await?
            .unwrap()
            .status
            .as_deref(),
        Some("pending")
    );
    drop((captured, held, original_body, original_narrow));
    tokio::time::timeout(Duration::from_secs(30), async {
        while body_files.iter().chain(&narrow_files).any(|p| p.exists()) {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;
    let (update, result) = latest.unwrap();
    assert_eq!(a.retained_body_frames(), (0, 0));
    assert_eq!(a.submit(&update, Duration::from_secs(10)).await?, result);
    assert_eq!(a.submit(&last, Duration::from_secs(10)).await?, last_result);
    stop.send(()).unwrap();
    task.await??;
    drop(a);
    c.shutdown().await?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = storage(root.path(), "annotations.lance").await?;
    let body = storage(root.path(), "body.lance").await?;
    let (a, runtime) = compact_owner(&c, &narrow, &body).await?;
    let (stop, task) = annotation_runtime::run_owner(collecting(runtime)?);
    check(&a, "A", "s", 1, "reviewed", Some(6.0)).await?;
    check(&a, "B", "s", 2, "pending", None).await?;
    check(&a, "C", "s", 3, "pending", None).await?;
    assert_eq!(
        ids(&selected_body(&c, &narrow, &body).await?).await?,
        original_ids
    );
    assert_eq!(a.submit(&update, Duration::from_secs(10)).await?, result);
    assert_eq!(a.submit(&last, Duration::from_secs(10)).await?, last_result);
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
