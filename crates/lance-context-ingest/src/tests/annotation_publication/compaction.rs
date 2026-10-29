//! Native indexed compaction, retained RF3 plans and acknowledged suffixes.
use super::*;
use crate::annotations::runtime::{Configuration, Runtime};
use crate::materializer::{Acquisition, Options};
use crate::tests::annotations::{producer, request, run, wire};
use crate::{annotations::Service, tests::annotation_runtime as runtime};
use fixture::*;
use lance::index::DatasetIndexExt;

mod fixture;
mod reclamation;
use crate::{
    annotations::{Completed, MutationResult},
    materializer,
};
use lance_context_api::annotation::TurnAnnotation;
use std::path::Path;

#[cfg(unix)]
mod process;

#[derive(serde::Serialize, serde::Deserialize)]
struct Expected {
    progress: p::Progress,
    manifest_head: u64,
    bodies: [Vec<u8>; 3],
    outcomes: [Completed; 3],
    current: TurnAnnotation,
}

fn fixture_limits() -> Limits {
    Limits {
        retained_bytes: 256 << 10,
        retained_entries: 128,
        value_bytes: 4096,
        readers: 16,
    }
}

async fn fixture_storage(root: &Path) -> Result<Storage> {
    Storage::resolve(
        root.join("turns.lance").to_str().unwrap().into(),
        Default::default(),
        Arc::default(),
    )
    .await
}

fn options() -> Options {
    Options {
        max_source_bytes: 1 << 20,
        min_source_bytes: 1 << 20,
        max_records: 2,
        max_batch_wait: Duration::from_millis(20),
        retained_manifest_versions: 2,
        compaction: Some(materializer::SmallFileCompaction {
            target_file_bytes: 64 << 10,
            min_fragments: 2,
        }),
        ..Default::default()
    }
}

async fn acquire(c: &network::Cluster) -> Result<Acquisition> {
    Ok(Acquisition {
        owner: Uuid::new_v4(),
        request: Uuid::new_v4(),
        expected_epoch: state(c).await?.authority.epoch,
    })
}

async fn compaction_owner(
    c: &network::Cluster,
    storage: &Storage,
    limits: &Limits,
) -> Result<(Service, Runtime)> {
    Runtime::connect(
        c.client.clone(),
        one_shard(),
        fixture_binding(),
        storage.clone(),
        Configuration {
            limits: limits.clone(),
            upload_budget: MergeMemoryBudget::new(32 << 20),
            apply_budget: MergeMemoryBudget::new(16 << 20),
            publication_budget: MergeMemoryBudget::new(32 << 20),
            publication: options(),
        },
        acquire(c).await?,
    )
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn annotation_compaction_replays_an_unprepared_maintenance_plan() -> Result<()> {
    recover(0).await
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn annotation_compaction_reuses_a_retained_preparation() -> Result<()> {
    recover(1).await
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn annotation_compaction_resolves_a_lost_selection_reply() -> Result<()> {
    recover(2).await
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn annotation_compaction_recovers_after_confirmation_before_retirement() -> Result<()> {
    recover(3).await
}

async fn recover(stage: u8) -> Result<()> {
    tokio::time::timeout(Duration::from_secs(90), recover_inner(stage)).await?
}

async fn recover_inner(stage: u8) -> Result<()> {
    let mut directory = tempfile::Builder::new();
    directory.prefix("annotation-compaction-");
    let root = if let Ok(parent) = std::env::var("INGEST_TEST_ARTIFACTS") {
        directory.disable_cleanup(true).tempdir_in(parent)?
    } else {
        directory.tempdir()?
    };
    recover_at(stage, root.path(), false).await
}

async fn recover_at(stage: u8, root: &Path, crash_boundary: bool) -> Result<()> {
    eprintln!(
        "annotation compaction fixture: {} stage={stage}",
        root.display()
    );
    let storage = fixture_storage(root).await?;
    let limits = fixture_limits();
    let c = network::Cluster::open_signed(root, one_shard()).await?;
    seed_fragments(&c, &storage, &limits, 8).await?;
    let (service, owner) = owner(&c, &storage, &limits).await?;
    let (stop, task) = runtime::run_owner(owner);
    let a = producer(&service, 1).await?;
    let b = producer(&service, 2).await?;
    let first = request(
        &service,
        &a,
        1,
        wire(&limits, &["t00000000"], 1, Some("reviewed"), 0, None)?,
    )
    .await?;
    let first_result = service.submit(&first, Duration::from_secs(30)).await?;
    assert_eq!(first_result.result, MutationResult::Applied);
    // Keep a distinct producer's rejection in the durable request checkpoint.
    let rejected = request(
        &service,
        &b,
        1,
        wire(&limits, &["absent"], 0, None, 1, Some(0.1))?,
    )
    .await?;
    let rejected_result = service.submit(&rejected, Duration::from_secs(30)).await?;
    assert!(matches!(
        rejected_result.result,
        MutationResult::Rejected(_)
    ));
    runtime::drained(&c, &service, rejected_result.index).await?;
    stop.send(()).unwrap();
    task.await??;
    drop(service);

    let ingress = Service::validated_ingress(
        c.client.clone(),
        one_shard(),
        fixture_binding(),
        &limits,
        MergeMemoryBudget::new(32 << 20),
        &MergeMemoryBudget::new(16 << 20),
    )
    .await?;
    let mut publisher = materializer::Publisher::acquire(
        ingress,
        storage.clone(),
        options(),
        MergeMemoryBudget::new(32 << 20),
        acquire(&c).await?,
    )
    .await?;
    let (service, driver) = Service::connect(
        c.client.clone(),
        one_shard(),
        fixture_binding(),
        &storage,
        limits.clone(),
        MergeMemoryBudget::new(32 << 20),
        MergeMemoryBudget::new(16 << 20),
    )
    .await?;
    let (stop, task) = run(driver);
    let d = producer(&service, 3).await?;
    let before = state(&c).await?;
    let plan = materializer::compaction::plan(&publisher, &before)
        .await?
        .context("fixture has no compaction work")?;
    let read_version = plan.read_version;
    command(
        &c,
        Command::Plan(p::PlanRequest {
            authority: publisher.authority().clone(),
            plan,
        }),
    )
    .await?;
    let prepared = if stage > 0 {
        publisher.prepare_next().await?
    } else {
        None
    };
    let suffix = request(
        &service,
        &d,
        1,
        wire(&limits, &["t00000000"], 0, None, 1, Some(0.9))?,
    )
    .await?;
    let suffix_result = service.submit(&suffix, Duration::from_secs(30)).await?;
    let current = service
        .get("t00000000", Duration::from_secs(30))
        .await?
        .unwrap();
    assert_eq!(current.status.as_deref(), Some("reviewed"));
    assert_eq!(current.score, Some(0.9));
    let base = read(&c, &storage, Some(read_version)).await?;
    if stage >= 2 {
        let view = state(&c).await?;
        let pending = view.progress.pending.as_ref().unwrap();
        let prepared = prepared.as_ref().unwrap();
        let original = materializer::stage::annotation_compaction_transaction(
            &base,
            pending,
            &prepared.descriptors,
        )
        .await?;
        let handler = p::lance::ManagedManifest::new(
            Arc::new(LostReply(Arc::new(c.client.manifest_catalog()))),
            storage.base.clone(),
            &one_shard(),
            p::lance::ManifestCommit {
                authority: publisher.authority().clone(),
                incarnation: fixture_binding().incarnation,
                transaction: pending.plan.transaction,
                plan_index: pending.index,
                prepared_index: prepared.index,
                descriptor_digest: prepared.descriptors.digest,
                predecessor: view.manifests.as_ref().unwrap().versions[&read_version].identity,
                target: pending.plan.target_version()?,
            },
        )?;
        assert!(lance::dataset::CommitBuilder::new(base.clone())
            .with_commit_handler(Arc::new(handler))
            .with_max_retries(0)
            .with_skip_auto_cleanup(true)
            .execute(original)
            .await
            .is_err());
        if stage == 3 {
            publisher.commit_pending().await?;
        }
    }
    let before_exit = state(&c).await?;
    let expected = Expected {
        progress: before_exit.progress.clone(),
        manifest_head: before_exit.manifests.as_ref().unwrap().head,
        bodies: [
            first.body().to_vec(),
            rejected.body().to_vec(),
            suffix.body().to_vec(),
        ],
        outcomes: [first_result, rejected_result, suffix_result],
        current,
    };
    if crash_boundary {
        // Persist the client's exact inputs/results before telling the parent to
        // kill this process. WAL replicas, applier and retained publication are
        // all still live; no shutdown or drop cleanup runs before SIGKILL.
        let proof = root.join("expected.json");
        std::fs::write(&proof, serde_json::to_vec(&expected)?)?;
        std::fs::File::open(&proof)?.sync_all()?;
        std::fs::write(root.join("ready"), b"ready")?;
        std::future::pending::<()>().await;
    }
    stop.send(()).unwrap();
    task.await??;
    drop((service, publisher, base, first, rejected, suffix, prepared));
    c.shutdown().await?;
    verify_recovery(root, expected).await
}

async fn verify_recovery(root: &Path, expected: Expected) -> Result<()> {
    // Fresh sessions prevent a warm reader cache from hiding missing blob files.
    let storage = fixture_storage(root).await?;
    let limits = fixture_limits();
    let c = network::Cluster::open_signed(root, one_shard())
        .await
        .context("reopening compaction replica cluster")?;
    let recovered = state(&c)
        .await
        .context("reading the reopened compaction catalog")?;
    assert_eq!(recovered.progress, expected.progress);
    assert_eq!(
        recovered.manifests.as_ref().unwrap().head,
        expected.manifest_head
    );
    let Expected {
        bodies: [first_body, rejected_body, suffix_body],
        outcomes: [first_result, rejected_result, suffix_result],
        current,
        ..
    } = expected;
    let (service, owner) = compaction_owner(&c, &storage, &limits)
        .await
        .context("acquiring the recovered compaction owner")?;
    let (stop, task) = runtime::run_owner(collector(owner)?);
    let a = producer(&service, 1)
        .await
        .context("restoring producer 1")?;
    let b = producer(&service, 2)
        .await
        .context("restoring producer 2")?;
    let d = producer(&service, 3)
        .await
        .context("restoring producer 3")?;
    // Each prepared request reserves workspace for the contract's 2048-row
    // maximum. Resolve and release one at a time instead of holding three full
    // reservations on the same bounded upload budget across subsequent writes.
    for (producer, body, outcome) in [
        (&a, first_body, &first_result),
        (&b, rejected_body.clone(), &rejected_result),
        (&d, suffix_body, &suffix_result),
    ] {
        let retry = request(&service, producer, 1, Bytes::from(body))
            .await
            .context("reserving recovered request workspace")?;
        assert_eq!(
            &service
                .submit(&retry, Duration::from_secs(30))
                .await
                .context("resolving the original recovered request")?,
            outcome
        );
    }
    assert_eq!(
        service.get("t00000000", Duration::from_secs(30)).await?,
        Some(current.clone())
    );
    runtime::drained(&c, &service, suffix_result.index)
        .await
        .context("draining the recovered suffix")?;
    let selected = read(&c, &storage, None).await?;
    assert_eq!(selected.manifest().fragments.len(), 1);
    assert_eq!(selected.count_rows(None).await?, 16);
    assert_eq!(
        selected
            .count_rows(Some("status = 'reviewed' AND score = 0.9".into()))
            .await?,
        1
    );
    let indices = selected.load_indices().await?;
    assert_eq!(indices.len(), 1);
    assert!(indices[0]
        .fragment_bitmap
        .as_ref()
        .unwrap()
        .contains(selected.manifest().fragments[0].id as u32));
    drop(selected);
    for round in 2..=5 {
        let req = request(
            &service,
            &d,
            round,
            wire(&limits, &["t00000000"], 0, None, 1, Some(round as f32))?,
        )
        .await?;
        let result = service.submit(&req, Duration::from_secs(30)).await?;
        runtime::drained(&c, &service, result.index).await?;
        assert_eq!(
            service
                .get("t00000000", Duration::from_secs(30))
                .await?
                .unwrap()
                .score,
            Some(round as f32)
        );
    }
    tokio::time::timeout(Duration::from_secs(30), async {
        while service.reclamation_observation().deleted == 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await?;
    let rejected = request(&service, &b, 1, Bytes::from(rejected_body)).await?;
    assert_eq!(
        service.submit(&rejected, Duration::from_secs(30)).await?,
        rejected_result
    );
    let latest = read(&c, &storage, None).await?;
    let batches = latest
        .scan()
        .project(&["body"])?
        .try_into_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    for batch in batches {
        for body in batch
            .column(0)
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .unwrap()
            .iter()
            .flatten()
        {
            assert_eq!(body, vec![0x5a; 1024].as_slice());
        }
    }
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await?;
    Ok(())
}
