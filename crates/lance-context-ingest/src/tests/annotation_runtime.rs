use super::annotations::{connect, producer, request, run, wire};
use super::*;
use crate::{
    annotation_publication as publication,
    annotations::{
        runtime::{Configuration, Runtime},
        MutationResult, Service,
    },
    materializer::{Acquisition, Options, Storage},
};
use lance_context_core::annotation_staging::Limits;

pub(super) mod automatic_reclamation;
#[cfg(feature = "service")]
mod lost_advance;
#[cfg(feature = "service")]
mod lost_plan;
#[cfg(feature = "service")]
pub(super) mod lost_prepare;
mod maintenance_ready;
mod open;

#[cfg(unix)]
mod process;

pub(super) async fn owner(
    c: &network::Cluster,
    storage: &Storage,
    limits: &Limits,
) -> Result<(Service, Runtime)> {
    owner_with_wait(c, storage, limits, Duration::from_millis(20)).await
}

async fn owner_with_wait(
    c: &network::Cluster,
    storage: &Storage,
    limits: &Limits,
    max_batch_wait: Duration,
) -> Result<(Service, Runtime)> {
    let state = annotation_publication::state(c).await?;
    Runtime::connect(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding(),
        storage.clone(),
        Configuration {
            limits: limits.clone(),
            upload_budget: MergeMemoryBudget::new(8 << 20),
            apply_budget: MergeMemoryBudget::new(8 << 20),
            publication_budget: MergeMemoryBudget::new(8 << 20),
            publication: Options {
                max_source_bytes: 1 << 20,
                min_source_bytes: 1 << 20,
                max_records: 2,
                max_batch_wait,
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

pub(super) async fn drained(c: &network::Cluster, service: &Service, through: u64) -> Result<()> {
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let (observed, pressured) = service.observation().await;
            if observed.materialized >= through
                && observed.retained_bytes == 0
                && observed.retained_metadata_bytes == 0
                && !pressured
                && service.source_observation().records == 0
            {
                // Releasing installed staging no longer waits for Close/Advance.
                // Tests that stop the owner or plan maintenance need the durable
                // publication boundary too, not just an empty local read view.
                let view = annotation_publication::state(c).await?;
                if view.progress.pending.is_none()
                    && view.progress.checkpoint.get(&10).copied().unwrap_or(0) >= through
                {
                    return Ok::<_, anyhow::Error>(());
                }
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await??;
    Ok(())
}

pub(super) fn run_owner(
    runtime: Runtime,
) -> (
    tokio::sync::oneshot::Sender<()>,
    tokio::task::JoinHandle<Result<()>>,
) {
    let (stop, stopped) = tokio::sync::oneshot::channel();
    let task = tokio::spawn(async move {
        let result = runtime
            .run(async {
                let _ = stopped.await;
            })
            .await;
        if let Err(error) = &result {
            eprintln!("annotation runtime stopped: {error:#}");
        }
        result
    });
    (stop, task)
}

#[tokio::test]
async fn annotation_runtime_replays_only_the_retained_unprepared_cut_then_drains() -> Result<()> {
    resume(0).await
}

#[tokio::test]
async fn annotation_runtime_reuses_prepared_files_then_drains() -> Result<()> {
    resume(1).await
}

#[tokio::test]
async fn annotation_runtime_resolves_a_selected_manifest_then_drains() -> Result<()> {
    resume(2).await
}

async fn resume(stage: u8) -> Result<()> {
    let root = tempfile::tempdir()?;
    let mut c =
        network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let storage = Storage::resolve(
        root.path().join("turns.lance").to_str().unwrap().into(),
        Default::default(),
        Arc::new(lance::session::Session::default()),
    )
    .await?;
    let limits = Limits {
        retained_bytes: 256 << 10,
        retained_entries: 128,
        value_bytes: 4096,
        readers: 16,
    };
    annotation_publication::seed_contract(&c, &storage, Service::contract(&binding(), &limits)?)
        .await?;
    let (service, driver) = connect(&c, &storage, &limits).await?;
    let (stop, task) = run(driver);
    let a = producer(&service, 1).await?;
    let b = producer(&service, 2).await?;
    let first = request(
        &service,
        &a,
        1,
        wire(&limits, &["T"], 1, Some("reviewed"), 0, None)?,
    )
    .await?;
    let first_result = service.submit(&first, Duration::from_secs(10)).await?;
    let frozen = service.freeze(Uuid::new_v4()).await?.unwrap();
    let original = if stage == 0 {
        annotation_publication::select_range(
            &c,
            frozen.sources().records.clone(),
            frozen.through(),
            Uuid::parse_str(frozen.identity())?,
        )
        .await?;
        None
    } else {
        let (prepared, selection, descriptors, commit) =
            annotation_checkpoint::prepare(&c, &frozen, 0).await?;
        let original = prepared.annotations().merge().transaction.clone();
        if stage == 2 {
            let result = publication::publish(
                Arc::new(annotation_publication::LostReply(Arc::new(
                    c.client.manifest_catalog(),
                ))),
                &annotation_publication::one_shard(),
                &storage,
                commit,
                &selection,
                &descriptors,
                prepared.annotations().base().clone(),
            )
            .await;
            assert!(result.is_err(), "the selected-manifest reply must be lost");
        }
        Some(original)
    };
    let suffix = request(
        &service,
        &b,
        1,
        wire(&limits, &["T"], 0, None, 1, Some(0.9))?,
    )
    .await?;
    let suffix_result = service.submit(&suffix, Duration::from_secs(10)).await?;
    stop.send(()).unwrap();
    task.await??;
    drop(frozen);
    c.shutdown().await?;
    c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let (service, runtime) = owner(&c, &storage, &limits).await?;
    let (stop, task) = run_owner(runtime);
    assert_eq!(
        service.submit(&first, Duration::from_secs(30)).await?,
        first_result
    );
    assert_eq!(
        service.submit(&suffix, Duration::from_secs(30)).await?,
        suffix_result
    );
    let current = service.get("T", Duration::from_secs(30)).await?.unwrap();
    assert_eq!(current.status.as_deref(), Some("reviewed"));
    assert_eq!(current.score, Some(0.9));
    drained(&c, &service, suffix_result.index).await?;
    if let Some(original) = original {
        let selected = annotation_publication::read(&c, &storage, Some(2)).await?;
        assert_eq!(
            selected.read_transaction().await?,
            Some(original),
            "recovery must reuse the original files and transaction"
        );
    }

    let rejected = request(
        &service,
        &a,
        2,
        wire(&limits, &["absent"], 2, None, 0, None)?,
    )
    .await?;
    let rejection = service.submit(&rejected, Duration::from_secs(30)).await?;
    assert!(matches!(rejection.result, MutationResult::Rejected(_)));
    drained(&c, &service, rejection.index).await?;
    assert_eq!(
        service.get("T", Duration::from_secs(30)).await?.unwrap(),
        current
    );
    let fetched = c
        .client
        .request_routed(Request::Fetch {
            group: 10,
            index: first_result.index,
        })
        .await?;
    assert!(matches!(fetched.reply, Reply::Record(None)));

    let mut last = None;
    for sequence in 3..=8 {
        let status = format!("review-{sequence}");
        let status_request = request(
            &service,
            &a,
            sequence,
            wire(&limits, &["T"], 1, Some(&status), 0, None)?,
        )
        .await?;
        let score = sequence as f32 / 10.;
        let score_request = request(
            &service,
            &b,
            sequence - 1,
            wire(&limits, &["T"], 0, None, 1, Some(score))?,
        )
        .await?;
        let (status_result, score_result) = tokio::join!(
            service.submit(&status_request, Duration::from_secs(30)),
            service.submit(&score_request, Duration::from_secs(30)),
        );
        let status_result = status_result?;
        let score_result = score_result?;
        let read = service.get("T", Duration::from_secs(30)).await?.unwrap();
        assert_eq!(read.status.as_deref(), Some(status.as_str()));
        assert_eq!(read.score, Some(score));
        drained(&c, &service, status_result.index.max(score_result.index)).await?;
        last = Some((
            status_request,
            status_result,
            score_request,
            score_result,
            read,
        ));
    }
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let (service, runtime) = owner(&c, &storage, &limits).await?;
    let (stop, task) = run_owner(runtime);
    let (status, status_result, score, score_result, current) = last.unwrap();
    assert_eq!(
        service.submit(&status, Duration::from_secs(30)).await?,
        status_result
    );
    assert_eq!(
        service.submit(&score, Duration::from_secs(30)).await?,
        score_result
    );
    assert_eq!(
        service.get("T", Duration::from_secs(30)).await?.unwrap(),
        current
    );
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
