use super::*;
use crate::annotations::{
    runtime::{Configuration, Runtime},
    single_body,
};
use crate::materializer::{Acquisition, Options};
use crate::tests::annotation_runtime::{drained, run_owner};

mod compaction;
mod credentials;
mod lookahead_pressure;
mod native_memory;
mod payload_memory;
#[cfg(unix)]
mod process;
mod staged_lookup;
mod upload_failure;

pub(super) async fn owner(
    c: &network::Cluster,
    storage: &Storage,
    spec: &TurnIngestionSpec,
) -> Result<(Service, Runtime)> {
    owner_with_wait(c, storage, spec, Duration::from_millis(20)).await
}

async fn owner_with_wait(
    c: &network::Cluster,
    storage: &Storage,
    spec: &TurnIngestionSpec,
    max_batch_wait: Duration,
) -> Result<(Service, Runtime)> {
    let view = annotation_publication::state(c).await?;
    Runtime::connect_body_shard(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding(),
        storage.clone(),
        Configuration {
            limits: fixture::limits(),
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
            expected_epoch: view.authority.epoch,
        },
        Shard { bucket: 0 },
        spec,
    )
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn full_body_rf3_first_create_drains_reads_and_recovers_without_auxiliary_table() -> Result<()>
{
    let input = bodies()?;
    let spec = TurnIngestionSpec::new(
        input[0].schema(),
        vec!["session_id".into(), "id".into()],
        "session_id",
    )?;
    let keys = input
        .iter()
        .map(|body| spec.prepare(body.clone(), 4096).map(|p| p.keys().to_vec()))
        .collect::<Result<Vec<_>>>()?;
    let root = tempfile::tempdir()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let body = fixture::storage(root.path(), "full-body.lance").await?;
    let identity = Identity {
        owner: Uuid::new_v4(),
        acquisition: Uuid::new_v4(),
        manifest: Uuid::new_v4(),
    };
    for _ in 0..2 {
        crate::bootstrap::provision_body_shard(
            c.client.clone(),
            annotation_publication::one_shard(),
            binding(),
            body.clone(),
            identity.clone(),
            &fixture::limits(),
            Shard { bucket: 0 },
            &spec,
        )
        .await?;
    }
    let encode_body = |i| {
        fixture::encoded(single_body::batch(
            &spec,
            &fixture::limits(),
            Shard { bucket: 0 },
            &input[i],
        )?)
    };
    let (service, runtime) = owner(&c, &body, &spec).await?;
    let (stop, task) = run_owner(runtime);
    let a = producer(&service, 1).await?;
    let b = producer(&service, 2).await?;
    let first = request(&service, &a, 1, encode_body(0)?).await?;
    let result = service.submit(&first, DEADLINE).await?;
    assert_eq!(result.result, MutationResult::Applied);
    assert_eq!(
        service
            .get_full_body(&keys[0][0], DEADLINE)
            .await?
            .unwrap()
            .batch(),
        &input[0].slice(0, 1)
    );
    let mut through = result.index;
    for (i, sequence) in [(1, 1), (2, 2), (3, 3)] {
        let append = request(&service, &b, sequence, encode_body(i)?).await?;
        let result = service.submit(&append, DEADLINE).await?;
        assert_eq!(result.result, MutationResult::Applied);
        through = result.index;
    }
    drained(&c, &service, through).await?;
    assert_eq!(service.retained_body_frames(), (0, 0));
    let selected = service.selected_full_body(DEADLINE).await?;
    assert_eq!(selected.uri, body.uri);
    assert_eq!(selected.incarnation, binding().incarnation);
    assert!(selected.through >= through);
    let durable = annotation_publication::read(&c, &body, Some(selected.manifest.version)).await?;
    assert_eq!(durable.count_rows(None).await?, 6);
    assert_eq!(
        Schema::from(durable.schema()),
        *single_body::storage_schema(&spec)?
    );
    assert!(
        lance_context_core::annotation_staging::BodySnapshot::read(&durable)
            .await?
            .is_none()
    );
    assert_eq!(
        service
            .get_full_body(&keys[0][0], DEADLINE)
            .await?
            .unwrap()
            .batch(),
        &input[0].slice(0, 1)
    );
    assert_eq!(
        service
            .get_full_body(&keys[3][0], DEADLINE)
            .await?
            .unwrap()
            .batch(),
        &input[3].slice(0, 1)
    );
    stop.send(()).unwrap();
    task.await??;
    drop(service);
    let (service, runtime) = owner(&c, &body, &spec).await?;
    // Recovery observes the installed published snapshot before more apply.
    let recovered = service.observation().await.0;
    assert_eq!(recovered.installed_base.rows, Some(6));
    assert_eq!(recovered.installed_base.version, selected.manifest.version);
    assert_eq!(recovered.installed_base.source, selected.source);
    assert_eq!(recovered.materialized, selected.through);
    let (stop, task) = run_owner(runtime);
    assert_eq!(
        service
            .get_full_body(&keys[0][0], DEADLINE)
            .await?
            .unwrap()
            .batch(),
        &input[0].slice(0, 1)
    );
    let repeated = request(&service, &a, 2, encode_body(0)?).await?;
    // Point reads do not perform ingestion's historical deduplication lookup.
    assert_eq!(service.body_lookup_observation().completed_lookups, 0);
    let result = service.submit(&repeated, DEADLINE).await?;
    drained(&c, &service, result.index).await?;
    let selected = service.selected_full_body(DEADLINE).await?;
    let replayed = annotation_publication::read(&c, &body, Some(selected.manifest.version)).await?;
    assert_eq!(replayed.manifest().fragments, durable.manifest().fragments);
    let lookup = service.body_lookup_observation();
    assert_eq!(lookup.completed_lookups, 1);
    assert_eq!(lookup.scans_started, 1);
    assert_eq!(lookup.requested_keys, keys[0].len() as u64);
    assert!(lookup.projected_rows >= keys[0].len() as u64);
    assert!(lookup.projected_array_bytes > 0);
    assert!(lookup.native_summaries <= lookup.completed_lookups);
    assert_eq!(service.submit(&repeated, DEADLINE).await?, result);
    assert_eq!(service.body_lookup_observation().completed_lookups, 1);
    // More concurrent lanes than the frame budget can admit at once. Queued
    // decoded frames must leave capacity for their own historical existence
    // lookup, otherwise an all-duplicate burst can stop apply and publication.
    let mut burst = Vec::new();
    for slot in 3..11 {
        burst.push(producer(&service, slot).await?);
    }
    let encoded = encode_body(0)?;
    // Admission and sending belong to the same future: retaining every upload
    // before sending any would deliberately fill the bounded upload budget.
    let completed = tokio::time::timeout(
        DEADLINE,
        futures::future::try_join_all(burst.iter().map(|lane| {
            let body = encoded.clone();
            let service = &service;
            async move {
                let request = request(service, lane, 1, body).await?;
                service.submit(&request, DEADLINE).await
            }
        })),
    )
    .await??;
    assert!(completed
        .iter()
        .all(|result| result.result == MutationResult::Applied));
    drained(
        &c,
        &service,
        completed.iter().map(|r| r.index).max().unwrap(),
    )
    .await?;
    assert_eq!(service.retained_body_frames(), (0, 0));
    let selected = service.selected_full_body(DEADLINE).await?;
    let after_burst =
        annotation_publication::read(&c, &body, Some(selected.manifest.version)).await?;
    assert_eq!(
        after_burst.manifest().fragments,
        durable.manifest().fragments
    );
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
