//! Recover a mixed cut after the real multipart object is written but its reply is lost.
use super::*;
use crate::{
    annotations::{mixed, runtime::Configuration, runtime::Runtime, Service},
    materializer::{Acquisition, Options, Storage},
    tests::{annotations::request, mixed_annotations as fixture},
};
use std::sync::atomic::Ordering;

const ROWS: usize = 7;
const BODY_BYTES: usize = 1 << 20;

fn large_binding() -> TableBinding {
    TableBinding {
        max_record_bytes: 8 << 20,
        ..binding()
    }
}

fn payload(row: usize) -> Vec<u8> {
    // Incompressible deterministic bytes keep the native file above the
    // multipart threshold. Verify complete bytes, not just a length or digest.
    let mut state = row as u64 + 1;
    (0..BODY_BYTES)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect()
}

async fn owner(
    c: &network::Cluster,
    narrow: &Storage,
    body: &Storage,
) -> Result<(Service, Runtime)> {
    let state = annotation_publication::state(c).await?;
    Runtime::connect_mixed(
        c.client.clone(),
        annotation_publication::one_shard(),
        large_binding(),
        narrow.clone(),
        body.clone(),
        Configuration {
            limits: fixture::limits(),
            upload_budget: MergeMemoryBudget::new(64 << 20),
            apply_budget: MergeMemoryBudget::new(64 << 20),
            publication_budget: MergeMemoryBudget::new(128 << 20),
            publication: Options {
                max_source_bytes: 8 << 20,
                min_source_bytes: 1,
                max_records: 2,
                max_batch_wait: Duration::from_millis(20),
                retained_manifest_versions: 4,
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

async fn current_values(service: &Service, pending_applied: bool) -> Result<()> {
    for row in 0..ROWS {
        let id = format!("large-{row}");
        let turn = service
            .get_turn(&id, Duration::from_secs(30))
            .await?
            .context("acknowledged turn missing")?;
        let expected_status = match row {
            0 => "reviewed",
            1 if pending_applied => "recovered",
            _ => "pending",
        };
        assert_eq!(turn.annotation.status.as_deref(), Some(expected_status));
        assert_eq!(turn.annotation.score, (row == 0).then_some(0.9));
        assert_eq!(
            turn.annotation.labels,
            std::collections::BTreeSet::from(["imported".into()])
        );
        let batch = turn.body();
        let bodies = batch
            .column_by_name("binary_payload")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::LargeBinaryArray>()
            .unwrap();
        assert_eq!(bodies.value(0), payload(row));
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mixed_lost_multipart_completion_recovers_acked_fields_without_replaying_requests(
) -> Result<()> {
    tokio::time::timeout(Duration::from_secs(120), exercise()).await?
}

async fn exercise() -> Result<()> {
    let mut builder = tempfile::Builder::new();
    builder.prefix("mixed-lost-multipart-");
    let root = if let Ok(parent) = std::env::var("INGEST_TEST_ARTIFACTS") {
        builder.disable_cleanup(true).tempdir_in(parent)?
    } else {
        builder.tempdir()?
    };
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = fixture::storage(root.path(), "annotations.lance").await?;
    let gates = Arc::new(store::Gates::default());
    let body = store::storage(&root.path().join("body.lance"), gates.clone()).await?;
    fixture::seed_with_binding(&c, &narrow, &body, large_binding()).await?;
    let base = annotation_publication::read(&c, &narrow, None).await?;
    let (a, runtime) = owner(&c, &narrow, &body).await?;
    let (_stop, task) = annotation_runtime::run_owner(runtime);
    let mut failed_upload = gates.lose_put_reply();

    let ids = (0..ROWS)
        .map(|row| format!("large-{row}"))
        .collect::<Vec<_>>();
    let names = ids.iter().map(String::as_str).collect::<Vec<_>>();
    let batch = fixture::create(&names, &vec!["s"; ROWS], &vec![0; ROWS])?;
    let mut columns = batch.columns().to_vec();
    columns[batch.schema().index_of("binary_payload")?] = Arc::new(
        arrow_array::LargeBinaryArray::from_iter_values((0..ROWS).map(payload)),
    );
    let encoded = fixture::encoded(RecordBatch::try_new(batch.schema(), columns)?)?;
    let producer = crate::tests::annotations::producer(&a, 1).await?;
    let first = request(&a, &producer, 1, encoded).await?;
    let first_result = a.submit(&first, Duration::from_secs(30)).await?;
    assert_eq!(first_result.result, annotations::MutationResult::Applied);
    failed_upload.entered().await?;
    assert!(gates.multipart_parts.load(Ordering::SeqCst) >= 2);
    assert_eq!(gates.multipart_completions.load(Ordering::SeqCst), 1);
    let orphan = gates.failed_upload.lock().unwrap().clone().unwrap();
    let orphan = std::path::Path::new("/").join(orphan.as_ref());
    assert!(std::fs::metadata(&orphan)?.len() > 5 << 20);

    let mut acknowledged = vec![(first, first_result)];
    for (slot, status, score) in [(2, Some("reviewed"), None), (3, None, Some(0.9))] {
        let producer = crate::tests::annotations::producer(&a, slot).await?;
        let patch = mixed::wire::patches(
            &fixture::patch(&["large-0"], status, score)?,
            &fixture::limits(),
        )?;
        let request = request(&a, &producer, 1, fixture::encoded(patch)?).await?;
        let completed = a.submit(&request, Duration::from_secs(30)).await?;
        assert_eq!(completed.result, annotations::MutationResult::Applied);
        acknowledged.push((request, completed));
    }
    current_values(&a, false).await?;
    let producer = crate::tests::annotations::producer(&a, 4).await?;
    let waiting = request(
        &a,
        &producer,
        1,
        fixture::encoded(mixed::wire::patches(
            &fixture::patch(&["large-1"], Some("recovered"), None)?,
            &fixture::limits(),
        )?)?,
    )
    .await?;
    // Admit the original proposal once. It has no application ACK and is not
    // resubmitted after the failed owner or the replica restart.
    let accepted = a.ingress().append(&waiting).await?;
    tokio::time::timeout(Duration::from_secs(10), async {
        while a.apply_backpressure().is_none() {
            tokio::task::yield_now().await;
        }
    })
    .await?;
    assert_eq!(a.source_observation().records, 2);
    assert!(a.observation().await.0.applied < accepted.position.index);
    let retained = annotation_publication::state(&c).await?;
    let plan = retained.progress.pending.clone().unwrap();
    assert!(retained.progress.prepared.is_none());
    assert_eq!(retained.manifests.as_ref().unwrap().head, base.version_id());
    assert_eq!(plan.plan.sources[&10].records.len(), 1);
    assert_eq!(plan.plan.sources[&10].records[0].0, acknowledged[0].1.index);

    failed_upload.release();
    let error = tokio::time::timeout(Duration::from_secs(30), task)
        .await??
        .unwrap_err();
    assert!(
        format!("{error:#}").contains("injected lost native PUT reply"),
        "{error:#}"
    );
    assert_eq!(gates.multipart_completions.load(Ordering::SeqCst), 1);
    let stopped = annotation_publication::state(&c).await?;
    assert_eq!(stopped.progress.pending.as_ref(), Some(&plan));
    assert!(stopped.progress.prepared.is_none());
    assert_eq!(stopped.manifests.as_ref().unwrap().head, base.version_id());
    for index in acknowledged
        .iter()
        .map(|(_, result)| result.index)
        .chain([accepted.position.index])
    {
        assert!(matches!(
            c.client
                .request_routed(Request::Fetch { group: 10, index })
                .await?
                .reply,
            Reply::Record(Some(_))
        ));
    }
    drop(a);
    c.shutdown().await?;
    drop((narrow, body));

    // Only this test's fresh RF3 stores reopen. A fresh owner reconstructs the
    // frozen cut and its applied suffix; no business request is replayed.
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = fixture::storage(root.path(), "annotations.lance").await?;
    let body = store::storage(
        &root.path().join("body.lance"),
        Arc::new(store::Gates::default()),
    )
    .await?;
    assert_eq!(
        annotation_publication::state(&c).await?.progress.pending,
        Some(plan)
    );
    assert!(orphan.is_file());
    let (a, runtime) = owner(&c, &narrow, &body).await?;
    let runtime =
        runtime.with_exclusive_reclamation(annotations::runtime::reclamation::Options {
            max_objects: 16,
            max_bytes: 16 << 20,
            delete_parallelism: 2,
            interval: Duration::from_millis(20),
        })?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    current_values(&a, true).await?;
    annotation_runtime::drained(&c, &a, accepted.position.index).await?;
    assert_eq!(a.retained_body_frames(), (0, 0));
    let selected = annotation_publication::read(&c, &narrow, None).await?;
    assert_eq!(selected.count_rows(None).await?, ROWS);
    let reference = lance_context_core::annotation_staging::BodySnapshot::read(&selected)
        .await?
        .unwrap();
    let selected_body =
        mixed::load_body(&body, &reference, &binding().incarnation.to_string()).await?;
    assert_eq!(selected_body.count_rows(None).await?, ROWS);
    tokio::time::timeout(Duration::from_secs(30), async {
        while orphan.exists() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await?;
    // Reclaimed failed-upload bytes cannot invalidate the selected body pair.
    current_values(&a, true).await?;
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
