//! Native upload succeeds but reports failure, before catalog Prepare exists.
use super::*;
use crate::tests::annotations::{producer, request, wire};

#[tokio::test]
async fn lost_native_put_reply_recovers_acked_fields_and_bounded_pending_work() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), exercise()).await?
}

async fn exercise() -> Result<()> {
    let mut builder = tempfile::Builder::new();
    builder.prefix("annotation-lost-native-put-");
    let root = if let Ok(parent) = std::env::var("INGEST_TEST_ARTIFACTS") {
        builder.disable_cleanup(true).tempdir_in(parent)?
    } else {
        builder.tempdir()?
    };
    let path = root.path().join("turns.lance");
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let gates = Arc::new(store::Gates::default());
    let storage = store::storage(&path, gates.clone()).await?;
    let limits = Limits {
        retained_bytes: 256 << 10,
        retained_entries: 128,
        value_bytes: 4096,
        readers: 16,
    };
    annotation_publication::seed_contract(
        &c,
        &storage,
        annotations::Service::contract(&binding(), &limits)?,
    )
    .await?;
    let base = annotation_publication::read(&c, &storage, None)
        .await?
        .version_id();
    let (service, runtime) = annotation_runtime::owner(&c, &storage, &limits).await?;
    let (_, _, read_cut) = runtime.http_binding();
    let (_stop, task) = annotation_runtime::run_owner(runtime);
    let mut upload = gates.lose_put_reply();
    let mut acknowledged = Vec::new();
    for (slot, status, score) in [
        (1, Some("reviewed"), None),
        (2, None, Some(0.9)),
        (3, Some("approved"), None),
    ] {
        let producer = producer(&service, slot)
            .await
            .with_context(|| format!("open annotation test producer slot {slot}"))?;
        let update = request(
            &service,
            &producer,
            1,
            wire(
                &limits,
                &["T"],
                u8::from(status.is_some()),
                status,
                u8::from(score.is_some()),
                score,
            )?,
        )
        .await?;
        let result = service.submit(&update, Duration::from_secs(30)).await?;
        assert_eq!(result.result, annotations::MutationResult::Applied);
        acknowledged.push((update, result));
        if slot == 1 {
            upload.entered().await?;
        }
    }
    let orphan = gates.failed_upload.lock().unwrap().clone().unwrap();
    let orphan = std::path::Path::new("/").join(orphan.as_ref());
    assert!(
        orphan.is_file(),
        "fault must follow an actual successful native upload"
    );
    let pending = annotation_publication::state(&c)
        .await?
        .progress
        .pending
        .unwrap();
    assert_eq!(
        pending.plan.sources[&10].records,
        vec![(acknowledged[0].1.index, acknowledged[0].1.meta.clone())]
    );
    assert!(pending.plan.sources[&10].through < acknowledged[1].1.index);

    // The active two-record window is full. A fourth committed request must
    // wait for apply without receiving an annotation ACK or growing staging.
    let producer = producer(&service, 4).await?;
    let waiting = request(
        &service,
        &producer,
        1,
        wire(&limits, &["U"], 1, Some("recovered"), 0, None)?,
    )
    .await?;
    let accepted = service.ingress().append(&waiting).await?;
    tokio::time::timeout(Duration::from_secs(5), async {
        while !service.observation().await.1 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;
    let view = service.observation().await.0;
    // Producer-open/control records may advance the processed cursor beyond
    // the last mutation; the waiting mutation itself must remain unapplied.
    assert!(view.applied >= acknowledged[2].1.index);
    assert!(view.applied < accepted.position.index);
    assert_eq!(view.frozen_entries, 1);
    assert_eq!(view.active_entries, 1);
    assert!(view.retained_bytes <= limits.retained_bytes);
    assert_eq!(service.source_observation().records, 2);
    assert!(
        tokio::time::timeout(
            Duration::from_millis(100),
            service.submit(&waiting, Duration::from_secs(30)),
        )
        .await
        .is_err(),
        "unapplied work received an annotation ACK"
    );
    let current = service
        .get_owned("T", read_cut, Duration::from_secs(1))
        .await?
        .unwrap();
    assert_eq!(current.status.as_deref(), Some("approved"));
    assert_eq!(current.score, Some(0.9));

    upload.release();
    let error = tokio::time::timeout(Duration::from_secs(30), task)
        .await??
        .unwrap_err();
    assert!(
        format!("{error:#}").contains("injected lost native PUT reply"),
        "{error:#}"
    );
    let failed = annotation_publication::state(&c).await?;
    assert_eq!(failed.manifests.as_ref().unwrap().head, base);
    assert_eq!(failed.progress.pending.as_ref(), Some(&pending));
    assert!(
        failed.progress.prepared.is_none(),
        "upload is not a prepared publication"
    );
    assert!(
        orphan.is_file(),
        "the failed upload must leave an actual orphan"
    );
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
    drop(service);
    c.shutdown().await?;

    // Open the original three replica disks and the same table, with a new
    // session. Recovery must use the retained plan and WAL, not old staging.
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let storage = store::storage(&path, Arc::new(store::Gates::default())).await?;
    assert_eq!(
        annotation_publication::state(&c)
            .await?
            .progress
            .pending
            .as_ref(),
        Some(&pending)
    );
    let (service, runtime) = annotation_runtime::owner(&c, &storage, &limits).await?;
    let (_, _, read_cut) = runtime.http_binding();
    assert!(
        orphan.is_file(),
        "the orphan must survive until collection runs"
    );
    let runtime =
        runtime.with_exclusive_reclamation(annotations::runtime::reclamation::Options {
            max_objects: 16,
            max_bytes: 16 << 10,
            delete_parallelism: 2,
            interval: Duration::from_millis(100),
        })?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    assert_eq!(
        service
            .get_owned("T", read_cut, Duration::from_secs(30))
            .await?,
        Some(current.clone())
    );
    let resolved = service.submit(&waiting, Duration::from_secs(30)).await?;
    assert_eq!(resolved.index, accepted.position.index);
    assert_eq!(resolved.result, annotations::MutationResult::Applied);
    annotation_runtime::drained(&c, &service, resolved.index).await?;
    for (request, result) in &acknowledged {
        assert_eq!(
            &service.submit(request, Duration::from_secs(30)).await?,
            result
        );
    }
    assert_eq!(
        service
            .get_owned("T", read_cut, Duration::from_secs(30))
            .await?,
        Some(current)
    );
    tokio::time::timeout(Duration::from_secs(30), async {
        while orphan.exists() || service.reclamation_observation().pending_objects != 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;
    for index in acknowledged
        .iter()
        .map(|(_, result)| result.index)
        .chain([resolved.index])
    {
        assert!(matches!(
            c.client
                .request_routed(Request::Fetch { group: 10, index })
                .await?
                .reply,
            Reply::Record(None)
        ));
    }
    stop.send(()).unwrap();
    task.await??;
    // The complete native rows retain original labels and body bytes.
    let base = annotation_publication::read(&c, &storage, None).await?;
    let batches = base
        .scan()
        .try_into_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    let mut seen = std::collections::BTreeSet::new();
    for batch in batches {
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let labels = batch
            .column_by_name("labels")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::ListArray>()
            .unwrap();
        let statuses = batch
            .column_by_name("status")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let scores = batch
            .column_by_name("score")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::Float32Array>()
            .unwrap();
        let bodies = batch
            .column_by_name("body")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::LargeBinaryArray>()
            .unwrap();
        for row in 0..batch.num_rows() {
            assert!(seen.insert(ids.value(row).to_owned()));
            assert!(!statuses.is_null(row) && !labels.is_null(row) && !bodies.is_null(row));
            let expected = if ids.value(row) == "T" {
                "approved"
            } else {
                "recovered"
            };
            assert_eq!(statuses.value(row), expected);
            assert_eq!(scores.is_null(row), ids.value(row) == "U");
            if ids.value(row) == "T" {
                assert_eq!(scores.value(row), 0.9);
            }
            let values = labels.value(row);
            let values = values.as_any().downcast_ref::<StringArray>().unwrap();
            assert_eq!(values.iter().collect::<Vec<_>>(), vec![Some("original")]);
            assert_eq!(bodies.value(row).len(), 256 << 10);
            assert!(bodies
                .value(row)
                .iter()
                .enumerate()
                .all(|(i, b)| *b == ((i * 31 + i / 17) % 251) as u8));
        }
    }
    assert_eq!(seen, ["T".to_owned(), "U".to_owned()].into());
    c.shutdown().await
}
