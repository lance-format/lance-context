use super::*;
use crate::{
    annotations::{self, Driver, MutationResult, Service},
    materializer::Storage,
};
use arrow_array::{
    builder::{ListBuilder, StringBuilder},
    Float32Array, UInt8Array,
};
use lance_context_core::annotation_staging::{Limits, RejectionReason};
use tokio::sync::oneshot;

pub(super) fn wire(
    limits: &Limits,
    ids: &[&str],
    status_op: u8,
    status: Option<&str>,
    score_op: u8,
    score: Option<f32>,
) -> Result<Bytes> {
    let schema = annotations::wire::schema(limits);
    let empty = || {
        let mut labels = ListBuilder::new(StringBuilder::new()).with_field(Arc::new(
            arrow_schema::Field::new("item", arrow_schema::DataType::Utf8, false),
        ));
        for _ in ids {
            labels.append(true);
        }
        Arc::new(labels.finish()) as ArrayRef
    };
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(ids.to_vec())),
            empty(),
            empty(),
            Arc::new(UInt8Array::from(vec![status_op; ids.len()])),
            Arc::new(StringArray::from(vec![status; ids.len()])),
            Arc::new(UInt8Array::from(vec![score_op; ids.len()])),
            Arc::new(Float32Array::from(vec![score; ids.len()])),
        ],
    )?;
    let mut bytes = Vec::new();
    let mut writer = FileWriter::try_new(&mut bytes, &schema)?;
    writer.write(&batch)?;
    writer.finish()?;
    drop(writer);
    Ok(Bytes::from(bytes))
}

pub(super) async fn connect(
    c: &network::Cluster,
    storage: &Storage,
    limits: &Limits,
) -> Result<(Service, Driver)> {
    Service::connect(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding(),
        storage,
        limits.clone(),
        MergeMemoryBudget::new(8 << 20),
        MergeMemoryBudget::new(8 << 20),
    )
    .await
}

pub(super) fn run(driver: Driver) -> (oneshot::Sender<()>, tokio::task::JoinHandle<Result<()>>) {
    let (stop, done) = oneshot::channel();
    (
        stop,
        tokio::spawn(driver.run(async {
            let _ = done.await;
        })),
    )
}

pub(super) async fn producer(service: &Service, slot: u32) -> Result<Producer> {
    service
        .open_producer(ProducerOpen {
            bucket: 0,
            slot,
            expected_generation: 0,
            operation: Uuid::from_u128(100 + slot as u128),
        })
        .await
}

pub(super) async fn request(
    service: &Service,
    producer: &Producer,
    sequence: u64,
    body: Bytes,
) -> Result<PreparedAppend> {
    let upload = service.reserve_upload(body.len()).await?;
    let original = body.as_ptr();
    let prepared = service.prepare(upload, body, producer, sequence).await?;
    assert_eq!(prepared.body().as_ptr(), original);
    Ok(prepared)
}

fn label_patch(limits: &Limits) -> Result<Bytes> {
    let schema = annotations::wire::schema(limits);
    let source = ipc::decode(wire(limits, &["T"], 0, None, 0, None)?, schema.clone())?;
    let mut columns = source[0].columns().to_vec();
    for (index, label) in [(1, "good"), (2, "original")] {
        let mut values = ListBuilder::new(StringBuilder::new()).with_field(Arc::new(
            arrow_schema::Field::new("item", arrow_schema::DataType::Utf8, false),
        ));
        values.values().append_value(label);
        values.append(true);
        columns[index] = Arc::new(values.finish());
    }
    let batch = RecordBatch::try_new(schema.clone(), columns)?;
    let mut bytes = Vec::new();
    let mut writer = FileWriter::try_new(&mut bytes, &schema)?;
    writer.write(&batch)?;
    writer.finish()?;
    drop(writer);
    Ok(Bytes::from(bytes))
}

#[tokio::test]
async fn quorum_ack_waits_for_apply_and_another_ingress_reads_without_a_tail_scan() -> Result<()> {
    let root = tempfile::tempdir()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let storage = Storage::resolve(
        root.path().join("turns.lance").to_str().unwrap().into(),
        Default::default(),
        Arc::new(lance::session::Session::default()),
    )
    .await?;
    let limits = Limits::default();
    annotation_publication::seed_contract(&c, &storage, Service::contract(&binding(), &limits)?)
        .await?;
    let (a, driver_a) = connect(&c, &storage, &limits).await?;
    let (b, driver_b) = connect(&c, &storage, &limits).await?;
    let producer_a = producer(&a, 1).await?;
    let first = Arc::new(
        request(
            &a,
            &producer_a,
            1,
            wire(&limits, &["T", "U"], 1, Some("reviewed"), 0, None)?,
        )
        .await?,
    );
    let submit = tokio::spawn({
        let a = a.clone();
        let first = first.clone();
        async move { a.submit(&first, Duration::from_secs(10)).await }
    });
    // Prove WAL durability independently while the application worker is paused.
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let response = c
                .client
                .request_routed(Request::Query {
                    group: 10,
                    query: query::Query::Producer {
                        table: binding().table,
                        bucket: 0,
                        producer: producer_a.id(),
                    },
                })
                .await?;
            if matches!(
                response.reply,
                Reply::Query(query::Outcome::Observed {
                    value: query::Value::Producer(Some(_)),
                    ..
                })
            ) {
                return Ok::<_, anyhow::Error>(());
            }
            tokio::task::yield_now().await;
        }
    })
    .await??;
    assert!(
        !submit.is_finished(),
        "quorum persistence alone must not return application success"
    );
    assert!(
        b.get("T", Duration::from_millis(20)).await.is_err(),
        "lagging ingress must not serve its old base"
    );
    let (stop_a, task_a) = run(driver_a);
    let (stop_b, task_b) = run(driver_b);
    let accepted = submit.await??;
    assert_eq!(accepted.result, MutationResult::Applied);
    assert_eq!(
        b.get("T", Duration::from_secs(5))
            .await?
            .unwrap()
            .status
            .as_deref(),
        Some("reviewed")
    );

    let producer_b = producer(&b, 2).await?;
    let later = request(
        &b,
        &producer_b,
        1,
        wire(&limits, &["T"], 1, Some("final"), 1, Some(0.9))?,
    )
    .await?;
    assert_eq!(
        b.submit(&later, Duration::from_secs(5)).await?.result,
        MutationResult::Applied
    );
    assert_eq!(a.submit(&first, Duration::from_secs(5)).await?, accepted);
    let current = b.get("T", Duration::from_secs(5)).await?.unwrap();
    assert_eq!(current.status.as_deref(), Some("final"));
    assert_eq!(current.score, Some(0.9));
    let changed = request(&a, &producer_a, 1, wire(&limits, &["T"], 2, None, 2, None)?).await?;
    assert!(a.submit(&changed, Duration::from_secs(5)).await.is_err());
    assert_eq!(b.get("T", Duration::from_secs(5)).await?.unwrap(), current);
    let bad = request(
        &a,
        &producer_a,
        2,
        wire(&limits, &["T", "missing"], 2, None, 2, None)?,
    )
    .await?;
    let rejection = a.submit(&bad, Duration::from_secs(5)).await?;
    assert!(
        matches!(rejection.result, MutationResult::Rejected(ref r) if r.row == 1 && r.reason == RejectionReason::MissingTurn)
    );
    assert_eq!(b.get("T", Duration::from_secs(5)).await?.unwrap(), current);
    let clear = request(&a, &producer_a, 3, wire(&limits, &["T"], 0, None, 2, None)?).await?;
    assert_eq!(
        a.submit(&clear, Duration::from_secs(5)).await?.result,
        MutationResult::Applied
    );
    assert_eq!(
        b.get("T", Duration::from_secs(5)).await?.unwrap().score,
        None
    );
    // Malformed operations fail before consuming a producer sequence.
    for bytes in [
        wire(&limits, &["T"], 2, Some("invalid"), 0, None)?,
        wire(&limits, &["T"], 0, None, 1, Some(f32::NAN))?,
    ] {
        assert!(request(&a, &producer_a, 4, bytes).await.is_err());
    }
    let labels = request(&a, &producer_a, 4, label_patch(&limits)?).await?;
    let last = a.submit(&labels, Duration::from_secs(5)).await?;
    assert_eq!(last.result, MutationResult::Applied);
    let current = b.get("T", Duration::from_secs(5)).await?.unwrap();
    assert_eq!(
        current.labels,
        std::collections::BTreeSet::from(["good".into()])
    );
    assert_eq!(current.status.as_deref(), Some("final"));
    stop_a.send(()).unwrap();
    stop_b.send(()).unwrap();
    task_a.await??;
    task_b.await??;
    drop((a, b));
    c.shutdown().await?;

    // Rebuild independent values and request outcomes from the original native
    // base and durable WAL. This is an orderly RF3 restart, not SIGKILL.
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let (recovered, driver) = connect(&c, &storage, &limits).await?;
    let (stop, task) = run(driver);
    assert_eq!(
        recovered.submit(&labels, Duration::from_secs(5)).await?,
        last
    );
    let current = recovered.get("T", Duration::from_secs(5)).await?.unwrap();
    assert_eq!(current.status.as_deref(), Some("final"));
    assert_eq!(current.score, None);
    assert_eq!(
        current.labels,
        std::collections::BTreeSet::from(["good".into()])
    );
    assert!(recovered
        .submit(&first, Duration::from_secs(5))
        .await
        .is_err());
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
