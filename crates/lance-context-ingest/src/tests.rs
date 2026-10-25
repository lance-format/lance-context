use super::*;
use arrow_array::{
    new_null_array, types::Int8Type, ArrayRef, DictionaryArray, Int32Array, Int8Array,
    LargeBinaryArray, RecordBatch, TimestampMicrosecondArray,
};
use arrow_ipc::writer::FileWriter;
use lance::Dataset;
use lance_context_wal::replication::publication as p;
use receipt::ReceiptKeys;
use std::collections::BTreeMap;
#[cfg(not(feature = "materializer"))]
use {
    arrow_array::RecordBatchIterator,
    lance::dataset::WriteParams,
    lance_table::io::commit::{ConditionalPutCommitHandler, ManifestNamingScheme},
};

#[cfg(feature = "materializer")]
mod annotation_checkpoint;
#[cfg(feature = "materializer")]
mod annotation_publication;
#[cfg(feature = "materializer")]
mod annotations;
#[cfg(feature = "materializer")]
mod bootstrap;
#[cfg(all(feature = "http-client", feature = "materializer"))]
mod http;
#[cfg(feature = "materializer")]
mod materializer;
mod network;
#[cfg(feature = "service")]
mod preparation;
#[cfg(feature = "materializer")]
mod reader;
#[cfg(feature = "worker")]
mod worker;

fn layout() -> Arc<Layout> {
    Arc::new(Layout {
        cluster: Uuid::from_u128(1),
        nodes: (1..=3)
            .map(|id| (id, format!("wal-{id}.example.invalid:7000")))
            .collect(),
        groups: BTreeMap::from([(10, vec![1, 2, 3]), (20, vec![1, 2, 3])]),
        buckets: vec![10, 20, 10, 20],
    })
}
fn binding() -> TableBinding {
    TableBinding {
        table: Uuid::from_u128(2),
        destination: Destination {
            namespace: Uuid::from_u128(3),
            dataset: "rollouts.lance".into(),
        },
        incarnation: Uuid::from_u128(4),
        tenant: "tenant".into(),
        qualified_table: "rollouts".into(),
        max_record_bytes: 1 << 20,
        max_rows: 8,
    }
}
fn scope(binding: &TableBinding) -> Arc<Scope> {
    Arc::new(Scope {
        cluster: layout().cluster,
        layout: layout().fingerprint().unwrap(),
        table: binding.table,
        destination: binding.destination.clone(),
        incarnation: binding.incarnation,
        contract: binding.contract().unwrap(),
    })
}
fn session(bucket: u32) -> String {
    (0..1000)
        .map(|i| format!("session-{i}"))
        .find(|s| layout().session_bucket("tenant", "rollouts", s).unwrap() == bucket)
        .unwrap()
}
fn batch(ids: &[&str], sessions: &[&str]) -> Result<RecordBatch> {
    let schema = Arc::new(rollout_schema());
    let n = ids.len();
    ensure!(n == sessions.len(), "test row mismatch");
    let payload = vec![0x5a; 16 << 10];
    let columns = schema
        .fields()
        .iter()
        .map(|field| -> Result<ArrayRef> {
            Ok(match field.name().as_str() {
                "id" => Arc::new(StringArray::from(ids.to_vec())),
                "rollout_id" => Arc::new(StringArray::from(sessions.to_vec())),
                "problem_id" => Arc::new(StringArray::from(vec!["problem"; n])),
                "sequence_order" => Arc::new(Int32Array::from_iter_values(0..n as i32)),
                "role" => Arc::new(DictionaryArray::<Int8Type>::try_new(
                    Int8Array::from(vec![0; n]),
                    Arc::new(StringArray::from(vec!["assistant"])),
                )?),
                "created_at" => Arc::new(TimestampMicrosecondArray::from(vec![1234; n])),
                "content_type" => Arc::new(StringArray::from(vec!["binary"; n])),
                "binary_payload" => Arc::new(LargeBinaryArray::from_iter_values(
                    (0..n).map(|_| payload.as_slice()),
                )),
                _ => new_null_array(field.data_type(), n),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(RecordBatch::try_new(schema, columns)?)
}
fn ipc_file(batches: &[RecordBatch]) -> Result<Bytes> {
    let mut body = Vec::new();
    let mut writer = FileWriter::try_new(&mut body, &rollout_schema())?;
    for batch in batches {
        writer.write(batch)?;
    }
    writer.finish()?;
    drop(writer);
    Ok(Bytes::from(body))
}
async fn prepare_local(body: Bytes, binding: &TableBinding, bucket: u32) -> Result<PreparedAppend> {
    let scope = scope(binding);
    let budget = MergeMemoryBudget::new(8 << 20);
    let reservation = budget
        .reserve(workspace(&scope.contract, body.len())?)
        .await;
    let producer = Producer {
        scope: scope.clone(),
        open: ProducerOpen {
            bucket,
            slot: 0,
            expected_generation: 0,
            operation: Uuid::from_u128(5),
        },
        id: producer::producer_id(0, 1),
        generation: 1,
    };
    prepare(
        Upload {
            scope,
            bytes: body.len(),
            reservation,
        },
        body,
        producer,
        1,
        Arc::new(rollout_schema()),
        &layout(),
    )
}

#[tokio::test]
async fn validates_whole_atomic_batch_and_preserves_original_buffer() -> Result<()> {
    let s = session(0);
    let body = ipc_file(&[batch(&["one", "two"], &[&s, &s])?])?;
    let prepared = prepare_local(body.clone(), &binding(), 0).await?;
    assert_eq!(prepared.body().as_ptr(), body.as_ptr());
    assert_eq!(prepared.rows(), 2);
    // Buckets 0 and 2 deliberately share the same physical group.
    let other = session(2);
    assert!(prepare_local(
        ipc_file(&[batch(&["one", "two"], &[&s, &other])?])?,
        &binding(),
        0
    )
    .await
    .is_err());
    assert!(prepare_local(
        ipc_file(&[batch(&["one"], &[&s])?, batch(&["one"], &[&s])?])?,
        &binding(),
        0
    )
    .await
    .is_err());
    assert!(
        prepare_local(ipc_file(&[batch(&[""], &[&s])?])?, &binding(), 0)
            .await
            .is_err()
    );
    assert!(
        prepare_local(ipc_file(&[batch(&["one"], &[""])?])?, &binding(), 0)
            .await
            .is_err()
    );
    let mut limited = binding();
    limited.max_rows = 1;
    assert!(prepare_local(
        ipc_file(&[batch(&["one"], &[&s])?, batch(&["two"], &[&s])?])?,
        &limited,
        0
    )
    .await
    .is_err());
    assert!(prepare_local(ipc_file(&[batch(&[], &[])?])?, &binding(), 0)
        .await
        .is_err());
    assert!(
        prepare_local(Bytes::from_static(b"invalid IPC"), &binding(), 0)
            .await
            .is_err()
    );
    Ok(())
}

#[cfg(feature = "materializer")]
async fn bootstrap(
    c: &network::Cluster,
    path: &std::path::Path,
    binding: &TableBinding,
) -> Result<()> {
    let storage = crate::materializer::Storage::resolve(
        path.to_str().context("test URI")?.into(),
        Default::default(),
        Arc::new(lance::session::Session::default()),
    )
    .await?;
    crate::bootstrap::provision(
        c.client.clone(),
        layout(),
        binding.clone(),
        storage,
        bootstrap::identity(),
    )
    .await?;
    Ok(())
}

#[cfg(not(feature = "materializer"))]
async fn bootstrap(
    c: &network::Cluster,
    path: &std::path::Path,
    binding: &TableBinding,
) -> Result<()> {
    let dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch(&[], &[])?)], Arc::new(rollout_schema())),
        path.to_str().context("test path")?,
        Some(WriteParams {
            commit_handler: Some(Arc::new(ConditionalPutCommitHandler)),
            ..Default::default()
        }),
    )
    .await?;
    let response = c
        .client
        .request_routed(Request::Commit(Box::new(Command::Acquire(p::Acquire {
            table: binding.table,
            destination: binding.destination.clone(),
            ingest: binding.contract()?,
            owner: Uuid::new_v4(),
            request: Uuid::new_v4(),
            expected_epoch: 0,
        }))))
        .await?;
    let Reply::Commit(Outcome::Committed(Applied::Publisher(p::Reply::Acquired(authority)))) =
        response.reply
    else {
        bail!("publisher allocation failed");
    };
    let location = dataset.manifest_location();
    let response = c
        .client
        .request_routed(Request::Commit(Box::new(Command::Bootstrap(
            p::BootstrapManifest {
                authority,
                incarnation: binding.incarnation,
                request: Uuid::new_v4(),
                version: 1,
                object: p::ManifestObject {
                    path: format!(
                        "_versions/{}",
                        location.path.filename().context("manifest name")?
                    ),
                    bytes: location.size.context("manifest size")?,
                    v2_names: location.naming_scheme == ManifestNamingScheme::V2,
                },
                transaction: Uuid::parse_str(
                    &dataset
                        .read_transaction_by_version(1)
                        .await?
                        .context("genesis transaction")?
                        .uuid,
                )?,
            },
        ))))
        .await?;
    ensure!(
        matches!(
            response.reply,
            Reply::Commit(Outcome::Committed(Applied::Publisher(
                p::Reply::ManifestReserved(_)
            )))
        ),
        "bootstrap not confirmed"
    );
    Ok(())
}
async fn connect(c: &network::Cluster, binding: TableBinding) -> Result<Ingress> {
    Ingress::connect(
        c.client.clone(),
        layout(),
        binding,
        MergeMemoryBudget::new(8 << 20),
    )
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn durable_core_ack_receipt_and_exact_retry_survive_three_replica_restart() -> Result<()> {
    let root = tempfile::tempdir()?;
    let c = network::Cluster::open(root.path(), layout()).await?;
    let b = binding();
    bootstrap(&c, &root.path().join("rollouts.lance"), &b).await?;
    let ingress = connect(&c, b.clone()).await?;
    let mut wrong = b.clone();
    wrong.incarnation = Uuid::new_v4();
    assert!(connect(&c, wrong).await.is_err());
    let mut wrong = b.clone();
    wrong.destination.dataset = "other.lance".into();
    assert!(connect(&c, wrong).await.is_err());
    let mut wrong = b.clone();
    wrong.max_rows += 1;
    assert!(connect(&c, wrong).await.is_err());
    assert!(Ingress::connect(
        c.client.clone(),
        layout(),
        b.clone(),
        MergeMemoryBudget::new(1 << 20)
    )
    .await
    .is_err());
    assert!(ingress.reserve_upload(0).await.is_err());
    assert!(ingress
        .reserve_upload(1 + b.max_record_bytes as usize)
        .await
        .is_err());
    let full_a = ingress.reserve_upload(b.max_record_bytes as usize).await?;
    let full_b = ingress.reserve_upload(b.max_record_bytes as usize).await?;
    assert!(tokio::time::timeout(
        Duration::from_millis(30),
        ingress.reserve_upload(b.max_record_bytes as usize)
    )
    .await
    .is_err());
    drop(full_a);
    let admitted = tokio::time::timeout(
        Duration::from_secs(1),
        ingress.reserve_upload(b.max_record_bytes as usize),
    )
    .await??;
    drop((full_b, admitted));
    assert_eq!(ingress.budget.reserved(), 0);
    // Acquire before buffering; the same budget remains held through prepare
    // and explicit retries, and is released only when the prepared body drops.
    let open = ProducerOpen {
        bucket: 0,
        slot: 0,
        expected_generation: 0,
        operation: Uuid::new_v4(),
    };
    let producer = ingress.open_producer(open.clone()).await?;
    let s = session(0);
    let body = ipc_file(&[batch(&["one", "two"], &[&s, &s])?])?;
    let prepared = ingress
        .prepare(
            ingress.reserve_upload(body.len()).await?,
            body.clone(),
            &producer,
            1,
        )
        .await?;
    assert_eq!(prepared.body().as_ptr(), body.as_ptr());
    assert!(ingress.budget.reserved() >= body.len());
    let accepted = ingress.append(&prepared).await?;
    let retry = ingress.append(&prepared).await?;
    assert_eq!(retry.position(), accepted.position());
    assert_eq!(
        ingress.publication_status(&accepted).await?,
        PublicationStatus::Pending
    );
    assert!(ingress
        .wait_published(&accepted, Duration::from_millis(30))
        .await
        .is_err());
    let altered = ipc_file(&[batch(&["changed", "two"], &[&s, &s])?])?;
    let altered = ingress
        .prepare(
            ingress.reserve_upload(altered.len()).await?,
            altered,
            &producer,
            1,
        )
        .await?;
    let error = ingress.append(&altered).await.unwrap_err();
    assert!(matches!(
        error.downcast_ref::<AppendFailure>(),
        Some(AppendFailure::Sequence { .. })
    ));
    drop(altered);
    let keys = ReceiptKeys::new(1, [(1, [0x41; 32])])?;
    let token = keys.encode(&accepted)?;
    let restored = ingress.restore_receipt(&keys, &token)?;
    assert_eq!(restored.meta(), accepted.meta());
    let bad_keys = ReceiptKeys::new(1, [(1, [0x42; 32])])?;
    assert!(ingress.restore_receipt(&bad_keys, &token).is_err());
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
    let mut corrupt = URL_SAFE_NO_PAD.decode(&token)?;
    corrupt[10] ^= 1;
    assert!(ingress
        .restore_receipt(&keys, &URL_SAFE_NO_PAD.encode(corrupt))
        .is_err());
    assert!(ingress
        .restore_receipt(&keys, &"a".repeat(64 << 10))
        .is_err());
    let mut foreign = ingress.clone();
    Arc::make_mut(&mut foreign.scope).incarnation = Uuid::new_v4();
    assert!(foreign.restore_receipt(&keys, &token).is_err());
    let position = accepted.position().clone();
    let meta = accepted.meta().clone();
    drop(prepared);
    assert_eq!(ingress.budget.reserved(), 0);
    drop(ingress);
    c.shutdown().await?;

    let c = network::Cluster::open(root.path(), layout()).await?;
    let ingress = connect(&c, b).await?;
    let rotated = ReceiptKeys::new(2, [(1, [0x41; 32]), (2, [0x43; 32])])?;
    let restored = ingress.restore_receipt(&rotated, &token)?;
    assert_eq!(restored.position(), &position);
    assert_eq!(restored.meta(), &meta);
    let new_token = rotated.encode(&restored)?;
    assert!(ingress.restore_receipt(&keys, &new_token).is_err());
    assert!(ingress.restore_receipt(&rotated, &new_token).is_ok());
    let retired = ReceiptKeys::new(2, [(2, [0x43; 32])])?;
    assert!(ingress.restore_receipt(&retired, &token).is_err());
    let producer = ingress.open_producer(open).await?;
    let prepared = ingress
        .prepare(
            ingress.reserve_upload(body.len()).await?,
            body.clone(),
            &producer,
            1,
        )
        .await?;
    let retry = ingress.append(&prepared).await?;
    assert_eq!(retry.position(), &position);
    for node in 1..=3 {
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                let read = c
                    .client
                    .request(
                        node,
                        Request::Fetch {
                            group: position.group,
                            index: position.index,
                        },
                    )
                    .await?;
                if let Reply::Record(Some(found)) = &read.reply {
                    assert_eq!(found, &meta);
                    assert_eq!(read.body, body);
                    break Ok::<_, anyhow::Error>(());
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await??;
    }
    assert_eq!(
        ingress.publication_status(&restored).await?,
        PublicationStatus::Pending
    );
    ingress.close_producer(&producer, Some(&retry)).await?;
    // Closing fences new writes while preserving reconciliation of the last
    // accepted tuple. Otherwise a lost final ACK could never be reconciled.
    assert_eq!(ingress.append(&prepared).await?.position(), &position);
    let next = ingress
        .prepare(
            ingress.reserve_upload(body.len()).await?,
            body,
            &producer,
            2,
        )
        .await?;
    let error = ingress.append(&next).await.unwrap_err();
    assert!(matches!(
        error.downcast_ref::<AppendFailure>(),
        Some(AppendFailure::ProducerFenced)
    ));
    c.shutdown().await
}

#[cfg(all(feature = "service", unix))]
mod application;
