//! Real Lance files and six persistent WAL processes. Collection is disabled;
//! this qualifies descriptor recovery, not the external commit/GC fence.
use super::*;
#[path = "managed_manifest.rs"]
mod managed;
use arrow_array::{Int64Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lance::{
    dataset::{
        transaction::{Operation, Transaction},
        CommitBuilder, InsertBuilder, WriteMode, WriteParams,
    },
    Dataset,
};
use lance_context_wal::ingest_contract::{IngestContract, WritePolicy};
use lance_table::{
    format::{Fragment, IndexMetadata, Manifest, Transaction as ManifestTransaction},
    io::commit::{
        CommitError, CommitHandler, ConditionalPutCommitHandler, ManifestLocation,
        ManifestNamingScheme, ManifestWriter,
    },
};

fn payload(id: i64) -> String {
    format!("{id}:{}", "source-bytes".repeat(4096))
}
fn batch(ids: &[i64]) -> Result<RecordBatch> {
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("payload", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(Int64Array::from(ids.to_vec())),
            Arc::new(StringArray::from(
                ids.iter().map(|id| payload(*id)).collect::<Vec<_>>(),
            )),
        ],
    )?)
}
async fn verify(uri: &str, expected: &[i64]) -> Result<()> {
    let dataset = Dataset::open(uri).await?;
    verify_dataset(dataset, expected).await
}
async fn verify_dataset(dataset: Dataset, expected: &[i64]) -> Result<()> {
    let batches: Vec<_> = dataset
        .scan()
        .try_into_stream()
        .await?
        .try_collect()
        .await?;
    let mut rows = BTreeMap::new();
    for b in batches {
        let ids = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let values = b.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        for i in 0..b.num_rows() {
            assert!(rows
                .insert(ids.value(i), values.value(i).to_owned())
                .is_none());
        }
    }
    assert_eq!(
        rows,
        expected.iter().map(|id| (*id, payload(*id))).collect()
    );
    Ok(())
}

#[derive(Debug)]
struct FixedTarget(u64);
#[async_trait::async_trait]
#[allow(clippy::too_many_arguments)]
impl CommitHandler for FixedTarget {
    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<IndexMetadata>>,
        path: &object_store::path::Path,
        store: &lance_io::object_store::ObjectStore,
        writer: ManifestWriter,
        naming: ManifestNamingScheme,
        transaction: Option<ManifestTransaction>,
    ) -> std::result::Result<ManifestLocation, CommitError> {
        if manifest.version != self.0 {
            return Err(CommitError::CommitConflict);
        }
        ConditionalPutCommitHandler
            .commit(manifest, indices, path, store, writer, naming, transaction)
            .await
    }
}

async fn recovered_publication(occupied: bool, managed: Option<managed::Mode>) -> Result<()> {
    let key = ring::signature::Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new())
        .map_err(|_| anyhow::anyhow!("test key generation failed"))?;
    let layout = Layout {
        cluster: Uuid::from_u128(901),
        nodes: (1..=6)
            .map(|id| (id, format!("wal-{id}.example.invalid:7800")))
            .collect(),
        groups: BTreeMap::from([(10, vec![1, 2, 3]), (20, vec![4, 5, 6])]),
        buckets: vec![10, 20],
    };
    let slots = u32::from(managed == Some(managed::Mode::ProducerSlots));
    let mut c = Cluster::with_producer_slots(layout, Some(key.as_ref()), slots)?;
    let uri = c
        .root
        .path()
        .join("dataset.lance")
        .to_str()
        .unwrap()
        .to_owned();
    let genesis = batch(&[0])?;
    let base = Dataset::write(
        RecordBatchIterator::new(vec![Ok(genesis.clone())], genesis.schema()),
        &uri,
        Some(WriteParams {
            commit_handler: Some(Arc::new(ConditionalPutCommitHandler)),
            ..Default::default()
        }),
    )
    .await?;
    let catalog = c.elect(1, 10, &[1, 2, 3])?;
    let source = c.elect(4, 20, &[4, 5, 6])?;
    let rows = batch(&[1, 2, 3, 4])?;
    let mut wire = Vec::new();
    {
        let mut encoder = arrow_ipc::writer::FileWriter::try_new(&mut wire, &rows.schema())?;
        encoder.write(&rows)?;
        encoder.finish()?;
    }
    let table = Uuid::from_u128(902);
    let producer = if slots > 0 {
        managed::producer_lifecycle::open(&mut c, source, table)?
    } else {
        Uuid::from_u128(903)
    };
    let input = Batch::new(table, 1, producer, 1, Bytes::from(wire));
    c.command(source, Request::Propose(20, 2000, input.clone()))?;
    c.pump(&[4, 5, 6])?;
    let Some(Outcome::Committed(Applied::Accepted {
        index: source_index,
    })) = c.outcome(2000)
    else {
        anyhow::bail!("WAL append missing");
    };
    if slots > 0 {
        managed::producer_lifecycle::close_before_publication(
            &mut c,
            source,
            &input,
            source_index,
        )?;
    }
    let mut acquire = Acquire {
        table,
        destination: Destination {
            namespace: Uuid::from_u128(904),
            dataset: "tables/recovered".into(),
        },
        ingest: IngestContract {
            tenant: "test".into(),
            qualified_table: "recovered".into(),
            schema: lance_context_wal::ingest_contract::arrow_schema_fingerprint(
                rows.schema().as_ref(),
            )?,
            policy: WritePolicy::ImmutableUniqueIds,
            max_record_bytes: 1024 * 1024,
            max_rows: 1024,
        },
        owner: Uuid::from_u128(905),
        request: Uuid::from_u128(906),
        expected_epoch: 0,
    };
    c.command(catalog, Request::Acquire(2001, acquire.clone()))?;
    c.pump(&[1, 2, 3])?;
    let Some(Outcome::Committed(Applied::Publisher(PublisherReply::Acquired(authority)))) =
        c.outcome(2001)
    else {
        anyhow::bail!("authority missing");
    };
    let bootstrap = if managed.is_some() {
        Some(managed::bootstrap(&mut c, catalog, &base, &authority).await?)
    } else {
        None
    };
    let transaction = Uuid::from_u128(907);
    c.command(
        catalog,
        Request::Plan(
            2002,
            PlanRequest {
                authority: authority.clone(),
                plan: Plan {
                    transaction,
                    read_version: 1,
                    next_group: 10,
                    sources: BTreeMap::from([(
                        20,
                        SourceRange {
                            after: 0,
                            through: source_index,
                            records: vec![(source_index, input.meta.clone())],
                        },
                    )]),
                },
            },
        ),
    )?;
    c.pump(&[1, 2, 3])?;
    let Some(Outcome::Committed(Applied::Publisher(PublisherReply::Planned { index: plan_index }))) =
        c.outcome(2002)
    else {
        anyhow::bail!("plan missing");
    };
    c.command(
        catalog,
        Request::Query(
            10,
            2003,
            Query::Pin {
                table,
                transaction,
                plan_index,
                source_group: 20,
            },
        ),
    )?;
    c.pump(&[1, 2, 3])?;
    let Some(ReadOutcome::Observed {
        value: Value::Pin(Some(permit)),
        ..
    }) = c.read_outcome(2003)
    else {
        anyhow::bail!("attestation missing");
    };
    c.command(source, Request::Pin(2004, *permit))?;
    c.pump(&[4, 5, 6])?;
    assert!(matches!(
        c.outcome(2004),
        Some(Outcome::Committed(Applied::Source(
            SourceReply::Pinned { .. }
        )))
    ));
    let Response::Record(Some(recovered_input)) = c
        .peers
        .get_mut(&source)
        .unwrap()
        .call(Request::Read(20, source_index))?
    else {
        anyhow::bail!("source payload missing");
    };
    assert_eq!(recovered_input, input);
    let decoded = lance_context_wal::ipc::decode(recovered_input.body, rows.schema())?;
    let staged = InsertBuilder::new(Arc::new(base))
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute_uncommitted(decoded)
        .await?;
    let Operation::Append { fragments } = staged.operation else {
        anyhow::bail!("not append fragments");
    };
    let descriptors = Descriptors::lance_fragments(Bytes::from(serde_json::to_vec(&fragments)?))?;
    c.command(
        catalog,
        Request::Prepare(
            2005,
            PrepareRequest {
                authority,
                transaction,
                plan_index,
                descriptors: descriptors.clone(),
            },
        ),
    )?;
    c.pump(&[1, 2, 3])?;
    let Some(Outcome::Committed(Applied::Publisher(PublisherReply::Prepared {
        index: prepared_index,
    }))) = c.outcome(2005)
    else {
        anyhow::bail!("preparation missing");
    };
    for (node, group) in [(catalog, 10), (source, 20)] {
        c.peers
            .get_mut(&node)
            .unwrap()
            .call(Request::PrepareCheckpoint(group, Uuid::new_v4()))?;
        c.command(node, Request::InstallCheckpoint)?;
        c.pump(&[1, 2, 3, 4, 5, 6])?;
    }
    // No publisher object or in-memory fragment vector supplies the retry.
    drop(fragments);
    c.queue.clear();
    c.peers.clear();
    for id in 1..=6 {
        c.start(id)?;
    }
    let catalog = c.elect(2, 10, &[1, 2, 3])?;
    acquire.owner = Uuid::from_u128(908);
    acquire.request = Uuid::from_u128(909);
    acquire.expected_epoch = 1;
    c.command(catalog, Request::Acquire(2010, acquire))?;
    c.pump(&[1, 2, 3])?;
    c.command(
        catalog,
        Request::Query(10, 2011, Query::Publisher { table }),
    )?;
    c.pump(&[1, 2, 3])?;
    let Some(ReadOutcome::Observed {
        value: Value::Publisher(Some(view)),
        ..
    }) = c.read_outcome(2011)
    else {
        anyhow::bail!("recovered plan missing");
    };
    let saved = view
        .progress
        .prepared
        .context("recovery lost descriptors")?;
    assert_eq!(saved.index, prepared_index);
    assert_eq!(saved.plan_index, plan_index);
    assert_eq!(saved.descriptors, descriptors);
    let base = Dataset::open(&uri).await?;
    if occupied {
        InsertBuilder::new(Arc::new(base.clone()))
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            })
            .execute(vec![batch(&[99])?])
            .await?;
    }
    saved.descriptors.validate(1024 * 1024)?;
    let decoded: Vec<Fragment> = serde_json::from_slice(&saved.descriptors.bytes)?;
    let mut tx = Transaction::new(1, Operation::Append { fragments: decoded }, None);
    tx.uuid = saved.transaction.to_string();
    if let Some(mode) = managed {
        return managed::publish_and_recover(
            c,
            managed::Attempt {
                uri,
                base,
                tx,
                prepared: saved,
                authority: view.authority,
                bootstrap: bootstrap.context("missing managed bootstrap")?,
                input,
            },
            mode,
        )
        .await;
    }
    let result = CommitBuilder::new(Arc::new(base))
        .with_commit_handler(Arc::new(FixedTarget(2)))
        .with_max_retries(0)
        .with_skip_auto_cleanup(true)
        .execute(tx)
        .await;
    if occupied {
        assert!(result.is_err(), "changed target must not rebase this plan");
    } else {
        result?;
    }
    // Treat the response as lost: recover again and inspect the actual immutable
    // target. No catalog resolution or source release is implemented here.
    c.queue.clear();
    c.peers.clear();
    for id in 1..=6 {
        c.start(id)?;
    }
    let latest = Dataset::open(&uri).await?;
    assert_eq!(latest.version().version, 2);
    let committed = latest
        .read_transaction_by_version(2)
        .await?
        .context("missing actual transaction")?;
    assert_eq!(committed.uuid == transaction.to_string(), !occupied);
    verify(&uri, if occupied { &[0, 99] } else { &[0, 1, 2, 3, 4] }).await?;
    for id in 1..=3 {
        let Response::Publisher(Some(view)) = c
            .peers
            .get_mut(&id)
            .unwrap()
            .call(Request::Publisher(table))?
        else {
            anyhow::bail!("lost pending plan");
        };
        assert!(view.progress.checkpoint.is_empty());
        assert_eq!(view.progress.prepared.unwrap().descriptors, descriptors);
    }
    for id in 4..=6 {
        verify_records(
            c.peers.get_mut(&id).unwrap(),
            20,
            std::slice::from_ref(&input),
        )?;
    }
    Ok(())
}

#[tokio::test]
async fn recovered_descriptors_publish_exact_lance_payload_after_all_process_restart() -> Result<()>
{
    recovered_publication(false, None).await
}

#[tokio::test]
async fn recovered_descriptors_do_not_retarget_an_occupied_lance_version() -> Result<()> {
    recovered_publication(true, None).await
}

#[tokio::test]
async fn catalog_manifest_authority_recovers_unique_lance_path_after_full_restart() -> Result<()> {
    recovered_publication(false, Some(managed::Mode::Normal)).await
}

#[tokio::test]
async fn catalog_manifest_authority_resolves_lost_commit_reply_after_full_restart() -> Result<()> {
    recovered_publication(false, Some(managed::Mode::LostReply)).await
}

#[tokio::test]
async fn catalog_manifest_authority_recovers_after_lost_reply_and_unavailable_observation(
) -> Result<()> {
    recovered_publication(false, Some(managed::Mode::LostReplyAndObservation)).await
}

#[tokio::test]
async fn catalog_rejects_old_owner_paused_after_real_manifest_upload() -> Result<()> {
    recovered_publication(false, Some(managed::Mode::Takeover)).await
}

#[tokio::test]
async fn catalog_manifest_retention_preserves_real_lance_after_history_limit_and_restart(
) -> Result<()> {
    recovered_publication(false, Some(managed::Mode::Retention)).await
}

#[tokio::test]
async fn producer_slot_reuse_preserves_real_lance_and_fences_old_writes_after_restart() -> Result<()>
{
    recovered_publication(false, Some(managed::Mode::ProducerSlots)).await
}
