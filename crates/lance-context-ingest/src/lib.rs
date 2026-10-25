//! Durable Arrow rollout ingestion over the independent WAL's authenticated RPCs.
//!
//! Use a newly bound immutable-ID rollout table. Resolve/authorize its namespace,
//! table generation and manifest incarnation before connecting. This does not
//! migrate or replace legacy merge/upsert tables, or authenticate an HTTP caller.
//! One atomic append must fit one virtual bucket; reject mixed buckets before ACK.
use anyhow::{bail, ensure, Context, Result};
use arrow_array::{Array, RecordBatch, StringArray};
use arrow_schema::SchemaRef;
use bytes::Bytes;
use lance_context_core::{rollout_schema, MergeMemoryBudget, MergeReservation};
use lance_context_wal::{
    ingest_contract::{arrow_schema_fingerprint, IngestContract, WritePolicy},
    ipc,
    placement::{Layout, Receipt},
    record::{Batch, BatchMeta, RecordKind},
    replication::{
        producer,
        publication::Destination,
        query,
        runtime::{
            client::{Client, Reply, Request},
            Command,
        },
        Applied, Outcome,
    },
};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, sync::Arc, time::Duration};
use uuid::Uuid;

#[cfg(feature = "materializer")]
pub mod annotation_publication;
#[cfg(feature = "materializer")]
pub mod annotations;
#[cfg(feature = "service")]
pub mod application;
#[cfg(feature = "materializer")]
pub mod bootstrap;
#[cfg(feature = "http")]
pub mod http;
#[cfg(feature = "materializer")]
pub mod materializer;
#[cfg(feature = "service")]
pub mod preparation;
#[cfg(feature = "materializer")]
pub mod reader;
pub mod receipt;
#[cfg(feature = "worker")]
pub mod worker;

/// Resolved by the application's authorized catalog, not an uploaded URI/UUID.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TableBinding {
    pub table: Uuid,
    pub destination: Destination,
    pub incarnation: Uuid,
    pub tenant: String,
    pub qualified_table: String,
    pub max_record_bytes: u32,
    pub max_rows: u32,
}
impl TableBinding {
    pub fn contract(&self) -> Result<IngestContract> {
        let value = IngestContract {
            tenant: self.tenant.clone(),
            qualified_table: self.qualified_table.clone(),
            schema: arrow_schema_fingerprint(&rollout_schema())?,
            policy: WritePolicy::ImmutableUniqueIds,
            max_record_bytes: self.max_record_bytes,
            max_rows: self.max_rows,
        };
        value.validate()?;
        Ok(value)
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Scope {
    cluster: Uuid,
    layout: [u8; 32],
    table: Uuid,
    destination: Destination,
    incarnation: Uuid,
    contract: IngestContract,
}
#[derive(Clone)]
pub struct Ingress {
    client: Arc<Client>,
    layout: Arc<Layout>,
    scope: Arc<Scope>,
    schema: SchemaRef,
    budget: Arc<MergeMemoryBudget>,
    #[cfg(feature = "materializer")]
    session_read_concurrency: usize,
    #[cfg(feature = "materializer")]
    session_scan_batch_rows: usize,
    #[cfg(feature = "materializer")]
    session_scan_fragment_readahead: usize,
}
/// Held before body buffering; retained by preparation and every ambiguous retry.
pub struct Upload {
    scope: Arc<Scope>,
    bytes: usize,
    reservation: MergeReservation,
}
/// Clone only the opaque Bytes internally; do not re-encode an ambiguous retry.
pub struct PreparedAppend {
    scope: Arc<Scope>,
    batch: Batch,
    rows: u32,
    _reservation: MergeReservation,
}
impl PreparedAppend {
    pub fn meta(&self) -> &BatchMeta {
        &self.batch.meta
    }
    pub fn rows(&self) -> u32 {
        self.rows
    }
    pub fn body(&self) -> &Bytes {
        &self.batch.body
    }
}
/// Constructor and receipt fields are private: a bare index is not membership.
#[derive(Clone, Debug)]
pub struct AcceptedAppend {
    scope: Arc<Scope>,
    position: Receipt,
    meta: BatchMeta,
    rows: u32,
}
impl AcceptedAppend {
    pub fn position(&self) -> &Receipt {
        &self.position
    }
    pub fn meta(&self) -> &BatchMeta {
        &self.meta
    }
    pub fn rows(&self) -> u32 {
        self.rows
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProducerOpen {
    /// Persist the complete tuple before the first open RPC; retry it unchanged.
    pub bucket: u32,
    pub slot: u32,
    pub expected_generation: u64,
    pub operation: Uuid,
}
#[derive(Clone, Debug)]
pub struct Producer {
    scope: Arc<Scope>,
    open: ProducerOpen,
    id: Uuid,
    generation: u64,
}
impl Producer {
    pub fn id(&self) -> Uuid {
        self.id
    }
    pub fn bucket(&self) -> u32 {
        self.open.bucket
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppendFailure {
    Sequence { expected: u64 },
    Capacity,
    ProducerFenced,
    Unknown,
}
impl std::fmt::Display for AppendFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sequence { expected } => {
                write!(f, "producer sequence rejected; expected {expected}")
            }
            Self::Capacity => f.write_str("WAL capacity exhausted"),
            Self::ProducerFenced => f.write_str("producer fenced"),
            Self::Unknown => {
                f.write_str("append outcome unknown; retain exact producer sequence and bytes")
            }
        }
    }
}
impl std::error::Error for AppendFailure {}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PublicationStatus {
    Pending,
    Published { minimum_read_version: u64 },
}

impl Ingress {
    pub async fn connect(
        client: Arc<Client>,
        layout: Arc<Layout>,
        binding: TableBinding,
        budget: Arc<MergeMemoryBudget>,
    ) -> Result<Self> {
        let contract = binding.contract()?;
        Self::connect_with_schema(
            client,
            layout,
            binding,
            budget,
            contract,
            Arc::new(rollout_schema()),
        )
        .await
    }

    async fn connect_with_schema(
        client: Arc<Client>,
        layout: Arc<Layout>,
        binding: TableBinding,
        budget: Arc<MergeMemoryBudget>,
        contract: IngestContract,
        schema: SchemaRef,
    ) -> Result<Self> {
        let fingerprint = layout.fingerprint()?;
        ensure!(
            client.identity() == (layout.cluster, fingerprint),
            "ingress and client layout differ"
        );
        ensure!(
            !binding.table.is_nil() && !binding.incarnation.is_nil() && client.producer_slots() > 0,
            "ingress requires a generation and bounded producer slots"
        );
        contract.validate()?;
        ensure!(
            contract.max_record_bytes as usize <= client.max_record_bytes(),
            "contract exceeds WAL record capacity"
        );
        ensure!(
            workspace(&contract, contract.max_record_bytes as usize)? <= budget.limit(),
            "contract exceeds shared ingress workspace"
        );
        let value = Self {
            client,
            layout: layout.clone(),
            scope: Arc::new(Scope {
                cluster: layout.cluster,
                layout: fingerprint,
                table: binding.table,
                destination: binding.destination,
                incarnation: binding.incarnation,
                contract,
            }),
            schema,
            budget,
            #[cfg(feature = "materializer")]
            session_read_concurrency: reader::default_concurrency(),
            #[cfg(feature = "materializer")]
            session_scan_batch_rows: reader::default_scan_batch_rows(),
            #[cfg(feature = "materializer")]
            session_scan_fragment_readahead: reader::default_scan_fragment_readahead(),
        };
        value.visibility().await?;
        Ok(value)
    }
    fn group(&self, bucket: u32) -> Result<u64> {
        self.layout.group_for_bucket(bucket)
    }
    pub fn bucket_for_session(&self, session: &str) -> Result<u32> {
        ensure!(!session.is_empty(), "empty session key");
        self.layout.session_bucket(
            &self.scope.contract.tenant,
            &self.scope.contract.qualified_table,
            session,
        )
    }
    async fn visibility(&self) -> Result<Box<query::Visibility>> {
        let group = self.client.catalog_group();
        let response = self
            .client
            .request_routed(Request::Query {
                group,
                query: query::Query::Visibility {
                    table: self.scope.table,
                },
            })
            .await?;
        let Reply::Query(query::Outcome::Observed {
            cut,
            value: query::Value::Visibility(Some(view)),
        }) = response.reply
        else {
            bail!("table binding unavailable at quorum");
        };
        ensure!(
            cut.cluster == self.scope.cluster
                && cut.layout == self.scope.layout
                && cut.group == group
                && cut.term > 0
                && cut.applied >= cut.quorum_index,
            "foreign visibility quorum scope"
        );
        ensure!(
            view.destination == self.scope.destination
                && view.ingest == self.scope.contract
                && view.incarnation == Some(self.scope.incarnation),
            "durable table binding differs from resolved rollout generation"
        );
        Ok(view)
    }
    pub async fn open_producer(&self, open: ProducerOpen) -> Result<Producer> {
        ensure!(
            !open.operation.is_nil()
                && open.slot < self.client.producer_slots()
                && open.expected_generation < u64::MAX,
            "invalid producer open identity"
        );
        let response = self
            .client
            .request_routed(Request::Commit(Box::new(Command::Producer(
                producer::Command {
                    cluster: self.scope.cluster,
                    layout: self.scope.layout,
                    group: self.group(open.bucket)?,
                    table: self.scope.table,
                    slot: open.slot,
                    action: producer::Action::Open {
                        expected_generation: open.expected_generation,
                        operation: open.operation,
                        bucket: open.bucket,
                    },
                },
            ))))
            .await?;
        let Reply::Commit(Outcome::Committed(Applied::Producer(producer::Reply::Opened {
            producer: id,
            generation,
            ..
        }))) = response.reply
        else {
            bail!("producer allocation not confirmed; reconcile the original open tuple");
        };
        ensure!(
            generation == open.expected_generation + 1
                && id == producer::producer_id(open.slot, generation),
            "foreign producer allocation"
        );
        Ok(Producer {
            scope: self.scope.clone(),
            open,
            id,
            generation,
        })
    }
    pub async fn close_producer(
        &self,
        producer: &Producer,
        last: Option<&AcceptedAppend>,
    ) -> Result<()> {
        ensure!(*producer.scope == *self.scope, "foreign producer scope");
        let expected_last = last
            .map(|last| {
                self.check_receipt(last)?;
                ensure!(
                    last.meta.producer == producer.id && last.meta.bucket == producer.open.bucket,
                    "foreign producer receipt"
                );
                Ok::<_, anyhow::Error>(producer::Last {
                    sequence: last.meta.sequence,
                    index: last.position.index,
                })
            })
            .transpose()?;
        let response = self
            .client
            .request_routed(Request::Commit(Box::new(Command::Producer(
                producer::Command {
                    cluster: self.scope.cluster,
                    layout: self.scope.layout,
                    group: self.group(producer.open.bucket)?,
                    table: self.scope.table,
                    slot: producer.open.slot,
                    action: producer::Action::Close {
                        generation: producer.generation,
                        expected_last,
                    },
                },
            ))))
            .await?;
        ensure!(
            matches!(response.reply, Reply::Commit(Outcome::Committed(Applied::Producer(producer::Reply::Closed { last }))) if last == expected_last),
            "producer close not confirmed; preserve its outstanding state"
        );
        Ok(())
    }
    /// Reserve before reading/buffering the IPC body. The row allowance also
    /// budgets the identity/routing workspace; this is not an allocator quota.
    pub async fn reserve_upload(&self, bytes: usize) -> Result<Upload> {
        ensure!(
            bytes > 0 && bytes <= self.scope.contract.max_record_bytes as usize,
            "upload exceeds registered record limit"
        );
        let amount = workspace(&self.scope.contract, bytes)?;
        Ok(Upload {
            scope: self.scope.clone(),
            bytes,
            reservation: self.budget.reserve(amount).await,
        })
    }
    /// Arrow validation and payload hashing stay off the async I/O executor.
    /// Cancellation keeps the reservation with the running blocking operation.
    pub async fn prepare(
        &self,
        upload: Upload,
        body: Bytes,
        producer: &Producer,
        sequence: u64,
    ) -> Result<PreparedAppend> {
        ensure!(
            Arc::ptr_eq(&upload.scope, &self.scope) && *producer.scope == *self.scope,
            "foreign preparation scope"
        );
        ensure!(
            sequence > 0 && sequence < u64::MAX && body.len() == upload.bytes,
            "invalid sequence or upload length"
        );
        let schema = self.schema.clone();
        let layout = self.layout.clone();
        let producer = producer.clone();
        tokio::task::spawn_blocking(move || {
            prepare(upload, body, producer, sequence, schema, &layout)
        })
        .await
        .context("ingress preparation task failed")?
    }
    pub async fn append(&self, prepared: &PreparedAppend) -> Result<AcceptedAppend> {
        ensure!(
            *prepared.scope == *self.scope,
            "prepared append belongs to another table"
        );
        let group = self.group(prepared.batch.meta.bucket)?;
        let response = self
            .client
            .request_routed(Request::Append {
                group,
                batch: prepared.batch.clone(),
            })
            .await
            .map_err(|error| anyhow::Error::new(AppendFailure::Unknown).context(error))?;
        let index = match response.reply {
            Reply::Commit(Outcome::Committed(Applied::Accepted { index })) => index,
            Reply::Commit(Outcome::Committed(Applied::Rejected { expected_sequence })) => {
                return Err(AppendFailure::Sequence {
                    expected: expected_sequence,
                }
                .into())
            }
            Reply::Commit(Outcome::Committed(Applied::Capacity)) => {
                return Err(AppendFailure::Capacity.into())
            }
            Reply::Commit(Outcome::Committed(Applied::ProducerFenced)) => {
                return Err(AppendFailure::ProducerFenced.into())
            }
            _ => return Err(AppendFailure::Unknown.into()),
        };
        ensure!(index > 0 && index < u64::MAX, "invalid accepted index");
        Ok(AcceptedAppend {
            scope: self.scope.clone(),
            position: Receipt {
                cluster: self.scope.cluster,
                layout: self.scope.layout,
                group,
                index,
            },
            meta: prepared.batch.meta.clone(),
            rows: prepared.rows,
        })
    }
    fn check_receipt(&self, accepted: &AcceptedAppend) -> Result<()> {
        ensure!(
            *accepted.scope == *self.scope
                && accepted.position.cluster == self.scope.cluster
                && accepted.position.layout == self.scope.layout
                && accepted.position.group == self.group(accepted.meta.bucket)?
                && accepted.position.index > 0
                && accepted.position.index < u64::MAX
                && accepted.meta.stream == self.scope.table
                && accepted.meta.kind == RecordKind::Data
                && !accepted.meta.producer.is_nil()
                && accepted.meta.sequence > 0
                && accepted.meta.sequence < u64::MAX
                && accepted.rows > 0
                && accepted.rows <= self.scope.contract.max_rows,
            "foreign or invalid accepted receipt"
        );
        accepted.meta.check_body_length(
            usize::try_from(accepted.meta.body_bytes)?,
            self.scope.contract.max_record_bytes as usize,
        )?;
        Ok(())
    }
    pub async fn publication_status(&self, accepted: &AcceptedAppend) -> Result<PublicationStatus> {
        self.check_receipt(accepted)?;
        let view = self.visibility().await?;
        if view
            .checkpoint
            .get(&accepted.position.group)
            .copied()
            .unwrap_or(0)
            < accepted.position.index
        {
            return Ok(PublicationStatus::Pending);
        }
        ensure!(
            view.minimum_read_version > 0,
            "published prefix has no Lance version"
        );
        Ok(PublicationStatus::Published {
            minimum_read_version: view.minimum_read_version,
        })
    }
    pub async fn wait_published(&self, accepted: &AcceptedAppend, limit: Duration) -> Result<u64> {
        tokio::time::timeout(limit, async {
            loop {
                if let PublicationStatus::Published {
                    minimum_read_version,
                } = self.publication_status(accepted).await?
                {
                    return Ok(minimum_read_version);
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .context("publication deadline expired; durable append remains accepted")?
    }
}
fn workspace(contract: &IngestContract, bytes: usize) -> Result<usize> {
    bytes
        .checked_mul(3)
        .and_then(|v| v.checked_add(contract.max_rows as usize * 64))
        .context("ingress workspace overflow")
}
fn prepare(
    upload: Upload,
    body: Bytes,
    producer: Producer,
    sequence: u64,
    schema: SchemaRef,
    layout: &Layout,
) -> Result<PreparedAppend> {
    let batches = ipc::decode(body.clone(), schema)?;
    let rows = validate_rows(
        &batches,
        &upload.scope.contract,
        layout,
        producer.open.bucket,
    )?;
    let batch = Batch::new(
        upload.scope.table,
        producer.open.bucket,
        producer.id,
        sequence,
        body,
    );
    Ok(PreparedAppend {
        scope: upload.scope,
        batch,
        rows,
        _reservation: upload.reservation,
    })
}

fn validate_rows(
    batches: &[RecordBatch],
    contract: &IngestContract,
    layout: &Layout,
    expected_bucket: u32,
) -> Result<u32> {
    let mut ids = HashSet::new();
    let mut rows = 0u32;
    let mut previous_session = None;
    for batch in batches {
        ensure!(
            batch.num_rows() <= (contract.max_rows - rows) as usize,
            "upload exceeds registered row limit"
        );
        for (field, column) in batch.schema().fields().iter().zip(batch.columns()) {
            column.to_data().validate_full()?;
            ensure!(
                field.is_nullable() || column.logical_null_count() == 0,
                "null in a required rollout field"
            );
        }
        let sessions = batch
            .column_by_name("rollout_id")
            .and_then(|a| a.as_any().downcast_ref::<StringArray>())
            .context("missing rollout_id column")?;
        let keys = batch
            .column_by_name("id")
            .and_then(|a| a.as_any().downcast_ref::<StringArray>())
            .context("missing id column")?;
        for (session, id) in sessions.iter().zip(keys.iter()) {
            let session = session
                .filter(|s| !s.is_empty())
                .context("empty session key")?;
            let id = id.filter(|s| !s.is_empty()).context("empty row id")?;
            ensure!(ids.insert(id), "duplicate row id within atomic append");
            // Consecutive rows in a rollout share routing. Borrow the decoded
            // string and hash only when the session changes; no string copy.
            if previous_session != Some(session) {
                let bucket =
                    layout.session_bucket(&contract.tenant, &contract.qualified_table, session)?;
                ensure!(
                    bucket == expected_bucket,
                    "atomic append spans producer bucket"
                );
                previous_session = Some(session);
            }
            rows += 1;
        }
    }
    ensure!(rows > 0, "empty append");
    Ok(rows)
}

#[cfg(test)]
mod tests;
