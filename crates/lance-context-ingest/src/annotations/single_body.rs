//! Full-schema immutable first-create ingestion through the existing shard
//! runtime. One native dataset holds body rows, covered WAL cut, and outcomes.
use super::*;
use crate::turn_ingestion::{TurnIngestionSpec, TurnLookup};
use lance_context_core::annotation_staging::{ImmutableBase, ImmutableRegistration};

const KEY: &str = "lance-context.immutable-turn-body.v1";

pub(super) fn budget_parts(limit: usize) -> Result<(usize, usize)> {
    const MIB: usize = 1 << 20;
    ensure!(
        limit >= 4 * MIB,
        "single body apply needs at least four MiB"
    );
    let lookup = (limit / (4 * MIB)).max(1) * MIB;
    Ok((limit - lookup, lookup))
}

mod read;
pub use read::Body;

#[allow(clippy::too_many_arguments)]
pub async fn connect_ingress(
    client: Arc<Client>,
    layout: Arc<Layout>,
    binding: TableBinding,
    limits: &Limits,
    budget: Arc<MergeMemoryBudget>,
    shard: Shard,
    spec: &TurnIngestionSpec,
) -> Result<Ingress> {
    layout.group_for_bucket(shard.bucket)?;
    let contract = contract(&binding, spec, limits, shard)?;
    Ingress::connect_with_schema(
        client,
        layout,
        binding,
        budget,
        contract,
        wire_schema(spec, limits, shard)?,
    )
    .await
}

pub(crate) fn is_schema(schema: &arrow_schema::Schema) -> bool {
    schema.metadata().get(KEY).map(String::as_str) == Some("1")
}

/// Full original columns with the durable identity specification in metadata.
/// Use this schema when provisioning a new single-body dataset.
pub fn storage_schema(spec: &TurnIngestionSpec) -> Result<SchemaRef> {
    let bound = spec.bind_schema(spec.body_schema())?;
    ensure!(
        !bound.metadata().contains_key(KEY),
        "reserved immutable body metadata"
    );
    let mut metadata = bound.metadata().clone();
    metadata.insert(KEY.into(), "1".into());
    Ok(Arc::new(arrow_schema::Schema::new_with_metadata(
        bound.fields().clone(),
        metadata,
    )))
}

pub fn wire_schema(spec: &TurnIngestionSpec, limits: &Limits, shard: Shard) -> Result<SchemaRef> {
    limits.validate()?;
    let body = storage_schema(spec)?;
    let mut metadata = body.metadata().clone();
    for (key, value) in wire::schema(limits).metadata() {
        ensure!(
            metadata.insert(key.clone(), value.clone()).is_none(),
            "reserved ingestion metadata"
        );
    }
    shard.bind_schema(Arc::new(arrow_schema::Schema::new_with_metadata(
        body.fields().clone(),
        metadata,
    )))
}

pub fn contract(
    binding: &TableBinding,
    spec: &TurnIngestionSpec,
    limits: &Limits,
    shard: Shard,
) -> Result<IngestContract> {
    Ok(IngestContract {
        policy: WritePolicy::TurnIngestion,
        schema: arrow_schema_fingerprint(wire_schema(spec, limits, shard)?.as_ref())?,
        ..binding.contract()?
    })
}

/// Attach the bound transport metadata while sharing every original array.
/// The service still validates all keys and their shard before durable append.
pub fn batch(
    spec: &TurnIngestionSpec,
    limits: &Limits,
    shard: Shard,
    body: &RecordBatch,
) -> Result<RecordBatch> {
    ensure!(
        body.schema() == *spec.body_schema(),
        "foreign full body input schema"
    );
    Ok(RecordBatch::try_new(
        wire_schema(spec, limits, shard)?,
        body.columns().to_vec(),
    )?)
}

pub(super) fn decode(
    body: Bytes,
    schema: SchemaRef,
    maximum: u32,
    key_bytes: usize,
) -> Result<mixed::Decoded> {
    ensure!(is_schema(&schema), "not an immutable body transport");
    let spec =
        TurnIngestionSpec::from_schema(&schema)?.context("body identity specification missing")?;
    let decoded = ipc::decode(body, schema)?;
    let count = decoded
        .iter()
        .try_fold(0usize, |n, batch| n.checked_add(batch.num_rows()))
        .context("body row count overflow")?;
    ensure!(
        count > 0 && count <= maximum as usize,
        "invalid body batch rows"
    );
    let stored = storage_schema(&spec)?;
    let mut rows = Vec::with_capacity(count);
    let mut bodies = Vec::with_capacity(decoded.len());
    for batch in decoded {
        for (field, array) in spec.body_schema().fields().iter().zip(batch.columns()) {
            array.to_data().validate_full()?;
            ensure!(
                field.is_nullable() || array.logical_null_count() == 0,
                "null required body field"
            );
        }
        let original = RecordBatch::try_new(spec.body_schema().clone(), batch.columns().to_vec())?;
        for row in 0..batch.num_rows() {
            let turn = spec.row_key(&original, row, key_bytes)?;
            let rollout_id = spec.routing_key(&turn)?;
            rows.push(mixed::Created {
                turn,
                rollout_id,
                annotation: TurnAnnotation::default(),
            });
        }
        bodies.push(RecordBatch::try_new(
            stored.clone(),
            batch.columns().to_vec(),
        )?);
    }
    Ok(mixed::Decoded::Creates { rows, bodies })
}

/// Exact selected full-body manifest and the WAL prefix it covers. This is
/// selection evidence, not a file-retention lease or a source checkpoint ACK.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelectedBody {
    pub cluster: Uuid,
    pub layout: [u8; 32],
    pub table: Uuid,
    pub incarnation: Uuid,
    pub schema: [u8; 32],
    pub source: StagingScope,
    pub through: u64,
    pub uri: String,
    pub manifest: lance_context_wal::replication::publication::ManifestRecord,
}

pub(super) struct Lookup {
    pub base: ImmutableBase,
    pub keys: TurnLookup,
}

impl Service {
    pub(crate) fn is_single_body(&self) -> bool {
        is_schema(&self.ingress.schema)
    }

    pub(super) async fn lookup_bodies(&self, keys: &[&str]) -> Result<Option<Lookup>> {
        self.shared
            .phases
            .measure(Phase::AnnotationRegistrationLookup, async {
                let spec = self
                    .ingress
                    .turn_spec
                    .as_ref()
                    .context("missing body spec")?;
                let Some(mut prepared) = spec.prepare_lookup(
                    keys,
                    self.lookup_budget
                        .as_ref()
                        .context("missing body lookup budget")?,
                )?
                else {
                    return Ok(None);
                };
                let Some(base) = self
                    .shared
                    .staging
                    .try_capture_immutable_base_with_staged(&mut prepared.keys)
                    .await?
                else {
                    return Ok(None);
                };
                let keys = match spec.lookup_prepared(base.dataset().clone(), prepared).await {
                    Ok(keys) => keys,
                    Err(error)
                        if error.chain().any(|cause| {
                            cause
                                .downcast_ref::<lance::Error>()
                                .is_some_and(lance::Error::is_backpressure)
                        }) =>
                    {
                        // Only the read-only historical lookup is retried.
                        // Its partial results, native scan and reader handle
                        // are dropped; Driver retains the committed request
                        // without advancing apply or treating keys as absent.
                        // Existing ApplyMemory pressure lets the independent
                        // publisher drain an already-applied prefix. The host
                        // must still leave a release path for native capacity.
                        return Ok(None);
                    }
                    Err(error) => return Err(error),
                };
                Ok(Some(Lookup { base, keys }))
            })
            .await
            .context("immutable body historical lookup failed")
    }

    pub(super) async fn apply_bodies(
        &self,
        after: u64,
        pending: &Pending,
        lookup: &Lookup,
    ) -> Result<Option<ApplyOutcome>> {
        let mixed::Decoded::Creates { rows, .. } = &pending.decoded else {
            bail!("immutable body cannot apply patches")
        };
        ensure!(
            lookup.keys.matches(lookup.base.dataset()),
            "foreign body lookup"
        );
        let rows = rows
            .iter()
            .map(|row| {
                Ok(ImmutableRegistration {
                    key: row.turn.as_str(),
                    exists_in_base: lookup.keys.get(&row.turn).context("unqueried body key")?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(self
            .shared
            .staging
            .register_immutable_committed(after, pending.index, &rows, &lookup.base)
            .await?)
    }

    pub async fn selected_full_body(&self, deadline: Duration) -> Result<SelectedBody> {
        ensure!(self.is_single_body(), "not a single body service");
        tokio::time::timeout(deadline, async {
            let (base, snapshot) =
                checkpoint::open(&self.ingress, &self.shared.narrow, None, self.bucket()).await?;
            let query::Value::Publisher(Some(view)) = crate::materializer::query(
                &self.ingress,
                self.ingress.client.catalog_group(),
                query::Query::Publisher {
                    table: self.ingress.scope.table,
                },
            )
            .await?
            else {
                bail!("body catalog unavailable")
            };
            ensure!(
                view.destination == self.ingress.scope.destination
                    && view.ingest == self.ingress.scope.contract
                    && view
                        .manifests
                        .as_ref()
                        .is_some_and(|m| m.incarnation == self.ingress.scope.incarnation),
                "foreign selected body binding"
            );
            let manifest = view
                .manifests
                .as_ref()
                .and_then(|m| m.versions.get(&base.version_id()))
                .context("selected full body no longer retained")?
                .clone();
            ensure!(
                manifest.transaction.to_string()
                    == base
                        .read_transaction()
                        .await?
                        .context("missing selected transaction")?
                        .uuid,
                "selected full body manifest differs"
            );
            Ok(SelectedBody {
                cluster: self.ingress.scope.cluster,
                layout: self.ingress.scope.layout,
                table: self.ingress.scope.table,
                incarnation: self.ingress.scope.incarnation,
                schema: self.ingress.scope.contract.schema,
                source: StagingScope {
                    table_instance: self.ingress.scope.incarnation.to_string(),
                    source_group: self.group()?,
                    shard: self.bucket(),
                },
                through: snapshot.through,
                uri: base.uri().to_owned(),
                manifest: manifest.as_ref().clone(),
            })
        })
        .await
        .context("selected full body read deadline exceeded")?
    }
}
