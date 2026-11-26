//! A physical annotation table owns one existing WAL bucket. Its routing policy
//! is part of the durable Arrow schema fingerprint, not a local restart option.
use super::*;

const KEY: &str = "lance-context.annotation-turn-shard-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Shard {
    pub bucket: u32,
}

impl Shard {
    /// Session routing for a table explicitly provisioned with a generic spec.
    pub fn for_session(layout: &Layout, binding: &TableBinding, session: &str) -> Result<Self> {
        Self::for_turn(layout, binding, session)
    }

    pub fn turn_schema(
        &self,
        spec: &crate::turn_ingestion::TurnIngestionSpec,
        limits: &Limits,
    ) -> Result<SchemaRef> {
        self.bind_schema(mixed::wire::schema_for(spec, limits)?)
    }

    pub fn turn_contract(
        &self,
        binding: &TableBinding,
        spec: &crate::turn_ingestion::TurnIngestionSpec,
        limits: &Limits,
    ) -> Result<IngestContract> {
        Ok(IngestContract {
            policy: WritePolicy::TurnIngestion,
            schema: arrow_schema_fingerprint(self.turn_schema(spec, limits)?.as_ref())?,
            ..binding.contract()?
        })
    }
    /// Reuse the immutable layout's stable length-prefixed key routing. Turn IDs
    /// are sufficient for both point reads and updates; no directory lookup or
    /// body/session projection is required. Changing the layout is a migration.
    pub fn for_turn(layout: &Layout, binding: &TableBinding, turn: &str) -> Result<Self> {
        ensure!(!turn.is_empty(), "empty annotation turn");
        Ok(Self {
            bucket: layout.session_bucket(&binding.tenant, &binding.qualified_table, turn)?,
        })
    }

    pub fn wire_schema(&self, limits: &Limits, mixed: bool) -> Result<SchemaRef> {
        self.bind_schema(if mixed {
            mixed::wire::schema(limits)
        } else {
            wire::schema(limits)
        })
    }

    pub fn contract(
        &self,
        binding: &TableBinding,
        limits: &Limits,
        mixed: bool,
    ) -> Result<IngestContract> {
        let mut contract = if mixed {
            Service::mixed_contract(binding, limits)?
        } else {
            Service::contract(binding, limits)?
        };
        contract.schema = arrow_schema_fingerprint(self.wire_schema(limits, mixed)?.as_ref())?;
        Ok(contract)
    }

    /// Bind an already partitioned batch before encoding, retaining its Arrow
    /// arrays. This helper does not authorize or validate the row placement;
    /// the owner validates every key before submitting the original bytes.
    pub fn bind_batch(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        Ok(RecordBatch::try_new(
            self.bind_schema(batch.schema())?,
            batch.columns().to_vec(),
        )?)
    }

    pub(super) fn bind_schema(&self, schema: SchemaRef) -> Result<SchemaRef> {
        ensure!(self.bucket < 65_536, "annotation shard out of bounds");
        ensure!(
            schema
                .metadata()
                .get("lance-context.annotation-mutations")
                .map(String::as_str)
                == Some("1"),
            "annotation shard requires a mutation schema"
        );
        let value = self.bucket.to_string();
        ensure!(
            schema.metadata().get(KEY).is_none_or(|old| old == &value),
            "annotation batch belongs to another shard"
        );
        let mut metadata = schema.metadata().clone();
        metadata.insert(KEY.into(), value);
        Ok(Arc::new(arrow_schema::Schema::new_with_metadata(
            schema.fields().clone(),
            metadata,
        )))
    }

    pub(crate) fn from_ingress(ingress: &Ingress) -> Result<Option<Self>> {
        let Some(value) = ingress.schema.metadata().get(KEY) else {
            ensure!(
                ingress.layout.buckets.len() == 1,
                "annotation shard binding missing"
            );
            return Ok(None);
        };
        let bucket: u32 = value.parse().context("invalid annotation shard binding")?;
        ensure!(
            value == &bucket.to_string(),
            "noncanonical annotation shard binding"
        );
        ingress.group(bucket)?;
        Ok(Some(Self { bucket }))
    }

    #[cfg(feature = "service")]
    pub(crate) fn same_wire_family(a: &arrow_schema::Schema, b: &arrow_schema::Schema) -> bool {
        a.fields() == b.fields()
            && a.metadata().len() == b.metadata().len()
            && a.metadata()
                .iter()
                .all(|(key, value)| key == KEY || b.metadata().get(key) == Some(value))
    }

    pub(super) fn validate_turn(
        &self,
        layout: &Layout,
        contract: &IngestContract,
        turn: &str,
    ) -> Result<()> {
        ensure!(
            !turn.is_empty()
                && layout.session_bucket(&contract.tenant, &contract.qualified_table, turn)?
                    == self.bucket,
            "annotation turn belongs to another shard"
        );
        Ok(())
    }
}

impl mixed::Decoded {
    pub(super) fn validate_shard(
        &self,
        shard: Option<Shard>,
        layout: &Layout,
        contract: &IngestContract,
        spec: Option<&crate::turn_ingestion::TurnIngestionSpec>,
    ) -> Result<()> {
        let Some(shard) = shard else {
            return Ok(());
        };
        match self {
            Self::Patches(rows) => {
                for row in rows {
                    if let Some(spec) = spec {
                        shard.validate_turn(layout, contract, &spec.routing_key(&row.turn)?)?;
                    } else {
                        shard.validate_turn(layout, contract, &row.turn)?;
                    }
                }
            }
            Self::Creates { rows, .. } => {
                for row in rows {
                    shard.validate_turn(
                        layout,
                        contract,
                        if spec.is_some() {
                            &row.rollout_id
                        } else {
                            &row.turn
                        },
                    )?;
                }
            }
        }
        Ok(())
    }
}
