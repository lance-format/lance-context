//! Full-body schema and transport bindings for immutable first-create ingestion.
//!
//! These helpers retain all original columns. They prepare schemas and batches;
//! ordered application, durable ACKs and publication belong to the owner runtime.
use super::*;
use crate::turn_ingestion::TurnIngestionSpec;

const KEY: &str = "lance-context.immutable-turn-body.v1";

#[cfg(test)]
mod tests;

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

/// Add the staging limits and shard to the transport schema. Their fingerprint
/// is bound at provisioning, rather than selected independently on restart.
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
/// The service must still validate all keys and their shard before durable append.
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
