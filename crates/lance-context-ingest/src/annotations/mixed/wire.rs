//! A fixed Arrow schema; body projection and row slicing share input arrays.
use super::*;
use arrow_array::{new_null_array, UInt8Array};
use arrow_schema::{DataType, Field, Schema};

mod generic;
pub use generic::{creates_for, patches_for, schema_for};

const PATCH_COLUMNS: usize = 7;
const OPERATION: usize = PATCH_COLUMNS;
pub const PATCH: u8 = 0;
pub const CREATE: u8 = 1;

pub fn schema(limits: &Limits) -> SchemaRef {
    let patch = super::super::wire::schema(limits);
    let mut fields = patch.fields().to_vec();
    fields.push(Arc::new(Field::new(
        "turn_operation",
        DataType::UInt8,
        false,
    )));
    for field in rollout_schema().fields() {
        if !["id", "score"].contains(&field.name().as_str()) {
            fields.push(Arc::new(field.as_ref().clone().with_nullable(true)));
        }
    }
    let mut metadata = patch.metadata().clone();
    metadata.insert("lance-context.mixed-turns".into(), "1".into());
    Arc::new(Schema::new_with_metadata(fields, metadata))
}

/// Wrap an ordinary annotation batch without copying its arrays. Immutable
/// body fields are null and cannot be smuggled into a patch operation.
pub fn patches(batch: &RecordBatch, limits: &Limits) -> Result<RecordBatch> {
    ensure!(
        batch.schema() == super::super::wire::schema(limits),
        "foreign patch schema"
    );
    let mixed = schema(limits);
    let mut columns = batch.columns().to_vec();
    columns.push(Arc::new(UInt8Array::from(vec![PATCH; batch.num_rows()])));
    for field in mixed.fields().iter().skip(OPERATION + 1) {
        columns.push(new_null_array(field.data_type(), batch.num_rows()));
    }
    Ok(RecordBatch::try_new(mixed, columns)?)
}

/// Combine an original rollout batch and its initial annotation operations.
/// The immutable arrays remain shared; the annotation score is authoritative.
pub fn creates(body: &RecordBatch, initial: &RecordBatch, limits: &Limits) -> Result<RecordBatch> {
    ensure!(
        body.schema().as_ref() == &rollout_schema(),
        "foreign rollout schema"
    );
    ensure!(
        initial.schema() == super::super::wire::schema(limits)
            && body.num_rows() == initial.num_rows(),
        "initial annotation batch differs"
    );
    ensure!(
        body.column_by_name("id") == initial.column_by_name("id")
            && body.column_by_name("score") == initial.column_by_name("score"),
        "create IDs/scores differ from initial annotations"
    );
    let mixed = schema(limits);
    let mut columns = initial.columns().to_vec();
    columns.push(Arc::new(UInt8Array::from(vec![CREATE; body.num_rows()])));
    for field in mixed.fields().iter().skip(OPERATION + 1) {
        columns.push(
            body.column_by_name(field.name())
                .context("missing body column")?
                .clone(),
        );
    }
    Ok(RecordBatch::try_new(mixed, columns)?)
}

pub(in crate::annotations) fn decode(
    body: Bytes,
    expected: SchemaRef,
    max_rows: u32,
    value_bytes: usize,
) -> Result<Decoded> {
    if let Some(spec) = crate::turn_ingestion::TurnIngestionSpec::from_schema(&expected)? {
        return generic::decode(body, expected, max_rows, value_bytes, &spec);
    }
    let batches = ipc::decode(body, expected)?;
    let rows = batches
        .iter()
        .try_fold(0usize, |n, b| n.checked_add(b.num_rows()))
        .context("mixed batch row overflow")?;
    ensure!(
        rows > 0 && rows <= max_rows as usize,
        "invalid mixed batch row count"
    );
    let mut operation = None;
    let mut narrow = Vec::with_capacity(batches.len());
    for batch in &batches {
        let operations = batch
            .column(OPERATION)
            .as_any()
            .downcast_ref::<UInt8Array>()
            .context("invalid mixed operation column")?;
        ensure!(operations.null_count() == 0, "null mixed operation");
        for value in operations.values() {
            ensure!(*value <= CREATE, "unknown mixed operation");
            ensure!(
                operation.is_none_or(|old| old == *value),
                "one WAL record must have one operation kind"
            );
            operation = Some(*value);
        }
        narrow.push(batch.project(&(0..PATCH_COLUMNS).collect::<Vec<_>>())?);
    }
    let patches = super::super::wire::decode_batches(&narrow, max_rows, value_bytes)?;
    if operation == Some(PATCH) {
        ensure!(
            batches.iter().all(|b| b
                .columns()
                .iter()
                .skip(OPERATION + 1)
                .all(|column| column.logical_null_count() == b.num_rows())),
            "patch includes immutable body fields"
        );
        return Ok(Decoded::Patches(patches));
    }
    let body_schema = Arc::new(rollout_schema());
    let mut bodies = Vec::with_capacity(batches.len());
    let mut groups = Vec::with_capacity(rows);
    for batch in &batches {
        let mut columns = Vec::with_capacity(body_schema.fields().len());
        for field in body_schema.fields() {
            let column = batch
                .column_by_name(field.name())
                .context("missing immutable field")?;
            column.to_data().validate_full()?;
            ensure!(
                field.is_nullable() || column.logical_null_count() == 0,
                "null required body field"
            );
            columns.push(column.clone());
        }
        let sessions = batch
            .column_by_name("rollout_id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .context("invalid rollout key")?;
        for session in sessions.iter() {
            let session = session
                .filter(|s| !s.is_empty() && s.len() <= value_bytes)
                .context("invalid rollout key length")?;
            groups.push(session.to_owned());
        }
        bodies.push(RecordBatch::try_new(body_schema.clone(), columns)?);
    }
    let rows = patches
        .into_iter()
        .zip(groups)
        .map(|(mutation, rollout_id)| {
            let mut annotation = TurnAnnotation::default();
            mutation.patch.apply_to(&mut annotation)?;
            Ok(Created {
                turn: mutation.turn,
                rollout_id,
                annotation,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Decoded::Creates { rows, bodies })
}
