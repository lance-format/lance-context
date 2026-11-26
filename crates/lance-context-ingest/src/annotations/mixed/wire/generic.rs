//! Keep arbitrary body fields in their own Arrow struct. In particular a body
//! `id`, `score` or operation-like field never aliases the narrow annotations.
use super::*;
use crate::turn_ingestion::TurnIngestionSpec;
use arrow_array::StructArray;

const BODY: usize = OPERATION + 1;

pub fn schema_for(spec: &TurnIngestionSpec, limits: &Limits) -> Result<SchemaRef> {
    limits.validate()?;
    let patch = super::super::super::wire::schema(limits);
    let mut fields = patch.fields().to_vec();
    fields.push(Arc::new(Field::new(
        "turn_operation",
        DataType::UInt8,
        false,
    )));
    fields.push(Arc::new(
        Field::new(
            "turn_body",
            DataType::Struct(spec.body_schema().fields().clone()),
            true,
        )
        .with_metadata(spec.body_schema().metadata().clone()),
    ));
    let mut metadata = patch.metadata().clone();
    metadata.insert("lance-context.mixed-turns".into(), "2".into());
    spec.bind_schema(&Schema::new_with_metadata(fields, metadata))
}

pub fn patches_for(
    spec: &TurnIngestionSpec,
    batch: &RecordBatch,
    limits: &Limits,
) -> Result<RecordBatch> {
    ensure!(
        batch.schema() == super::super::super::wire::schema(limits),
        "foreign patch schema"
    );
    let schema = schema_for(spec, limits)?;
    let mut columns = batch.columns().to_vec();
    columns.push(Arc::new(UInt8Array::from(vec![PATCH; batch.num_rows()])));
    columns.push(new_null_array(
        schema.field(BODY).data_type(),
        batch.num_rows(),
    ));
    Ok(RecordBatch::try_new(schema, columns)?)
}

/// Initial annotations use the internal business key, while all original body
/// fields and buffers are preserved. Duplicate bodies remain for ordered apply
/// to select the first committed occurrence across producers and restarts.
pub fn creates_for(
    spec: &TurnIngestionSpec,
    body: &RecordBatch,
    initial: &RecordBatch,
    limits: &Limits,
) -> Result<RecordBatch> {
    ensure!(
        body.schema() == *spec.body_schema(),
        "foreign turn body schema"
    );
    ensure!(
        initial.schema() == super::super::super::wire::schema(limits)
            && initial.num_rows() == body.num_rows(),
        "initial annotations differ"
    );
    let ids = initial
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .context("invalid initial keys")?;
    for row in 0..body.num_rows() {
        ensure!(
            !ids.is_null(row) && spec.row_key(body, row, limits.value_bytes)? == ids.value(row),
            "initial annotation key differs from body identity"
        );
    }
    let mut columns = initial.columns().to_vec();
    columns.push(Arc::new(UInt8Array::from(vec![CREATE; body.num_rows()])));
    columns.push(Arc::new(StructArray::try_new(
        spec.body_schema().fields().clone(),
        body.columns().to_vec(),
        None,
    )?));
    Ok(RecordBatch::try_new(schema_for(spec, limits)?, columns)?)
}

pub(super) fn decode(
    body: Bytes,
    expected: SchemaRef,
    max_rows: u32,
    value_bytes: usize,
    spec: &TurnIngestionSpec,
) -> Result<Decoded> {
    let batches = ipc::decode(body, expected)?;
    let rows = batches
        .iter()
        .try_fold(0usize, |n, b| n.checked_add(b.num_rows()))
        .context("turn row overflow")?;
    ensure!(
        rows > 0 && rows <= max_rows as usize,
        "invalid turn row count"
    );
    let mut kind = None;
    let mut narrow = Vec::with_capacity(batches.len());
    let mut bodies = Vec::with_capacity(batches.len());
    for batch in &batches {
        ensure!(
            batch.num_columns() == BODY + 1,
            "invalid generic turn columns"
        );
        let operations = batch
            .column(OPERATION)
            .as_any()
            .downcast_ref::<UInt8Array>()
            .context("invalid turn operation")?;
        ensure!(operations.null_count() == 0, "null turn operation");
        for &op in operations.values() {
            ensure!(
                op <= CREATE && kind.is_none_or(|old| old == op),
                "mixed or unknown turn operation"
            );
            kind = Some(op);
        }
        let body = batch
            .column(BODY)
            .as_any()
            .downcast_ref::<StructArray>()
            .context("invalid turn body")?;
        if batch.num_rows() > 0 && kind == Some(CREATE) {
            ensure!(body.null_count() == 0, "null create body");
            for (field, array) in spec.body_schema().fields().iter().zip(body.columns()) {
                array.to_data().validate_full()?;
                ensure!(
                    field.is_nullable() || array.logical_null_count() == 0,
                    "null required body field"
                );
            }
            bodies.push(RecordBatch::try_new(
                spec.body_schema().clone(),
                body.columns().to_vec(),
            )?);
        } else if batch.num_rows() > 0 {
            ensure!(
                body.null_count() == batch.num_rows()
                    && body
                        .columns()
                        .iter()
                        .all(|a| a.logical_null_count() == batch.num_rows()),
                "patch includes body fields"
            );
        }
        narrow.push(batch.project(&(0..PATCH_COLUMNS).collect::<Vec<_>>())?);
    }
    let mutations = super::super::super::wire::decode_batches(&narrow, max_rows, value_bytes)?;
    if kind == Some(PATCH) {
        return Ok(Decoded::Patches(mutations));
    }
    let mut mutations = mutations.into_iter();
    let mut created = Vec::with_capacity(rows);
    for body in &bodies {
        let sessions = body
            .column_by_name(spec.routing_column())
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .context("invalid routing column")?;
        for row in 0..body.num_rows() {
            let mutation = mutations.next().context("missing initial annotation")?;
            ensure!(
                spec.row_key(body, row, value_bytes)? == mutation.turn,
                "body and annotation identities differ"
            );
            let mut annotation = TurnAnnotation::default();
            mutation.patch.apply_to(&mut annotation)?;
            created.push(Created {
                turn: mutation.turn,
                rollout_id: sessions.value(row).into(),
                annotation,
            });
        }
    }
    ensure!(mutations.next().is_none(), "extra initial annotations");
    Ok(Decoded::Creates {
        rows: created,
        bodies,
    })
}
