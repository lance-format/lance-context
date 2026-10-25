//! Batched Arrow mutations. Keep/Set/Clear are explicit opcodes, not null guesses.
use super::*;
use arrow_array::{Float32Array, ListArray, UInt8Array};
use arrow_schema::{DataType, Field, Schema};
use lance_context_api::annotation::{AnnotationFieldUpdate, TurnAnnotationPatch};
use std::collections::{BTreeSet, HashMap};

pub(super) struct Mutation {
    pub turn: String,
    pub patch: TurnAnnotationPatch,
}

/// Stable rejection limits are part of the schema fingerprint, so independent
/// appliers cannot silently change mutation outcomes after restart. Reader
/// admission is local and deliberately excluded.
pub fn schema(limits: &Limits) -> SchemaRef {
    let labels = DataType::List(Arc::new(Field::new("item", DataType::Utf8, false)));
    Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("add_labels", labels.clone(), false),
            Field::new("remove_labels", labels, false),
            Field::new("status_op", DataType::UInt8, false),
            Field::new("status", DataType::Utf8, true),
            Field::new("score_op", DataType::UInt8, false),
            Field::new("score", DataType::Float32, true),
        ],
        HashMap::from([
            ("lance-context.annotation-mutations".into(), "1".into()),
            (
                "lance-context.annotation-value-bytes".into(),
                limits.value_bytes.to_string(),
            ),
            (
                "lance-context.annotation-retained-bytes".into(),
                limits.retained_bytes.to_string(),
            ),
            (
                "lance-context.annotation-retained-entries".into(),
                limits.retained_entries.to_string(),
            ),
        ]),
    ))
}

fn column<T: 'static>(batch: &RecordBatch, index: usize) -> Result<&T> {
    batch
        .column(index)
        .as_any()
        .downcast_ref()
        .context("invalid annotation column")
}

fn labels(array: &ListArray, row: usize) -> Result<BTreeSet<String>> {
    let values = array.value(row);
    let strings = values
        .as_any()
        .downcast_ref::<StringArray>()
        .context("invalid label array")?;
    strings
        .iter()
        .map(|s| {
            Ok(s.filter(|s| !s.is_empty())
                .context("null/empty annotation label")?
                .to_owned())
        })
        .collect()
}

fn field<T>(op: u8, value: Option<T>) -> Result<AnnotationFieldUpdate<T>> {
    match (op, value) {
        (0, None) => Ok(AnnotationFieldUpdate::Keep),
        (1, Some(value)) => Ok(AnnotationFieldUpdate::Set(value)),
        (2, None) => Ok(AnnotationFieldUpdate::Clear),
        _ => bail!("invalid annotation opcode/value combination"),
    }
}

pub(super) fn decode(
    body: Bytes,
    schema: SchemaRef,
    max_rows: u32,
    value_bytes: usize,
) -> Result<Vec<Mutation>> {
    let batches = ipc::decode(body, schema)?;
    let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
    ensure!(
        rows > 0 && rows <= max_rows as usize,
        "invalid annotation batch row count"
    );
    let mut output = Vec::with_capacity(rows);
    for batch in batches {
        for (field, column) in batch.schema().fields().iter().zip(batch.columns()) {
            column.to_data().validate_full()?;
            ensure!(
                field.is_nullable() || column.null_count() == 0,
                "null annotation input field"
            );
        }
        let ids: &StringArray = column(&batch, 0)?;
        let adds: &ListArray = column(&batch, 1)?;
        let removes: &ListArray = column(&batch, 2)?;
        let status_ops: &UInt8Array = column(&batch, 3)?;
        let statuses: &StringArray = column(&batch, 4)?;
        let score_ops: &UInt8Array = column(&batch, 5)?;
        let scores: &Float32Array = column(&batch, 6)?;
        for row in 0..batch.num_rows() {
            let turn = ids.value(row);
            ensure!(!turn.is_empty(), "empty annotation turn");
            // Bound owned patch allocation before copying any row strings.
            let mut charge = 512usize
                .checked_add(turn.len())
                .context("annotation input overflow")?;
            if !statuses.is_null(row) {
                charge = charge
                    .checked_add(statuses.value(row).len())
                    .context("annotation input overflow")?;
            }
            for array in [adds, removes] {
                let values = array.value(row);
                let strings = values
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .context("invalid label array")?;
                for label in strings.iter() {
                    charge = charge
                        .checked_add(512)
                        .and_then(|n| n.checked_add(label.map_or(0, str::len)))
                        .context("annotation input overflow")?;
                    ensure!(
                        charge <= value_bytes,
                        "annotation patch exceeds value budget"
                    );
                }
            }
            ensure!(
                charge <= value_bytes,
                "annotation patch exceeds value budget"
            );
            let patch = TurnAnnotationPatch {
                add_labels: labels(adds, row)?,
                remove_labels: labels(removes, row)?,
                status: field(
                    status_ops.value(row),
                    (!statuses.is_null(row)).then(|| statuses.value(row).to_owned()),
                )?,
                score: field(
                    score_ops.value(row),
                    (!scores.is_null(row)).then(|| scores.value(row)),
                )?,
            };
            patch.validate()?;
            output.push(Mutation {
                turn: turn.to_owned(),
                patch,
            });
        }
    }
    Ok(output)
}
