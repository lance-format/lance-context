//! Decode an original-schema point response from the owner HTTP route.
use super::*;
use crate::turn_ingestion::TurnIngestionSpec;

/// Verify one original-schema row and its complete requested business key.
///
/// The caller must bound the HTTP input and admit its input/decode workspace
/// before calling, retaining that admission while using the result. Uncompressed
/// binary values can share the input; alignment repair can allocate.
pub fn decode(bytes: Bytes, spec: &TurnIngestionSpec, key: &str) -> Result<RecordBatch> {
    let mut batches = ipc::decode(bytes, spec.body_schema().clone())?;
    ensure!(
        batches.len() == 1,
        "full body response must contain one batch"
    );
    let batch = batches.pop().unwrap();
    ensure!(
        batch.num_rows() == 1,
        "full body response must contain one row"
    );
    for (field, array) in batch.schema().fields().iter().zip(batch.columns()) {
        array.to_data().validate_full()?;
        ensure!(
            field.is_nullable() || array.logical_null_count() == 0,
            "null required body field"
        );
    }
    ensure!(
        spec.row_key(&batch, 0, key.len())? == key,
        "foreign full body key"
    );
    Ok(batch)
}
