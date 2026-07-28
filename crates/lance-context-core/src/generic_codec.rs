//! Schema-driven encode/decode: rows as maps, columns from a [`SchemaSpec`].
//!
//! The built-in stores convert records to Arrow with hand-written builders — 37
//! of them in the rollout store alone — and decode by naming every column again.
//! That is what makes adding a column a five-to-seven-place edit. Here the
//! schema drives both directions, so an arbitrary user schema needs no code.
//!
//! # Nested values are decoded by name
//!
//! The built-in decoders read struct children *positionally*
//! (`relationships`, `state_metadata`), so reordering fields silently
//! assigns data to the wrong field. Everything here is keyed by name in both
//! directions.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::builder::{
    BooleanBuilder, FixedSizeListBuilder, Float32Builder, Float64Builder, Int32Builder,
    Int64Builder, LargeBinaryBuilder, LargeStringBuilder, ListBuilder, StringBuilder,
    TimestampMicrosecondBuilder,
};
use arrow_array::{
    Array, ArrayRef, BooleanArray, FixedSizeListArray, Float32Array, Float64Array, Int32Array,
    Int64Array, LargeBinaryArray, LargeStringArray, ListArray, RecordBatch, StringArray,
    TimestampMicrosecondArray,
};
use arrow_schema::{ArrowError, Schema};
use serde_json::{Map, Number, Value};

use lance_context_api::schema_spec::{ColumnSpec, ColumnType, SchemaSpec, ID_COLUMN};

/// One row: column name to value. Absent keys are written as null, which is why
/// a nullable column can simply be omitted.
pub type Row = Map<String, Value>;

/// Encode rows into a [`RecordBatch`] matching `schema`.
///
/// Values are matched to columns **by name**; a key with no matching column is
/// an error rather than being silently dropped, so a typo in a field name
/// surfaces at the write instead of turning into missing data.
///
/// Binary columns accept either a JSON array of byte values or a base64 string,
/// since JSON has no byte-string type.
///
/// # Errors
///
/// - a row is missing a non-nullable column, or supplies null for one
/// - a value's JSON type does not match the declared column type
/// - a vector's length does not match the declared dimension
/// - a row carries a key that the schema does not declare
pub fn rows_to_batch(
    spec: &SchemaSpec,
    schema: Arc<Schema>,
    rows: &[Row],
) -> Result<RecordBatch, ArrowError> {
    for (index, row) in rows.iter().enumerate() {
        for key in row.keys() {
            if spec.column(key).is_none() {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "row {index}: column '{key}' is not declared in the store schema"
                )));
            }
        }
    }

    // Build every declared column, then assemble in the dataset's field order.
    // Keying by name (rather than pushing positionally) is what lets the
    // physical schema reorder or omit columns without breaking the write.
    let mut arrays: HashMap<&str, ArrayRef> = HashMap::with_capacity(spec.columns.len());
    for (name, column) in &spec.columns {
        arrays.insert(name.as_str(), build_column(name, column, rows)?);
    }

    let columns = schema
        .fields()
        .iter()
        .map(|field| {
            arrays.get(field.name().as_str()).cloned().ok_or_else(|| {
                ArrowError::SchemaError(format!(
                    "dataset column '{}' is not declared in the store schema",
                    field.name()
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    RecordBatch::try_new(schema, columns)
}

/// Decode a [`RecordBatch`] back into rows.
///
/// Only columns present in the batch are decoded, so a projected scan (one that
/// dropped blob columns, say) round-trips to rows without those keys rather
/// than erroring. Null values are omitted from the row entirely.
///
/// # Errors
///
/// Returns an error if a column's array type does not match the schema.
pub fn batch_to_rows(spec: &SchemaSpec, batch: &RecordBatch) -> Result<Vec<Row>, ArrowError> {
    let mut rows = vec![Row::new(); batch.num_rows()];

    for (name, column) in &spec.columns {
        let Some(array) = batch.column_by_name(name) else {
            // Projected out; not an error.
            continue;
        };
        decode_column(name, column, array.as_ref(), &mut rows)?;
    }

    Ok(rows)
}

/// Extract the `id` of every row, in order — the merge key the storage layer
/// needs, without decoding the rest of the batch.
///
/// # Errors
///
/// Returns an error if the batch has no `id` column or it is not a string.
pub fn ids_from_batch(batch: &RecordBatch) -> Result<Vec<String>, ArrowError> {
    let array = batch
        .column_by_name(ID_COLUMN)
        .ok_or_else(|| ArrowError::SchemaError(format!("batch has no '{ID_COLUMN}' column")))?;
    let ids = array
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            ArrowError::SchemaError(format!("'{ID_COLUMN}' column is not a string array"))
        })?;
    Ok((0..ids.len())
        .map(|row| ids.value(row).to_string())
        .collect())
}

// ------------------------------------------------------------------ encoding

/// Fetch a row's value for `name`, enforcing nullability.
///
/// `Ok(None)` means "write a null here", which is valid only for a nullable
/// column. An explicit JSON `null` is treated identically to an absent key.
fn value_for<'a>(
    name: &str,
    column: &ColumnSpec,
    row: &'a Row,
    index: usize,
) -> Result<Option<&'a Value>, ArrowError> {
    match row.get(name) {
        Some(Value::Null) | None if !column.nullable => Err(ArrowError::InvalidArgumentError(
            format!("row {index}: column '{name}' is not nullable and was not supplied"),
        )),
        Some(Value::Null) | None => Ok(None),
        Some(value) => Ok(Some(value)),
    }
}

fn type_error(name: &str, index: usize, expected: &str, got: &Value) -> ArrowError {
    ArrowError::InvalidArgumentError(format!(
        "row {index}: column '{name}' expects {expected}, got {got}"
    ))
}

/// Error for a bad *element* inside a list, where the row index is not
/// threaded through the append closure.
fn element_error(name: &str, expected: &str, got: &Value) -> ArrowError {
    ArrowError::InvalidArgumentError(format!(
        "column '{name}': expects {expected}, got element {got}"
    ))
}

fn build_column(name: &str, column: &ColumnSpec, rows: &[Row]) -> Result<ArrayRef, ArrowError> {
    Ok(match &column.column_type {
        ColumnType::String { large: false } => {
            let mut b = StringBuilder::new();
            for (index, row) in rows.iter().enumerate() {
                match value_for(name, column, row, index)? {
                    Some(Value::String(text)) => b.append_value(text),
                    Some(other) => return Err(type_error(name, index, "a string", other)),
                    None => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        ColumnType::String { large: true } => {
            let mut b = LargeStringBuilder::new();
            for (index, row) in rows.iter().enumerate() {
                match value_for(name, column, row, index)? {
                    Some(Value::String(text)) => b.append_value(text),
                    Some(other) => return Err(type_error(name, index, "a string", other)),
                    None => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        ColumnType::Int32 => {
            let mut b = Int32Builder::new();
            for (index, row) in rows.iter().enumerate() {
                match value_for(name, column, row, index)? {
                    Some(value) => b.append_value(
                        as_i64(name, index, value)?
                            .try_into()
                            .map_err(|_| type_error(name, index, "a 32-bit integer", value))?,
                    ),
                    None => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        ColumnType::Int64 => {
            let mut b = Int64Builder::new();
            for (index, row) in rows.iter().enumerate() {
                match value_for(name, column, row, index)? {
                    Some(value) => b.append_value(as_i64(name, index, value)?),
                    None => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        ColumnType::Float32 => {
            let mut b = Float32Builder::new();
            for (index, row) in rows.iter().enumerate() {
                match value_for(name, column, row, index)? {
                    Some(value) => b.append_value(as_f64(name, index, value)? as f32),
                    None => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        ColumnType::Float64 => {
            let mut b = Float64Builder::new();
            for (index, row) in rows.iter().enumerate() {
                match value_for(name, column, row, index)? {
                    Some(value) => b.append_value(as_f64(name, index, value)?),
                    None => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        ColumnType::Bool => {
            let mut b = BooleanBuilder::new();
            for (index, row) in rows.iter().enumerate() {
                match value_for(name, column, row, index)? {
                    Some(Value::Bool(flag)) => b.append_value(*flag),
                    Some(other) => return Err(type_error(name, index, "a boolean", other)),
                    None => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        ColumnType::Timestamp => {
            let mut b = TimestampMicrosecondBuilder::new();
            for (index, row) in rows.iter().enumerate() {
                match value_for(name, column, row, index)? {
                    Some(value) => b.append_value(as_timestamp_micros(name, index, value)?),
                    None => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        ColumnType::Binary { .. } => {
            let mut b = LargeBinaryBuilder::new();
            for (index, row) in rows.iter().enumerate() {
                match value_for(name, column, row, index)? {
                    Some(value) => b.append_value(as_bytes(name, index, value)?),
                    None => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        ColumnType::Vector { dim, .. } => {
            let mut b = FixedSizeListBuilder::new(Float32Builder::new(), *dim);
            for (index, row) in rows.iter().enumerate() {
                match value_for(name, column, row, index)? {
                    Some(Value::Array(items)) => {
                        if items.len() != *dim as usize {
                            return Err(ArrowError::InvalidArgumentError(format!(
                                "row {index}: column '{name}' expects a vector of {dim} \
                                 values, got {}",
                                items.len()
                            )));
                        }
                        for item in items {
                            b.values().append_value(as_f64(name, index, item)? as f32);
                        }
                        b.append(true);
                    }
                    Some(other) => {
                        return Err(type_error(name, index, "an array of numbers", other))
                    }
                    None => {
                        // A FixedSizeList null still needs its slots filled.
                        for _ in 0..*dim {
                            b.values().append_null();
                        }
                        b.append(false);
                    }
                }
            }
            Arc::new(b.finish())
        }
        ColumnType::List { item } => build_list_column(name, column, item, rows)?,
    })
}

/// Build a `List<item>` column. Element types are the scalar subset;
/// [`SchemaSpec::validate`] has already rejected nested lists and vector lists.
fn build_list_column(
    name: &str,
    column: &ColumnSpec,
    item: &ColumnType,
    rows: &[Row],
) -> Result<ArrayRef, ArrowError> {
    /// Drive a `ListBuilder`, appending each element through `$append`.
    macro_rules! build_list {
        ($builder:expr, $append:expr) => {{
            let mut b = ListBuilder::new($builder);
            for (index, row) in rows.iter().enumerate() {
                match value_for(name, column, row, index)? {
                    Some(Value::Array(items)) => {
                        for element in items {
                            let append: &dyn Fn(&mut _, &Value) -> Result<(), ArrowError> =
                                &$append;
                            append(b.values(), element)?;
                        }
                        b.append(true);
                    }
                    Some(other) => return Err(type_error(name, index, "an array", other)),
                    None => b.append(false),
                }
            }
            Arc::new(b.finish()) as ArrayRef
        }};
    }

    Ok(match item {
        ColumnType::String { large: false } => build_list!(
            StringBuilder::new(),
            |values: &mut StringBuilder, element: &Value| match element {
                Value::String(text) => {
                    values.append_value(text);
                    Ok(())
                }
                other => Err(element_error(name, "an array of strings", other)),
            }
        ),
        ColumnType::String { large: true } => build_list!(
            LargeStringBuilder::new(),
            |values: &mut LargeStringBuilder, element: &Value| match element {
                Value::String(text) => {
                    values.append_value(text);
                    Ok(())
                }
                other => Err(element_error(name, "an array of strings", other)),
            }
        ),
        ColumnType::Int32 => build_list!(
            Int32Builder::new(),
            |values: &mut Int32Builder, element: &Value| {
                let raw = as_i64(name, 0, element)?;
                values
                    .append_value(raw.try_into().map_err(|_| {
                        element_error(name, "an array of 32-bit integers", element)
                    })?);
                Ok(())
            }
        ),
        ColumnType::Int64 => build_list!(
            Int64Builder::new(),
            |values: &mut Int64Builder, element: &Value| {
                values.append_value(as_i64(name, 0, element)?);
                Ok(())
            }
        ),
        ColumnType::Float32 => build_list!(
            Float32Builder::new(),
            |values: &mut Float32Builder, element: &Value| {
                values.append_value(as_f64(name, 0, element)? as f32);
                Ok(())
            }
        ),
        ColumnType::Float64 => build_list!(
            Float64Builder::new(),
            |values: &mut Float64Builder, element: &Value| {
                values.append_value(as_f64(name, 0, element)?);
                Ok(())
            }
        ),
        ColumnType::Bool => build_list!(
            BooleanBuilder::new(),
            |values: &mut BooleanBuilder, element: &Value| match element {
                Value::Bool(flag) => {
                    values.append_value(*flag);
                    Ok(())
                }
                other => Err(element_error(name, "an array of booleans", other)),
            }
        ),
        ColumnType::Timestamp => build_list!(
            TimestampMicrosecondBuilder::new(),
            |values: &mut TimestampMicrosecondBuilder, element: &Value| {
                values.append_value(as_timestamp_micros(name, 0, element)?);
                Ok(())
            }
        ),
        ColumnType::Binary { .. } => build_list!(
            LargeBinaryBuilder::new(),
            |values: &mut LargeBinaryBuilder, element: &Value| {
                values.append_value(as_bytes(name, 0, element)?);
                Ok(())
            }
        ),
        // Rejected by `SchemaSpec::validate`, so unreachable in practice.
        ColumnType::List { .. } | ColumnType::Vector { .. } => {
            return Err(ArrowError::SchemaError(format!(
                "column '{name}': unsupported list element type"
            )))
        }
    })
}

fn as_i64(name: &str, index: usize, value: &Value) -> Result<i64, ArrowError> {
    value
        .as_i64()
        .ok_or_else(|| type_error(name, index, "an integer", value))
}

fn as_f64(name: &str, index: usize, value: &Value) -> Result<f64, ArrowError> {
    value
        .as_f64()
        .ok_or_else(|| type_error(name, index, "a number", value))
}

/// Timestamps accept microseconds-since-epoch or an RFC 3339 string.
fn as_timestamp_micros(name: &str, index: usize, value: &Value) -> Result<i64, ArrowError> {
    match value {
        Value::Number(number) => number
            .as_i64()
            .ok_or_else(|| type_error(name, index, "microseconds since epoch", value)),
        Value::String(text) => chrono::DateTime::parse_from_rfc3339(text)
            .map(|parsed| parsed.timestamp_micros())
            .map_err(|_| type_error(name, index, "an RFC 3339 timestamp", value)),
        other => Err(type_error(
            name,
            index,
            "microseconds since epoch or an RFC 3339 string",
            other,
        )),
    }
}

/// Binary accepts a base64 string or an array of byte values — JSON has no
/// native byte-string type, and both spellings appear in practice.
fn as_bytes(name: &str, index: usize, value: &Value) -> Result<Vec<u8>, ArrowError> {
    use base64::Engine;
    match value {
        Value::String(encoded) => base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map_err(|_| type_error(name, index, "base64-encoded bytes", value)),
        Value::Array(items) => items
            .iter()
            .map(|item| {
                let byte = as_i64(name, index, item)?;
                u8::try_from(byte)
                    .map_err(|_| type_error(name, index, "an array of bytes (0-255)", item))
            })
            .collect(),
        other => Err(type_error(
            name,
            index,
            "base64-encoded bytes or an array of byte values",
            other,
        )),
    }
}

// ------------------------------------------------------------------ decoding

/// Downcast `array`, or report the mismatch against the declared type.
fn downcast<'a, A: 'static>(name: &str, array: &'a dyn Array) -> Result<&'a A, ArrowError> {
    array.as_any().downcast_ref::<A>().ok_or_else(|| {
        ArrowError::SchemaError(format!(
            "column '{name}' has array type {:?}, which does not match the store schema",
            array.data_type()
        ))
    })
}

fn decode_column(
    name: &str,
    column: &ColumnSpec,
    array: &dyn Array,
    rows: &mut [Row],
) -> Result<(), ArrowError> {
    /// Decode every non-null slot through `$convert`.
    macro_rules! decode {
        ($ty:ty, $convert:expr) => {{
            let typed = downcast::<$ty>(name, array)?;
            for (index, row) in rows.iter_mut().enumerate() {
                if typed.is_null(index) {
                    continue;
                }
                #[allow(clippy::redundant_closure_call)]
                row.insert(name.to_string(), ($convert)(typed.value(index)));
            }
        }};
    }

    match &column.column_type {
        ColumnType::String { large: false } => {
            decode!(StringArray, |value: &str| Value::String(value.to_string()))
        }
        ColumnType::String { large: true } => {
            decode!(LargeStringArray, |value: &str| Value::String(
                value.to_string()
            ))
        }
        ColumnType::Int32 => decode!(Int32Array, |value: i32| Value::Number(value.into())),
        ColumnType::Int64 => decode!(Int64Array, |value: i64| Value::Number(value.into())),
        ColumnType::Float32 => decode!(Float32Array, |value: f32| number(f64::from(value))),
        ColumnType::Float64 => decode!(Float64Array, number),
        ColumnType::Bool => decode!(BooleanArray, Value::Bool),
        ColumnType::Timestamp => {
            decode!(TimestampMicrosecondArray, |value: i64| Value::Number(
                value.into()
            ))
        }
        ColumnType::Binary { .. } => {
            // Base64 on the way out, matching the preferred input spelling.
            use base64::Engine;
            decode!(LargeBinaryArray, |value: &[u8]| Value::String(
                base64::engine::general_purpose::STANDARD.encode(value)
            ))
        }
        ColumnType::Vector { .. } => {
            let typed = downcast::<FixedSizeListArray>(name, array)?;
            for (index, row) in rows.iter_mut().enumerate() {
                if typed.is_null(index) {
                    continue;
                }
                let values = typed.value(index);
                let floats = downcast::<Float32Array>(name, values.as_ref())?;
                let vector = (0..floats.len())
                    .map(|slot| number(f64::from(floats.value(slot))))
                    .collect();
                row.insert(name.to_string(), Value::Array(vector));
            }
        }
        ColumnType::List { item } => {
            let typed = downcast::<ListArray>(name, array)?;
            for (index, row) in rows.iter_mut().enumerate() {
                if typed.is_null(index) {
                    continue;
                }
                let values = typed.value(index);
                row.insert(
                    name.to_string(),
                    Value::Array(decode_list_values(name, item, values.as_ref())?),
                );
            }
        }
    }
    Ok(())
}

fn decode_list_values(
    name: &str,
    item: &ColumnType,
    values: &dyn Array,
) -> Result<Vec<Value>, ArrowError> {
    macro_rules! collect {
        ($ty:ty, $convert:expr) => {{
            let typed = downcast::<$ty>(name, values)?;
            (0..typed.len())
                .map(|slot| {
                    if typed.is_null(slot) {
                        Value::Null
                    } else {
                        #[allow(clippy::redundant_closure_call)]
                        ($convert)(typed.value(slot))
                    }
                })
                .collect()
        }};
    }

    Ok(match item {
        ColumnType::String { large: false } => {
            collect!(StringArray, |v: &str| Value::String(v.to_string()))
        }
        ColumnType::String { large: true } => {
            collect!(LargeStringArray, |v: &str| Value::String(v.to_string()))
        }
        ColumnType::Int32 => collect!(Int32Array, |v: i32| Value::Number(v.into())),
        ColumnType::Int64 => collect!(Int64Array, |v: i64| Value::Number(v.into())),
        ColumnType::Float32 => collect!(Float32Array, |v: f32| number(f64::from(v))),
        ColumnType::Float64 => collect!(Float64Array, number),
        ColumnType::Bool => collect!(BooleanArray, Value::Bool),
        ColumnType::Timestamp => {
            collect!(TimestampMicrosecondArray, |v: i64| Value::Number(v.into()))
        }
        ColumnType::Binary { .. } => {
            use base64::Engine;
            collect!(LargeBinaryArray, |v: &[u8]| Value::String(
                base64::engine::general_purpose::STANDARD.encode(v)
            ))
        }
        ColumnType::List { .. } | ColumnType::Vector { .. } => {
            return Err(ArrowError::SchemaError(format!(
                "column '{name}': unsupported list element type"
            )))
        }
    })
}

/// JSON has no NaN or infinity, so non-finite floats decode to null rather than
/// producing invalid JSON.
fn number(value: f64) -> Value {
    Number::from_f64(value).map_or(Value::Null, Value::Number)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_context_api::schema_spec::ColumnSpec;
    use serde_json::json;

    fn spec() -> SchemaSpec {
        SchemaSpec::new(vec![
            (
                ID_COLUMN.to_string(),
                ColumnSpec::required(ColumnType::String { large: false }),
            ),
            ("count".to_string(), ColumnSpec::new(ColumnType::Int64)),
            ("score".to_string(), ColumnSpec::new(ColumnType::Float32)),
            ("active".to_string(), ColumnSpec::new(ColumnType::Bool)),
            (
                "tags".to_string(),
                ColumnSpec::new(ColumnType::List {
                    item: Box::new(ColumnType::String { large: false }),
                }),
            ),
            (
                "embedding".to_string(),
                ColumnSpec::new(ColumnType::Vector {
                    dim: 3,
                    metric: None,
                }),
            ),
            (
                "blob".to_string(),
                ColumnSpec::new(ColumnType::Binary { blob: true }),
            ),
            ("at".to_string(), ColumnSpec::new(ColumnType::Timestamp)),
        ])
    }

    fn row(value: Value) -> Row {
        value.as_object().unwrap().clone()
    }

    #[test]
    fn rows_round_trip_through_arrow() {
        let spec = spec();
        let schema = Arc::new(spec.to_arrow().unwrap());
        let rows = vec![
            row(json!({
                "id": "r1",
                "count": 42,
                "score": 0.5,
                "active": true,
                "tags": ["a", "b"],
                "embedding": [1.0, 2.0, 3.0],
                "blob": [0, 1, 255],
                "at": 1_700_000_000_000_000i64,
            })),
            // Every nullable column omitted.
            row(json!({"id": "r2"})),
        ];

        let batch = rows_to_batch(&spec, schema, &rows).unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(ids_from_batch(&batch).unwrap(), vec!["r1", "r2"]);

        let decoded = batch_to_rows(&spec, &batch).unwrap();
        assert_eq!(decoded[0]["id"], json!("r1"));
        assert_eq!(decoded[0]["count"], json!(42));
        assert_eq!(decoded[0]["active"], json!(true));
        assert_eq!(decoded[0]["tags"], json!(["a", "b"]));
        assert_eq!(decoded[0]["embedding"], json!([1.0, 2.0, 3.0]));
        assert_eq!(decoded[0]["at"], json!(1_700_000_000_000_000i64));
        // Binary comes back base64-encoded.
        assert_eq!(decoded[0]["blob"], json!("AAH/"));

        // Nulls are omitted rather than emitted as JSON null.
        assert_eq!(decoded[1].keys().collect::<Vec<_>>(), vec!["id"]);
    }

    #[test]
    fn binary_accepts_base64_and_byte_arrays_alike() {
        let spec = spec();
        let schema = Arc::new(spec.to_arrow().unwrap());
        let rows = vec![
            row(json!({"id": "a", "blob": "AAH/"})),
            row(json!({"id": "b", "blob": [0, 1, 255]})),
        ];
        let batch = rows_to_batch(&spec, schema, &rows).unwrap();
        let decoded = batch_to_rows(&spec, &batch).unwrap();
        assert_eq!(decoded[0]["blob"], decoded[1]["blob"]);
    }

    #[test]
    fn missing_required_column_is_rejected() {
        let spec = spec();
        let schema = Arc::new(spec.to_arrow().unwrap());
        let err = rows_to_batch(&spec, schema, &[row(json!({"count": 1}))]).unwrap_err();
        assert!(err.to_string().contains("'id' is not nullable"), "{err}");
    }

    #[test]
    fn undeclared_column_is_rejected_not_dropped() {
        // Silently dropping an unknown key turns a field-name typo into
        // missing data that surfaces much later.
        let spec = spec();
        let schema = Arc::new(spec.to_arrow().unwrap());
        let err = rows_to_batch(&spec, schema, &[row(json!({"id": "r1", "typo": 1}))]).unwrap_err();
        assert!(err.to_string().contains("not declared"), "{err}");
    }

    #[test]
    fn type_mismatches_are_rejected() {
        let spec = spec();
        let schema = Arc::new(spec.to_arrow().unwrap());
        for bad in [
            json!({"id": "r1", "count": "not-a-number"}),
            json!({"id": "r1", "active": "yes"}),
            json!({"id": 7}),
            json!({"id": "r1", "tags": "not-a-list"}),
        ] {
            assert!(
                rows_to_batch(&spec, schema.clone(), &[row(bad.clone())]).is_err(),
                "{bad} should be rejected"
            );
        }
    }

    #[test]
    fn vector_length_must_match_the_declared_dimension() {
        let spec = spec();
        let schema = Arc::new(spec.to_arrow().unwrap());
        let err = rows_to_batch(
            &spec,
            schema,
            &[row(json!({"id": "r1", "embedding": [1.0, 2.0]}))],
        )
        .unwrap_err();
        assert!(err.to_string().contains("vector of 3 values"), "{err}");
    }

    #[test]
    fn timestamps_accept_rfc3339_strings() {
        let spec = spec();
        let schema = Arc::new(spec.to_arrow().unwrap());
        let batch = rows_to_batch(
            &spec,
            schema,
            &[row(json!({"id": "r1", "at": "2023-11-14T22:13:20Z"}))],
        )
        .unwrap();
        let decoded = batch_to_rows(&spec, &batch).unwrap();
        assert_eq!(decoded[0]["at"], json!(1_700_000_000_000_000i64));
    }

    #[test]
    fn projected_batches_decode_without_the_missing_columns() {
        // A list-style scan drops blob columns; decoding must not fail on them.
        let spec = spec();
        let schema = Arc::new(spec.to_arrow().unwrap());
        let batch = rows_to_batch(
            &spec,
            schema,
            &[row(json!({"id": "r1", "blob": [1, 2, 3], "count": 5}))],
        )
        .unwrap();

        let projected = batch.project(&[0, 1]).unwrap();
        let decoded = batch_to_rows(&spec, &projected).unwrap();
        assert_eq!(decoded[0]["id"], json!("r1"));
        assert_eq!(decoded[0]["count"], json!(5));
        assert!(!decoded[0].contains_key("blob"));
    }

    #[test]
    fn column_order_follows_the_dataset_not_the_row() {
        // Assembly is keyed by name, so a physical schema whose field order
        // differs from the declaration still encodes correctly.
        let spec = spec();
        let arrow = spec.to_arrow().unwrap();
        let reversed: Vec<_> = arrow.fields().iter().rev().cloned().collect();
        let reversed = Arc::new(Schema::new(reversed));

        let batch =
            rows_to_batch(&spec, reversed, &[row(json!({"id": "r1", "count": 9}))]).unwrap();
        assert_eq!(batch.schema().field(0).name(), "at");
        let decoded = batch_to_rows(&spec, &batch).unwrap();
        assert_eq!(decoded[0]["id"], json!("r1"));
        assert_eq!(decoded[0]["count"], json!(9));
    }
}
