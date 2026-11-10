//! Versioned JSON documents with automatic, additive typed-column extraction.
//!
//! This initial adapter commits batches directly to Lance. It does not attach a
//! MemWAL: coordinating schema changes with resident writers is separate work.
//! Use one writer per store. Readers retain a consistent dataset snapshot until
//! explicitly refreshed. Generated record IDs are independent of JSON contents.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, LargeStringArray, RecordBatch,
    RecordBatchIterator, StringArray, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use datafusion::prelude::{col, lit};
use futures::TryStreamExt;
use lance::dataset::{builder::DatasetBuilder, NewColumnTransform, WriteParams};
use lance::io::{ObjectStoreParams, StorageOptionsAccessor};
use lance::{Dataset, Error, Result};
use serde_json::Value;

use crate::shredding::{ShredField, ShredLayout, ShredPolicy, ShredType, ShreddedDocument};

const FORMAT_KEY: &str = "lance-context:document-format";
const POLICY_KEY: &str = "lance-context:shred-policy";
const FIELD_KEY: &str = "lance-context:shred-field";
const ID: &str = "id";
const RESIDUAL: &str = "residual";

/// Embedded document store. All extraction paths are discovered automatically.
pub struct DocumentStore {
    dataset: Dataset,
    layout: ShredLayout,
    policy: ShredPolicy,
    pinned: bool,
}

impl DocumentStore {
    pub async fn create(
        uri: &str,
        policy: ShredPolicy,
        storage_options: Option<HashMap<String, String>>,
    ) -> Result<Self> {
        policy.validate().map_err(Error::invalid_input)?;
        let schema = Arc::new(
            Schema::new(vec![
                Field::new(ID, DataType::Utf8, false),
                Field::new(RESIDUAL, DataType::LargeUtf8, false),
            ])
            .with_metadata(HashMap::from([
                (FORMAT_KEY.into(), "1".into()),
                (
                    POLICY_KEY.into(),
                    serde_json::to_string(&policy).map_err(invalid)?,
                ),
            ])),
        );
        let reader = RecordBatchIterator::new(
            Vec::<std::result::Result<RecordBatch, arrow_schema::ArrowError>>::new(),
            schema,
        );
        let params = WriteParams {
            store_params: storage_options.map(|options| ObjectStoreParams {
                storage_options_accessor: Some(Arc::new(
                    StorageOptionsAccessor::with_static_options(options),
                )),
                ..Default::default()
            }),
            ..Default::default()
        };
        let dataset = Dataset::write(reader, uri, Some(params)).await?;
        Self::from_dataset(dataset, false)
    }

    pub async fn open(uri: &str, storage_options: Option<HashMap<String, String>>) -> Result<Self> {
        let mut builder = DatasetBuilder::from_uri(uri);
        if let Some(options) = storage_options {
            builder = builder.with_storage_options(options);
        }
        Self::from_dataset(builder.load().await?, false)
    }

    fn from_dataset(dataset: Dataset, pinned: bool) -> Result<Self> {
        let schema = Schema::from(dataset.schema());
        if schema.metadata().get(FORMAT_KEY).map(String::as_str) != Some("1") {
            return Err(Error::invalid_input(
                "unsupported or missing document format metadata",
            ));
        }
        let policy: ShredPolicy = serde_json::from_str(
            schema
                .metadata()
                .get(POLICY_KEY)
                .ok_or_else(|| Error::invalid_input("missing shredding policy metadata"))?,
        )
        .map_err(invalid)?;
        policy.validate().map_err(Error::invalid_input)?;
        if schema.fields().len() < 2
            || schema.field(0) != &Field::new(ID, DataType::Utf8, false)
            || schema.field(1) != &Field::new(RESIDUAL, DataType::LargeUtf8, false)
        {
            return Err(Error::invalid_input(
                "invalid document identity/residual schema",
            ));
        }
        let mut fields = Vec::new();
        for physical in schema.fields().iter().skip(2) {
            let field: ShredField = serde_json::from_str(
                physical
                    .metadata()
                    .get(FIELD_KEY)
                    .ok_or_else(|| Error::invalid_input("missing shredded field metadata"))?,
            )
            .map_err(invalid)?;
            if physical.as_ref() != &physical_field(&field)? {
                return Err(Error::invalid_input(
                    "shredded field metadata does not match physical schema",
                ));
            }
            fields.push(field);
        }
        let layout = ShredLayout { fields };
        layout.validate().map_err(Error::invalid_input)?;
        if layout.fields.len() > policy.max_columns {
            return Err(Error::invalid_input(
                "persisted layout exceeds column budget",
            ));
        }
        Ok(Self {
            dataset,
            layout,
            policy,
            pinned,
        })
    }

    pub fn layout(&self) -> &ShredLayout {
        &self.layout
    }
    pub fn policy(&self) -> &ShredPolicy {
        &self.policy
    }
    pub fn version(&self) -> u64 {
        self.dataset.version().version
    }

    /// Add documents and return their generated IDs after a durable commit.
    /// A schema-only commit may precede the atomic document append. If append
    /// fails, existing documents remain valid with nulls in the new columns.
    pub async fn add(&mut self, documents: Vec<Value>) -> Result<Vec<String>> {
        if self.pinned {
            return Err(Error::invalid_input(
                "cannot write through a pinned document snapshot",
            ));
        }
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        for document in &documents {
            validate_depth(document, 0)?;
        }
        let next = self
            .layout
            .evolve(&documents, &self.policy)
            .map_err(Error::invalid_input)?;
        if next.fields.len() > self.layout.fields.len() {
            let fields = next
                .fields
                .iter()
                .skip(self.layout.fields.len())
                .map(physical_field)
                .collect::<Result<Vec<_>>>()?;
            self.dataset
                .add_columns(
                    NewColumnTransform::AllNulls(Arc::new(Schema::new(fields))),
                    None,
                    None,
                )
                .await?;
            self.layout = next;
        }
        let ids: Vec<_> = documents.iter().map(|_| crate::generate_id()).collect();
        let encoded = documents
            .into_iter()
            .map(|document| self.layout.shred(document).map_err(Error::invalid_input))
            .collect::<Result<Vec<_>>>()?;
        let schema = Arc::new(Schema::from(self.dataset.schema()));
        let batch = encode(&ids, &encoded, &self.layout, schema.clone())?;
        self.dataset
            .append(RecordBatchIterator::new(vec![Ok(batch)], schema), None)
            .await?;
        Ok(ids)
    }

    pub async fn get(&self, id: &str) -> Result<Option<Value>> {
        let mut scanner = self.dataset.scan();
        scanner.filter_expr(col(ID).eq(lit(id)));
        let batches: Vec<RecordBatch> = scanner.try_into_stream().await?.try_collect().await?;
        for batch in batches {
            if batch.num_rows() > 0 {
                return Ok(Some(decode(&batch, 0, &self.layout)?));
            }
        }
        Ok(None)
    }

    /// Read a logical object path. An extracted scalar reads just its physical
    /// column; older rows and incompatible shapes fall back to reconstruction.
    /// Missing records/paths return `None`; explicit JSON null returns `Some(Null)`.
    pub async fn get_path(&self, id: &str, path: &[String]) -> Result<Option<Value>> {
        if let Some(index) = self
            .layout
            .fields
            .iter()
            .position(|field| field.path == path)
        {
            let name = column_name(&self.layout.fields[index]);
            let mut scanner = self.dataset.scan();
            scanner
                .project(&[name.as_str()])?
                .filter_expr(col(ID).eq(lit(id)));
            let batches: Vec<RecordBatch> = scanner.try_into_stream().await?.try_collect().await?;
            for batch in batches {
                if batch.num_rows() > 0 {
                    if let Some(value) =
                        scalar_value(batch.column(0), 0, self.layout.fields[index].data_type)?
                    {
                        return Ok(Some(value));
                    }
                }
            }
        }
        let Some(mut value) = self.get(id).await? else {
            return Ok(None);
        };
        for component in path {
            let Some(object) = value.as_object_mut() else {
                return Ok(None);
            };
            let Some(child) = object.remove(component) else {
                return Ok(None);
            };
            value = child;
        }
        Ok(Some(value))
    }

    pub async fn checkout(&mut self, version: u64) -> Result<()> {
        let replacement = Self::from_dataset(self.dataset.checkout_version(version).await?, true)?;
        *self = replacement;
        Ok(())
    }

    pub async fn refresh(&mut self) -> Result<()> {
        let mut dataset = self.dataset.clone();
        dataset.checkout_latest().await?;
        *self = Self::from_dataset(dataset, false)?;
        Ok(())
    }
}

fn invalid(error: impl std::fmt::Display) -> Error {
    Error::invalid_input(error.to_string())
}
fn column_name(field: &ShredField) -> String {
    // Stable names prevent concurrent schema changes from assigning different
    // logical paths to the same physical column, even on stale writer handles.
    let encoded = serde_json::to_vec(field).expect("serializable shred field");
    format!(
        "shred_{}",
        uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, &encoded).simple()
    )
}

fn physical_field(field: &ShredField) -> Result<Field> {
    let data_type = match field.data_type {
        ShredType::String => DataType::LargeUtf8,
        ShredType::Int64 => DataType::Int64,
        ShredType::UInt64 => DataType::UInt64,
        ShredType::Float64 => DataType::Float64,
        ShredType::Bool => DataType::Boolean,
    };
    Ok(
        Field::new(column_name(field), data_type, true).with_metadata(HashMap::from([(
            FIELD_KEY.into(),
            serde_json::to_string(field).map_err(invalid)?,
        )])),
    )
}

fn encode(
    ids: &[String],
    documents: &[ShreddedDocument],
    layout: &ShredLayout,
    schema: Arc<Schema>,
) -> Result<RecordBatch> {
    let residuals = documents
        .iter()
        .map(|document| serde_json::to_string(&document.residual).map_err(invalid))
        .collect::<Result<Vec<_>>>()?;
    let mut arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from_iter_values(ids)),
        Arc::new(LargeStringArray::from_iter_values(residuals)),
    ];
    for (index, field) in layout.fields.iter().enumerate() {
        let values = documents
            .iter()
            .map(|document| document.values[index].as_ref());
        let array: ArrayRef = match field.data_type {
            ShredType::String => Arc::new(LargeStringArray::from_iter(
                values.map(|value| value.and_then(Value::as_str)),
            )),
            ShredType::Int64 => Arc::new(Int64Array::from_iter(
                values.map(|value| value.and_then(Value::as_i64)),
            )),
            ShredType::UInt64 => Arc::new(UInt64Array::from_iter(
                values.map(|value| value.and_then(Value::as_u64)),
            )),
            ShredType::Float64 => Arc::new(Float64Array::from_iter(
                values.map(|value| value.and_then(Value::as_f64)),
            )),
            ShredType::Bool => Arc::new(BooleanArray::from_iter(
                values.map(|value| value.and_then(Value::as_bool)),
            )),
        };
        arrays.push(array);
    }
    Ok(RecordBatch::try_new(schema, arrays)?)
}

fn scalar_value(array: &ArrayRef, row: usize, data_type: ShredType) -> Result<Option<Value>> {
    if array.is_null(row) {
        return Ok(None);
    }
    macro_rules! read {
        ($array_type:ty) => {
            array
                .as_any()
                .downcast_ref::<$array_type>()
                .ok_or_else(|| Error::invalid_input("invalid shredded scalar array"))?
                .value(row)
        };
    }
    Ok(Some(match data_type {
        ShredType::String => Value::from(read!(LargeStringArray)),
        ShredType::Int64 => Value::from(read!(Int64Array)),
        ShredType::UInt64 => Value::from(read!(UInt64Array)),
        ShredType::Float64 => {
            let value = read!(Float64Array);
            if !value.is_finite() {
                return Err(Error::invalid_input("non-finite shredded JSON number"));
            }
            Value::from(value)
        }
        ShredType::Bool => Value::from(read!(BooleanArray)),
    }))
}

fn decode(batch: &RecordBatch, row: usize, layout: &ShredLayout) -> Result<Value> {
    let residual = batch
        .column_by_name(RESIDUAL)
        .and_then(|array| array.as_any().downcast_ref::<LargeStringArray>())
        .ok_or_else(|| Error::invalid_input("missing document residual"))?;
    if residual.is_null(row) {
        return Err(Error::invalid_input("null document residual"));
    }
    let residual: Value = serde_json::from_str(residual.value(row)).map_err(invalid)?;
    let values = layout
        .fields
        .iter()
        .map(|field| {
            let array = batch
                .column_by_name(&column_name(field))
                .ok_or_else(|| Error::invalid_input("missing shredded column"))?;
            scalar_value(array, row, field.data_type)
        })
        .collect::<Result<Vec<_>>>()?;
    layout
        .reconstruct(ShreddedDocument { residual, values })
        .map_err(Error::invalid_input)
}

// Match the JSON reader's nesting limit before accepting a write, including
// arrays that the selector leaves opaque. Programmatically built Values can
// otherwise serialize successfully and fail on every subsequent read.
fn validate_depth(value: &Value, depth: usize) -> Result<()> {
    if matches!(value, Value::Object(_) | Value::Array(_)) && depth >= 127 {
        return Err(Error::invalid_input(
            "document exceeds 127 JSON nesting levels",
        ));
    }
    match value {
        Value::Array(values) => {
            for value in values {
                validate_depth(value, depth + 1)?;
            }
        }
        Value::Object(values) => {
            for value in values.values() {
                validate_depth(value, depth + 1)?;
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests;
