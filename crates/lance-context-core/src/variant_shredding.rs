//! Automatic layout selection with Apache Arrow's standard Variant kernels.
//!
//! Unlike the JSON-residual baseline, this module emits `arrow.parquet.variant`
//! arrays with standard binary metadata/value and shredded typed_value children.
//! No custom Variant codec or reconstruction format is used.

use std::{collections::BTreeMap, sync::Arc};

use arrow_array::{Array, ArrayRef, BooleanArray, StructArray};
use arrow_cast::cast;
use arrow_schema::{ArrowError, DataType, Field, Fields};
use parquet_variant::{ObjectFieldBuilder, Variant, VariantBuilderExt};
use parquet_variant_compute::{
    shred_variant, unshred_variant, VariantArray, VariantArrayBuilder, VariantValueArrayBuilder,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::shredding::{ShredField, ShredLayout, ShredPolicy, ShredType};

type Result<T> = std::result::Result<T, ArrowError>;

/// Build standard Variant values with Arrow's builder. In Arrow 58 the JSON
/// string parser rounds unsigned integers above i64 through f64; the typed
/// builder instead encodes u64 losslessly as a standard scale-zero decimal.
/// None denotes SQL null; Some(Value::Null) denotes Variant null.
pub fn json_values_to_variant<'a>(
    documents: impl IntoIterator<Item = Option<&'a Value>>,
) -> Result<VariantArray> {
    fn append(value: &Value, builder: &mut impl VariantBuilderExt, depth: usize) -> Result<()> {
        if depth >= 128 {
            return Err(invalid("document exceeds 127 JSON nesting levels"));
        }
        match value {
            Value::Null => builder.append_value(Variant::Null),
            Value::Bool(value) => builder.append_value(*value),
            Value::String(value) => builder.append_value(value.as_str()),
            Value::Number(value) => {
                if let Some(value) = value.as_i64() {
                    builder.append_value(value);
                } else if let Some(value) = value.as_u64() {
                    builder.append_value(value);
                } else {
                    builder.append_value(
                        value
                            .as_f64()
                            .ok_or_else(|| invalid("invalid JSON number"))?,
                    );
                }
            }
            Value::Array(values) => {
                let mut list = builder.try_new_list()?;
                for value in values {
                    append(value, &mut list, depth + 1)?;
                }
                list.finish();
            }
            Value::Object(values) => {
                let mut object = builder.try_new_object()?;
                for (key, value) in values {
                    append(
                        value,
                        &mut ObjectFieldBuilder::new(key, &mut object),
                        depth + 1,
                    )?;
                }
                object.finish();
            }
        }
        Ok(())
    }
    let documents = documents.into_iter();
    let mut builder = VariantArrayBuilder::new(documents.size_hint().0);
    for document in documents {
        match document {
            Some(document) => append(document, &mut builder, 0)?,
            None => builder.append_null(),
        }
    }
    Ok(builder.build())
}

/// Reconstruct JSON using the upstream unshred and JSON kernels.
pub fn variant_to_json_values(input: &VariantArray) -> Result<Vec<Option<Value>>> {
    use parquet_variant_json::VariantToJson;
    let plain = unshred_variant(input)?;
    (0..plain.len())
        .map(|row| {
            if plain.is_null(row) {
                Ok(None)
            } else {
                plain.try_value(row)?.to_json_value().map(Some)
            }
        })
        .collect()
}

/// A bounded, automatically selected target for the upstream shredding kernel.
/// Selection is independent of the baseline's JSON residual encoding. Each
/// inference can select a new layout, including new fields and changed types.
/// Persisted stores must migrate data before replacing their physical layout.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AutoVariantLayout {
    fields: Vec<ShredField>,
}

impl AutoVariantLayout {
    /// Choose fields from a bounded sample; callers provide limits, not paths.
    /// Arrays remain standard Variant residuals. Unsigned integers outside i64
    /// remain residuals because Variant has no unsigned integer shred type.
    pub fn infer(documents: &[Value], policy: &ShredPolicy) -> Result<Self> {
        let discovered = ShredLayout::default()
            .evolve(documents, policy)
            .map_err(ArrowError::InvalidArgumentError)?;
        let mut fields: Vec<ShredField> = Vec::new();
        for field in discovered.fields {
            if field.data_type == ShredType::UInt64
                || fields.iter().any(|other| {
                    other.path.starts_with(&field.path) || field.path.starts_with(&other.path)
                })
            {
                continue;
            }
            fields.push(field);
        }
        Ok(Self { fields })
    }

    pub fn fields(&self) -> &[ShredField] {
        &self.fields
    }

    /// Reconsider sampled paths on every append. String paths take priority;
    /// new sampled paths precede old paths of the same kind. Keep old paths in
    /// remaining slots and sort the result to avoid order-only migrations.
    /// Stores must rewrite atomically when this returns a different layout.
    pub fn evolve(&self, documents: &[Value], policy: &ShredPolicy) -> Result<Self> {
        self.target_type()?;
        let proposed = Self::infer(documents, policy)?;
        let mut candidates = proposed.fields;
        candidates.extend(self.fields.iter().cloned());
        candidates.sort_by_key(|field| field.data_type != ShredType::String);
        let mut fields: Vec<ShredField> = Vec::new();
        for field in candidates {
            if fields.len() == policy.max_columns {
                break;
            }
            if fields.iter().any(|other| {
                other.path.starts_with(&field.path) || field.path.starts_with(&other.path)
            }) {
                continue;
            }
            fields.push(field);
        }
        fields.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(Self { fields })
    }

    pub fn target_type(&self) -> Result<DataType> {
        let layout = ShredLayout {
            fields: self.fields.clone(),
        };
        layout
            .validate()
            .map_err(ArrowError::InvalidArgumentError)?;
        let mut root = TargetNode::default();
        for field in &self.fields {
            let mut node = &mut root;
            for key in &field.path {
                if node.scalar.is_some() {
                    return Err(invalid("overlapping Variant target paths"));
                }
                node = node.children.entry(key.clone()).or_default();
            }
            if !node.children.is_empty() {
                return Err(invalid("overlapping Variant target paths"));
            }
            node.scalar = Some(match field.data_type {
                ShredType::String => DataType::Utf8,
                ShredType::Int64 => DataType::Int64,
                ShredType::Float64 => DataType::Float64,
                ShredType::Bool => DataType::Boolean,
                ShredType::UInt64 => return Err(invalid("Variant has no unsigned shred type")),
            });
        }
        Ok(root.data_type())
    }

    /// Run Arrow's kernel, then restore only fields it coerced across JSON types.
    /// Incompatible values live in their standard field-level residuals while
    /// compatible siblings remain shredded. Already-shredded input is first
    /// reconstructed by Arrow's unshred kernel.
    pub fn shred(&self, input: &VariantArray) -> Result<VariantArray> {
        let target = self.target_type()?;
        let input = unshred_variant(input)?;
        if self.fields.is_empty() {
            return normalize(&input);
        }
        let originals = (0..input.len())
            .map(|row| {
                if input.is_null(row) {
                    Ok(None)
                } else {
                    input.try_value(row).map(Some)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let shredded = shred_variant(&input, &target)?;
        let restored = restore_coerced_fields(&originals, &target, shredded.inner())?;
        normalize(&VariantArray::try_new(&restored)?)
    }
}

#[derive(Default)]
struct TargetNode {
    scalar: Option<DataType>,
    children: BTreeMap<String, TargetNode>,
}
impl TargetNode {
    fn data_type(self) -> DataType {
        self.scalar.unwrap_or_else(|| {
            DataType::Struct(
                self.children
                    .into_iter()
                    .map(|(name, child)| Arc::new(Field::new(name, child.data_type(), true)))
                    .collect::<Fields>(),
            )
        })
    }
}

// Arrow's default safe casts still allow true -> 1, 1.5 -> 1, and number ->
// string. Keep its object routing, missing/null handling and binary encoding,
// but replace such casts with the original value at the affected leaf only.
// Value builders retain the original metadata dictionary, including object keys.
fn restore_coerced_fields(
    originals: &[Option<Variant<'_, '_>>],
    target: &DataType,
    shredded: &StructArray,
) -> Result<StructArray> {
    let typed_index = shredded
        .fields()
        .iter()
        .position(|field| field.name() == "typed_value")
        .ok_or_else(|| invalid("missing shredded typed_value"))?;
    let typed = shredded.column(typed_index);
    let mut columns = shredded.columns().to_vec();
    if let DataType::Struct(fields) = target {
        let children = typed
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| invalid("expected shredded object"))?;
        let mut restored = Vec::with_capacity(fields.len());
        for (index, field) in fields.iter().enumerate() {
            let values = originals
                .iter()
                .map(|value| match value {
                    Some(Variant::Object(object)) => object.get(field.name()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let child = children
                .column(index)
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| invalid("expected shredded field"))?;
            restored.push(
                Arc::new(restore_coerced_fields(&values, field.data_type(), child)?) as ArrayRef,
            );
        }
        columns[typed_index] = Arc::new(StructArray::try_new(
            children.fields().clone(),
            restored,
            children.nulls().cloned(),
        )?);
    } else {
        let restore = originals
            .iter()
            .enumerate()
            .map(|(row, value)| {
                typed.is_valid(row) && value.as_ref().is_some_and(|value| !fits(value, target))
            })
            .collect::<Vec<_>>();
        if !restore.iter().any(|restore| *restore) {
            return Ok(shredded.clone());
        }
        let mut residual = VariantValueArrayBuilder::new(originals.len());
        for (value, restore) in originals.iter().zip(&restore) {
            if *restore {
                residual.append_value(value.as_ref().unwrap().clone());
            } else {
                residual.append_null();
            }
        }
        let value_index = shredded
            .fields()
            .iter()
            .position(|field| field.name() == "value")
            .ok_or_else(|| invalid("missing shredded residual"))?;
        let original: ArrayRef = Arc::new(residual.build()?);
        let original = cast(&original, columns[value_index].data_type())?;
        let restore = BooleanArray::from(restore);
        columns[value_index] = arrow_select::zip::zip(&restore, &original, &columns[value_index])?;
        columns[typed_index] = arrow_select::zip::zip(
            &restore,
            &arrow_array::new_null_array(typed.data_type(), typed.len()),
            typed,
        )?;
    }
    StructArray::try_new(
        shredded.fields().clone(),
        columns,
        shredded.nulls().cloned(),
    )
}

fn fits(value: &Variant<'_, '_>, target: &DataType) -> bool {
    matches!(
        (value, target),
        (Variant::Null, _)
            | (Variant::String(_) | Variant::ShortString(_), DataType::Utf8)
            | (
                Variant::Int8(_) | Variant::Int16(_) | Variant::Int32(_) | Variant::Int64(_),
                DataType::Int64,
            )
            | (Variant::Double(_), DataType::Float64)
            | (
                Variant::BooleanTrue | Variant::BooleanFalse,
                DataType::Boolean
            )
    )
}

// Normalize views before Lance's write boundary, keeping this adapter independent
// of an upstream Lance upgrade. Arrow cast recurses through structs and preserves
// field metadata; the Variant extension belongs to the enclosing field().
fn normalize(input: &VariantArray) -> Result<VariantArray> {
    fn physical_type(data_type: &DataType) -> DataType {
        match data_type {
            DataType::BinaryView => DataType::Binary,
            DataType::Utf8View => DataType::Utf8,
            DataType::Struct(fields) => DataType::Struct(
                fields
                    .iter()
                    .map(|field| {
                        Arc::new(
                            field
                                .as_ref()
                                .clone()
                                .with_data_type(physical_type(field.data_type())),
                        )
                    })
                    .collect(),
            ),
            other => other.clone(),
        }
    }
    let input: ArrayRef = Arc::new(input.inner().clone());
    let normalized = cast(&input, &physical_type(input.data_type()))?;
    VariantArray::try_new(normalized.as_ref())
}

fn invalid(message: impl Into<String>) -> ArrowError {
    ArrowError::InvalidArgumentError(message.into())
}

#[cfg(test)]
mod tests;
