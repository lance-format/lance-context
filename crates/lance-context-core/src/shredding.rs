//! Automatic, bounded JSON field selection and lossless residual encoding.
//!
//! Layouts are additive: an existing column never changes meaning. A later
//! batch can promote new paths; older rows still decode with their old layout
//! or with nulls in the newly added columns. Arrays and nonmatching types stay
//! in the residual. No document text is duplicated in the encoded record.

use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Selection limits, independent of any user-declared JSON paths.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShredPolicy {
    pub max_columns: usize,
    pub sample_rows: usize,
    /// Maximum JSON nodes visited across the sample, including containers.
    pub sample_nodes: usize,
    pub max_candidates: usize,
    pub max_depth: usize,
    pub max_path_bytes: usize,
    /// Promote a scalar type seen in at least this fraction of sampled rows.
    pub min_frequency: f64,
    /// Also promote a rare field when its sampled scalar bytes reach this size.
    pub min_value_bytes: usize,
}

impl Default for ShredPolicy {
    fn default() -> Self {
        Self {
            max_columns: 32,
            sample_rows: 256,
            sample_nodes: 100_000,
            max_candidates: 4096,
            max_depth: 32,
            max_path_bytes: 1024,
            min_frequency: 0.5,
            min_value_bytes: 64 * 1024,
        }
    }
}

impl ShredPolicy {
    pub fn validate(&self) -> Result<(), String> {
        if self.max_columns == 0
            || self.sample_rows == 0
            || self.sample_nodes == 0
            || self.max_candidates == 0
            || self.max_depth == 0
            || self.max_depth > 128
            || self.max_path_bytes == 0
            || self.min_value_bytes == 0
            || !self.min_frequency.is_finite()
            || !(0.0..=1.0).contains(&self.min_frequency)
        {
            return Err("invalid automatic shredding policy limits".into());
        }
        Ok(())
    }
}

/// Scalar storage type. Integers never pass through floating-point conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShredType {
    String,
    Int64,
    UInt64,
    Float64,
    Bool,
}

impl ShredType {
    pub fn of(value: &Value) -> Option<Self> {
        match value {
            Value::String(_) => Some(Self::String),
            Value::Bool(_) => Some(Self::Bool),
            Value::Number(n) if n.is_i64() => Some(Self::Int64),
            Value::Number(n) if n.is_u64() => Some(Self::UInt64),
            Value::Number(number) => number
                .as_f64()
                .and_then(serde_json::Number::from_f64)
                .filter(|round_trip| round_trip == number)
                .map(|_| Self::Float64),
            _ => None,
        }
    }
}

/// An automatically chosen logical object path and its physical scalar type.
/// Path components are literal keys, so dots, slashes, and empty keys are safe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShredField {
    pub path: Vec<String>,
    pub data_type: ShredType,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShredLayout {
    pub fields: Vec<ShredField>,
}

#[derive(Default)]
struct Statistics {
    occurrences: usize,
    bytes: usize,
}

/// The extracted values correspond positionally to a validated layout.
/// `None` means the value was not extracted; explicit JSON null stays residual.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShreddedDocument {
    pub residual: Value,
    pub values: Vec<Option<Value>>,
}

impl ShredLayout {
    pub fn validate(&self) -> Result<(), String> {
        let mut paths = HashSet::new();
        for field in &self.fields {
            if field.path.is_empty() || !paths.insert(&field.path) {
                return Err("shred layout contains an empty or duplicate path".into());
            }
        }
        Ok(())
    }

    /// Select additional columns from this batch without changing existing ones.
    /// Rank by sampled value bytes, then frequency, then path/type for ties.
    /// This gives large, infrequent text a chance to outrank tiny common fields.
    pub fn evolve(&self, documents: &[Value], policy: &ShredPolicy) -> Result<Self, String> {
        self.validate()?;
        policy.validate()?;
        if self.fields.len() > policy.max_columns {
            return Err("existing layout exceeds the configured column budget".into());
        }
        let mut next = self.clone();
        if next.fields.len() == policy.max_columns {
            return Ok(next);
        }
        let existing: HashSet<_> = self.fields.iter().map(|field| &field.path).collect();
        let mut stats = BTreeMap::<(Vec<String>, ShredType), Statistics>::new();
        let mut remaining = policy.sample_nodes;
        let mut sampled = 0;
        for document in documents.iter().take(policy.sample_rows) {
            if remaining == 0 {
                break;
            }
            sampled += 1;
            profile(
                document,
                &mut Vec::new(),
                0,
                &mut remaining,
                &mut stats,
                policy,
            );
        }
        let mut candidates: Vec<_> = stats
            .into_iter()
            .filter(|((path, _), stat)| {
                !existing.contains(path)
                    && (stat.bytes >= policy.min_value_bytes
                        || stat.occurrences as f64 >= sampled as f64 * policy.min_frequency)
            })
            .collect();
        candidates.sort_by(|(left_key, left), (right_key, right)| {
            right
                .bytes
                .cmp(&left.bytes)
                .then_with(|| right.occurrences.cmp(&left.occurrences))
                .then_with(|| left_key.cmp(right_key))
        });
        for ((path, data_type), _) in candidates {
            if next.fields.len() == policy.max_columns {
                break;
            }
            // One stable type per path; other types stay in the residual.
            if !next.fields.iter().any(|field| field.path == path) {
                next.fields.push(ShredField { path, data_type });
            }
        }
        Ok(next)
    }

    /// Consume the document so extracted strings are moved, not copied.
    pub fn shred(&self, mut document: Value) -> Result<ShreddedDocument, String> {
        self.validate()?;
        let mut values = Vec::with_capacity(self.fields.len());
        for field in &self.fields {
            let (key, parents) = field.path.split_last().expect("validated path");
            let parent = object_at_mut(&mut document, parents);
            let value = match parent {
                Some(object)
                    if object.get(key).and_then(ShredType::of) == Some(field.data_type) =>
                {
                    object.remove(key)
                }
                _ => None,
            };
            values.push(value);
        }
        Ok(ShreddedDocument {
            residual: document,
            values,
        })
    }

    /// Restore the logical JSON value, rejecting malformed or conflicting data.
    pub fn reconstruct(&self, encoded: ShreddedDocument) -> Result<Value, String> {
        self.validate()?;
        if encoded.values.len() != self.fields.len() {
            return Err("shredded value count does not match layout".into());
        }
        let mut document = encoded.residual;
        for (field, value) in self.fields.iter().zip(encoded.values) {
            let Some(value) = value else { continue };
            if ShredType::of(&value) != Some(field.data_type) {
                return Err("shredded value has the wrong scalar type".into());
            }
            let (key, parents) = field.path.split_last().expect("validated path");
            let object = object_at_mut(&mut document, parents)
                .ok_or("shredded path has no residual object parent")?;
            if object.contains_key(key) {
                return Err("shredded value conflicts with residual field".into());
            }
            object.insert(key.clone(), value);
        }
        Ok(document)
    }
}

fn object_at_mut<'a>(
    document: &'a mut Value,
    path: &[String],
) -> Option<&'a mut serde_json::Map<String, Value>> {
    let mut node = document;
    for key in path {
        node = node.as_object_mut()?.get_mut(key)?;
    }
    node.as_object_mut()
}

fn profile(
    value: &Value,
    path: &mut Vec<String>,
    path_bytes: usize,
    remaining: &mut usize,
    stats: &mut BTreeMap<(Vec<String>, ShredType), Statistics>,
    policy: &ShredPolicy,
) {
    if *remaining == 0 {
        return;
    }
    *remaining -= 1;
    if let Some(data_type) = ShredType::of(value) {
        if path.is_empty() {
            return;
        }
        let key = (path.clone(), data_type);
        if stats.len() == policy.max_candidates && !stats.contains_key(&key) {
            return;
        }
        let stat = stats.entry(key).or_default();
        stat.occurrences += 1;
        let bytes = match value {
            Value::String(text) => text.len(),
            Value::Bool(_) => 1,
            _ => 8,
        };
        stat.bytes = stat.bytes.saturating_add(bytes);
    } else if let Value::Object(object) = value {
        if path.len() == policy.max_depth {
            return;
        }
        for (key, child) in object {
            // Charge skipped keys too: a huge object cannot bypass the node budget.
            if *remaining == 0 {
                break;
            }
            let bytes = path_bytes.saturating_add(key.len());
            if bytes > policy.max_path_bytes {
                *remaining -= 1;
                continue;
            }
            path.push(key.clone());
            profile(child, path, bytes, remaining, stats, policy);
            path.pop();
        }
    }
}

#[cfg(test)]
mod tests;
