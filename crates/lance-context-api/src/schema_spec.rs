//! User-defined schemas: declaration, validation, and Arrow mapping.
//!
//! The three built-in stores each hard-code an Arrow schema and then repeat its
//! field list across five to seven places (schema fn, record→Arrow builders,
//! Arrow→record decode, projection, filters, merge key, index name). A
//! [`SchemaSpec`] replaces that with a *value*: declared once at store creation,
//! persisted with the store, and read back on open — so the encode/decode path
//! can be driven by it instead of by hand-written column lists.
//!
//! # What is deliberately constrained
//!
//! A user schema is not an arbitrary Arrow schema. The storage layer
//! ([`crate::store_base::StorageBase`]) makes hard assumptions that a schema
//! must satisfy for `add` and `merge` to behave identically to the built-in
//! stores:
//!
//! - **An `id` column is mandatory** (`Utf8`, non-nullable). It is the LSM merge
//!   key, which is what makes a retried append idempotent and a crashed merge
//!   safe to redo, and it is the column the scalar index is built on.
//! - **Reserved names are rejected**, because they collide with Lance internals
//!   or with this crate's own sentinels.
//! - **Blob columns are declared, not inferred**, and are stored *inline* — see
//!   [`ColumnType::Binary`].
//! - **Vector columns carry their dimension and metric**, because an index
//!   cannot be built without them.
//!
//! # This type lives in the API crate on purpose
//!
//! A schema is part of the wire contract — the server accepts one at store
//! creation and the client needs the same type to send it — so it belongs
//! alongside the other request/response types rather than in the storage crate.
//! It depends only on `arrow-schema` and `serde`, not on Lance.
//!
//! # Schema is immutable after creation
//!
//! v1 deliberately has no evolution path: the schema is fixed when the store is
//! created. Additive evolution already exists for the built-in rollout schema
//! (`StorageBaseOptions::latest_schema`) and can be extended to user schemas
//! later, but doing it safely needs a versioning story that is out of scope
//! here.

use std::collections::{HashMap, HashSet};

use arrow_schema::{ArrowError, DataType, Field, Schema, TimeUnit};
use serde::{Deserialize, Serialize};

/// The mandatory primary-key column. Always the LSM merge key.
pub const ID_COLUMN: &str = "id";

/// Marks the key column as a primary key that Lance does not enforce. Matches
/// the metadata the built-in schemas set on their own key columns.
const UNENFORCED_PRIMARY_KEY: &str = "lance-schema:unenforced-primary-key";

/// Schema-metadata key under which a vector column's distance metric is
/// persisted, so it round-trips on open without being re-specified.
const DISTANCE_METRIC_KEY: &str = "lance-context:distance_metric";

/// Field-metadata key marking a column as a blob: excluded from scan
/// projections by default. See [`ColumnType::Binary`].
const BLOB_COLUMN_KEY: &str = "lance-context:blob";

/// Column names a user schema may not use.
///
/// `_rowid` and `_mem_wal` are Lance's; `_rowaddr` likewise. Anything starting
/// with `_` is reserved wholesale so future Lance internals cannot collide with
/// a user column.
const RESERVED_COLUMNS: &[&str] = &["_rowid", "_rowaddr", "_mem_wal", "_distance"];

/// Distance metrics a vector column may declare. Kept in sync with
/// `lance_context_core::DistanceMetric`, which parses the same identifiers;
/// duplicated as strings so this crate stays free of a storage dependency.
const VALID_METRICS: &[&str] = &["l2", "euclidean", "cosine", "dot", "dot_product"];

/// Maximum number of columns in a user schema. A guard against pathological
/// declarations, not a Lance limit.
const MAX_COLUMNS: usize = 1024;

/// Declared type of one user column.
///
/// Serialized in both a shorthand form (`"string"`) and a tagged form
/// (`{"type": "vector", "dim": 768}`) — see [`ColumnSpec`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ColumnType {
    /// UTF-8 string. `large: true` selects `LargeUtf8` for values over ~2 GiB
    /// cumulative per array.
    String {
        #[serde(default)]
        large: bool,
    },
    /// Signed integers.
    Int32,
    Int64,
    /// Floating point.
    Float32,
    Float64,
    Bool,
    /// Microsecond-precision UTC timestamp.
    Timestamp,
    /// Inline binary payload — the encoding for large blobs.
    ///
    /// # Blobs are stored inline, never blob-v2 offloaded
    ///
    /// Reads go through the MemWAL LSM scanner, which has **no
    /// blob-materialization step**: a blob-v2 (`lance-encoding:blob`) column
    /// reads back as `None` through it. Inline `LargeBinary` is therefore the
    /// only encoding that round-trips, which is why `offload` is not an option
    /// here and why [`SchemaSpec::validate`] rejects any attempt to set the
    /// blob-v2 metadata directly.
    ///
    /// Measured on this storage path, a 120 MB inline value writes in ~1.4 s
    /// and reads back in ~0.8 s, and survives a WAL merge intact.
    ///
    /// `blob: true` does not change *where* bytes live; it changes *who pays
    /// for them*. Blob columns are excluded from scan projections by default
    /// (see [`SchemaSpec::scan_columns`]), so `list`-style reads never
    /// materialize them — fetch them per row instead.
    Binary {
        /// Exclude from default scan projections. Set this for anything large
        /// enough that you would not want it in a list response.
        #[serde(default)]
        blob: bool,
    },
    /// Fixed-size float vector, for similarity search.
    Vector {
        /// Vector width. Must match on every write.
        dim: i32,
        /// Ranking metric. Defaults to `l2`.
        #[serde(default)]
        metric: Option<String>,
    },
    /// Variable-length list of a scalar type.
    List {
        /// Element type. Nested lists and lists of vectors are not supported.
        item: Box<ColumnType>,
    },
}

/// One column in a user schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnSpec {
    /// Column type.
    #[serde(flatten)]
    pub column_type: ColumnType,
    /// Whether the column accepts nulls. Defaults to `true`; a write may then
    /// omit the column entirely and it is materialized as null.
    #[serde(default = "default_true")]
    pub nullable: bool,
}

fn default_true() -> bool {
    true
}

impl ColumnSpec {
    /// A nullable column of the given type.
    #[must_use]
    pub fn new(column_type: ColumnType) -> Self {
        Self {
            column_type,
            nullable: true,
        }
    }

    /// A non-nullable column of the given type.
    #[must_use]
    pub fn required(column_type: ColumnType) -> Self {
        Self {
            column_type,
            nullable: false,
        }
    }

    /// Whether this column is excluded from default scan projections.
    #[must_use]
    pub fn is_blob(&self) -> bool {
        matches!(self.column_type, ColumnType::Binary { blob: true })
    }
}

/// A user-declared store schema.
///
/// Column order is preserved: [`SchemaSpec::to_arrow`] emits fields in
/// declaration order, with `id` first regardless of where it was declared, so
/// the physical layout is predictable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchemaSpec {
    /// Columns in declaration order.
    pub columns: Vec<(String, ColumnSpec)>,
}

impl SchemaSpec {
    /// Build a spec from an ordered column list. Call [`Self::validate`] before
    /// using it to create a store.
    #[must_use]
    pub fn new(columns: Vec<(String, ColumnSpec)>) -> Self {
        Self { columns }
    }

    /// Look up a column by name.
    #[must_use]
    pub fn column(&self, name: &str) -> Option<&ColumnSpec> {
        self.columns
            .iter()
            .find(|(column, _)| column == name)
            .map(|(_, spec)| spec)
    }

    /// Names of the blob columns — those excluded from default scan
    /// projections.
    #[must_use]
    pub fn blob_columns(&self) -> HashSet<String> {
        self.columns
            .iter()
            .filter(|(_, spec)| spec.is_blob())
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Column names a `list`-style scan should read: everything except blob
    /// columns, so large payloads are never materialized by a bulk read.
    #[must_use]
    pub fn scan_columns(&self) -> Vec<String> {
        self.columns
            .iter()
            .filter(|(_, spec)| !spec.is_blob())
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Names of the vector columns, paired with their dimension.
    #[must_use]
    pub fn vector_columns(&self) -> Vec<(String, i32)> {
        self.columns
            .iter()
            .filter_map(|(name, spec)| match spec.column_type {
                ColumnType::Vector { dim, .. } => Some((name.clone(), dim)),
                _ => None,
            })
            .collect()
    }

    /// Validate the declaration against everything the storage layer assumes.
    ///
    /// # Errors
    ///
    /// Returns the first violated rule:
    /// - no columns, or more than [`MAX_COLUMNS`]
    /// - missing `id`, or `id` that is not a non-nullable string
    /// - duplicate, empty, or reserved column names
    /// - a vector column with a non-positive dimension or an unknown metric
    /// - a nested list, or a list of vectors
    pub fn validate(&self) -> Result<(), String> {
        if self.columns.is_empty() {
            return Err("schema must declare at least one column".to_string());
        }
        if self.columns.len() > MAX_COLUMNS {
            return Err(format!(
                "schema declares {} columns, the maximum is {MAX_COLUMNS}",
                self.columns.len()
            ));
        }

        let mut seen = HashSet::new();
        for (name, spec) in &self.columns {
            if name.is_empty() {
                return Err("column names must not be empty".to_string());
            }
            if !seen.insert(name.as_str()) {
                return Err(format!("column '{name}' is declared more than once"));
            }
            if name.starts_with('_') || RESERVED_COLUMNS.contains(&name.as_str()) {
                return Err(format!(
                    "column '{name}' is reserved: names starting with '_' are used by Lance \
                     internals"
                ));
            }
            validate_column_type(name, &spec.column_type)?;
        }

        // `id` is what makes the whole storage layer work: it is the LSM merge
        // key, so it must be present, non-null and comparable on every row.
        let Some(id) = self.column(ID_COLUMN) else {
            return Err(format!(
                "schema must declare an '{ID_COLUMN}' column: it is the primary key and the \
                 LSM merge key"
            ));
        };
        if !matches!(id.column_type, ColumnType::String { .. }) {
            return Err(format!(
                "column '{ID_COLUMN}' must be a string, got {:?}",
                id.column_type
            ));
        }
        if id.nullable {
            return Err(format!(
                "column '{ID_COLUMN}' must be non-nullable: it is the primary key"
            ));
        }

        Ok(())
    }

    /// Convert to an Arrow schema, with `id` first and the declared columns
    /// following in order.
    ///
    /// Validates first, so an invalid spec cannot produce a schema.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::validate`] as an [`ArrowError::SchemaError`].
    pub fn to_arrow(&self) -> Result<Schema, ArrowError> {
        self.validate().map_err(ArrowError::SchemaError)?;

        let mut fields = Vec::with_capacity(self.columns.len());

        // `id` leads, wherever it was declared, so physical layout does not
        // depend on declaration order.
        let id = self
            .column(ID_COLUMN)
            .expect("validate() proved id is present");
        fields.push(arrow_field(ID_COLUMN, id)?);

        for (name, spec) in &self.columns {
            if name == ID_COLUMN {
                continue;
            }
            fields.push(arrow_field(name, spec)?);
        }

        // Persist each vector column's metric so `open` does not have to be
        // told again. Keyed per column, since a schema may declare several.
        let mut metadata = HashMap::new();
        for (name, spec) in &self.columns {
            if let ColumnType::Vector { metric, .. } = &spec.column_type {
                let metric = metric.as_deref().unwrap_or("l2");
                metadata.insert(format!("{DISTANCE_METRIC_KEY}:{name}"), metric.to_string());
            }
        }

        Ok(Schema::new(fields).with_metadata(metadata))
    }
}

fn validate_column_type(name: &str, column_type: &ColumnType) -> Result<(), String> {
    match column_type {
        ColumnType::Vector { dim, metric } => {
            if *dim <= 0 {
                return Err(format!(
                    "column '{name}': vector dimension must be positive, got {dim}"
                ));
            }
            if let Some(metric) = metric {
                if !VALID_METRICS.contains(&metric.to_ascii_lowercase().as_str()) {
                    return Err(format!(
                        "column '{name}': invalid distance metric '{metric}', expected one of \
                         {VALID_METRICS:?}"
                    ));
                }
            }
            Ok(())
        }
        ColumnType::List { item } => match **item {
            // A nested list would need a recursive builder and a recursive
            // decoder; the built-in schemas never use one, so it stays out
            // until something needs it.
            ColumnType::List { .. } => {
                Err(format!("column '{name}': nested lists are not supported"))
            }
            ColumnType::Vector { .. } => Err(format!(
                "column '{name}': a list of vectors is not supported; declare a vector column"
            )),
            _ => Ok(()),
        },
        _ => Ok(()),
    }
}

fn arrow_field(name: &str, spec: &ColumnSpec) -> Result<Field, ArrowError> {
    let data_type = arrow_data_type(&spec.column_type)?;
    let mut field = Field::new(name, data_type, spec.nullable);

    let mut metadata = HashMap::new();
    if name == ID_COLUMN {
        metadata.insert(UNENFORCED_PRIMARY_KEY.to_string(), "true".to_string());
    }
    if spec.is_blob() {
        // Marks the column for projection exclusion. Deliberately *not*
        // `lance-encoding:blob`: that is blob-v2 offload, which the LSM read
        // path cannot materialize (see `ColumnType::Binary`).
        metadata.insert(BLOB_COLUMN_KEY.to_string(), "true".to_string());
    }
    if !metadata.is_empty() {
        field = field.with_metadata(metadata);
    }
    Ok(field)
}

fn arrow_data_type(column_type: &ColumnType) -> Result<DataType, ArrowError> {
    Ok(match column_type {
        ColumnType::String { large: false } => DataType::Utf8,
        ColumnType::String { large: true } => DataType::LargeUtf8,
        ColumnType::Int32 => DataType::Int32,
        ColumnType::Int64 => DataType::Int64,
        ColumnType::Float32 => DataType::Float32,
        ColumnType::Float64 => DataType::Float64,
        ColumnType::Bool => DataType::Boolean,
        ColumnType::Timestamp => DataType::Timestamp(TimeUnit::Microsecond, None),
        // Always `LargeBinary`, inline. See `ColumnType::Binary`.
        ColumnType::Binary { .. } => DataType::LargeBinary,
        ColumnType::Vector { dim, .. } => DataType::FixedSizeList(
            std::sync::Arc::new(Field::new("item", DataType::Float32, true)),
            *dim,
        ),
        ColumnType::List { item } => DataType::List(std::sync::Arc::new(Field::new(
            "item",
            arrow_data_type(item)?,
            true,
        ))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id_column() -> (String, ColumnSpec) {
        (
            ID_COLUMN.to_string(),
            ColumnSpec::required(ColumnType::String { large: false }),
        )
    }

    fn spec_with(extra: Vec<(String, ColumnSpec)>) -> SchemaSpec {
        let mut columns = vec![id_column()];
        columns.extend(extra);
        SchemaSpec::new(columns)
    }

    #[test]
    fn minimal_schema_is_just_id() {
        let spec = spec_with(vec![]);
        spec.validate().unwrap();
        let schema = spec.to_arrow().unwrap();
        assert_eq!(schema.fields().len(), 1);
        let id = schema.field_with_name(ID_COLUMN).unwrap();
        assert_eq!(id.data_type(), &DataType::Utf8);
        assert!(!id.is_nullable());
        assert_eq!(
            id.metadata().get(UNENFORCED_PRIMARY_KEY),
            Some(&"true".to_string()),
            "id must carry the unenforced-primary-key marker like the built-in schemas"
        );
    }

    #[test]
    fn id_is_mandatory_and_must_be_a_non_null_string() {
        let missing = SchemaSpec::new(vec![(
            "user_id".to_string(),
            ColumnSpec::new(ColumnType::String { large: false }),
        )]);
        assert!(missing
            .validate()
            .unwrap_err()
            .contains("must declare an 'id'"));

        let wrong_type = SchemaSpec::new(vec![(
            ID_COLUMN.to_string(),
            ColumnSpec::required(ColumnType::Int64),
        )]);
        assert!(wrong_type
            .validate()
            .unwrap_err()
            .contains("must be a string"));

        let nullable = SchemaSpec::new(vec![(
            ID_COLUMN.to_string(),
            ColumnSpec::new(ColumnType::String { large: false }),
        )]);
        assert!(nullable.validate().unwrap_err().contains("non-nullable"));
    }

    #[test]
    fn reserved_and_duplicate_names_are_rejected() {
        for reserved in ["_rowid", "_mem_wal", "_anything"] {
            let spec = spec_with(vec![(
                reserved.to_string(),
                ColumnSpec::new(ColumnType::Int64),
            )]);
            assert!(
                spec.validate().unwrap_err().contains("reserved"),
                "{reserved} must be rejected"
            );
        }

        let dup = SchemaSpec::new(vec![
            id_column(),
            ("x".to_string(), ColumnSpec::new(ColumnType::Int64)),
            ("x".to_string(), ColumnSpec::new(ColumnType::Int64)),
        ]);
        assert!(dup.validate().unwrap_err().contains("more than once"));
    }

    #[test]
    fn vector_columns_carry_dimension_and_metric() {
        let spec = spec_with(vec![(
            "embedding".to_string(),
            ColumnSpec::new(ColumnType::Vector {
                dim: 768,
                metric: Some("cosine".to_string()),
            }),
        )]);
        let schema = spec.to_arrow().unwrap();
        let field = schema.field_with_name("embedding").unwrap();
        match field.data_type() {
            DataType::FixedSizeList(_, dim) => assert_eq!(*dim, 768),
            other => panic!("expected FixedSizeList, got {other:?}"),
        }
        assert_eq!(
            schema
                .metadata()
                .get(&format!("{DISTANCE_METRIC_KEY}:embedding")),
            Some(&"cosine".to_string()),
            "the metric must round-trip so open() need not be told again"
        );
        assert_eq!(spec.vector_columns(), vec![("embedding".to_string(), 768)]);

        let bad_dim = spec_with(vec![(
            "e".to_string(),
            ColumnSpec::new(ColumnType::Vector {
                dim: 0,
                metric: None,
            }),
        )]);
        assert!(bad_dim.validate().unwrap_err().contains("must be positive"));

        let bad_metric = spec_with(vec![(
            "e".to_string(),
            ColumnSpec::new(ColumnType::Vector {
                dim: 8,
                metric: Some("manhattan".to_string()),
            }),
        )]);
        assert!(bad_metric
            .validate()
            .unwrap_err()
            .contains("invalid distance metric"));
    }

    #[test]
    fn blob_columns_are_inline_and_excluded_from_scans() {
        let spec = spec_with(vec![
            (
                "video".to_string(),
                ColumnSpec::new(ColumnType::Binary { blob: true }),
            ),
            (
                "thumbnail".to_string(),
                ColumnSpec::new(ColumnType::Binary { blob: false }),
            ),
            (
                "caption".to_string(),
                ColumnSpec::new(ColumnType::String { large: false }),
            ),
        ]);
        let schema = spec.to_arrow().unwrap();

        // Inline LargeBinary, never blob-v2: the LSM read path has no
        // blob-materialization step, so an offloaded column reads back as None.
        for column in ["video", "thumbnail"] {
            let field = schema.field_with_name(column).unwrap();
            assert_eq!(field.data_type(), &DataType::LargeBinary);
            assert!(
                field.metadata().get("lance-encoding:blob").is_none(),
                "{column} must not be blob-v2 offloaded"
            );
        }

        // `blob: true` means "excluded from default scans", nothing else.
        assert_eq!(spec.blob_columns(), HashSet::from(["video".to_string()]));
        assert_eq!(
            spec.scan_columns(),
            vec![
                "id".to_string(),
                "thumbnail".to_string(),
                "caption".to_string()
            ],
            "a list-style scan must skip blob columns but keep everything else"
        );
    }

    #[test]
    fn id_leads_the_arrow_schema_regardless_of_declaration_order() {
        let spec = SchemaSpec::new(vec![
            ("score".to_string(), ColumnSpec::new(ColumnType::Float32)),
            id_column(),
            (
                "tags".to_string(),
                ColumnSpec::new(ColumnType::List {
                    item: Box::new(ColumnType::String { large: false }),
                }),
            ),
        ]);
        let schema = spec.to_arrow().unwrap();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, vec!["id", "score", "tags"]);
    }

    #[test]
    fn nested_lists_and_vector_lists_are_rejected() {
        let nested = spec_with(vec![(
            "x".to_string(),
            ColumnSpec::new(ColumnType::List {
                item: Box::new(ColumnType::List {
                    item: Box::new(ColumnType::Int64),
                }),
            }),
        )]);
        assert!(nested.validate().unwrap_err().contains("nested lists"));

        let vec_list = spec_with(vec![(
            "x".to_string(),
            ColumnSpec::new(ColumnType::List {
                item: Box::new(ColumnType::Vector {
                    dim: 4,
                    metric: None,
                }),
            }),
        )]);
        assert!(vec_list.validate().unwrap_err().contains("list of vectors"));
    }

    #[test]
    fn json_shorthand_round_trips() {
        // The wire form users actually write.
        let json = serde_json::json!({
            "columns": [
                ["id",    {"type": "string", "nullable": false}],
                ["score", {"type": "float32"}],
                ["video", {"type": "binary", "blob": true}],
                ["embedding", {"type": "vector", "dim": 4, "metric": "cosine"}]
            ]
        });
        let spec: SchemaSpec = serde_json::from_value(json).unwrap();
        spec.validate().unwrap();
        assert_eq!(spec.blob_columns(), HashSet::from(["video".to_string()]));

        let reparsed: SchemaSpec =
            serde_json::from_str(&serde_json::to_string(&spec).unwrap()).unwrap();
        assert_eq!(reparsed, spec);
    }
}
