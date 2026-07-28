use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use arrow_array::{
    Array, Int32Array, LargeStringArray, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use futures::TryStreamExt;
use lance::dataset::{builder::DatasetBuilder, Dataset, WriteMode, WriteParams};
use lance::io::{ObjectStoreParams, StorageOptionsAccessor};
use lance::{Error as LanceError, Result as LanceResult};
use uuid::Uuid;

use crate::storage::join_uri;
use crate::store::{ContextStore, ContextStoreOptions};

const MANIFEST_TABLE_NAME: &str = "__manifest";
const PARTITION_TABLE_NAME: &str = "dataset";

/// Complete selector values for one context partition.
pub type PartitionSelector = BTreeMap<String, String>;

/// Phase-1 partition specification for a context namespace.
///
/// This intentionally models only identity partition fields. More advanced
/// Lance partition transforms (`bucket`, `truncate`, time transforms) can be
/// layered in once the namespace resolver grows past the single-partition path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionSpec {
    version: i32,
    fields: Vec<String>,
}

impl PartitionSpec {
    /// Create a v1 identity partition spec from ordered field names.
    pub fn new<I, S>(fields: I) -> LanceResult<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::with_version(1, fields)
    }

    /// Create an identity partition spec with an explicit version.
    pub fn with_version<I, S>(version: i32, fields: I) -> LanceResult<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        if version <= 0 {
            return Err(invalid_input("partition spec version must be positive"));
        }

        let mut seen = HashSet::new();
        let mut normalized = Vec::new();
        for field in fields {
            let field = field.into();
            if field.trim().is_empty() {
                return Err(invalid_input("partition field names must be non-empty"));
            }
            if !seen.insert(field.clone()) {
                return Err(invalid_input(format!(
                    "duplicate partition field '{field}'"
                )));
            }
            normalized.push(field);
        }
        if normalized.is_empty() {
            return Err(invalid_input("partition spec requires at least one field"));
        }

        Ok(Self {
            version,
            fields: normalized,
        })
    }

    #[must_use]
    pub fn tenant() -> Self {
        Self {
            version: 1,
            fields: vec!["tenant".to_string()],
        }
    }

    #[must_use]
    pub fn tenant_source() -> Self {
        Self {
            version: 1,
            fields: vec!["tenant".to_string(), "source".to_string()],
        }
    }

    #[must_use]
    pub fn version(&self) -> i32 {
        self.version
    }

    #[must_use]
    pub fn fields(&self) -> &[String] {
        &self.fields
    }
}

/// Manifest row describing one resolved context partition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionInfo {
    pub partition_id: String,
    pub spec_version: i32,
    pub selector: PartitionSelector,
    pub dataset_uri: String,
}

/// Thin resolver for scoped context partitions under one namespace root.
///
/// Phase 1 supports opening exactly one partition from a complete selector. The
/// returned value is today's [`ContextStore`], preserving existing per-context
/// behavior while putting each selector in a physically separate Lance dataset.
#[derive(Debug, Clone)]
pub struct ContextNamespace {
    root_uri: String,
    partition_spec: PartitionSpec,
    context_options: ContextStoreOptions,
}

impl ContextNamespace {
    /// Create or open a namespace root with default context-store options.
    pub async fn create(
        root_uri: impl Into<String>,
        partition_spec: PartitionSpec,
    ) -> LanceResult<Self> {
        Self::create_with_options(root_uri, partition_spec, ContextStoreOptions::default()).await
    }

    /// Create or open a namespace root with explicit context-store options.
    pub async fn create_with_options(
        root_uri: impl Into<String>,
        partition_spec: PartitionSpec,
        context_options: ContextStoreOptions,
    ) -> LanceResult<Self> {
        let namespace = Self::new(root_uri, partition_spec, context_options)?;
        namespace.ensure_manifest().await?;
        Ok(namespace)
    }

    /// Build a namespace resolver without touching storage.
    pub fn new(
        root_uri: impl Into<String>,
        partition_spec: PartitionSpec,
        context_options: ContextStoreOptions,
    ) -> LanceResult<Self> {
        let root_uri = normalize_root_uri(root_uri.into())?;
        Ok(Self {
            root_uri,
            partition_spec,
            context_options,
        })
    }

    #[must_use]
    pub fn root_uri(&self) -> &str {
        &self.root_uri
    }

    #[must_use]
    pub fn partition_spec(&self) -> &PartitionSpec {
        &self.partition_spec
    }

    #[must_use]
    pub fn manifest_uri(&self) -> String {
        join_uri(&self.root_uri, MANIFEST_TABLE_NAME)
    }

    /// Resolve a complete selector to its opaque partition dataset URI.
    pub fn resolve_partition(&self, selector: &PartitionSelector) -> LanceResult<PartitionInfo> {
        self.validate_selector(selector)?;
        let selector_json = selector_json(selector)?;
        let partition_id =
            partition_id(&self.root_uri, self.partition_spec.version, &selector_json);
        let dataset_uri = join_uri(
            &self.root_uri,
            &format!(
                "v{}/{}/{}",
                self.partition_spec.version, partition_id, PARTITION_TABLE_NAME
            ),
        );

        Ok(PartitionInfo {
            partition_id,
            spec_version: self.partition_spec.version,
            selector: selector.clone(),
            dataset_uri,
        })
    }

    /// Open one partition as a normal [`ContextStore`].
    pub async fn context(&self, selector: &PartitionSelector) -> LanceResult<ContextStore> {
        let partition = self.resolve_partition(selector)?;
        self.ensure_manifest_entry(&partition).await?;
        ContextStore::open_with_options(&partition.dataset_uri, self.context_options.clone()).await
    }

    /// List partition mappings currently recorded in `__manifest`.
    pub async fn partitions(&self) -> LanceResult<Vec<PartitionInfo>> {
        let manifest = self.ensure_manifest().await?;
        read_manifest(&manifest).await
    }

    async fn ensure_manifest(&self) -> LanceResult<Dataset> {
        match load_dataset(&self.manifest_uri(), self.context_options.storage_options()).await {
            Ok(dataset) => Ok(dataset),
            Err(LanceError::DatasetNotFound { .. }) => {
                create_manifest(&self.manifest_uri(), self.context_options.storage_options()).await
            }
            Err(err) => Err(err),
        }
    }

    async fn ensure_manifest_entry(&self, partition: &PartitionInfo) -> LanceResult<()> {
        let manifest = self.ensure_manifest().await?;
        for existing in read_manifest(&manifest).await? {
            if existing.partition_id == partition.partition_id {
                if existing == *partition {
                    return Ok(());
                }
                return Err(invalid_input(format!(
                    "partition id '{}' already maps to a different selector",
                    partition.partition_id
                )));
            }
        }

        append_manifest_entry(
            &self.manifest_uri(),
            self.context_options.storage_options(),
            partition,
        )
        .await
    }

    fn validate_selector(&self, selector: &PartitionSelector) -> LanceResult<()> {
        for field in &self.partition_spec.fields {
            match selector.get(field) {
                Some(value) if !value.is_empty() => {}
                Some(_) => {
                    return Err(invalid_input(format!(
                        "partition selector value for '{field}' must be non-empty"
                    )));
                }
                None => {
                    return Err(invalid_input(format!(
                        "partition selector is missing required field '{field}'"
                    )));
                }
            }
        }

        for field in selector.keys() {
            if !self
                .partition_spec
                .fields
                .iter()
                .any(|expected| expected == field)
            {
                return Err(invalid_input(format!(
                    "partition selector contains unknown field '{field}'"
                )));
            }
        }

        Ok(())
    }
}

fn manifest_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("partition_id", DataType::Utf8, false),
        Field::new("spec_version", DataType::Int32, false),
        Field::new("selector_json", DataType::LargeUtf8, false),
        Field::new("dataset_uri", DataType::Utf8, false),
    ]))
}

async fn load_dataset(
    uri: &str,
    storage_options: Option<HashMap<String, String>>,
) -> LanceResult<Dataset> {
    if let Some(options) = storage_options {
        DatasetBuilder::from_uri(uri)
            .with_storage_options(options)
            .load()
            .await
    } else {
        Dataset::open(uri).await
    }
}

async fn create_manifest(
    uri: &str,
    storage_options: Option<HashMap<String, String>>,
) -> LanceResult<Dataset> {
    let schema = manifest_schema();
    let empty_batch = RecordBatch::new_empty(schema.clone());
    let batches = RecordBatchIterator::new(
        vec![Ok::<RecordBatch, ArrowError>(empty_batch)].into_iter(),
        schema,
    );
    let params = write_params(WriteMode::Create, storage_options);
    Dataset::write(batches, uri, Some(params)).await
}

async fn append_manifest_entry(
    uri: &str,
    storage_options: Option<HashMap<String, String>>,
    partition: &PartitionInfo,
) -> LanceResult<()> {
    let selector_json = selector_json(&partition.selector)?;
    let schema = manifest_schema();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec![partition.partition_id.as_str()])),
            Arc::new(Int32Array::from(vec![partition.spec_version])),
            Arc::new(LargeStringArray::from(vec![selector_json.as_str()])),
            Arc::new(StringArray::from(vec![partition.dataset_uri.as_str()])),
        ],
    )?;
    let batches = RecordBatchIterator::new(
        vec![Ok::<RecordBatch, ArrowError>(batch)].into_iter(),
        schema,
    );
    let params = write_params(WriteMode::Append, storage_options);
    Dataset::write(batches, uri, Some(params)).await?;
    Ok(())
}

fn write_params(mode: WriteMode, storage_options: Option<HashMap<String, String>>) -> WriteParams {
    let mut params = WriteParams {
        mode,
        ..Default::default()
    };

    if let Some(options) = storage_options {
        let store_params = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                options,
            ))),
            ..Default::default()
        };
        params.store_params = Some(store_params);
    }

    params
}

async fn read_manifest(dataset: &Dataset) -> LanceResult<Vec<PartitionInfo>> {
    let mut scanner = dataset.scan();
    scanner.project(&[
        "partition_id",
        "spec_version",
        "selector_json",
        "dataset_uri",
    ])?;
    let mut stream = scanner.try_into_stream().await?;
    let mut partitions = Vec::new();
    while let Some(batch) = stream.try_next().await? {
        partitions.extend(manifest_batch_to_partitions(&batch)?);
    }
    Ok(partitions)
}

fn manifest_batch_to_partitions(batch: &RecordBatch) -> LanceResult<Vec<PartitionInfo>> {
    let partition_id = column_as::<StringArray>(batch, "partition_id")?;
    let spec_version = column_as::<Int32Array>(batch, "spec_version")?;
    let selector_json = column_as::<LargeStringArray>(batch, "selector_json")?;
    let dataset_uri = column_as::<StringArray>(batch, "dataset_uri")?;

    let mut partitions = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        partitions.push(PartitionInfo {
            partition_id: partition_id.value(row).to_string(),
            spec_version: spec_version.value(row),
            selector: serde_json::from_str(selector_json.value(row)).map_err(|err| {
                LanceError::from(ArrowError::InvalidArgumentError(format!(
                    "invalid partition selector JSON in manifest: {err}"
                )))
            })?,
            dataset_uri: dataset_uri.value(row).to_string(),
        });
    }
    Ok(partitions)
}

fn column_as<'a, A>(batch: &'a RecordBatch, name: &str) -> LanceResult<&'a A>
where
    A: Array + 'static,
{
    batch
        .column_by_name(name)
        .and_then(|col| col.as_ref().as_any().downcast_ref::<A>())
        .ok_or_else(|| {
            LanceError::from(ArrowError::InvalidArgumentError(format!(
                "column '{name}' has unexpected data type"
            )))
        })
}

fn selector_json(selector: &PartitionSelector) -> LanceResult<String> {
    serde_json::to_string(selector).map_err(|err| {
        LanceError::from(ArrowError::InvalidArgumentError(format!(
            "partition selector is not JSON serializable: {err}"
        )))
    })
}

fn partition_id(root_uri: &str, spec_version: i32, selector_json: &str) -> String {
    let key = format!("{root_uri}\n{spec_version}\n{selector_json}");
    let uuid = Uuid::new_v5(&Uuid::NAMESPACE_URL, key.as_bytes())
        .simple()
        .to_string();
    uuid[..16].to_string()
}

fn normalize_root_uri(root_uri: String) -> LanceResult<String> {
    let root_uri = root_uri.trim();
    if root_uri.is_empty() {
        return Err(invalid_input("namespace root URI must be non-empty"));
    }
    if root_uri == "/" {
        return Ok(root_uri.to_string());
    }
    Ok(root_uri.trim_end_matches('/').to_string())
}

fn invalid_input(message: impl Into<String>) -> LanceError {
    LanceError::from(ArrowError::InvalidArgumentError(message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{ContextRecord, LIFECYCLE_ACTIVE};
    use crate::serde::CONTENT_TYPE_TEXT;
    use chrono::Utc;
    use tempfile::TempDir;

    fn selector(pairs: &[(&str, &str)]) -> PartitionSelector {
        pairs
            .iter()
            .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
            .collect()
    }

    fn record(id: &str, tenant: &str, source: &str) -> ContextRecord {
        ContextRecord {
            id: id.to_string(),
            external_id: None,
            run_id: format!("run-{id}"),
            bot_id: Some("support-bot".to_string()),
            session_id: None,
            tenant: Some(tenant.to_string()),
            source: Some(source.to_string()),
            created_at: Utc::now(),
            role: "user".to_string(),
            state_metadata: None,
            metadata: None,
            relationships: Vec::new(),
            expires_at: None,
            retention_policy: None,
            lifecycle_status: LIFECYCLE_ACTIVE.to_string(),
            retired_at: None,
            retired_reason: None,
            supersedes_id: None,
            superseded_by_id: None,
            content_type: CONTENT_TYPE_TEXT.to_string(),
            text_payload: Some("hello".to_string()),
            binary_payload: None,
            payload_uri: None,
            payload_size: None,
            payload_checksum: None,
            embedding: None,
        }
    }

    #[tokio::test]
    async fn namespace_context_records_manifest_and_opens_partition() {
        let dir = TempDir::new().unwrap();
        let root_uri = dir.path().to_string_lossy().to_string();
        let namespace = ContextNamespace::create(root_uri, PartitionSpec::tenant_source())
            .await
            .unwrap();
        let selector = selector(&[("tenant", "acme"), ("source", "memory")]);
        let partition = namespace.resolve_partition(&selector).unwrap();

        let store = namespace.context(&selector).await.unwrap();
        store
            .add(&[record("rec-1", "acme", "memory")])
            .await
            .unwrap();

        let records = store
            .list_filtered(
                None,
                None,
                Some(&crate::record::RecordFilters {
                    tenant: Some("acme".to_string()),
                    source: Some("memory".to_string()),
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        assert_eq!(records.len(), 1);

        let partitions = namespace.partitions().await.unwrap();
        assert_eq!(partitions, vec![partition]);

        namespace.context(&selector).await.unwrap();
        assert_eq!(namespace.partitions().await.unwrap().len(), 1);
    }

    #[test]
    fn namespace_rejects_partial_or_unknown_selectors() {
        let namespace = ContextNamespace::new(
            "/tmp/context-ns",
            PartitionSpec::tenant_source(),
            ContextStoreOptions::default(),
        )
        .unwrap();

        let err = namespace
            .resolve_partition(&selector(&[("tenant", "acme")]))
            .unwrap_err();
        assert!(err.to_string().contains("source"));

        let err = namespace
            .resolve_partition(&selector(&[
                ("tenant", "acme"),
                ("source", "memory"),
                ("session_id", "s1"),
            ]))
            .unwrap_err();
        assert!(err.to_string().contains("session_id"));
    }
}
