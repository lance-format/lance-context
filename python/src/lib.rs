#![recursion_limit = "256"]

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use chrono::{DateTime, SecondsFormat, Utc};
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyModule, PyType};
use pyo3::IntoPyObject;
use serde_json::{Map, Value};
use tokio::runtime::Runtime;

use lance_context::{
    datagen_event_to_dto, folded_item_to_dto, AddRolloutRequest, ColumnSpec,
    CreateDatagenStoreRequest, CreateGenericStoreRequest, CreateRolloutStoreRequest,
    DatagenErrorInfo, DatagenEventDto, DatagenFailureDto, DatagenFieldChange, DatagenFieldStateDto,
    DatagenItemNode as UnifiedDatagenItemNode, DatagenItemTree as UnifiedDatagenItemTree,
    DatagenStepCursorDto, DatagenStore as UnifiedDatagenStore, DatagenStoreApi,
    DatagenStreamPosition, DatagenStreamPositionDto,
    DatagenStreamWriter as CoreDatagenStreamWriter, DatagenValueDto, DatagenWriteContext,
    FoldedDatagenItemDto, GenericStore as UnifiedGenericStore, GenericStoreApi, RolloutRecordDto,
    RolloutStore as UnifiedRolloutStore, RolloutStoreApi, SchemaSpec,
};
use lance_context_api::{
    AddRecordRequest, CompactRequest, CompactResponse, CompactStatsResponse, ContextError,
    ContextStoreApi, RecordDto, RecordPatchDto, RelationshipDto, RetrieveRequest,
    RetrieveResultDto, SearchRequest, SearchResultDto, StateMetadataDto, UpdateRecordRequest,
    UpsertRecordRequest,
};
use lance_context_client::RemoteContextStore;
use lance_context_core::serde::CONTENT_TYPE_TEXT;
use lance_context_core::{
    datagen_event_id as core_datagen_event_id, CompactionConfig, CompactionMetrics,
    CompactionStats, Context as RustContext, ContextNamespace as RustContextNamespace,
    ContextRecord, ContextStore, ContextStoreOptions, DatagenBlobValue as CoreDatagenBlobValue,
    DatagenItemId as CoreDatagenItemId, DatagenNewStream, DatagenStepId, DatagenStepKind,
    DatagenTerminal, DatagenValue as CoreDatagenValue, DistanceMetric, EvalConfig, EvalQuerySet,
    ExportConfig, ExportTask, FieldOp, GroupBy, IdIndexType, LifecycleQueryOptions, PartitionInfo,
    PartitionSelector, PartitionSpec, PreferenceForm, ReadProjection, RecordFilters, RecordPatch,
    Relationship, RetrievalMode, RetrieveResult, SearchResult, SplitConfig, StateMetadata,
    DATAGEN_SCHEMA_VERSION, LIFECYCLE_ACTIVE,
};

const DEFAULT_BINARY_CONTENT_TYPE: &str = "application/octet-stream";
const BINARY_PLACEHOLDER: &str = "[binary]";

struct PreparedRecord {
    record: ContextRecord,
    role: String,
    inner_content: String,
    data_type: Option<String>,
}

struct RecordInput {
    role: String,
    data_type: Option<String>,
    embedding: Option<Vec<f32>>,
    bot_id: Option<String>,
    session_id: Option<String>,
    tenant: Option<String>,
    source: Option<String>,
    external_id: Option<String>,
    run_id: Option<String>,
    created_at: Option<DateTime<Utc>>,
    state_metadata: Option<StateMetadata>,
    metadata_json: Option<String>,
    relationships: Vec<Relationship>,
    lifecycle: LifecycleFields,
    payload_uri: Option<String>,
    payload_size: Option<i64>,
    payload_checksum: Option<String>,
}

#[derive(Default)]
struct LifecycleFields {
    expires_at: Option<DateTime<Utc>>,
    retention_policy: Option<String>,
    lifecycle_status: Option<String>,
    retired_at: Option<DateTime<Utc>>,
    retired_reason: Option<String>,
    supersedes_id: Option<String>,
    superseded_by_id: Option<String>,
}

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Generate a new time-ordered id (UUIDv7) as a string.
///
/// Ids sort roughly in creation order, making them a good default primary key
/// for append-heavy tables such as the rollout store.
#[pyfunction]
fn generate_id() -> String {
    lance_context_core::generate_id()
}

#[pyclass]
struct Context {
    inner: RustContext,
    store: ContextStore,
    runtime: Arc<Runtime>,
    run_id: String,
}

#[pyclass]
struct ContextNamespace {
    inner: RustContextNamespace,
    runtime: Arc<Runtime>,
}

#[pyclass]
struct RemoteContext {
    store: RemoteContextStore,
    runtime: Arc<Runtime>,
}

fn storage_options_from_dict<'py>(
    dict: Option<&Bound<'py, PyDict>>,
) -> PyResult<Option<HashMap<String, String>>> {
    let Some(dict) = dict else {
        return Ok(None);
    };

    let mut options = HashMap::new();
    for (key, value) in dict.iter() {
        let key_str = key.extract::<String>()?;
        if value.is_none() {
            continue;
        }
        let string_value = if let Ok(boolean) = value.extract::<bool>() {
            if boolean {
                "true".to_string()
            } else {
                "false".to_string()
            }
        } else if let Ok(number) = value.extract::<i64>() {
            number.to_string()
        } else if let Ok(float_val) = value.extract::<f64>() {
            float_val.to_string()
        } else {
            value.str()?.to_string()
        };
        options.insert(key_str, string_value);
    }

    if options.is_empty() {
        Ok(None)
    } else {
        Ok(Some(options))
    }
}

fn compaction_config_from_dict<'py>(
    dict: Option<&Bound<'py, PyDict>>,
) -> PyResult<CompactionConfig> {
    let Some(dict) = dict else {
        return Ok(CompactionConfig::default());
    };

    let mut config = CompactionConfig::default();

    if let Some(enabled) = dict.get_item("enabled")? {
        config.enabled = enabled.extract()?;
    }
    if let Some(min_frags) = dict.get_item("min_fragments")? {
        config.min_fragments = min_frags.extract()?;
    }
    if let Some(target_rows) = dict.get_item("target_rows_per_fragment")? {
        config.target_rows_per_fragment = target_rows.extract()?;
    }
    if let Some(max_rows) = dict.get_item("max_rows_per_group")? {
        config.max_rows_per_group = max_rows.extract()?;
    }
    if let Some(materialize) = dict.get_item("materialize_deletions")? {
        config.materialize_deletions = materialize.extract()?;
    }
    if let Some(threshold) = dict.get_item("materialize_deletions_threshold")? {
        config.materialize_deletions_threshold = threshold.extract()?;
    }
    if let Some(threads) = dict.get_item("num_threads")? {
        config.num_threads = Some(threads.extract()?);
    }
    if let Some(interval) = dict.get_item("check_interval_secs")? {
        config.check_interval_secs = interval.extract()?;
    }
    if let Some(quiet) = dict.get_item("quiet_hours")? {
        let quiet_list: Vec<(u8, u8)> = quiet.extract()?;
        config.quiet_hours = quiet_list;
    }

    Ok(config)
}

fn context_options_from_py<'py>(
    storage_options: Option<&Bound<'py, PyDict>>,
    compaction_config: Option<&Bound<'py, PyDict>>,
    blob_columns: Option<Vec<String>>,
    id_index_type: Option<String>,
    embedding_dim: Option<i32>,
    distance_metric: Option<String>,
) -> PyResult<ContextStoreOptions> {
    let blob_set: HashSet<String> = blob_columns.unwrap_or_default().into_iter().collect();

    let id_idx = match id_index_type.as_deref() {
        Some("btree") => IdIndexType::BTree,
        Some("zonemap") => IdIndexType::ZoneMap,
        Some("none") | None => IdIndexType::None,
        Some(other) => {
            return Err(PyRuntimeError::new_err(format!(
                "invalid id_index_type '{}': valid values are 'btree', 'zonemap'",
                other
            )))
        }
    };

    let metric = match distance_metric.as_deref() {
        Some(value) => Some(DistanceMetric::parse(value).map_err(to_py_err)?),
        None => None,
    };

    Ok(ContextStoreOptions {
        storage_options: storage_options_from_dict(storage_options)?,
        compaction: compaction_config_from_dict(compaction_config)?,
        embedding_dim,
        blob_columns: blob_set,
        id_index_type: id_idx,
        distance_metric: metric,
        ..Default::default()
    })
}

fn metadata_from_json(metadata_json: Option<String>) -> PyResult<Option<Value>> {
    metadata_json
        .map(|value| serde_json::from_str(&value).map_err(to_py_err))
        .transpose()
}

fn relationships_from_json(relationships_json: Option<String>) -> PyResult<Vec<Relationship>> {
    relationships_json
        .map(|value| serde_json::from_str(&value).map_err(to_py_err))
        .transpose()
        .map(|value| value.unwrap_or_default())
}

fn filters_from_json(filters_json: Option<String>) -> PyResult<Option<RecordFilters>> {
    let Some(filters_json) = filters_json else {
        return Ok(None);
    };
    let value: Value = serde_json::from_str(&filters_json).map_err(to_py_err)?;
    RecordFilters::from_json_value(value)
        .map(Some)
        .map_err(PyRuntimeError::new_err)
}

#[allow(clippy::too_many_arguments)]
fn export_config(
    task: &str,
    group_by: &str,
    preference_form: &str,
    filters_json: Option<String>,
    dedup_threshold: Option<f32>,
    decontaminate_against: Option<Vec<Vec<f32>>>,
    decontaminate_threshold: Option<f32>,
    min_reward: Option<f64>,
    version: Option<u64>,
    include_expired: bool,
    include_retired: bool,
    split_eval_fraction: Option<f64>,
    split_by: Option<String>,
    split_seed: Option<u64>,
) -> PyResult<ExportConfig> {
    let task = match task {
        "sft" => ExportTask::Sft,
        "preference" => ExportTask::Preference,
        "rollout" => ExportTask::Rollout,
        other => {
            return Err(PyRuntimeError::new_err(format!(
                "invalid export task '{other}'; use 'sft', 'preference', or 'rollout'"
            )));
        }
    };
    let group_by = parse_group_by(group_by)?;
    let preference_form = match preference_form {
        "paired" => PreferenceForm::Paired,
        "unpaired" => PreferenceForm::Unpaired,
        "ranked" => PreferenceForm::Ranked,
        other => {
            return Err(PyRuntimeError::new_err(format!(
                "invalid preference_form '{other}'; use 'paired', 'unpaired', or 'ranked'"
            )));
        }
    };
    let filters_summary = filters_json
        .as_ref()
        .map(|raw| serde_json::from_str::<Value>(raw))
        .transpose()
        .map_err(to_py_err)?;
    let split = match split_eval_fraction {
        Some(eval_fraction) => Some(SplitConfig {
            eval_fraction,
            by: match split_by {
                Some(by) => parse_group_by(&by)?,
                None => GroupBy::SessionId,
            },
            seed: split_seed.unwrap_or(0),
        }),
        None => None,
    };

    Ok(ExportConfig {
        task,
        group_by,
        preference_form,
        filters: filters_from_json(filters_json)?,
        lifecycle: LifecycleQueryOptions::new(include_expired, include_retired),
        dedup_threshold,
        decontaminate_against: decontaminate_against.unwrap_or_default(),
        decontaminate_threshold,
        min_reward,
        version,
        filters_summary,
        split,
        emit_stats: false,
    })
}

fn parse_group_by(group_by: &str) -> PyResult<GroupBy> {
    Ok(match group_by {
        "session_id" => GroupBy::SessionId,
        "run_id" => GroupBy::RunId,
        "tenant" => GroupBy::Tenant,
        "source" => GroupBy::Source,
        "bot_id" => GroupBy::BotId,
        "none" => GroupBy::None,
        "external_id_prefix" => GroupBy::ExternalIdPrefix("#".to_string()),
        other if other.starts_with("external_id_prefix:") => {
            GroupBy::ExternalIdPrefix(other["external_id_prefix:".len()..].to_string())
        }
        other => {
            return Err(PyRuntimeError::new_err(format!(
                "invalid group_by '{other}'"
            )));
        }
    })
}

fn eval_config(
    k: usize,
    mode: &str,
    filters_json: Option<String>,
    include_expired: bool,
    include_retired: bool,
) -> PyResult<EvalConfig> {
    let mode = match mode {
        "vector" => RetrievalMode::Vector,
        "hybrid" => RetrievalMode::Hybrid,
        other => {
            return Err(PyRuntimeError::new_err(format!(
                "invalid eval mode '{other}'; use 'vector' or 'hybrid'"
            )));
        }
    };
    Ok(EvalConfig {
        k,
        mode,
        filters: filters_from_json(filters_json)?,
        lifecycle: LifecycleQueryOptions::new(include_expired, include_retired),
    })
}

fn selector_from_dict(dict: &Bound<'_, PyDict>) -> PyResult<PartitionSelector> {
    let mut selector = BTreeMap::new();
    for (key, value) in dict.iter() {
        if value.is_none() {
            continue;
        }
        selector.insert(key.extract::<String>()?, value.extract::<String>()?);
    }
    Ok(selector)
}

#[pymethods]
impl Context {
    #[classmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (uri, *, storage_options=None, compaction_config=None, blob_columns=None, id_index_type=None, embedding_dim=None, distance_metric=None))]
    fn create(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        uri: &str,
        storage_options: Option<&Bound<'_, PyDict>>,
        compaction_config: Option<&Bound<'_, PyDict>>,
        blob_columns: Option<Vec<String>>,
        id_index_type: Option<String>,
        embedding_dim: Option<i32>,
        distance_metric: Option<String>,
    ) -> PyResult<Self> {
        let runtime = Arc::new(Runtime::new().map_err(to_py_err)?);
        let options = context_options_from_py(
            storage_options,
            compaction_config,
            blob_columns,
            id_index_type,
            embedding_dim,
            distance_metric,
        )?;

        let store_res =
            py.allow_threads(|| runtime.block_on(ContextStore::open_with_options(uri, options)));
        let store = store_res.map_err(to_py_err)?;
        let run_id = new_run_id();
        Ok(Self {
            inner: RustContext::new(uri),
            store,
            runtime,
            run_id,
        })
    }

    fn uri(&self) -> &str {
        self.inner.uri()
    }

    fn branch(&self) -> &str {
        self.inner.branch()
    }

    fn entries(&self) -> u64 {
        self.inner.entries()
    }

    fn version(&self) -> u64 {
        self.store.version()
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (role, content, data_type = None, embedding = None, bot_id = None, session_id = None, tenant = None, source = None, external_id = None, run_id = None, created_at = None, state_metadata = None, metadata_json = None, expires_at = None, retention_policy = None, lifecycle_status = None, retired_at = None, retired_reason = None, supersedes_id = None, superseded_by_id = None, relationships_json = None, payload_uri = None, payload_size = None, payload_checksum = None))]
    fn add(
        &mut self,
        py: Python<'_>,
        role: &str,
        content: &Bound<'_, PyAny>,
        data_type: Option<&str>,
        embedding: Option<Vec<f32>>,
        bot_id: Option<String>,
        session_id: Option<String>,
        tenant: Option<String>,
        source: Option<String>,
        external_id: Option<String>,
        run_id: Option<String>,
        created_at: Option<String>,
        state_metadata: Option<&Bound<'_, PyDict>>,
        metadata_json: Option<String>,
        expires_at: Option<String>,
        retention_policy: Option<String>,
        lifecycle_status: Option<String>,
        retired_at: Option<String>,
        retired_reason: Option<String>,
        supersedes_id: Option<String>,
        superseded_by_id: Option<String>,
        relationships_json: Option<String>,
        payload_uri: Option<String>,
        payload_size: Option<i64>,
        payload_checksum: Option<String>,
    ) -> PyResult<()> {
        let lifecycle = LifecycleFields {
            expires_at: parse_optional_datetime(expires_at, "expires_at")?,
            retention_policy,
            lifecycle_status,
            retired_at: parse_optional_datetime(retired_at, "retired_at")?,
            retired_reason,
            supersedes_id,
            superseded_by_id,
        };
        let prepared = self.prepare_record(
            content,
            RecordInput {
                role: role.to_string(),
                data_type: data_type.map(str::to_string),
                embedding,
                bot_id,
                session_id,
                tenant,
                source,
                external_id,
                run_id,
                created_at: parse_optional_datetime(created_at, "created_at")?,
                state_metadata: state_metadata_from_dict(state_metadata)?,
                metadata_json,
                relationships: relationships_from_json(relationships_json)?,
                lifecycle,
                payload_uri,
                payload_size,
                payload_checksum,
            },
            1,
        )?;

        let add_res = py.allow_threads(|| {
            self.runtime
                .block_on(self.store.add(std::slice::from_ref(&prepared.record)))
        });
        add_res.map_err(to_py_err)?;
        self.inner.add(
            &prepared.role,
            &prepared.inner_content,
            prepared.data_type.as_deref(),
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (role, content, data_type = None, embedding = None, bot_id = None, session_id = None, tenant = None, source = None, external_id = None, run_id = None, created_at = None, state_metadata = None, metadata_json = None, expires_at = None, retention_policy = None, lifecycle_status = None, retired_at = None, retired_reason = None, relationships_json = None, payload_uri = None, payload_size = None, payload_checksum = None, key = "external_id"))]
    fn upsert(
        &mut self,
        py: Python<'_>,
        role: &str,
        content: &Bound<'_, PyAny>,
        data_type: Option<&str>,
        embedding: Option<Vec<f32>>,
        bot_id: Option<String>,
        session_id: Option<String>,
        tenant: Option<String>,
        source: Option<String>,
        external_id: Option<String>,
        run_id: Option<String>,
        created_at: Option<String>,
        state_metadata: Option<&Bound<'_, PyDict>>,
        metadata_json: Option<String>,
        expires_at: Option<String>,
        retention_policy: Option<String>,
        lifecycle_status: Option<String>,
        retired_at: Option<String>,
        retired_reason: Option<String>,
        relationships_json: Option<String>,
        payload_uri: Option<String>,
        payload_size: Option<i64>,
        payload_checksum: Option<String>,
        key: &str,
    ) -> PyResult<PyObject> {
        if key != "external_id" {
            return Err(PyRuntimeError::new_err(format!(
                "upsert key '{key}' is not supported; use 'external_id'"
            )));
        }
        if external_id.as_deref().is_none_or(str::is_empty) {
            return Err(PyRuntimeError::new_err(
                "upsert requires external_id".to_string(),
            ));
        }

        let lifecycle = LifecycleFields {
            expires_at: parse_optional_datetime(expires_at, "expires_at")?,
            retention_policy,
            lifecycle_status,
            retired_at: parse_optional_datetime(retired_at, "retired_at")?,
            retired_reason,
            supersedes_id: None,
            superseded_by_id: None,
        };
        let prepared = self.prepare_record(
            content,
            RecordInput {
                role: role.to_string(),
                data_type: data_type.map(str::to_string),
                embedding,
                bot_id,
                session_id,
                tenant,
                source,
                external_id,
                run_id,
                created_at: parse_optional_datetime(created_at, "created_at")?,
                state_metadata: state_metadata_from_dict(state_metadata)?,
                metadata_json,
                relationships: relationships_from_json(relationships_json)?,
                lifecycle,
                payload_uri,
                payload_size,
                payload_checksum,
            },
            1,
        )?;

        let result = py.allow_threads(|| {
            self.runtime
                .block_on(self.store.upsert_by_external_id(prepared.record.clone()))
        });
        let result = result.map_err(to_py_err)?;
        self.inner.add(
            &prepared.role,
            &prepared.inner_content,
            prepared.data_type.as_deref(),
        );

        let dict = PyDict::new(py);
        dict.set_item("inserted", result.inserted)?;
        dict.set_item("replaced_id", result.replaced_id)?;
        dict.set_item("version", result.version)?;
        dict.set_item("record", record_to_py(py, result.record)?)?;
        Ok(dict.into_pyobject(py)?.unbind().into())
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (id = None, external_id = None, bot_id = None, session_id = None, tenant = None, source = None, metadata_json = None, relationships_json = None, expires_at = None, retention_policy = None, lifecycle_status = None, retired_at = None, retired_reason = None, embedding = None, payload_uri = None, payload_size = None, payload_checksum = None))]
    fn update(
        &mut self,
        py: Python<'_>,
        id: Option<String>,
        external_id: Option<String>,
        bot_id: Option<String>,
        session_id: Option<String>,
        tenant: Option<String>,
        source: Option<String>,
        metadata_json: Option<String>,
        relationships_json: Option<String>,
        expires_at: Option<String>,
        retention_policy: Option<String>,
        lifecycle_status: Option<String>,
        retired_at: Option<String>,
        retired_reason: Option<String>,
        embedding: Option<Vec<f32>>,
        payload_uri: Option<String>,
        payload_size: Option<i64>,
        payload_checksum: Option<String>,
    ) -> PyResult<PyObject> {
        let patch = RecordPatch {
            bot_id,
            session_id,
            tenant,
            source,
            state_metadata: None,
            metadata: metadata_from_json(metadata_json)?,
            relationships: relationships_patch_from_json(relationships_json)?,
            expires_at: parse_optional_datetime(expires_at, "expires_at")?,
            retention_policy,
            lifecycle_status,
            retired_at: parse_optional_datetime(retired_at, "retired_at")?,
            retired_reason,
            embedding,
            payload_uri,
            payload_size,
            payload_checksum,
        };
        if patch.is_empty() {
            return Err(PyRuntimeError::new_err(
                "update requires at least one patch field",
            ));
        }

        let result = match (id, external_id) {
            (Some(id), None) => py.allow_threads(|| {
                self.runtime
                    .block_on(self.store.update_by_id(&id, patch))
                    .map_err(to_py_err)
            }),
            (None, Some(external_id)) => py.allow_threads(|| {
                self.runtime
                    .block_on(self.store.update_by_external_id(&external_id, patch))
                    .map_err(to_py_err)
            }),
            (None, None) => Err(PyRuntimeError::new_err(
                "update() requires either id or external_id",
            )),
            (Some(_), Some(_)) => Err(PyRuntimeError::new_err(
                "update() accepts only one of id or external_id",
            )),
        }?;

        let dict = PyDict::new(py);
        match result {
            Some(result) => {
                dict.set_item("updated", true)?;
                dict.set_item("replaced_id", Some(result.replaced_id))?;
                dict.set_item("version", result.version)?;
                dict.set_item("record", record_to_py(py, result.record)?)?;
            }
            None => {
                dict.set_item("updated", false)?;
                dict.set_item("replaced_id", Option::<String>::None)?;
                dict.set_item("version", self.store.version())?;
                dict.set_item("record", Option::<PyObject>::None)?;
            }
        }
        Ok(dict.into_pyobject(py)?.unbind().into())
    }

    #[pyo3(signature = (records))]
    fn add_many(&mut self, py: Python<'_>, records: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut prepared = Vec::new();
        for (index, item) in records.try_iter()?.enumerate() {
            let item = item?;
            let dict = item
                .downcast::<PyDict>()
                .map_err(|_| PyTypeError::new_err(format!("records[{index}] must be a dict")))?;
            prepared.push(self.prepare_record_from_dict(dict, index)?);
        }

        if prepared.is_empty() {
            return Ok(());
        }

        let context_records: Vec<ContextRecord> =
            prepared.iter().map(|item| item.record.clone()).collect();
        let add_res = py.allow_threads(|| self.runtime.block_on(self.store.add(&context_records)));
        add_res.map_err(to_py_err)?;

        for item in prepared {
            self.inner
                .add(&item.role, &item.inner_content, item.data_type.as_deref());
        }
        Ok(())
    }

    #[pyo3(signature = (records, key = "external_id"))]
    fn upsert_many(
        &mut self,
        py: Python<'_>,
        records: &Bound<'_, PyAny>,
        key: &str,
    ) -> PyResult<Vec<PyObject>> {
        if key != "external_id" {
            return Err(PyRuntimeError::new_err(format!(
                "upsert key '{key}' is not supported; use 'external_id'"
            )));
        }

        let mut prepared = Vec::new();
        for (index, item) in records.try_iter()?.enumerate() {
            let item = item?;
            let dict = item
                .downcast::<PyDict>()
                .map_err(|_| PyTypeError::new_err(format!("records[{index}] must be a dict")))?;
            let record = self.prepare_record_from_dict(dict, index)?;
            if record
                .record
                .external_id
                .as_deref()
                .is_none_or(str::is_empty)
            {
                return Err(PyRuntimeError::new_err(format!(
                    "upsert_many requires external_id (records[{index}])"
                )));
            }
            prepared.push(record);
        }

        if prepared.is_empty() {
            return Ok(Vec::new());
        }

        let context_records: Vec<ContextRecord> =
            prepared.iter().map(|item| item.record.clone()).collect();
        let results = py.allow_threads(|| {
            self.runtime
                .block_on(self.store.upsert_many_by_external_id(context_records))
        });
        let results = results.map_err(to_py_err)?;

        for item in &prepared {
            self.inner
                .add(&item.role, &item.inner_content, item.data_type.as_deref());
        }

        let mut out = Vec::with_capacity(results.len());
        for result in results {
            let dict = PyDict::new(py);
            dict.set_item("inserted", result.inserted)?;
            dict.set_item("replaced_id", result.replaced_id)?;
            dict.set_item("version", result.version)?;
            dict.set_item("record", record_to_py(py, result.record)?)?;
            out.push(dict.into_pyobject(py)?.unbind().into());
        }
        Ok(out)
    }

    #[pyo3(signature = (label = None))]
    fn snapshot(&mut self, label: Option<&str>) -> String {
        self.inner.snapshot(label)
    }

    fn fork(&self, py: Python<'_>, branch_name: &str) -> PyResult<Self> {
        // Opens a second handle on the same dataset rather than cloning this
        // one: `ContextStore` owns a resident MemWAL writer (and a `Drop` that
        // seals it), so it is deliberately not `Clone` -- two owners would mean
        // two writers racing for one shard. A fork branches the in-memory
        // `Context` and shares the underlying dataset, which a fresh handle
        // gives it.
        let uri = self.store.uri().to_string();
        let store = py.allow_threads(|| self.runtime.block_on(ContextStore::open(&uri)));
        Ok(Self {
            inner: self.inner.fork(branch_name),
            store: store.map_err(to_py_err)?,
            runtime: Arc::clone(&self.runtime),
            run_id: new_run_id(),
        })
    }

    fn checkout(&mut self, py: Python<'_>, version_id: u64) -> PyResult<()> {
        let res = py.allow_threads(|| self.runtime.block_on(self.store.checkout(version_id)));
        res.map_err(to_py_err)?;
        self.run_id = new_run_id();
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (query, limit = None, filters_json = None, include_expired = false, include_retired = false, include_relationships = false, include_binary = true, include_embedding = true))]
    fn search(
        &self,
        py: Python<'_>,
        query: Vec<f32>,
        limit: Option<usize>,
        filters_json: Option<String>,
        include_expired: bool,
        include_retired: bool,
        include_relationships: bool,
        include_binary: bool,
        include_embedding: bool,
    ) -> PyResult<Vec<PyObject>> {
        let filters = filters_from_json(filters_json)?;
        let options = LifecycleQueryOptions::new(include_expired, include_retired);
        let projection = ReadProjection {
            text: true,
            binary: include_binary,
            embedding: include_embedding,
        };
        let hits_res = py.allow_threads(|| {
            self.runtime.block_on(self.store.search_filtered_projected(
                &query,
                limit,
                filters.as_ref(),
                options,
                projection,
            ))
        });
        let hits = hits_res.map_err(to_py_err)?;
        hits.into_iter()
            .map(|hit| search_hit_to_py(py, hit, include_relationships))
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (text = None, vector = None, limit = None, filters_json = None, include_expired = false, include_retired = false, include_relationships = false, fusion = None))]
    fn retrieve(
        &self,
        py: Python<'_>,
        text: Option<String>,
        vector: Option<Vec<f32>>,
        limit: Option<usize>,
        filters_json: Option<String>,
        include_expired: bool,
        include_retired: bool,
        include_relationships: bool,
        fusion: Option<String>,
    ) -> PyResult<Vec<PyObject>> {
        if fusion.as_deref().is_some_and(|value| value != "rrf") {
            return Err(PyRuntimeError::new_err(
                "retrieve fusion currently supports only 'rrf'",
            ));
        }

        let filters = filters_from_json(filters_json)?;
        let options = LifecycleQueryOptions::new(include_expired, include_retired);
        let hits_res = py.allow_threads(|| {
            self.runtime
                .block_on(self.store.retrieve_filtered_with_options(
                    text.as_deref(),
                    vector.as_deref(),
                    limit,
                    filters.as_ref(),
                    options,
                ))
        });
        let hits = hits_res.map_err(to_py_err)?;
        hits.into_iter()
            .map(|hit| retrieve_hit_to_py(py, hit, include_relationships))
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (output_path, task = "sft", group_by = "session_id", preference_form = "paired", filters_json = None, dedup_threshold = None, decontaminate_against = None, decontaminate_threshold = None, min_reward = None, version = None, include_expired = false, include_retired = false, split_eval_fraction = None, split_by = None, split_seed = None, emit_stats = false))]
    fn export_training(
        &mut self,
        py: Python<'_>,
        output_path: &str,
        task: &str,
        group_by: &str,
        preference_form: &str,
        filters_json: Option<String>,
        dedup_threshold: Option<f32>,
        decontaminate_against: Option<Vec<Vec<f32>>>,
        decontaminate_threshold: Option<f32>,
        min_reward: Option<f64>,
        version: Option<u64>,
        include_expired: bool,
        include_retired: bool,
        split_eval_fraction: Option<f64>,
        split_by: Option<String>,
        split_seed: Option<u64>,
        emit_stats: bool,
    ) -> PyResult<String> {
        let mut config = export_config(
            task,
            group_by,
            preference_form,
            filters_json,
            dedup_threshold,
            decontaminate_against,
            decontaminate_threshold,
            min_reward,
            version,
            include_expired,
            include_retired,
            split_eval_fraction,
            split_by,
            split_seed,
        )?;
        config.emit_stats = emit_stats;
        let manifest = py
            .allow_threads(|| {
                self.runtime
                    .block_on(self.store.export_training(&config, output_path))
            })
            .map_err(to_py_err)?;
        serde_json::to_string(&manifest).map_err(to_py_err)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (query_set_json, k = 10, mode = "vector", filters_json = None, include_expired = false, include_retired = false))]
    fn evaluate(
        &self,
        py: Python<'_>,
        query_set_json: &str,
        k: usize,
        mode: &str,
        filters_json: Option<String>,
        include_expired: bool,
        include_retired: bool,
    ) -> PyResult<String> {
        let query_set: EvalQuerySet = serde_json::from_str(query_set_json).map_err(to_py_err)?;
        let config = eval_config(k, mode, filters_json, include_expired, include_retired)?;
        let report = py
            .allow_threads(|| {
                self.runtime
                    .block_on(self.store.evaluate(&query_set, &config))
            })
            .map_err(to_py_err)?;
        serde_json::to_string(&report).map_err(to_py_err)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (query_set_json, baseline_version, candidate_version, k = 10, mode = "vector", filters_json = None, include_expired = false, include_retired = false))]
    fn evaluate_versions(
        &mut self,
        py: Python<'_>,
        query_set_json: &str,
        baseline_version: u64,
        candidate_version: u64,
        k: usize,
        mode: &str,
        filters_json: Option<String>,
        include_expired: bool,
        include_retired: bool,
    ) -> PyResult<String> {
        let query_set: EvalQuerySet = serde_json::from_str(query_set_json).map_err(to_py_err)?;
        let config = eval_config(k, mode, filters_json, include_expired, include_retired)?;
        let report = py
            .allow_threads(|| {
                self.runtime.block_on(self.store.evaluate_versions(
                    &query_set,
                    &config,
                    baseline_version,
                    candidate_version,
                ))
            })
            .map_err(to_py_err)?;
        serde_json::to_string(&report).map_err(to_py_err)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (limit = None, offset = None, filters_json = None, include_expired = false, include_retired = false, include_binary = true, include_embedding = true))]
    fn list(
        &self,
        py: Python<'_>,
        limit: Option<usize>,
        offset: Option<usize>,
        filters_json: Option<String>,
        include_expired: bool,
        include_retired: bool,
        include_binary: bool,
        include_embedding: bool,
    ) -> PyResult<Vec<PyObject>> {
        let filters = filters_from_json(filters_json)?;
        let options = LifecycleQueryOptions::new(include_expired, include_retired);
        let projection = ReadProjection {
            text: true,
            binary: include_binary,
            embedding: include_embedding,
        };
        // Release GIL during data retrieval
        let records = py.allow_threads(|| {
            self.runtime
                .block_on(self.store.list_filtered_projected(
                    limit,
                    offset,
                    filters.as_ref(),
                    options,
                    projection,
                ))
                .map_err(to_py_err)
        })?;

        records
            .into_iter()
            .map(|record| record_to_py(py, record))
            .collect()
    }

    fn get_blob(&self, py: Python<'_>, id: &str) -> PyResult<Option<PyObject>> {
        let blob = py.allow_threads(|| {
            self.runtime
                .block_on(self.store.get_blob(id))
                .map_err(to_py_err)
        })?;
        Ok(blob.map(|bytes| PyBytes::new(py, &bytes).into()))
    }

    #[pyo3(signature = (target_id, relation = None, limit = None, include_expired = false, include_retired = false))]
    fn related(
        &self,
        py: Python<'_>,
        target_id: &str,
        relation: Option<&str>,
        limit: Option<usize>,
        include_expired: bool,
        include_retired: bool,
    ) -> PyResult<Vec<PyObject>> {
        let options = LifecycleQueryOptions::new(include_expired, include_retired);
        let records = py.allow_threads(|| {
            self.runtime
                .block_on(
                    self.store
                        .list_related_with_options(target_id, relation, limit, options),
                )
                .map_err(to_py_err)
        })?;

        records
            .into_iter()
            .map(|record| record_to_py(py, record))
            .collect()
    }

    #[pyo3(signature = (id = None, external_id = None))]
    fn get(
        &self,
        py: Python<'_>,
        id: Option<String>,
        external_id: Option<String>,
    ) -> PyResult<Option<PyObject>> {
        let record = match (id, external_id) {
            (Some(id), None) => py.allow_threads(|| {
                self.runtime
                    .block_on(self.store.get_by_id(&id))
                    .map_err(to_py_err)
            })?,
            (None, Some(external_id)) => py.allow_threads(|| {
                self.runtime
                    .block_on(self.store.get_by_external_id(&external_id))
                    .map_err(to_py_err)
            })?,
            (None, None) => {
                return Err(PyRuntimeError::new_err(
                    "get() requires either id or external_id",
                ));
            }
            (Some(_), Some(_)) => {
                return Err(PyRuntimeError::new_err(
                    "get() accepts only one of id or external_id",
                ));
            }
        };

        record.map(|record| record_to_py(py, record)).transpose()
    }

    /// Resolve a record's external payload reference to its bytes on demand,
    /// using the context's configured ``storage_options``. Returns ``None`` if
    /// no record with ``id`` exists; raises if the record has no external
    /// payload reference.
    #[pyo3(signature = (id))]
    fn fetch_payload(&self, py: Python<'_>, id: &str) -> PyResult<Option<Py<PyBytes>>> {
        let bytes = py.allow_threads(|| {
            self.runtime
                .block_on(self.store.fetch_payload(id))
                .map_err(to_py_err)
        })?;
        Ok(bytes.map(|bytes| PyBytes::new(py, &bytes).unbind()))
    }

    /// Offload caller-provided bytes to an object at ``uri`` using the context's
    /// configured ``storage_options``; returns the number of bytes written.
    /// Pair with a subsequent ``add(..., payload_uri=uri)``.
    #[pyo3(signature = (uri, data))]
    fn put_payload(&self, py: Python<'_>, uri: &str, data: &[u8]) -> PyResult<u64> {
        py.allow_threads(|| {
            self.runtime
                .block_on(self.store.put_payload(uri, data))
                .map_err(to_py_err)
        })
    }

    #[pyo3(signature = (id = None, external_id = None))]
    fn delete(
        &mut self,
        py: Python<'_>,
        id: Option<String>,
        external_id: Option<String>,
    ) -> PyResult<bool> {
        match (id, external_id) {
            (Some(id), None) => py.allow_threads(|| {
                self.runtime
                    .block_on(self.store.delete_by_id(&id))
                    .map_err(to_py_err)
            }),
            (None, Some(external_id)) => py.allow_threads(|| {
                self.runtime
                    .block_on(self.store.delete_by_external_id(&external_id))
                    .map_err(to_py_err)
            }),
            (None, None) => Err(PyRuntimeError::new_err(
                "delete() requires either id or external_id",
            )),
            (Some(_), Some(_)) => Err(PyRuntimeError::new_err(
                "delete() accepts only one of id or external_id",
            )),
        }
    }

    fn migrate_relationships(&mut self, py: Python<'_>) -> PyResult<bool> {
        py.allow_threads(|| {
            self.runtime
                .block_on(self.store.migrate_relationships_column())
                .map_err(to_py_err)
        })
    }

    #[pyo3(signature = (target_rows_per_fragment=None, materialize_deletions=None))]
    fn compact(
        &mut self,
        py: Python<'_>,
        target_rows_per_fragment: Option<usize>,
        materialize_deletions: Option<bool>,
    ) -> PyResult<PyObject> {
        // Prepare config before releasing GIL
        let config = if target_rows_per_fragment.is_some() || materialize_deletions.is_some() {
            let mut cfg = self.store.compaction_config.clone();
            if let Some(rows) = target_rows_per_fragment {
                cfg.target_rows_per_fragment = rows;
            }
            if let Some(mat) = materialize_deletions {
                cfg.materialize_deletions = mat;
            }
            Some(cfg)
        } else {
            None
        };

        // Release GIL during expensive compaction operation
        let metrics = py.allow_threads(|| {
            self.runtime
                .block_on(self.store.compact(config))
                .map_err(to_py_err)
        })?;

        compaction_metrics_to_py(py, metrics)
    }

    fn compaction_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        // Release GIL during stats query
        let stats = py.allow_threads(|| {
            self.runtime
                .block_on(self.store.compaction_stats())
                .map_err(to_py_err)
        })?;

        compaction_stats_to_py(py, stats)
    }
}

#[pymethods]
impl ContextNamespace {
    #[classmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (root_uri, fields, *, storage_options=None, compaction_config=None, blob_columns=None, id_index_type=None, embedding_dim=None, distance_metric=None))]
    fn create(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        root_uri: &str,
        fields: Vec<String>,
        storage_options: Option<&Bound<'_, PyDict>>,
        compaction_config: Option<&Bound<'_, PyDict>>,
        blob_columns: Option<Vec<String>>,
        id_index_type: Option<String>,
        embedding_dim: Option<i32>,
        distance_metric: Option<String>,
    ) -> PyResult<Self> {
        let runtime = Arc::new(Runtime::new().map_err(to_py_err)?);
        let spec = PartitionSpec::new(fields).map_err(to_py_err)?;
        let options = context_options_from_py(
            storage_options,
            compaction_config,
            blob_columns,
            id_index_type,
            embedding_dim,
            distance_metric,
        )?;
        let inner = py.allow_threads(|| {
            runtime.block_on(RustContextNamespace::create_with_options(
                root_uri, spec, options,
            ))
        });
        Ok(Self {
            inner: inner.map_err(to_py_err)?,
            runtime,
        })
    }

    fn root_uri(&self) -> &str {
        self.inner.root_uri()
    }

    fn manifest_uri(&self) -> String {
        self.inner.manifest_uri()
    }

    fn partition_uri(&self, selector: &Bound<'_, PyDict>) -> PyResult<String> {
        let selector = selector_from_dict(selector)?;
        Ok(self
            .inner
            .resolve_partition(&selector)
            .map_err(to_py_err)?
            .dataset_uri)
    }

    fn context(&self, py: Python<'_>, selector: &Bound<'_, PyDict>) -> PyResult<Context> {
        let selector = selector_from_dict(selector)?;
        let partition = self.inner.resolve_partition(&selector).map_err(to_py_err)?;
        let store = py.allow_threads(|| self.runtime.block_on(self.inner.context(&selector)));
        Ok(Context {
            inner: RustContext::new(partition.dataset_uri),
            store: store.map_err(to_py_err)?,
            runtime: Arc::clone(&self.runtime),
            run_id: new_run_id(),
        })
    }

    fn partitions(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        let partitions = py.allow_threads(|| self.runtime.block_on(self.inner.partitions()));
        partitions
            .map_err(to_py_err)?
            .into_iter()
            .map(|partition| partition_info_to_py(py, partition))
            .collect()
    }
}

impl Context {
    fn prepare_record_from_dict(
        &self,
        dict: &Bound<'_, PyDict>,
        index: usize,
    ) -> PyResult<PreparedRecord> {
        let role = required_item(dict, "role", index)?.extract::<String>()?;
        let content = required_item(dict, "content", index)?;
        let data_type = optional_item(dict, "data_type")?.map(|value| value.extract::<String>());
        let embedding = optional_item(dict, "embedding")?.map(|value| value.extract::<Vec<f32>>());
        let bot_id = optional_item(dict, "bot_id")?.map(|value| value.extract::<String>());
        let session_id = optional_item(dict, "session_id")?.map(|value| value.extract::<String>());
        let tenant = optional_item(dict, "tenant")?.map(|value| value.extract::<String>());
        let source = optional_item(dict, "source")?.map(|value| value.extract::<String>());
        let external_id =
            optional_item(dict, "external_id")?.map(|value| value.extract::<String>());
        let run_id = optional_item(dict, "run_id")?.map(|value| value.extract::<String>());
        let created_at = optional_item(dict, "created_at")?.map(|value| value.extract::<String>());
        let state_metadata = match optional_item(dict, "state_metadata")? {
            Some(value) => {
                let metadata = value.downcast::<PyDict>().map_err(|_| {
                    PyTypeError::new_err(format!("records[{index}].state_metadata must be a dict"))
                })?;
                state_metadata_from_dict(Some(metadata))?
            }
            None => None,
        };
        let metadata_json =
            optional_item(dict, "metadata_json")?.map(|value| value.extract::<String>());
        let relationships_json =
            optional_item(dict, "relationships_json")?.map(|value| value.extract::<String>());
        let expires_at = optional_item(dict, "expires_at")?.map(|value| value.extract::<String>());
        let retention_policy =
            optional_item(dict, "retention_policy")?.map(|value| value.extract::<String>());
        let lifecycle_status =
            optional_item(dict, "lifecycle_status")?.map(|value| value.extract::<String>());
        let retired_at = optional_item(dict, "retired_at")?.map(|value| value.extract::<String>());
        let retired_reason =
            optional_item(dict, "retired_reason")?.map(|value| value.extract::<String>());
        let supersedes_id =
            optional_item(dict, "supersedes_id")?.map(|value| value.extract::<String>());
        let superseded_by_id =
            optional_item(dict, "superseded_by_id")?.map(|value| value.extract::<String>());
        let payload_uri =
            optional_item(dict, "payload_uri")?.map(|value| value.extract::<String>());
        let payload_size = optional_item(dict, "payload_size")?.map(|value| value.extract::<i64>());
        let payload_checksum =
            optional_item(dict, "payload_checksum")?.map(|value| value.extract::<String>());

        let lifecycle = LifecycleFields {
            expires_at: parse_optional_datetime(expires_at.transpose()?, "expires_at")?,
            retention_policy: retention_policy.transpose()?,
            lifecycle_status: lifecycle_status.transpose()?,
            retired_at: parse_optional_datetime(retired_at.transpose()?, "retired_at")?,
            retired_reason: retired_reason.transpose()?,
            supersedes_id: supersedes_id.transpose()?,
            superseded_by_id: superseded_by_id.transpose()?,
        };

        self.prepare_record(
            &content,
            RecordInput {
                role,
                data_type: data_type.transpose()?,
                embedding: embedding.transpose()?,
                bot_id: bot_id.transpose()?,
                session_id: session_id.transpose()?,
                tenant: tenant.transpose()?,
                source: source.transpose()?,
                external_id: external_id.transpose()?,
                run_id: run_id.transpose()?,
                created_at: parse_optional_datetime(created_at.transpose()?, "created_at")?,
                state_metadata,
                metadata_json: metadata_json.transpose()?,
                relationships: relationships_from_json(relationships_json.transpose()?)?,
                lifecycle,
                payload_uri: payload_uri.transpose()?,
                payload_size: payload_size.transpose()?,
                payload_checksum: payload_checksum.transpose()?,
            },
            index as u64 + 1,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_record(
        &self,
        content: &Bound<'_, PyAny>,
        input: RecordInput,
        offset: u64,
    ) -> PyResult<PreparedRecord> {
        let RecordInput {
            role,
            data_type,
            embedding,
            bot_id,
            session_id,
            tenant,
            source,
            external_id,
            run_id,
            created_at,
            state_metadata,
            metadata_json,
            relationships,
            lifecycle,
            payload_uri,
            payload_size,
            payload_checksum,
        } = input;

        let (content_type, text_payload, binary_payload, inner_content) =
            match content.extract::<&[u8]>() {
                Ok(bytes) => (
                    data_type
                        .clone()
                        .unwrap_or_else(|| DEFAULT_BINARY_CONTENT_TYPE.to_string()),
                    None,
                    Some(bytes.to_vec()),
                    BINARY_PLACEHOLDER.to_string(),
                ),
                Err(_) => {
                    let content_str = content.str()?.to_string();
                    (
                        data_type
                            .clone()
                            .unwrap_or_else(|| CONTENT_TYPE_TEXT.to_string()),
                        Some(content_str.clone()),
                        None,
                        content_str,
                    )
                }
            };

        let record_run_id = run_id.unwrap_or_else(|| self.run_id.clone());
        let record_id = format!("{}-{}", record_run_id, self.inner.entries() + offset);
        let metadata = metadata_from_json(metadata_json)?;
        Ok(PreparedRecord {
            record: ContextRecord {
                id: record_id,
                external_id,
                run_id: record_run_id,
                bot_id,
                session_id,
                tenant,
                source,
                created_at: created_at.unwrap_or_else(Utc::now),
                role: role.clone(),
                state_metadata,
                metadata,
                relationships,
                expires_at: lifecycle.expires_at,
                retention_policy: lifecycle.retention_policy,
                lifecycle_status: lifecycle
                    .lifecycle_status
                    .unwrap_or_else(|| LIFECYCLE_ACTIVE.to_string()),
                retired_at: lifecycle.retired_at,
                retired_reason: lifecycle.retired_reason,
                supersedes_id: lifecycle.supersedes_id,
                superseded_by_id: lifecycle.superseded_by_id,
                content_type,
                text_payload,
                binary_payload,
                payload_uri,
                payload_size,
                payload_checksum,
                embedding,
            },
            role,
            inner_content,
            data_type,
        })
    }
}

#[pymethods]
impl RemoteContext {
    #[classmethod]
    fn connect(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        base_url: &str,
        name: &str,
    ) -> PyResult<Self> {
        let runtime = Arc::new(Runtime::new().map_err(to_py_err)?);
        let store_res =
            py.allow_threads(|| runtime.block_on(RemoteContextStore::connect(base_url, name)));
        let store = store_res.map_err(to_py_err)?;
        Ok(Self { store, runtime })
    }

    fn version(&self) -> u64 {
        self.store.version()
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (role, content, data_type = None, embedding = None, bot_id = None, session_id = None, external_id = None, state_metadata = None, metadata_json = None, expires_at = None, retention_policy = None, supersedes_id = None, relationships_json = None, payload_uri = None, payload_size = None, payload_checksum = None))]
    fn add(
        &mut self,
        py: Python<'_>,
        role: &str,
        content: &Bound<'_, PyAny>,
        data_type: Option<&str>,
        embedding: Option<Vec<f32>>,
        bot_id: Option<String>,
        session_id: Option<String>,
        external_id: Option<String>,
        state_metadata: Option<&Bound<'_, PyDict>>,
        metadata_json: Option<String>,
        expires_at: Option<String>,
        retention_policy: Option<String>,
        supersedes_id: Option<String>,
        relationships_json: Option<String>,
        payload_uri: Option<String>,
        payload_size: Option<i64>,
        payload_checksum: Option<String>,
    ) -> PyResult<PyObject> {
        let (content_type, text_payload, binary_payload) = content_to_payloads(content, data_type)?;
        let req = AddRecordRequest {
            role: role.to_string(),
            content_type,
            text_payload,
            binary_payload,
            payload_uri,
            payload_size,
            payload_checksum,
            embedding,
            bot_id,
            session_id,
            external_id,
            state_metadata: dto_state_metadata_from_dict(state_metadata)?,
            metadata: metadata_from_json(metadata_json)?,
            relationships: dto_relationships_from_json(relationships_json)?,
            expires_at: parse_optional_datetime(expires_at, "expires_at")?,
            retention_policy,
            supersedes_id,
            tenant: None,
            source: None,
        };
        let resp = py
            .allow_threads(|| {
                self.runtime
                    .block_on(self.store.add(std::slice::from_ref(&req)))
            })
            .map_err(to_py_err)?;
        let dict = PyDict::new(py);
        dict.set_item("version", resp.version)?;
        dict.set_item("ids", resp.ids)?;
        dict.set_item("count", resp.count)?;
        Ok(dict.into_pyobject(py)?.unbind().into())
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (role, content, data_type = None, embedding = None, bot_id = None, session_id = None, external_id = None, metadata_json = None, expires_at = None, retention_policy = None, supersedes_id = None, relationships_json = None, payload_uri = None, payload_size = None, payload_checksum = None, key = "external_id"))]
    fn upsert(
        &mut self,
        py: Python<'_>,
        role: &str,
        content: &Bound<'_, PyAny>,
        data_type: Option<&str>,
        embedding: Option<Vec<f32>>,
        bot_id: Option<String>,
        session_id: Option<String>,
        external_id: Option<String>,
        metadata_json: Option<String>,
        expires_at: Option<String>,
        retention_policy: Option<String>,
        supersedes_id: Option<String>,
        relationships_json: Option<String>,
        payload_uri: Option<String>,
        payload_size: Option<i64>,
        payload_checksum: Option<String>,
        key: &str,
    ) -> PyResult<PyObject> {
        if key != "external_id" {
            return Err(PyRuntimeError::new_err(format!(
                "upsert key '{key}' is not supported; use 'external_id'"
            )));
        }
        if external_id.as_deref().is_none_or(str::is_empty) {
            return Err(PyRuntimeError::new_err(
                "upsert requires external_id".to_string(),
            ));
        }

        let (content_type, text_payload, binary_payload) = content_to_payloads(content, data_type)?;
        let record = AddRecordRequest {
            role: role.to_string(),
            content_type,
            text_payload,
            binary_payload,
            payload_uri,
            payload_size,
            payload_checksum,
            embedding,
            bot_id,
            session_id,
            external_id,
            state_metadata: None,
            metadata: metadata_from_json(metadata_json)?,
            relationships: dto_relationships_from_json(relationships_json)?,
            expires_at: parse_optional_datetime(expires_at, "expires_at")?,
            retention_policy,
            supersedes_id,
            tenant: None,
            source: None,
        };
        let req = UpsertRecordRequest {
            record,
            key: key.to_string(),
        };
        let resp = py
            .allow_threads(|| self.runtime.block_on(self.store.upsert(&req)))
            .map_err(to_py_err)?;
        let dict = PyDict::new(py);
        dict.set_item("inserted", resp.inserted)?;
        dict.set_item("replaced_id", resp.replaced_id)?;
        dict.set_item("version", resp.version)?;
        dict.set_item("record", dto_record_to_py(py, resp.record)?)?;
        Ok(dict.into_pyobject(py)?.unbind().into())
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (id = None, external_id = None, bot_id = None, session_id = None, metadata_json = None, relationships_json = None, expires_at = None, retention_policy = None, lifecycle_status = None, retired_at = None, retired_reason = None, embedding = None, payload_uri = None, payload_size = None, payload_checksum = None))]
    fn update(
        &mut self,
        py: Python<'_>,
        id: Option<String>,
        external_id: Option<String>,
        bot_id: Option<String>,
        session_id: Option<String>,
        metadata_json: Option<String>,
        relationships_json: Option<String>,
        expires_at: Option<String>,
        retention_policy: Option<String>,
        lifecycle_status: Option<String>,
        retired_at: Option<String>,
        retired_reason: Option<String>,
        embedding: Option<Vec<f32>>,
        payload_uri: Option<String>,
        payload_size: Option<i64>,
        payload_checksum: Option<String>,
    ) -> PyResult<PyObject> {
        let patch = RecordPatchDto {
            bot_id,
            session_id,
            state_metadata: None,
            metadata: metadata_from_json(metadata_json)?,
            relationships: dto_relationships_patch_from_json(relationships_json)?,
            expires_at: parse_optional_datetime(expires_at, "expires_at")?,
            retention_policy,
            lifecycle_status,
            retired_at: parse_optional_datetime(retired_at, "retired_at")?,
            retired_reason,
            embedding,
            tenant: None,
            source: None,
            payload_uri,
            payload_size,
            payload_checksum,
        };
        if patch.is_empty() {
            return Err(PyRuntimeError::new_err(
                "update requires at least one patch field",
            ));
        }
        match (id.is_some(), external_id.is_some()) {
            (true, true) => {
                return Err(PyRuntimeError::new_err(
                    "update() accepts only one of id or external_id",
                ))
            }
            (false, false) => {
                return Err(PyRuntimeError::new_err(
                    "update() requires either id or external_id",
                ))
            }
            _ => {}
        }

        let req = UpdateRecordRequest {
            id,
            external_id,
            patch,
        };
        let resp = py
            .allow_threads(|| self.runtime.block_on(self.store.update(&req)))
            .map_err(to_py_err)?;
        let dict = PyDict::new(py);
        dict.set_item("updated", resp.updated)?;
        dict.set_item("replaced_id", resp.replaced_id)?;
        dict.set_item("version", resp.version)?;
        match resp.record {
            Some(record) => dict.set_item("record", dto_record_to_py(py, record)?)?,
            None => dict.set_item("record", py.None())?,
        }
        Ok(dict.into_pyobject(py)?.unbind().into())
    }

    #[pyo3(signature = (id = None, external_id = None))]
    fn get(
        &self,
        py: Python<'_>,
        id: Option<String>,
        external_id: Option<String>,
    ) -> PyResult<Option<PyObject>> {
        let record = match (id, external_id) {
            (Some(id), None) => py.allow_threads(|| {
                self.runtime
                    .block_on(self.store.get(&id))
                    .map_err(to_py_err)
            })?,
            (None, Some(external_id)) => py.allow_threads(|| {
                self.runtime
                    .block_on(self.store.get_by_external_id(&external_id))
                    .map_err(to_py_err)
            })?,
            (None, None) => {
                return Err(PyRuntimeError::new_err(
                    "get() requires either id or external_id",
                ));
            }
            (Some(_), Some(_)) => {
                return Err(PyRuntimeError::new_err(
                    "get() accepts only one of id or external_id",
                ));
            }
        };
        record
            .map(|record| dto_record_to_py(py, record))
            .transpose()
    }

    /// Resolve a record's external payload reference to its bytes via the server,
    /// which fetches from object storage using the context's ``storage_options``.
    #[pyo3(signature = (id))]
    fn fetch_payload(&self, py: Python<'_>, id: &str) -> PyResult<Py<PyBytes>> {
        let bytes = py.allow_threads(|| {
            self.runtime
                .block_on(self.store.fetch_payload(id))
                .map_err(to_py_err)
        })?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }

    #[pyo3(signature = (id = None, external_id = None))]
    fn delete(
        &mut self,
        py: Python<'_>,
        id: Option<String>,
        external_id: Option<String>,
    ) -> PyResult<bool> {
        let resp = match (id, external_id) {
            (Some(id), None) => py.allow_threads(|| {
                self.runtime
                    .block_on(self.store.delete_by_id(&id))
                    .map_err(to_py_err)
            })?,
            (None, Some(external_id)) => py.allow_threads(|| {
                self.runtime
                    .block_on(self.store.delete_by_external_id(&external_id))
                    .map_err(to_py_err)
            })?,
            (None, None) => {
                return Err(PyRuntimeError::new_err(
                    "delete() requires either id or external_id",
                ));
            }
            (Some(_), Some(_)) => {
                return Err(PyRuntimeError::new_err(
                    "delete() accepts only one of id or external_id",
                ));
            }
        };
        Ok(resp.deleted)
    }

    #[pyo3(signature = (limit = None, offset = None, filters_json = None, include_expired = false, include_retired = false))]
    fn list(
        &self,
        py: Python<'_>,
        limit: Option<usize>,
        offset: Option<usize>,
        filters_json: Option<String>,
        include_expired: bool,
        include_retired: bool,
    ) -> PyResult<Vec<PyObject>> {
        let filters = filters_value_from_json(filters_json)?;
        let records = py.allow_threads(|| {
            self.runtime
                .block_on(
                    self.store
                        .list(limit, offset, filters, include_expired, include_retired),
                )
                .map_err(to_py_err)
        })?;
        records
            .into_iter()
            .map(|record| dto_record_to_py(py, record))
            .collect()
    }

    #[pyo3(signature = (target_id, relation = None, limit = None, include_expired = false, include_retired = false))]
    fn related(
        &self,
        py: Python<'_>,
        target_id: &str,
        relation: Option<&str>,
        limit: Option<usize>,
        include_expired: bool,
        include_retired: bool,
    ) -> PyResult<Vec<PyObject>> {
        let records = py.allow_threads(|| {
            self.runtime
                .block_on(self.store.related(
                    target_id,
                    relation,
                    limit,
                    include_expired,
                    include_retired,
                ))
                .map_err(to_py_err)
        })?;
        records
            .into_iter()
            .map(|record| dto_record_to_py(py, record))
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (query, limit = None, filters_json = None, include_expired = false, include_retired = false, include_relationships = false))]
    fn search(
        &self,
        py: Python<'_>,
        query: Vec<f32>,
        limit: Option<usize>,
        filters_json: Option<String>,
        include_expired: bool,
        include_retired: bool,
        include_relationships: bool,
    ) -> PyResult<Vec<PyObject>> {
        let req = SearchRequest {
            query,
            limit: limit.unwrap_or(10),
            filters: filters_value_from_json(filters_json)?,
            include_expired,
            include_retired,
            include_relationships,
        };
        let hits = py
            .allow_threads(|| self.runtime.block_on(self.store.search(&req)))
            .map_err(to_py_err)?;
        hits.into_iter()
            .map(|hit| dto_search_hit_to_py(py, hit, include_relationships))
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (text = None, vector = None, limit = None, filters_json = None, include_expired = false, include_retired = false, include_relationships = false, fusion = None))]
    fn retrieve(
        &self,
        py: Python<'_>,
        text: Option<String>,
        vector: Option<Vec<f32>>,
        limit: Option<usize>,
        filters_json: Option<String>,
        include_expired: bool,
        include_retired: bool,
        include_relationships: bool,
        fusion: Option<String>,
    ) -> PyResult<Vec<PyObject>> {
        if fusion.as_deref().is_some_and(|value| value != "rrf") {
            return Err(PyRuntimeError::new_err(
                "retrieve fusion currently supports only 'rrf'",
            ));
        }
        let req = RetrieveRequest {
            text,
            vector,
            limit: limit.unwrap_or(10),
            filters: filters_value_from_json(filters_json)?,
            include_expired,
            include_retired,
            include_relationships,
            fusion: fusion.unwrap_or_else(|| "rrf".to_string()),
        };
        let hits = py
            .allow_threads(|| self.runtime.block_on(self.store.retrieve(&req)))
            .map_err(to_py_err)?;
        hits.into_iter()
            .map(|hit| dto_retrieve_hit_to_py(py, hit, include_relationships))
            .collect()
    }

    fn checkout(&mut self, py: Python<'_>, version_id: u64) -> PyResult<()> {
        py.allow_threads(|| {
            self.runtime
                .block_on(self.store.checkout(version_id))
                .map_err(to_py_err)
        })
    }

    #[pyo3(signature = (target_rows_per_fragment=None, materialize_deletions=None))]
    fn compact(
        &mut self,
        py: Python<'_>,
        target_rows_per_fragment: Option<usize>,
        materialize_deletions: Option<bool>,
    ) -> PyResult<PyObject> {
        let options = if target_rows_per_fragment.is_some() || materialize_deletions.is_some() {
            Some(CompactRequest {
                target_rows_per_fragment,
                materialize_deletions,
            })
        } else {
            None
        };
        let resp = py
            .allow_threads(|| self.runtime.block_on(self.store.compact(options)))
            .map_err(to_py_err)?;
        dto_compact_response_to_py(py, resp)
    }

    fn compaction_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let stats = py
            .allow_threads(|| self.runtime.block_on(self.store.compaction_stats()))
            .map_err(to_py_err)?;
        dto_compact_stats_to_py(py, stats)
    }
}

fn required_item<'py>(
    dict: &Bound<'py, PyDict>,
    key: &str,
    index: usize,
) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(key)?.ok_or_else(|| {
        PyRuntimeError::new_err(format!("records[{index}] is missing required key '{key}'"))
    })
}

fn optional_item<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    Ok(dict.get_item(key)?.filter(|value| !value.is_none()))
}

fn state_metadata_from_dict(dict: Option<&Bound<'_, PyDict>>) -> PyResult<Option<StateMetadata>> {
    let Some(dict) = dict else {
        return Ok(None);
    };

    Ok(Some(StateMetadata {
        step: optional_item(dict, "step")?
            .map(|value| value.extract::<i32>())
            .transpose()?,
        active_plan_id: optional_item(dict, "active_plan_id")?
            .map(|value| value.extract::<String>())
            .transpose()?,
        tokens_used: optional_item(dict, "tokens_used")?
            .map(|value| value.extract::<i32>())
            .transpose()?,
        custom: optional_item(dict, "custom")?
            .map(|value| value.extract::<String>())
            .transpose()?,
    }))
}

fn relationships_patch_from_json(value: Option<String>) -> PyResult<Option<Vec<Relationship>>> {
    value
        .map(|value| relationships_from_json(Some(value)))
        .transpose()
}

fn parse_optional_datetime(
    value: Option<String>,
    field_name: &str,
) -> PyResult<Option<DateTime<Utc>>> {
    value
        .map(|value| {
            DateTime::parse_from_rfc3339(&value)
                .map(|dt| dt.with_timezone(&Utc))
                .map_err(|err| {
                    PyTypeError::new_err(format!(
                        "{field_name} must be an RFC3339 timestamp: {err}"
                    ))
                })
        })
        .transpose()
}

fn compaction_metrics_to_py(py: Python<'_>, metrics: CompactionMetrics) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("fragments_removed", metrics.fragments_removed)?;
    dict.set_item("fragments_added", metrics.fragments_added)?;
    dict.set_item("files_removed", metrics.files_removed)?;
    dict.set_item("files_added", metrics.files_added)?;
    Ok(dict.into_pyobject(py)?.unbind().into())
}

fn compaction_stats_to_py(py: Python<'_>, stats: CompactionStats) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("total_fragments", stats.total_fragments)?;
    dict.set_item("is_compacting", stats.is_compacting)?;
    dict.set_item(
        "last_compaction",
        stats
            .last_compaction
            .map(|dt| dt.to_rfc3339_opts(SecondsFormat::Micros, true)),
    )?;
    dict.set_item("last_error", stats.last_error)?;
    dict.set_item("total_compactions", stats.total_compactions)?;
    Ok(dict.into_pyobject(py)?.unbind().into())
}

fn new_run_id() -> String {
    format!(
        "run-{}-{}",
        Utc::now().timestamp_micros(),
        std::process::id()
    )
}

fn search_hit_to_py(
    py: Python<'_>,
    hit: SearchResult,
    include_relationships: bool,
) -> PyResult<PyObject> {
    let SearchResult { record, distance } = hit;
    let mut record = record;
    if !include_relationships {
        record.relationships.clear();
    }
    let dict = record_to_py(py, record)?;
    let dict_ref = dict.downcast_bound::<PyDict>(py)?;
    dict_ref.set_item("distance", distance)?;
    Ok(dict)
}

fn retrieve_hit_to_py(
    py: Python<'_>,
    hit: RetrieveResult,
    include_relationships: bool,
) -> PyResult<PyObject> {
    let RetrieveResult {
        record,
        score,
        vector_distance,
        text_score,
        matched_channels,
    } = hit;
    let mut record = record;
    if !include_relationships {
        record.relationships.clear();
    }
    let dict = record_to_py(py, record)?;
    let dict_ref = dict.downcast_bound::<PyDict>(py)?;
    dict_ref.set_item("score", score)?;
    dict_ref.set_item("vector_distance", vector_distance)?;
    dict_ref.set_item("text_score", text_score)?;
    dict_ref.set_item("matched_channels", matched_channels)?;
    Ok(dict)
}

fn record_to_py(py: Python<'_>, record: ContextRecord) -> PyResult<PyObject> {
    let ContextRecord {
        id,
        external_id,
        run_id,
        bot_id,
        session_id,
        tenant,
        source,
        created_at,
        role,
        state_metadata,
        metadata,
        relationships,
        expires_at,
        retention_policy,
        lifecycle_status,
        retired_at,
        retired_reason,
        supersedes_id,
        superseded_by_id,
        content_type,
        text_payload,
        binary_payload,
        payload_uri,
        payload_size,
        payload_checksum,
        embedding,
    } = record;

    let dict = PyDict::new(py);
    dict.set_item("id", id)?;
    dict.set_item("external_id", external_id)?;
    dict.set_item("run_id", run_id)?;
    dict.set_item("bot_id", bot_id)?;
    dict.set_item("session_id", session_id)?;
    dict.set_item("tenant", tenant)?;
    dict.set_item("source", source)?;
    dict.set_item(
        "created_at",
        created_at.to_rfc3339_opts(SecondsFormat::Micros, true),
    )?;
    dict.set_item("role", role)?;

    let state_obj: PyObject = match state_metadata {
        Some(metadata) => {
            let state_dict = PyDict::new(py);
            state_dict.set_item("step", metadata.step)?;
            state_dict.set_item("active_plan_id", metadata.active_plan_id)?;
            state_dict.set_item("tokens_used", metadata.tokens_used)?;
            state_dict.set_item("custom", metadata.custom)?;
            state_dict.into_pyobject(py)?.unbind().into()
        }
        None => py.None().into_pyobject(py)?.unbind(),
    };
    dict.set_item("state_metadata", state_obj)?;
    let metadata_obj: PyObject = match metadata {
        Some(metadata) => json_value_to_py(py, &metadata)?,
        None => py.None().into_pyobject(py)?.unbind(),
    };
    dict.set_item("metadata", metadata_obj)?;
    dict.set_item("relationships", relationships_to_py(py, relationships)?)?;
    dict.set_item(
        "expires_at",
        expires_at.map(|dt| dt.to_rfc3339_opts(SecondsFormat::Micros, true)),
    )?;
    dict.set_item("retention_policy", retention_policy)?;
    dict.set_item("lifecycle_status", lifecycle_status)?;
    dict.set_item(
        "retired_at",
        retired_at.map(|dt| dt.to_rfc3339_opts(SecondsFormat::Micros, true)),
    )?;
    dict.set_item("retired_reason", retired_reason)?;
    dict.set_item("supersedes_id", supersedes_id)?;
    dict.set_item("superseded_by_id", superseded_by_id)?;
    dict.set_item("content_type", content_type)?;
    dict.set_item("text_payload", text_payload)?;
    match binary_payload {
        Some(payload) => dict.set_item("binary_payload", PyBytes::new(py, &payload))?,
        None => dict.set_item("binary_payload", py.None())?,
    }
    dict.set_item("payload_uri", payload_uri)?;
    dict.set_item("payload_size", payload_size)?;
    dict.set_item("payload_checksum", payload_checksum)?;
    dict.set_item("embedding", embedding)?;
    Ok(dict.into_pyobject(py)?.unbind().into())
}

fn relationships_to_py(py: Python<'_>, relationships: Vec<Relationship>) -> PyResult<PyObject> {
    let list = PyList::empty(py);
    for relationship in relationships {
        let dict = PyDict::new(py);
        dict.set_item("target_id", relationship.target_id)?;
        dict.set_item("relation", relationship.relation)?;
        dict.set_item("weight", relationship.weight)?;
        list.append(dict)?;
    }
    Ok(list.into_pyobject(py)?.unbind().into())
}

fn partition_info_to_py(py: Python<'_>, partition: PartitionInfo) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    let selector = PyDict::new(py);
    for (key, value) in partition.selector {
        selector.set_item(key, value)?;
    }
    dict.set_item("partition_id", partition.partition_id)?;
    dict.set_item("spec_version", partition.spec_version)?;
    dict.set_item("selector", selector)?;
    dict.set_item("dataset_uri", partition.dataset_uri)?;
    Ok(dict.into_pyobject(py)?.unbind().into())
}

fn json_value_to_py(py: Python<'_>, value: &Value) -> PyResult<PyObject> {
    let json = PyModule::import(py, "json")?;
    Ok(json.call_method1("loads", (value.to_string(),))?.unbind())
}

fn to_py_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

create_exception!(
    _internal,
    ContextStoreError,
    PyRuntimeError,
    "Base class for every error a context store raises."
);
create_exception!(
    _internal,
    NotFoundError,
    ContextStoreError,
    "The requested store, record, or id does not exist."
);
create_exception!(
    _internal,
    AlreadyExistsError,
    ContextStoreError,
    "The store or record being created already exists."
);
create_exception!(
    _internal,
    InvalidRequestError,
    ContextStoreError,
    "The request was rejected as malformed. Deterministic — retrying cannot help."
);
create_exception!(
    _internal,
    InternalError,
    ContextStoreError,
    "A transport or storage fault. The retryable case: the same call may yet succeed."
);
create_exception!(
    _internal,
    CompactionInProgressError,
    ContextStoreError,
    "A compaction holds the store. Retry once it finishes."
);

/// Map a [`ContextError`] onto the exception class that says whether retrying can help.
///
/// Every variant subclasses `RuntimeError`, which is what this crate raised before these
/// classes existed, so `except RuntimeError` keeps catching all of them.
///
/// `ContextError` already separates a transport or storage fault from a request the store
/// refused outright; flattening both to `RuntimeError` forced callers to retry a
/// deterministic rejection until their budget ran out. `InternalError` is the retryable
/// one — the rest are verdicts about the request itself and will fail again identically.
fn ctx_to_py_err(err: ContextError) -> PyErr {
    match err {
        ContextError::NotFound(msg) => NotFoundError::new_err(msg),
        ContextError::AlreadyExists(msg) => AlreadyExistsError::new_err(msg),
        ContextError::InvalidRequest(msg) => InvalidRequestError::new_err(msg),
        ContextError::Internal(msg) => InternalError::new_err(msg),
        ContextError::CompactionInProgress => {
            CompactionInProgressError::new_err(ContextError::CompactionInProgress.to_string())
        }
    }
}

fn content_to_payloads(
    content: &Bound<'_, PyAny>,
    data_type: Option<&str>,
) -> PyResult<(String, Option<String>, Option<Vec<u8>>)> {
    match content.extract::<&[u8]>() {
        Ok(bytes) => Ok((
            data_type
                .map(str::to_string)
                .unwrap_or_else(|| DEFAULT_BINARY_CONTENT_TYPE.to_string()),
            None,
            Some(bytes.to_vec()),
        )),
        Err(_) => {
            let content_str = content.str()?.to_string();
            Ok((
                data_type
                    .map(str::to_string)
                    .unwrap_or_else(|| CONTENT_TYPE_TEXT.to_string()),
                Some(content_str),
                None,
            ))
        }
    }
}

fn dto_state_metadata_from_dict(
    dict: Option<&Bound<'_, PyDict>>,
) -> PyResult<Option<StateMetadataDto>> {
    let Some(dict) = dict else {
        return Ok(None);
    };

    Ok(Some(StateMetadataDto {
        step: optional_item(dict, "step")?
            .map(|value| value.extract::<i32>())
            .transpose()?,
        active_plan_id: optional_item(dict, "active_plan_id")?
            .map(|value| value.extract::<String>())
            .transpose()?,
        tokens_used: optional_item(dict, "tokens_used")?
            .map(|value| value.extract::<i32>())
            .transpose()?,
        custom: optional_item(dict, "custom")?
            .map(|value| value.extract::<String>())
            .transpose()?,
    }))
}

fn dto_relationships_from_json(value: Option<String>) -> PyResult<Vec<RelationshipDto>> {
    value
        .map(|value| serde_json::from_str(&value).map_err(to_py_err))
        .transpose()
        .map(|value| value.unwrap_or_default())
}

fn dto_relationships_patch_from_json(
    value: Option<String>,
) -> PyResult<Option<Vec<RelationshipDto>>> {
    value
        .map(|value| serde_json::from_str::<Vec<RelationshipDto>>(&value).map_err(to_py_err))
        .transpose()
}

fn filters_value_from_json(filters_json: Option<String>) -> PyResult<Option<Value>> {
    filters_json
        .map(|value| serde_json::from_str(&value).map_err(to_py_err))
        .transpose()
}

fn dto_record_to_py(py: Python<'_>, record: RecordDto) -> PyResult<PyObject> {
    let RecordDto {
        id,
        external_id,
        run_id,
        bot_id,
        session_id,
        created_at,
        role,
        content_type,
        text_payload,
        binary_payload,
        payload_uri,
        payload_size,
        payload_checksum,
        embedding,
        state_metadata,
        metadata,
        relationships,
        expires_at,
        retention_policy,
        lifecycle_status,
        retired_at,
        retired_reason,
        supersedes_id,
        superseded_by_id,
        tenant,
        source,
    } = record;

    let dict = PyDict::new(py);
    dict.set_item("id", id)?;
    dict.set_item("external_id", external_id)?;
    dict.set_item("run_id", run_id)?;
    dict.set_item("bot_id", bot_id)?;
    dict.set_item("session_id", session_id)?;
    dict.set_item(
        "created_at",
        created_at.to_rfc3339_opts(SecondsFormat::Micros, true),
    )?;
    dict.set_item("role", role)?;

    let state_obj: PyObject = match state_metadata {
        Some(metadata) => {
            let state_dict = PyDict::new(py);
            state_dict.set_item("step", metadata.step)?;
            state_dict.set_item("active_plan_id", metadata.active_plan_id)?;
            state_dict.set_item("tokens_used", metadata.tokens_used)?;
            state_dict.set_item("custom", metadata.custom)?;
            state_dict.into_pyobject(py)?.unbind().into()
        }
        None => py.None().into_pyobject(py)?.unbind(),
    };
    dict.set_item("state_metadata", state_obj)?;
    let metadata_obj: PyObject = match metadata {
        Some(metadata) => json_value_to_py(py, &metadata)?,
        None => py.None().into_pyobject(py)?.unbind(),
    };
    dict.set_item("metadata", metadata_obj)?;
    dict.set_item("relationships", dto_relationships_to_py(py, relationships)?)?;
    dict.set_item(
        "expires_at",
        expires_at.map(|dt| dt.to_rfc3339_opts(SecondsFormat::Micros, true)),
    )?;
    dict.set_item("retention_policy", retention_policy)?;
    dict.set_item("lifecycle_status", lifecycle_status)?;
    dict.set_item(
        "retired_at",
        retired_at.map(|dt| dt.to_rfc3339_opts(SecondsFormat::Micros, true)),
    )?;
    dict.set_item("retired_reason", retired_reason)?;
    dict.set_item("supersedes_id", supersedes_id)?;
    dict.set_item("superseded_by_id", superseded_by_id)?;
    dict.set_item("tenant", tenant)?;
    dict.set_item("source", source)?;
    dict.set_item("content_type", content_type)?;
    dict.set_item("text_payload", text_payload)?;
    match binary_payload {
        Some(payload) => dict.set_item("binary_payload", PyBytes::new(py, &payload))?,
        None => dict.set_item("binary_payload", py.None())?,
    }
    dict.set_item("payload_uri", payload_uri)?;
    dict.set_item("payload_size", payload_size)?;
    dict.set_item("payload_checksum", payload_checksum)?;
    dict.set_item("embedding", embedding)?;
    Ok(dict.into_pyobject(py)?.unbind().into())
}

fn dto_relationships_to_py(
    py: Python<'_>,
    relationships: Vec<RelationshipDto>,
) -> PyResult<PyObject> {
    let list = PyList::empty(py);
    for relationship in relationships {
        let dict = PyDict::new(py);
        dict.set_item("target_id", relationship.target_id)?;
        dict.set_item("relation", relationship.relation)?;
        dict.set_item("weight", relationship.weight)?;
        list.append(dict)?;
    }
    Ok(list.into_pyobject(py)?.unbind().into())
}

fn dto_search_hit_to_py(
    py: Python<'_>,
    hit: SearchResultDto,
    include_relationships: bool,
) -> PyResult<PyObject> {
    let SearchResultDto { record, distance } = hit;
    let mut record = record;
    if !include_relationships {
        record.relationships.clear();
    }
    let dict = dto_record_to_py(py, record)?;
    let dict_ref = dict.downcast_bound::<PyDict>(py)?;
    dict_ref.set_item("distance", distance)?;
    Ok(dict)
}

fn dto_retrieve_hit_to_py(
    py: Python<'_>,
    hit: RetrieveResultDto,
    include_relationships: bool,
) -> PyResult<PyObject> {
    let RetrieveResultDto {
        record,
        score,
        vector_distance,
        text_score,
        matched_channels,
    } = hit;
    let mut record = record;
    if !include_relationships {
        record.relationships.clear();
    }
    let dict = dto_record_to_py(py, record)?;
    let dict_ref = dict.downcast_bound::<PyDict>(py)?;
    dict_ref.set_item("score", score)?;
    dict_ref.set_item("vector_distance", vector_distance)?;
    dict_ref.set_item("text_score", text_score)?;
    dict_ref.set_item("matched_channels", matched_channels)?;
    Ok(dict)
}

fn dto_compact_response_to_py(py: Python<'_>, resp: CompactResponse) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("fragments_removed", resp.fragments_removed)?;
    dict.set_item("fragments_added", resp.fragments_added)?;
    dict.set_item("files_removed", resp.files_removed)?;
    dict.set_item("files_added", resp.files_added)?;
    Ok(dict.into_pyobject(py)?.unbind().into())
}

fn dto_compact_stats_to_py(py: Python<'_>, stats: CompactStatsResponse) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("total_fragments", stats.total_fragments)?;
    dict.set_item("is_compacting", stats.is_compacting)?;
    dict.set_item(
        "last_compaction",
        stats
            .last_compaction
            .map(|dt| dt.to_rfc3339_opts(SecondsFormat::Micros, true)),
    )?;
    dict.set_item("last_error", stats.last_error)?;
    dict.set_item("total_compactions", stats.total_compactions)?;
    Ok(dict.into_pyobject(py)?.unbind().into())
}

// ---------------------------------------------------------------------------
// Rollout store binding (local + remote via the unified enum)
// ---------------------------------------------------------------------------

/// A rollout store, either an embedded Lance dataset (`open`) or a handle to a
/// remote `lance-context-server` (`connect` / `connect_or_create`).
///
/// Rollout rows carry ~35 fields, so records cross the FFI boundary as JSON
/// (a `records_json` array of objects matching `AddRolloutRequest`), and read
/// results come back as a JSON array string. The Python wrapper in `api.py`
/// hides this and exposes dict/list ergonomics.
#[pyclass]
struct RolloutStore {
    store: UnifiedRolloutStore,
    runtime: Arc<Runtime>,
}

impl RolloutStore {
    fn from_store(store: UnifiedRolloutStore, runtime: Arc<Runtime>) -> Self {
        Self { store, runtime }
    }
}

#[pymethods]
impl RolloutStore {
    /// Open (or create) an embedded rollout dataset at `uri`.
    #[classmethod]
    #[pyo3(signature = (uri, storage_options = None))]
    fn open(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        uri: &str,
        storage_options: Option<HashMap<String, String>>,
    ) -> PyResult<Self> {
        let runtime = Arc::new(Runtime::new().map_err(to_py_err)?);
        let store_res = py.allow_threads(|| {
            runtime.block_on(UnifiedRolloutStore::open_with_options(uri, storage_options))
        });
        let store = store_res.map_err(to_py_err)?;
        Ok(Self::from_store(store, runtime))
    }

    /// Connect to an existing rollout store on a remote server.
    #[classmethod]
    fn connect(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        base_url: &str,
        name: &str,
    ) -> PyResult<Self> {
        let runtime = Arc::new(Runtime::new().map_err(to_py_err)?);
        let store_res =
            py.allow_threads(|| runtime.block_on(UnifiedRolloutStore::connect(base_url, name)));
        let store = store_res.map_err(to_py_err)?;
        Ok(Self::from_store(store, runtime))
    }

    /// Connect to a remote rollout store, creating it if it does not exist.
    #[classmethod]
    #[pyo3(signature = (base_url, name, storage_options = None))]
    fn connect_or_create(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        base_url: &str,
        name: &str,
        storage_options: Option<HashMap<String, String>>,
    ) -> PyResult<Self> {
        let req = CreateRolloutStoreRequest {
            name: name.to_string(),
            storage_options,
        };
        let runtime = Arc::new(Runtime::new().map_err(to_py_err)?);
        let store_res = py.allow_threads(|| {
            runtime.block_on(UnifiedRolloutStore::connect_or_create(base_url, &req))
        });
        let store = store_res.map_err(to_py_err)?;
        Ok(Self::from_store(store, runtime))
    }

    /// Current store version (base dataset version).
    fn version(&self) -> u64 {
        self.store.version()
    }

    /// Seal the local MemWAL memtable so previously added rows become visible
    /// to subsequent reads. No-op for remote stores.
    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.runtime.block_on(self.store.flush()))
            .map_err(to_py_err)
    }

    /// Append rollout rows given as a JSON array of `AddRolloutRequest` objects.
    /// Returns a dict `{version, ids, count}`.
    fn add(&mut self, py: Python<'_>, records_json: &str) -> PyResult<PyObject> {
        let records: Vec<AddRolloutRequest> = serde_json::from_str(records_json)
            .map_err(|e| PyRuntimeError::new_err(format!("invalid records JSON: {e}")))?;
        if records.is_empty() {
            return Err(PyRuntimeError::new_err(
                "records must not be empty".to_string(),
            ));
        }
        let resp = py
            .allow_threads(|| self.runtime.block_on(self.store.add(&records)))
            .map_err(to_py_err)?;
        let dict = PyDict::new(py);
        dict.set_item("version", resp.version)?;
        dict.set_item("ids", resp.ids)?;
        dict.set_item("count", resp.count)?;
        Ok(dict.into_pyobject(py)?.unbind().into())
    }

    /// List rollout rows (artifact bytes projected out). Returns a JSON array
    /// string of records for the Python wrapper to parse into dicts.
    #[pyo3(signature = (limit = None, offset = None, filters_json = None))]
    fn list(
        &self,
        py: Python<'_>,
        limit: Option<usize>,
        offset: Option<usize>,
        filters_json: Option<String>,
    ) -> PyResult<String> {
        let filters = filters_json
            .map(|raw| {
                serde_json::from_str(&raw)
                    .map_err(|err| PyRuntimeError::new_err(format!("invalid filters JSON: {err}")))
            })
            .transpose()?;
        let records = py
            .allow_threads(|| {
                self.runtime
                    .block_on(self.store.list_filtered(limit, offset, filters))
            })
            .map_err(to_py_err)?;
        rollout_records_to_json(&records)
    }

    /// Return one complete trajectory in deterministic message order.
    fn get_trajectory(&self, py: Python<'_>, rollout_id: &str) -> PyResult<String> {
        let records = py
            .allow_threads(|| self.runtime.block_on(self.store.get_trajectory(rollout_id)))
            .map_err(to_py_err)?;
        rollout_records_to_json(&records)
    }

    /// Fetch a single rollout row by id, or `None`. Returns a JSON object
    /// string (artifact bytes projected out).
    fn get(&self, py: Python<'_>, id: &str) -> PyResult<Option<String>> {
        let record = py
            .allow_threads(|| self.runtime.block_on(self.store.get(id)))
            .map_err(to_py_err)?;
        match record {
            Some(r) => Ok(Some(rollout_record_to_json(&r)?)),
            None => Ok(None),
        }
    }

    /// Fetch a single artifact row's inline bytes on demand, or `None`.
    fn get_blob(&self, py: Python<'_>, id: &str) -> PyResult<Option<Py<PyBytes>>> {
        let bytes = py
            .allow_threads(|| self.runtime.block_on(self.store.get_blob(id)))
            .map_err(to_py_err)?;
        Ok(bytes.map(|b| PyBytes::new(py, &b).unbind()))
    }

    /// Checkout a base-table version (time travel over the base table only).
    fn checkout(&mut self, py: Python<'_>, version: u64) -> PyResult<()> {
        py.allow_threads(|| self.runtime.block_on(self.store.checkout(version)))
            .map_err(to_py_err)
    }
}

fn rollout_records_to_json(records: &[RolloutRecordDto]) -> PyResult<String> {
    serde_json::to_string(records).map_err(to_py_err)
}

fn rollout_record_to_json(record: &RolloutRecordDto) -> PyResult<String> {
    serde_json::to_string(record).map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// Datagen store binding (append-only delta-log, fold model)
// ---------------------------------------------------------------------------

/// A single embedded datagen checkpoint log (`open`).
///
/// Events cross the FFI boundary as plain dicts (not pickle): the executor
/// builds each `DatagenEvent` field-by-field, `append_checkpoint` persists one
/// atomic step boundary, and reads (`fold_item`) return the folded item state as
/// a dict. Mirrors the concurrency contract of the rollout store: open a fresh
/// handle per writer, never share a handle across concurrent appends.
#[pyclass]
struct DatagenStore {
    store: UnifiedDatagenStore,
    runtime: Arc<Runtime>,
}

impl DatagenStore {
    fn from_store(store: UnifiedDatagenStore, runtime: Arc<Runtime>) -> Self {
        Self { store, runtime }
    }
}

#[pymethods]
impl DatagenStore {
    /// Open (or create) an embedded datagen log at `uri`. `shard_id` gives this
    /// writer instance a stable identity for multi-writer fencing.
    #[classmethod]
    #[pyo3(signature = (uri, storage_options = None, shard_id = None))]
    fn open(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        uri: &str,
        storage_options: Option<HashMap<String, String>>,
        shard_id: Option<String>,
    ) -> PyResult<Self> {
        let _ = shard_id;
        let runtime = Arc::new(Runtime::new().map_err(to_py_err)?);
        let store_res = py.allow_threads(|| {
            runtime.block_on(UnifiedDatagenStore::open_with_options(uri, storage_options))
        });
        let store = store_res.map_err(ctx_to_py_err)?;
        Ok(Self::from_store(store, runtime))
    }

    /// Connect to an existing datagen store on a remote server.
    #[classmethod]
    fn connect(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        base_url: &str,
        name: &str,
    ) -> PyResult<Self> {
        let runtime = Arc::new(Runtime::new().map_err(to_py_err)?);
        let store_res =
            py.allow_threads(|| runtime.block_on(UnifiedDatagenStore::connect(base_url, name)));
        let store = store_res.map_err(ctx_to_py_err)?;
        Ok(Self::from_store(store, runtime))
    }

    /// Connect to a remote datagen store, creating it if it does not exist.
    #[classmethod]
    #[pyo3(signature = (base_url, name, storage_options = None))]
    fn connect_or_create(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        base_url: &str,
        name: &str,
        storage_options: Option<HashMap<String, String>>,
    ) -> PyResult<Self> {
        let req = CreateDatagenStoreRequest {
            name: name.to_string(),
            storage_options,
        };
        let runtime = Arc::new(Runtime::new().map_err(to_py_err)?);
        let store_res = py.allow_threads(|| {
            runtime.block_on(UnifiedDatagenStore::connect_or_create(base_url, &req))
        });
        let store = store_res.map_err(ctx_to_py_err)?;
        Ok(Self::from_store(store, runtime))
    }

    /// Current store version (base dataset version).
    fn version(&self) -> u64 {
        self.store.version()
    }

    /// Append one completed step boundary atomically. `events` is a list of
    /// event dicts sharing one item/checkpoint/writer attempt, with exactly one
    /// STEP_COMPLETED. Returns the new store version.
    fn append_checkpoint(&mut self, py: Python<'_>, events: &Bound<'_, PyList>) -> PyResult<u64> {
        let parsed = events_from_pylist(events)?;
        let resp = py
            .allow_threads(|| self.runtime.block_on(self.store.append_checkpoint(&parsed)))
            .map_err(ctx_to_py_err)?;
        Ok(resp.version)
    }

    /// Append raw events as one MemWAL generation (no single-STEP_COMPLETED
    /// constraint). Returns the new store version.
    fn append(&mut self, py: Python<'_>, events: &Bound<'_, PyList>) -> PyResult<u64> {
        let parsed = events_from_pylist(events)?;
        let resp = py
            .allow_threads(|| self.runtime.block_on(self.store.append(&parsed)))
            .map_err(ctx_to_py_err)?;
        Ok(resp.version)
    }

    /// Fold an item's events into its latest state, or `None` if never started.
    /// `load_blobs` materializes blob-field bytes inline; the default leaves them
    /// lazy, to be resolved through `get_blob`.
    #[pyo3(signature = (item_id, load_blobs = false))]
    fn fold_item(
        &self,
        py: Python<'_>,
        item_id: &str,
        load_blobs: bool,
    ) -> PyResult<Option<PyObject>> {
        let item = py
            .allow_threads(|| {
                self.runtime
                    .block_on(self.store.fold_item_with_blobs(item_id, load_blobs))
            })
            .map_err(ctx_to_py_err)?;
        match item {
            None => Ok(None),
            Some(item) => Ok(Some(folded_item_to_py(py, &item)?)),
        }
    }

    /// Aggregate the whole store into a run overview dict: root-item counts by
    /// status, completed-step counts, and a failure roll-up by `run_id` (each
    /// bucket carrying a small sample of failing root item ids).
    fn overview(&self, py: Python<'_>) -> PyResult<PyObject> {
        let overview = py
            .allow_threads(|| self.runtime.block_on(self.store.overview()))
            .map_err(ctx_to_py_err)?;
        run_overview_to_py(py, &overview)
    }

    /// Classify each root item id by folded lifecycle status. Missing ids (never
    /// started) are absent from the returned dict.
    fn root_item_statuses(&self, py: Python<'_>, root_item_ids: Vec<String>) -> PyResult<PyObject> {
        let statuses = py
            .allow_threads(|| {
                self.runtime
                    .block_on(self.store.root_item_statuses(&root_item_ids))
            })
            .map_err(ctx_to_py_err)?;
        let dict = PyDict::new(py);
        for (item_id, status) in statuses.statuses.iter() {
            dict.set_item(item_id, status)?;
        }
        Ok(dict.into_pyobject(py)?.unbind().into())
    }

    /// All failure records for an item (the failure lens), oldest first.
    fn item_failures(&self, py: Python<'_>, item_id: &str) -> PyResult<PyObject> {
        let failures = py
            .allow_threads(|| self.runtime.block_on(self.store.item_failures(item_id)))
            .map_err(ctx_to_py_err)?;
        let list = PyList::empty(py);
        for failure in &failures {
            list.append(failure_to_py(py, failure)?)?;
        }
        Ok(list.into_pyobject(py)?.unbind().into())
    }

    /// Materialize one FIELD_* event's blob bytes by event id, or `None`.
    fn get_blob(&self, py: Python<'_>, event_id: &str) -> PyResult<Option<Py<PyBytes>>> {
        let bytes = py
            .allow_threads(|| self.runtime.block_on(self.store.get_blob(event_id)))
            .map_err(ctx_to_py_err)?;
        Ok(bytes.map(|b| PyBytes::new(py, &b).unbind()))
    }

    /// Assemble the inspection tree rooted at `root_item_id`: every projected
    /// descendant folded to latest state and linked parent->child. Returns a dict
    /// `{"roots": [item_id, ...], "nodes": {item_id: {"item": folded, "children":
    /// [item_id, ...]}}}`.
    fn item_tree(&self, py: Python<'_>, root_item_id: &str) -> PyResult<PyObject> {
        let tree = py
            .allow_threads(|| self.runtime.block_on(self.store.item_tree(root_item_id)))
            .map_err(ctx_to_py_err)?;
        item_tree_to_py(py, &tree)
    }

    /// Open a fresh stream: persist ITEM_CREATED and return a writer positioned to
    /// continue after it. `run_id`/`writer_epoch` stamp every event the writer
    /// emits; `query_tags` (JSON) is captured onto ITEM_CREATED.
    #[pyo3(signature = (item_id, run_id, writer_epoch, parent_item_id = None, query_tags = None))]
    fn open_stream(
        &mut self,
        py: Python<'_>,
        item_id: &str,
        run_id: &str,
        writer_epoch: &str,
        parent_item_id: Option<&str>,
        query_tags: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<DatagenStreamWriter> {
        let stream = DatagenNewStream {
            item_id: CoreDatagenItemId::parse(item_id).map_err(to_py_err)?,
            parent_item_id: parent_item_id
                .map(CoreDatagenItemId::parse)
                .transpose()
                .map_err(to_py_err)?,
            query_tags: query_tags.map(py_any_to_json).transpose()?,
        };
        let context = DatagenWriteContext {
            run_id: run_id.to_string(),
            writer_epoch: writer_epoch.to_string(),
        };
        let writer = py
            .allow_threads(|| {
                self.runtime
                    .block_on(self.store.open_stream(&stream, &context))
            })
            .map_err(ctx_to_py_err)?;
        Ok(DatagenStreamWriter { inner: writer })
    }

    /// Rebuild a writer to resume an already-started item. Pure — emits nothing.
    /// Returns `None` if the item never started.
    fn resume_stream(
        &self,
        py: Python<'_>,
        item_id: &str,
        run_id: &str,
        writer_epoch: &str,
    ) -> PyResult<Option<DatagenStreamWriter>> {
        let context = DatagenWriteContext {
            run_id: run_id.to_string(),
            writer_epoch: writer_epoch.to_string(),
        };
        let writer = py
            .allow_threads(|| {
                self.runtime
                    .block_on(self.store.resume_stream(item_id, &context))
            })
            .map_err(ctx_to_py_err)?;
        Ok(writer.map(|inner| DatagenStreamWriter { inner }))
    }
}

/// A structured datagen item id. Root ids come from the executor's source key; sub-item
/// ids extend a parent with one fan-out segment (`5/expand:0`). `str(id)` is the stored
/// path form. Pure and client-side — composing an id does no I/O.
#[pyclass]
#[derive(Clone)]
struct DatagenItemId {
    inner: CoreDatagenItemId,
}

#[pymethods]
impl DatagenItemId {
    /// Build a root id from the executor's source key.
    #[staticmethod]
    fn from_source_key(key: &str) -> Self {
        Self {
            inner: CoreDatagenItemId::from_source_key(key),
        }
    }

    /// Parse a stored path string (`"5/expand:0"`) back into a structured id.
    #[staticmethod]
    fn parse(path: &str) -> PyResult<Self> {
        Ok(Self {
            inner: CoreDatagenItemId::parse(path).map_err(to_py_err)?,
        })
    }

    /// Extend this id with one fan-out segment -> the sub-item's id.
    fn child(&self, origin_step: &str, branch_idx: i64) -> Self {
        Self {
            inner: self.inner.child(origin_step, branch_idx),
        }
    }

    /// The parent stream's id (`None` on a root).
    fn parent(&self) -> Option<Self> {
        self.inner.parent().map(|inner| Self { inner })
    }

    /// The root of this id's tree (== self if root).
    fn root(&self) -> Self {
        Self {
            inner: self.inner.root(),
        }
    }

    /// The fan-out step that created this sub-item (`None` on a root).
    #[getter]
    fn origin_step(&self) -> Option<String> {
        self.inner.origin_step().map(str::to_string)
    }

    /// Which branch this sub-item is (`None` on a root).
    #[getter]
    fn branch_idx(&self) -> Option<i64> {
        self.inner.branch_idx()
    }

    /// Whether this id names a root stream.
    #[getter]
    fn is_root(&self) -> bool {
        self.inner.is_root()
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("DatagenItemId({:?})", self.inner.to_string())
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }
}

/// A per-stream write handle. Owns the bookkeeping columns the caller should never
/// touch (`item_seq`, `attempt`, `checkpoint_id`, `event_id`), stamping them onto
/// every event it emits. Each method returns the event dict(s) to persist — the
/// caller hands them to `DatagenStore.append`/`append_checkpoint`. Pure and
/// client-side, so it is identical for embedded and remote stores.
#[pyclass]
struct DatagenStreamWriter {
    inner: CoreDatagenStreamWriter,
}

#[pymethods]
impl DatagenStreamWriter {
    /// The item this writer streams to.
    #[getter]
    fn item_id(&self) -> String {
        self.inner.item_id().to_string()
    }

    /// The attempt number stamped onto emitted events (0 fresh, `last_attempt + 1`
    /// on resume).
    #[getter]
    fn attempt(&self) -> i32 {
        self.inner.attempt()
    }

    /// Emit STEP_STARTED for a driver frame (Sequence/Loop). Returns one event dict.
    fn step_started(&mut self, py: Python<'_>, position: &Bound<'_, PyDict>) -> PyResult<PyObject> {
        let position = position_from_dict(position)?;
        let event = self.inner.step_started(&position);
        event_dto_to_py(py, &datagen_event_to_dto(&event))
    }

    /// Emit a checkpoint boundary: the step's field writes plus its STEP_COMPLETED,
    /// all sharing one `checkpoint_id`. Returns a list of event dicts (the atomic
    /// unit `append_checkpoint` persists).
    fn step_completed(
        &mut self,
        py: Python<'_>,
        position: &Bound<'_, PyDict>,
        fields: &Bound<'_, PyList>,
    ) -> PyResult<PyObject> {
        let position = position_from_dict(position)?;
        let changes = field_changes_from_pylist(fields)?;
        let events = self.inner.step_completed(&position, &changes);
        let list = PyList::empty(py);
        for event in &events {
            list.append(event_dto_to_py(py, &datagen_event_to_dto(event))?)?;
        }
        Ok(list.into_pyobject(py)?.unbind().into())
    }

    /// Emit TERMINAL — the item reached a lifecycle end. `terminal` is
    /// `"completed"` or `"filtered"`. Returns one event dict.
    fn item_terminal(&mut self, py: Python<'_>, terminal: &str) -> PyResult<PyObject> {
        let terminal = match terminal {
            "completed" => DatagenTerminal::Completed,
            "filtered" => DatagenTerminal::Filtered,
            other => {
                return Err(PyValueError::new_err(format!(
                    "terminal must be 'completed' or 'filtered', got '{other}'"
                )))
            }
        };
        let event = self.inner.item_terminal(terminal);
        event_dto_to_py(py, &datagen_event_to_dto(&event))
    }

    /// Emit FAILED at a step position (failure lens only; does not terminate the
    /// item). Returns one event dict.
    #[pyo3(signature = (position, error_type, error_dump = None, traceback = None))]
    fn item_failed(
        &mut self,
        py: Python<'_>,
        position: &Bound<'_, PyDict>,
        error_type: &str,
        error_dump: Option<String>,
        traceback: Option<String>,
    ) -> PyResult<PyObject> {
        let position = position_from_dict(position)?;
        let error = DatagenErrorInfo {
            error_type: error_type.to_string(),
            error_dump,
            traceback,
        };
        let event = self.inner.item_failed(&position, &error);
        event_dto_to_py(py, &datagen_event_to_dto(&event))
    }
}

fn position_from_dict(dict: &Bound<'_, PyDict>) -> PyResult<DatagenStreamPosition> {
    let step_name = required_item(dict, "step_name", 0)?.extract::<String>()?;
    let step_kind_raw = required_item(dict, "step_kind", 0)?.extract::<String>()?;
    let step_kind = DatagenStepKind::parse(&step_kind_raw).map_err(to_py_err)?;
    let index = required_item(dict, "index", 0)?.extract::<i64>()?;
    let enclosing = optional_item(dict, "enclosing")?
        .map(|value| value.extract::<String>())
        .transpose()?;
    let selector = optional_item(dict, "selector")?
        .map(|value| value.extract::<String>())
        .transpose()?;
    Ok(DatagenStreamPosition {
        step: DatagenStepId {
            name: step_name,
            kind: step_kind,
        },
        index,
        enclosing,
        selector,
    })
}

fn field_changes_from_pylist(fields: &Bound<'_, PyList>) -> PyResult<Vec<DatagenFieldChange>> {
    fields
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let dict = item
                .downcast::<PyDict>()
                .map_err(|_| PyTypeError::new_err(format!("fields[{index}] must be a dict")))?;
            field_change_from_dict(dict)
        })
        .collect()
}

fn field_change_from_dict(dict: &Bound<'_, PyDict>) -> PyResult<DatagenFieldChange> {
    let name = required_item(dict, "name", 0)?.extract::<String>()?;
    let field_type = required_item(dict, "field_type", 0)?.extract::<String>()?;
    let codec_version = optional_item(dict, "codec_version")?
        .map(|value| value.extract::<i32>())
        .transpose()?
        .unwrap_or(0);
    let op = match optional_item(dict, "op")?
        .map(|value| value.extract::<String>())
        .transpose()?
        .as_deref()
    {
        Some("append") => FieldOp::Append,
        Some("set") | None => FieldOp::Set,
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "field op must be 'set' or 'append', got '{other}'"
            )))
        }
    };
    let value_dict = required_item(dict, "value", 0)?;
    let value_dict = value_dict
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("field 'value' must be a dict"))?;
    let value = core_value_from_dict(value_dict)?;
    Ok(DatagenFieldChange {
        name,
        field_type,
        codec_version,
        op,
        value,
    })
}

fn core_value_from_dict(dict: &Bound<'_, PyDict>) -> PyResult<CoreDatagenValue> {
    let kind = dict
        .get_item("kind")?
        .ok_or_else(|| PyRuntimeError::new_err("field value is missing 'kind'"))?
        .extract::<String>()?;
    let inner = || {
        dict.get_item("value")?
            .ok_or_else(|| PyRuntimeError::new_err("field value is missing 'value'"))
    };
    match kind.as_str() {
        "int" => Ok(CoreDatagenValue::Int(inner()?.extract::<i64>()?)),
        "float" => Ok(CoreDatagenValue::Float(inner()?.extract::<f64>()?)),
        "bool" => Ok(CoreDatagenValue::Bool(inner()?.extract::<bool>()?)),
        "str" => Ok(CoreDatagenValue::Str(inner()?.extract::<String>()?)),
        "json" => Ok(CoreDatagenValue::Json(py_any_to_json(&inner()?)?)),
        "blob" => {
            let bytes = dict
                .get_item("bytes")?
                .filter(|value| !value.is_none())
                .map(|value| value.extract::<Vec<u8>>())
                .transpose()?
                .ok_or_else(|| PyRuntimeError::new_err("blob field value is missing 'bytes'"))?;
            let size = bytes.len() as i64;
            Ok(CoreDatagenValue::Blob(CoreDatagenBlobValue {
                bytes: Some(bytes),
                size,
                checksum: None,
            }))
        }
        other => Err(PyRuntimeError::new_err(format!(
            "unsupported datagen value kind '{other}'"
        ))),
    }
}

/// Deterministic idempotency key for a checkpoint event, so retried batches
/// dedup instead of double-appending.
#[pyfunction]
fn datagen_event_id(item_id: &str, checkpoint_id: &str, ordinal: u32) -> String {
    core_datagen_event_id(item_id, checkpoint_id, ordinal)
}

fn events_from_pylist(events: &Bound<'_, PyList>) -> PyResult<Vec<DatagenEventDto>> {
    events
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let dict = item
                .downcast::<PyDict>()
                .map_err(|_| PyTypeError::new_err(format!("events[{index}] must be a dict")))?;
            event_from_dict(dict, index)
        })
        .collect()
}

fn event_from_dict(dict: &Bound<'_, PyDict>, index: usize) -> PyResult<DatagenEventDto> {
    let value = optional_item(dict, "value")?
        .map(|value| {
            let value_dict = value
                .downcast::<PyDict>()
                .map_err(|_| PyTypeError::new_err("event value must be a dict"))?;
            value_from_dict(value_dict)
        })
        .transpose()?;
    let query_tags = optional_item(dict, "query_tags")?
        .map(|value| py_any_to_json(&value))
        .transpose()?;
    let event_ts = optional_item(dict, "event_ts")?
        .map(|value| parse_optional_datetime(Some(value.extract::<String>()?), "event_ts"))
        .transpose()?
        .flatten();

    Ok(DatagenEventDto {
        event_id: required_item(dict, "event_id", index)?.extract::<String>()?,
        item_id: required_item(dict, "item_id", index)?.extract::<String>()?,
        root_item_id: required_item(dict, "root_item_id", index)?.extract::<String>()?,
        parent_item_id: optional_item(dict, "parent_item_id")?
            .map(|value| value.extract::<String>())
            .transpose()?,
        item_seq: required_item(dict, "item_seq", index)?.extract::<i64>()?,
        checkpoint_id: required_item(dict, "checkpoint_id", index)?.extract::<String>()?,
        event_type: required_item(dict, "event_type", index)?.extract::<String>()?,
        step_name: optional_item(dict, "step_name")?
            .map(|value| value.extract::<String>())
            .transpose()?,
        step_kind: optional_item(dict, "step_kind")?
            .map(|value| value.extract::<String>())
            .transpose()?,
        step_index: optional_item(dict, "step_index")?
            .map(|value| value.extract::<i64>())
            .transpose()?,
        enclosing_step: optional_item(dict, "enclosing_step")?
            .map(|value| value.extract::<String>())
            .transpose()?,
        selector_step: optional_item(dict, "selector_step")?
            .map(|value| value.extract::<String>())
            .transpose()?,
        attempt: optional_item(dict, "attempt")?
            .map(|value| value.extract::<i32>())
            .transpose()?
            .unwrap_or(0),
        run_id: required_item(dict, "run_id", index)?.extract::<String>()?,
        writer_epoch: required_item(dict, "writer_epoch", index)?.extract::<String>()?,
        field_name: optional_item(dict, "field_name")?
            .map(|value| value.extract::<String>())
            .transpose()?,
        field_type: optional_item(dict, "field_type")?
            .map(|value| value.extract::<String>())
            .transpose()?,
        codec_version: optional_item(dict, "codec_version")?
            .map(|value| value.extract::<i32>())
            .transpose()?,
        value,
        query_tags,
        status: optional_item(dict, "status")?
            .map(|value| value.extract::<String>())
            .transpose()?,
        error_type: optional_item(dict, "error_type")?
            .map(|value| value.extract::<String>())
            .transpose()?,
        error_dump: optional_item(dict, "error_dump")?
            .map(|value| value.extract::<String>())
            .transpose()?,
        traceback: optional_item(dict, "traceback")?
            .map(|value| value.extract::<String>())
            .transpose()?,
        event_ts,
        schema_version: optional_item(dict, "schema_version")?
            .map(|value| value.extract::<i32>())
            .transpose()?
            .unwrap_or(DATAGEN_SCHEMA_VERSION),
    })
}

fn value_from_dict(dict: &Bound<'_, PyDict>) -> PyResult<DatagenValueDto> {
    let kind = dict
        .get_item("kind")?
        .ok_or_else(|| PyRuntimeError::new_err("event value is missing 'kind'"))?
        .extract::<String>()?;
    let inner = || {
        dict.get_item("value")?
            .ok_or_else(|| PyRuntimeError::new_err("event value is missing 'value'"))
    };
    let mut dto = DatagenValueDto {
        kind: kind.clone(),
        value: None,
        bytes: None,
        size: None,
        checksum: None,
    };
    match kind.as_str() {
        "int" => dto.value = Some(Value::from(inner()?.extract::<i64>()?)),
        "float" => dto.value = Some(Value::from(inner()?.extract::<f64>()?)),
        "bool" => dto.value = Some(Value::from(inner()?.extract::<bool>()?)),
        "str" => dto.value = Some(Value::from(inner()?.extract::<String>()?)),
        "json" => dto.value = Some(py_any_to_json(&inner()?)?),
        "blob" => {
            let bytes = dict
                .get_item("bytes")?
                .filter(|value| !value.is_none())
                .map(|value| value.extract::<Vec<u8>>())
                .transpose()?;
            let size = match dict.get_item("size")? {
                Some(value) if !value.is_none() => Some(value.extract::<i64>()?),
                _ => bytes.as_ref().map(|b| b.len() as i64),
            };
            let checksum = dict
                .get_item("checksum")?
                .filter(|value| !value.is_none())
                .map(|value| value.extract::<String>())
                .transpose()?;
            dto.bytes = bytes;
            dto.size = size;
            dto.checksum = checksum;
        }
        other => {
            return Err(PyRuntimeError::new_err(format!(
                "unsupported datagen value kind '{other}'"
            )))
        }
    }
    Ok(dto)
}

fn py_any_to_json(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    let py = value.py();
    let json = PyModule::import(py, "json")?;
    let text = json.call_method1("dumps", (value,))?.extract::<String>()?;
    serde_json::from_str(&text).map_err(to_py_err)
}

fn value_to_py(py: Python<'_>, value: &DatagenValueDto) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("kind", &value.kind)?;
    if value.kind == "blob" {
        dict.set_item("bytes", value.bytes.as_ref().map(|b| PyBytes::new(py, b)))?;
        dict.set_item("size", value.size)?;
        dict.set_item("checksum", value.checksum.clone())?;
    } else if let Some(inner) = &value.value {
        dict.set_item("value", json_value_to_py(py, inner)?)?;
    }
    Ok(dict.into_pyobject(py)?.unbind().into())
}

fn folded_item_to_py(py: Python<'_>, item: &FoldedDatagenItemDto) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("item_id", &item.item_id)?;
    dict.set_item("root_item_id", &item.root_item_id)?;
    dict.set_item("parent_item_id", item.parent_item_id.clone())?;
    dict.set_item("status", &item.status)?;
    dict.set_item("last_item_seq", item.last_item_seq)?;
    dict.set_item("last_attempt", item.last_attempt)?;

    let fields = PyDict::new(py);
    for (name, state) in &item.fields {
        fields.set_item(name, field_state_to_py(py, state)?)?;
    }
    dict.set_item("fields", fields)?;

    let trajectory = PyList::empty(py);
    for cursor in &item.trajectory {
        trajectory.append(cursor_to_py(py, cursor)?)?;
    }
    dict.set_item("trajectory", trajectory)?;

    let started = PyList::empty(py);
    for position in &item.started {
        started.append(position_to_py(py, position)?)?;
    }
    dict.set_item("started", started)?;

    let completed = PyList::empty(py);
    for position in &item.completed {
        completed.append(position_to_py(py, position)?)?;
    }
    dict.set_item("completed", completed)?;

    dict.set_item(
        "query_tags",
        match &item.query_tags {
            Some(tags) => Some(json_value_to_py(py, tags)?),
            None => None,
        },
    )?;

    let blob_event_ids = PyDict::new(py);
    for (field_name, event_id) in &item.blob_event_ids {
        blob_event_ids.set_item(field_name, event_id)?;
    }
    dict.set_item("blob_event_ids", blob_event_ids)?;
    Ok(dict.into_pyobject(py)?.unbind().into())
}

fn field_state_to_py(py: Python<'_>, state: &DatagenFieldStateDto) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("mode", &state.mode)?;
    if let Some(value) = &state.value {
        dict.set_item("value", value_to_py(py, value)?)?;
    }
    if !state.values.is_empty() {
        let list = PyList::empty(py);
        for value in &state.values {
            list.append(value_to_py(py, value)?)?;
        }
        dict.set_item("values", list)?;
    }
    Ok(dict.into_pyobject(py)?.unbind().into())
}

fn cursor_to_py(py: Python<'_>, cursor: &DatagenStepCursorDto) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("step_name", &cursor.step_name)?;
    dict.set_item("step_kind", &cursor.step_kind)?;
    dict.set_item("step_index", cursor.step_index)?;
    dict.set_item("enclosing_step", cursor.enclosing_step.clone())?;
    dict.set_item("selector_step", cursor.selector_step.clone())?;
    dict.set_item("item_seq", cursor.item_seq)?;
    Ok(dict.into_pyobject(py)?.unbind().into())
}

fn position_to_py(py: Python<'_>, position: &DatagenStreamPositionDto) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("step_name", &position.step_name)?;
    dict.set_item("step_kind", &position.step_kind)?;
    dict.set_item("step_index", position.step_index)?;
    dict.set_item("enclosing_step", position.enclosing_step.clone())?;
    dict.set_item("selector_step", position.selector_step.clone())?;
    Ok(dict.into_pyobject(py)?.unbind().into())
}

fn failure_to_py(py: Python<'_>, failure: &DatagenFailureDto) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("at", cursor_to_py(py, &failure.at)?)?;
    dict.set_item("run_id", &failure.run_id)?;
    dict.set_item("attempt", failure.attempt)?;
    dict.set_item("error_type", &failure.error_type)?;
    dict.set_item("error_dump", failure.error_dump.clone())?;
    dict.set_item("traceback", failure.traceback.clone())?;
    Ok(dict.into_pyobject(py)?.unbind().into())
}

/// One event as the dict shape `event_from_dict` accepts, so writer-emitted events
/// round-trip straight back into `append`/`append_checkpoint`.
fn event_dto_to_py(py: Python<'_>, event: &DatagenEventDto) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("event_id", &event.event_id)?;
    dict.set_item("item_id", &event.item_id)?;
    dict.set_item("root_item_id", &event.root_item_id)?;
    dict.set_item("parent_item_id", event.parent_item_id.clone())?;
    dict.set_item("item_seq", event.item_seq)?;
    dict.set_item("checkpoint_id", &event.checkpoint_id)?;
    dict.set_item("event_type", &event.event_type)?;
    dict.set_item("step_name", event.step_name.clone())?;
    dict.set_item("step_kind", event.step_kind.clone())?;
    dict.set_item("step_index", event.step_index)?;
    dict.set_item("enclosing_step", event.enclosing_step.clone())?;
    dict.set_item("selector_step", event.selector_step.clone())?;
    dict.set_item("attempt", event.attempt)?;
    dict.set_item("run_id", &event.run_id)?;
    dict.set_item("writer_epoch", &event.writer_epoch)?;
    dict.set_item("field_name", event.field_name.clone())?;
    dict.set_item("field_type", event.field_type.clone())?;
    dict.set_item("codec_version", event.codec_version)?;
    dict.set_item(
        "value",
        match &event.value {
            Some(value) => Some(value_to_py(py, value)?),
            None => None,
        },
    )?;
    dict.set_item(
        "query_tags",
        match &event.query_tags {
            Some(tags) => Some(json_value_to_py(py, tags)?),
            None => None,
        },
    )?;
    dict.set_item("status", event.status.clone())?;
    dict.set_item("error_type", event.error_type.clone())?;
    dict.set_item("error_dump", event.error_dump.clone())?;
    dict.set_item("traceback", event.traceback.clone())?;
    dict.set_item("event_ts", event.event_ts.map(|ts| ts.to_rfc3339()))?;
    dict.set_item("schema_version", event.schema_version)?;
    Ok(dict.into_pyobject(py)?.unbind().into())
}

/// One tree node: the folded item plus its direct children's item_ids.
fn item_node_to_py(py: Python<'_>, node: &UnifiedDatagenItemNode) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item(
        "item",
        folded_item_to_py(py, &folded_item_to_dto(&node.item))?,
    )?;
    let children = PyList::empty(py);
    for child in &node.children {
        children.append(child.to_string())?;
    }
    dict.set_item("children", children)?;
    Ok(dict.into_pyobject(py)?.unbind().into())
}

fn item_tree_to_py(py: Python<'_>, tree: &UnifiedDatagenItemTree) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    let roots = PyList::empty(py);
    for root in tree.roots() {
        roots.append(root.to_string())?;
    }
    dict.set_item("roots", roots)?;
    let nodes = PyDict::new(py);
    for item in tree.items() {
        let node = tree.node(&item.item_id).expect("item is in tree");
        nodes.set_item(item.item_id.to_string(), item_node_to_py(py, node)?)?;
    }
    dict.set_item("nodes", nodes)?;
    Ok(dict.into_pyobject(py)?.unbind().into())
}

fn run_overview_to_py(
    py: Python<'_>,
    overview: &lance_context::DatagenRunOverviewDto,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("items", overview.items)?;
    dict.set_item("running", overview.running)?;
    dict.set_item("completed", overview.completed)?;
    dict.set_item("filtered", overview.filtered)?;
    dict.set_item("failures", overview.failures)?;
    dict.set_item(
        "failures_by_error_type",
        counts_to_py(py, &overview.failures_by_error_type)?,
    )?;
    dict.set_item(
        "completed_steps",
        counts_to_py(py, &overview.completed_steps)?,
    )?;
    let by_run = PyDict::new(py);
    for (run_id, bucket) in &overview.failures_by_run {
        let entry = PyDict::new(py);
        entry.set_item("failures", bucket.failures)?;
        entry.set_item(
            "failures_by_error_type",
            counts_to_py(py, &bucket.failures_by_error_type)?,
        )?;
        let samples = PyList::empty(py);
        for root in &bucket.sample_root_item_ids {
            samples.append(root)?;
        }
        entry.set_item("sample_root_item_ids", samples)?;
        by_run.set_item(run_id, entry)?;
    }
    dict.set_item("failures_by_run", by_run)?;
    Ok(dict.into_pyobject(py)?.unbind().into())
}

fn counts_to_py(
    py: Python<'_>,
    counts: &std::collections::BTreeMap<String, usize>,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    for (key, count) in counts {
        dict.set_item(key, count)?;
    }
    Ok(dict.into_pyobject(py)?.unbind().into())
}

/// A store over a user-declared schema. Rows are plain dicts; the schema is
/// declared once at creation and persisted in the dataset.
#[pyclass]
struct GenericStore {
    store: UnifiedGenericStore,
    runtime: Arc<Runtime>,
}

impl GenericStore {
    fn from_store(store: UnifiedGenericStore, runtime: Arc<Runtime>) -> Self {
        Self { store, runtime }
    }
}

/// Convert a Python dict describing a schema into a [`SchemaSpec`].
///
/// Accepts the same JSON shape the REST API takes, so the two surfaces cannot
/// drift: `{"id": {"type": "string", "nullable": false}, ...}`, or the
/// shorthand `{"id": "string"}` for a nullable column of that type.
fn schema_spec_from_py(schema: &Bound<'_, PyDict>) -> PyResult<SchemaSpec> {
    let mut columns = Vec::with_capacity(schema.len());
    for (key, value) in schema.iter() {
        let name: String = key.extract()?;
        // Shorthand: a bare type name means a nullable column of that type.
        let spec_json = if let Ok(type_name) = value.extract::<String>() {
            serde_json::json!({ "type": type_name })
        } else {
            let encoded = py_dict_to_json(&value)
                .map_err(|e| PyValueError::new_err(format!("column '{name}': {e}")))?;
            serde_json::from_str(&encoded)
                .map_err(|e| PyValueError::new_err(format!("column '{name}': invalid spec: {e}")))?
        };
        let column: ColumnSpec = serde_json::from_value(spec_json)
            .map_err(|e| PyValueError::new_err(format!("column '{name}': invalid spec: {e}")))?;
        columns.push((name, column));
    }
    let spec = SchemaSpec::new(columns);
    spec.validate().map_err(PyValueError::new_err)?;
    Ok(spec)
}

/// Serialize an arbitrary Python object to JSON via the `json` module, so
/// nested dicts describing a column type round-trip without a hand-written
/// converter.
fn py_dict_to_json(value: &Bound<'_, PyAny>) -> PyResult<String> {
    let json = value.py().import("json")?;
    json.call_method1("dumps", (value,))?.extract()
}

/// Convert a Python mapping into one row.
fn row_from_py(row: &Bound<'_, PyAny>) -> PyResult<Map<String, Value>> {
    let encoded = py_dict_to_json(row)?;
    let value: Value = serde_json::from_str(&encoded).map_err(to_py_err)?;
    match value {
        Value::Object(map) => Ok(map),
        other => Err(PyValueError::new_err(format!(
            "each row must be a mapping, got {other}"
        ))),
    }
}

/// Convert a decoded row back into a Python dict.
fn row_to_py(py: Python<'_>, row: &Map<String, Value>) -> PyResult<PyObject> {
    let json = py.import("json")?;
    let encoded = serde_json::to_string(row).map_err(to_py_err)?;
    Ok(json.call_method1("loads", (encoded,))?.unbind())
}

#[pymethods]
impl GenericStore {
    /// Open (or create) an embedded store at `uri` with the given schema.
    ///
    /// `schema` maps column name to type, e.g.
    /// `{"id": {"type": "string", "nullable": False}, "score": "float32"}`.
    /// An `id` column is required.
    #[classmethod]
    #[pyo3(signature = (uri, schema, storage_options = None, seal_on_add = false))]
    fn open(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        uri: &str,
        schema: &Bound<'_, PyDict>,
        storage_options: Option<HashMap<String, String>>,
        seal_on_add: bool,
    ) -> PyResult<Self> {
        let spec = schema_spec_from_py(schema)?;
        let runtime = Arc::new(Runtime::new().map_err(to_py_err)?);
        let store_res = py.allow_threads(|| {
            runtime.block_on(UnifiedGenericStore::open_with_options(
                uri,
                spec,
                storage_options,
                seal_on_add,
            ))
        });
        Ok(Self::from_store(store_res.map_err(to_py_err)?, runtime))
    }

    /// Open an existing embedded store, reading its schema from the dataset.
    #[classmethod]
    #[pyo3(signature = (uri, storage_options = None, seal_on_add = false))]
    fn open_existing(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        uri: &str,
        storage_options: Option<HashMap<String, String>>,
        seal_on_add: bool,
    ) -> PyResult<Self> {
        let runtime = Arc::new(Runtime::new().map_err(to_py_err)?);
        let store_res = py.allow_threads(|| {
            runtime.block_on(UnifiedGenericStore::open_existing(
                uri,
                storage_options,
                seal_on_add,
            ))
        });
        Ok(Self::from_store(store_res.map_err(to_py_err)?, runtime))
    }

    /// Connect to an existing store on a remote server.
    #[classmethod]
    fn connect(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        base_url: &str,
        name: &str,
    ) -> PyResult<Self> {
        let runtime = Arc::new(Runtime::new().map_err(to_py_err)?);
        let store_res =
            py.allow_threads(|| runtime.block_on(UnifiedGenericStore::connect(base_url, name)));
        Ok(Self::from_store(store_res.map_err(to_py_err)?, runtime))
    }

    /// Connect to a remote store, creating it with `schema` if absent.
    #[classmethod]
    #[pyo3(signature = (base_url, name, schema, storage_options = None, seal_on_add = false))]
    fn connect_or_create(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        base_url: &str,
        name: &str,
        schema: &Bound<'_, PyDict>,
        storage_options: Option<HashMap<String, String>>,
        seal_on_add: bool,
    ) -> PyResult<Self> {
        let spec = schema_spec_from_py(schema)?;
        let req = CreateGenericStoreRequest {
            name: name.to_string(),
            schema: spec,
            storage_options,
            seal_on_add,
        };
        let runtime = Arc::new(Runtime::new().map_err(to_py_err)?);
        let store_res = py.allow_threads(|| {
            runtime.block_on(UnifiedGenericStore::connect_or_create(base_url, &req))
        });
        Ok(Self::from_store(store_res.map_err(to_py_err)?, runtime))
    }

    /// The store's schema, as a dict keyed by column name.
    fn schema(&self, py: Python<'_>) -> PyResult<PyObject> {
        let spec = GenericStoreApi::spec(&self.store);
        let json = py.import("json")?;
        let encoded = serde_json::to_string(spec).map_err(to_py_err)?;
        Ok(json.call_method1("loads", (encoded,))?.unbind())
    }

    /// Base dataset version.
    fn version(&self) -> u64 {
        GenericStoreApi::version(&self.store)
    }

    /// Append rows. Each row is a dict keyed by column name; omitted nullable
    /// columns are written as null, and an undeclared column is an error.
    fn add(&self, py: Python<'_>, rows: &Bound<'_, PyAny>) -> PyResult<u64> {
        let mut parsed = Vec::new();
        for row in rows.try_iter()? {
            parsed.push(row_from_py(&row?)?);
        }
        let res = py.allow_threads(|| {
            self.runtime
                .block_on(GenericStoreApi::add(&self.store, &parsed))
        });
        Ok(res.map_err(to_py_err)?.version)
    }

    /// Read rows. Blob columns are excluded; use `get` with `columns` to fetch
    /// them one row at a time.
    #[pyo3(signature = (limit = None, offset = None, filter = None))]
    fn list(
        &self,
        py: Python<'_>,
        limit: Option<usize>,
        offset: Option<usize>,
        filter: Option<String>,
    ) -> PyResult<Vec<PyObject>> {
        let rows = py.allow_threads(|| {
            self.runtime.block_on(async {
                match &filter {
                    Some(filter) => {
                        GenericStoreApi::list_filtered(&self.store, filter, limit, offset).await
                    }
                    None => GenericStoreApi::list(&self.store, limit, offset).await,
                }
            })
        });
        rows.map_err(to_py_err)?
            .iter()
            .map(|row| row_to_py(py, row))
            .collect()
    }

    /// Fetch one row by id, or `None`. `columns` selects what to read; omit it
    /// to read everything except blob columns.
    #[pyo3(signature = (id, columns = None))]
    fn get(
        &self,
        py: Python<'_>,
        id: &str,
        columns: Option<Vec<String>>,
    ) -> PyResult<Option<PyObject>> {
        let row = py.allow_threads(|| {
            self.runtime
                .block_on(GenericStoreApi::get(&self.store, id, columns.as_deref()))
        });
        match row.map_err(to_py_err)? {
            Some(row) => Ok(Some(row_to_py(py, &row)?)),
            None => Ok(None),
        }
    }

    /// Seal the active memtable so added rows become readable. Unnecessary when
    /// the store was opened with `seal_on_add=True`.
    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.runtime.block_on(GenericStoreApi::flush(&self.store)))
            .map_err(to_py_err)
    }

    /// Merge pending WAL generations into the base table (embedded stores only).
    fn cleanup_wal(&mut self, py: Python<'_>) -> PyResult<usize> {
        py.allow_threads(|| self.runtime.block_on(self.store.cleanup_wal()))
            .map_err(to_py_err)
    }
}

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(generate_id, m)?)?;
    m.add_function(wrap_pyfunction!(datagen_event_id, m)?)?;
    m.add_class::<Context>()?;
    m.add_class::<ContextNamespace>()?;
    m.add_class::<RemoteContext>()?;
    m.add_class::<RolloutStore>()?;
    m.add_class::<DatagenStore>()?;
    m.add_class::<DatagenStreamWriter>()?;
    m.add_class::<DatagenItemId>()?;
    m.add_class::<GenericStore>()?;
    m.add("ContextStoreError", m.py().get_type::<ContextStoreError>())?;
    m.add("NotFoundError", m.py().get_type::<NotFoundError>())?;
    m.add(
        "AlreadyExistsError",
        m.py().get_type::<AlreadyExistsError>(),
    )?;
    m.add(
        "InvalidRequestError",
        m.py().get_type::<InvalidRequestError>(),
    )?;
    m.add("InternalError", m.py().get_type::<InternalError>())?;
    m.add(
        "CompactionInProgressError",
        m.py().get_type::<CompactionInProgressError>(),
    )?;
    Ok(())
}
