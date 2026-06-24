#![recursion_limit = "256"]

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use chrono::{DateTime, SecondsFormat, Utc};
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyModule, PyType};
use pyo3::IntoPyObject;
use serde_json::Value;
use tokio::runtime::Runtime;

use lance_context_api::{
    AddRecordRequest, CompactRequest, CompactResponse, CompactStatsResponse, ContextStoreApi,
    RecordDto, RecordPatchDto, RelationshipDto, RetrieveRequest, RetrieveResultDto, SearchRequest,
    SearchResultDto, StateMetadataDto, UpdateRecordRequest, UpsertRecordRequest,
};
use lance_context_client::RemoteContextStore;
use lance_context_core::serde::CONTENT_TYPE_TEXT;
use lance_context_core::{
    CompactionConfig, CompactionMetrics, CompactionStats, Context as RustContext,
    ContextNamespace as RustContextNamespace, ContextRecord, ContextStore, ContextStoreOptions,
    DistanceMetric, IdIndexType, LifecycleQueryOptions, PartitionInfo, PartitionSelector,
    PartitionSpec, RecordFilters, RecordPatch, Relationship, RetrieveResult, SearchResult,
    StateMetadata, LIFECYCLE_ACTIVE,
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
    #[pyo3(signature = (role, content, data_type = None, embedding = None, bot_id = None, session_id = None, tenant = None, source = None, external_id = None, run_id = None, created_at = None, state_metadata = None, metadata_json = None, expires_at = None, retention_policy = None, lifecycle_status = None, retired_at = None, retired_reason = None, supersedes_id = None, superseded_by_id = None, relationships_json = None))]
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
    #[pyo3(signature = (role, content, data_type = None, embedding = None, bot_id = None, session_id = None, tenant = None, source = None, external_id = None, run_id = None, created_at = None, state_metadata = None, metadata_json = None, expires_at = None, retention_policy = None, lifecycle_status = None, retired_at = None, retired_reason = None, relationships_json = None, key = "external_id"))]
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
    #[pyo3(signature = (id = None, external_id = None, bot_id = None, session_id = None, tenant = None, source = None, metadata_json = None, relationships_json = None, expires_at = None, retention_policy = None, lifecycle_status = None, retired_at = None, retired_reason = None, embedding = None))]
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

    fn fork(&self, branch_name: &str) -> Self {
        Self {
            inner: self.inner.fork(branch_name),
            store: self.store.clone(),
            runtime: Arc::clone(&self.runtime),
            run_id: new_run_id(),
        }
    }

    fn checkout(&mut self, py: Python<'_>, version_id: u64) -> PyResult<()> {
        let res = py.allow_threads(|| self.runtime.block_on(self.store.checkout(version_id)));
        res.map_err(to_py_err)?;
        self.run_id = new_run_id();
        Ok(())
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
        let filters = filters_from_json(filters_json)?;
        let options = LifecycleQueryOptions::new(include_expired, include_retired);
        let hits_res = py.allow_threads(|| {
            self.runtime
                .block_on(self.store.search_filtered_with_options(
                    &query,
                    limit,
                    filters.as_ref(),
                    options,
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
        let filters = filters_from_json(filters_json)?;
        let options = LifecycleQueryOptions::new(include_expired, include_retired);
        // Release GIL during data retrieval
        let records = py.allow_threads(|| {
            self.runtime
                .block_on(self.store.list_filtered_with_options(
                    limit,
                    offset,
                    filters.as_ref(),
                    options,
                ))
                .map_err(to_py_err)
        })?;

        records
            .into_iter()
            .map(|record| record_to_py(py, record))
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
    #[pyo3(signature = (role, content, data_type = None, embedding = None, bot_id = None, session_id = None, external_id = None, state_metadata = None, metadata_json = None, expires_at = None, retention_policy = None, supersedes_id = None, relationships_json = None))]
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
    ) -> PyResult<PyObject> {
        let (content_type, text_payload, binary_payload) = content_to_payloads(content, data_type)?;
        let req = AddRecordRequest {
            role: role.to_string(),
            content_type,
            text_payload,
            binary_payload,
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
    #[pyo3(signature = (role, content, data_type = None, embedding = None, bot_id = None, session_id = None, external_id = None, metadata_json = None, expires_at = None, retention_policy = None, supersedes_id = None, relationships_json = None, key = "external_id"))]
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
    #[pyo3(signature = (id = None, external_id = None, bot_id = None, session_id = None, metadata_json = None, relationships_json = None, expires_at = None, retention_policy = None, lifecycle_status = None, retired_at = None, retired_reason = None, embedding = None))]
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

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_class::<Context>()?;
    m.add_class::<ContextNamespace>()?;
    m.add_class::<RemoteContext>()?;
    Ok(())
}
