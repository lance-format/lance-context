#![recursion_limit = "256"]

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use chrono::{DateTime, SecondsFormat, Utc};
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyModule, PyType};
use pyo3::IntoPyObject;
use serde_json::Value;
use tokio::runtime::Runtime;

use lance_context::serde::CONTENT_TYPE_TEXT;
use lance_context::{
    CompactionConfig, CompactionMetrics, CompactionStats, Context as RustContext, ContextRecord,
    ContextStore, ContextStoreOptions, IdIndexType, LifecycleQueryOptions, MetadataFilter,
    RecordFilters, SearchResult, LIFECYCLE_ACTIVE,
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
    external_id: Option<String>,
    metadata_json: Option<String>,
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

fn metadata_from_json(metadata_json: Option<String>) -> PyResult<Option<Value>> {
    metadata_json
        .map(|value| serde_json::from_str(&value).map_err(to_py_err))
        .transpose()
}

fn filters_from_json(filters_json: Option<String>) -> PyResult<Option<RecordFilters>> {
    let Some(filters_json) = filters_json else {
        return Ok(None);
    };
    let value: Value = serde_json::from_str(&filters_json).map_err(to_py_err)?;
    let Value::Object(object) = value else {
        return Err(PyRuntimeError::new_err("filters must be a JSON object"));
    };

    let mut filters = RecordFilters::default();
    for (key, value) in object {
        match key.as_str() {
            "bot_id" => filters.bot_id = filter_string(key.as_str(), value)?,
            "session_id" => filters.session_id = filter_string(key.as_str(), value)?,
            "role" => filters.role = filter_string(key.as_str(), value)?,
            "content_type" => filters.content_type = filter_string(key.as_str(), value)?,
            "created_at" => apply_created_at_filter(&mut filters, value)?,
            "created_at_start" | "created_after" | "created_at_gte" => {
                filters.created_at_start = Some(parse_filter_datetime(&key, &value)?);
            }
            "created_at_end" | "created_before" | "created_at_lte" => {
                filters.created_at_end = Some(parse_filter_datetime(&key, &value)?);
            }
            _ => {
                let filter = match value {
                    Value::Object(mut object)
                        if object.len() == 1 && object.contains_key("contains") =>
                    {
                        MetadataFilter::Contains(object.remove("contains").unwrap())
                    }
                    value => MetadataFilter::Equals(value),
                };
                filters.metadata.insert(key, filter);
            }
        }
    }

    Ok(Some(filters))
}

fn filter_string(name: &str, value: Value) -> PyResult<Option<String>> {
    match value {
        Value::Null => Ok(None),
        Value::String(value) => Ok(Some(value)),
        _ => Err(PyRuntimeError::new_err(format!(
            "filter '{name}' must be a string or null"
        ))),
    }
}

fn apply_created_at_filter(filters: &mut RecordFilters, value: Value) -> PyResult<()> {
    let Value::Object(object) = value else {
        return Err(PyRuntimeError::new_err(
            "filter 'created_at' must be an object with gte/lte bounds",
        ));
    };

    for (key, value) in object {
        match key.as_str() {
            "gte" | "start" | "after" => {
                filters.created_at_start = Some(parse_filter_datetime(&key, &value)?);
            }
            "lte" | "end" | "before" => {
                filters.created_at_end = Some(parse_filter_datetime(&key, &value)?);
            }
            other => {
                return Err(PyRuntimeError::new_err(format!(
                    "unsupported created_at filter operator '{other}'"
                )));
            }
        }
    }

    Ok(())
}

fn parse_filter_datetime(name: &str, value: &Value) -> PyResult<DateTime<Utc>> {
    let Some(value) = value.as_str() else {
        return Err(PyRuntimeError::new_err(format!(
            "filter '{name}' must be an ISO-8601 timestamp string"
        )));
    };
    DateTime::parse_from_rfc3339(value)
        .map(|value| value.with_timezone(&Utc))
        .map_err(to_py_err)
}

#[pymethods]
impl Context {
    #[classmethod]
    #[pyo3(signature = (uri, *, storage_options=None, compaction_config=None, blob_columns=None, id_index_type=None))]
    fn create(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        uri: &str,
        storage_options: Option<&Bound<'_, PyDict>>,
        compaction_config: Option<&Bound<'_, PyDict>>,
        blob_columns: Option<Vec<String>>,
        id_index_type: Option<String>,
    ) -> PyResult<Self> {
        let runtime = Arc::new(Runtime::new().map_err(to_py_err)?);

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

        let options = ContextStoreOptions {
            storage_options: storage_options_from_dict(storage_options)?,
            compaction: compaction_config_from_dict(compaction_config)?,
            blob_columns: blob_set,
            id_index_type: id_idx,
        };

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
    #[pyo3(signature = (role, content, data_type = None, embedding = None, bot_id = None, session_id = None, external_id = None, metadata_json = None, expires_at = None, retention_policy = None, lifecycle_status = None, retired_at = None, retired_reason = None, supersedes_id = None, superseded_by_id = None))]
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
        metadata_json: Option<String>,
        expires_at: Option<String>,
        retention_policy: Option<String>,
        lifecycle_status: Option<String>,
        retired_at: Option<String>,
        retired_reason: Option<String>,
        supersedes_id: Option<String>,
        superseded_by_id: Option<String>,
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
                external_id,
                metadata_json,
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

    #[pyo3(signature = (query, limit = None, filters_json = None, include_expired = false, include_retired = false))]
    fn search(
        &self,
        py: Python<'_>,
        query: Vec<f32>,
        limit: Option<usize>,
        filters_json: Option<String>,
        include_expired: bool,
        include_retired: bool,
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
            .map(|hit| search_hit_to_py(py, hit))
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
        let external_id =
            optional_item(dict, "external_id")?.map(|value| value.extract::<String>());
        let metadata_json =
            optional_item(dict, "metadata_json")?.map(|value| value.extract::<String>());
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
                external_id: external_id.transpose()?,
                metadata_json: metadata_json.transpose()?,
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
            external_id,
            metadata_json,
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

        let record_id = format!("{}-{}", self.run_id, self.inner.entries() + offset);
        let metadata = metadata_from_json(metadata_json)?;
        Ok(PreparedRecord {
            record: ContextRecord {
                id: record_id,
                external_id,
                run_id: self.run_id.clone(),
                bot_id,
                session_id,
                created_at: Utc::now(),
                role: role.clone(),
                state_metadata: None,
                metadata,
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

fn search_hit_to_py(py: Python<'_>, hit: SearchResult) -> PyResult<PyObject> {
    let SearchResult { record, distance } = hit;
    let dict = record_to_py(py, record)?;
    let dict_ref = dict.downcast_bound::<PyDict>(py)?;
    dict_ref.set_item("distance", distance)?;
    Ok(dict)
}

fn record_to_py(py: Python<'_>, record: ContextRecord) -> PyResult<PyObject> {
    let ContextRecord {
        id,
        external_id,
        run_id,
        bot_id,
        session_id,
        created_at,
        role,
        state_metadata,
        metadata,
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

fn json_value_to_py(py: Python<'_>, value: &Value) -> PyResult<PyObject> {
    let json = PyModule::import(py, "json")?;
    Ok(json.call_method1("loads", (value.to_string(),))?.unbind())
}

fn to_py_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_class::<Context>()?;
    Ok(())
}
