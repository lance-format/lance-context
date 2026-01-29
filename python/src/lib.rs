use std::collections::HashMap;
use std::sync::Arc;

use chrono::{SecondsFormat, Utc};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyType};
use pyo3::IntoPyObject;
use tokio::runtime::Runtime;

use lance_context::serde::CONTENT_TYPE_TEXT;
use lance_context::{
    CompactionConfig, CompactionMetrics, CompactionStats, Context as RustContext, ContextRecord,
    ContextStore, ContextStoreOptions, SearchResult,
};

const DEFAULT_BINARY_CONTENT_TYPE: &str = "application/octet-stream";
const BINARY_PLACEHOLDER: &str = "[binary]";

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

#[pymethods]
impl Context {
    #[classmethod]
    #[pyo3(signature = (uri, *, storage_options=None, compaction_config=None))]
    fn create(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        uri: &str,
        storage_options: Option<&Bound<'_, PyDict>>,
        compaction_config: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let runtime = Arc::new(Runtime::new().map_err(to_py_err)?);

        let options = ContextStoreOptions {
            storage_options: storage_options_from_dict(storage_options)?,
            compaction: compaction_config_from_dict(compaction_config)?,
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
    #[pyo3(signature = (role, content, data_type = None, embedding = None, bot_id = None, session_id = None))]
    fn add(
        &mut self,
        py: Python<'_>,
        role: &str,
        content: &Bound<'_, PyAny>,
        data_type: Option<&str>,
        embedding: Option<Vec<f32>>,
        bot_id: Option<String>,
        session_id: Option<String>,
    ) -> PyResult<()> {
        let (content_type, text_payload, binary_payload, inner_content) =
            match content.extract::<&[u8]>() {
                Ok(bytes) => (
                    data_type.unwrap_or(DEFAULT_BINARY_CONTENT_TYPE).to_string(),
                    None,
                    Some(bytes.to_vec()),
                    BINARY_PLACEHOLDER.to_string(),
                ),
                Err(_) => {
                    let content_str = content.str()?.to_string();
                    (
                        data_type.unwrap_or(CONTENT_TYPE_TEXT).to_string(),
                        Some(content_str.clone()),
                        None,
                        content_str,
                    )
                }
            };

        let record_id = format!("{}-{}", self.run_id, self.inner.entries() + 1);
        let record = ContextRecord {
            id: record_id,
            run_id: self.run_id.clone(),
            bot_id,
            session_id,
            created_at: Utc::now(),
            role: role.to_string(),
            state_metadata: None,
            content_type,
            text_payload,
            binary_payload,
            embedding,
        };

        let add_res = py.allow_threads(|| {
            self.runtime
                .block_on(self.store.add(std::slice::from_ref(&record)))
        });
        add_res.map_err(to_py_err)?;
        self.inner.add(role, &inner_content, data_type);
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

    #[pyo3(signature = (query, limit = None))]
    fn search(
        &self,
        py: Python<'_>,
        query: Vec<f32>,
        limit: Option<usize>,
    ) -> PyResult<Vec<PyObject>> {
        let hits_res = py.allow_threads(|| self.runtime.block_on(self.store.search(&query, limit)));
        let hits = hits_res.map_err(to_py_err)?;
        hits.into_iter()
            .map(|hit| search_hit_to_py(py, hit))
            .collect()
    }

    #[pyo3(signature = (limit = None, offset = None))]
    fn list(
        &self,
        py: Python<'_>,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> PyResult<Vec<PyObject>> {
        // Release GIL during data retrieval
        let records = py.allow_threads(|| {
            self.runtime
                .block_on(self.store.list(limit, offset))
                .map_err(to_py_err)
        })?;

        records
            .into_iter()
            .map(|record| record_to_py(py, record))
            .collect()
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
        run_id,
        bot_id,
        session_id,
        created_at,
        role,
        state_metadata,
        content_type,
        text_payload,
        binary_payload,
        embedding,
    } = record;

    let dict = PyDict::new(py);
    dict.set_item("id", id)?;
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
    dict.set_item("content_type", content_type)?;
    dict.set_item("text_payload", text_payload)?;
    match binary_payload {
        Some(payload) => dict.set_item("binary_payload", PyBytes::new(py, &payload))?,
        None => dict.set_item("binary_payload", py.None())?,
    }
    dict.set_item("embedding", embedding)?;
    Ok(dict.into_pyobject(py)?.unbind().into())
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
