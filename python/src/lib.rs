use std::sync::Arc;

use chrono::Utc;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use tokio::runtime::Runtime;

use lance_context::serde::CONTENT_TYPE_TEXT;
use lance_context::{Context as RustContext, ContextRecord, ContextStore};

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

#[pymethods]
impl Context {
    #[classmethod]
    fn create(_cls: &Bound<'_, PyType>, uri: &str) -> PyResult<Self> {
        let runtime = Arc::new(Runtime::new().map_err(to_py_err)?);
        let store = runtime
            .block_on(ContextStore::open(uri))
            .map_err(to_py_err)?;
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

    #[pyo3(signature = (role, content, data_type = None))]
    fn add(
        &mut self,
        role: &str,
        content: &Bound<'_, PyAny>,
        data_type: Option<&str>,
    ) -> PyResult<()> {
        let (content_type, text_payload, binary_payload, inner_content) =
            match content.extract::<&[u8]>() {
                Ok(bytes) => (
                    data_type
                        .unwrap_or(DEFAULT_BINARY_CONTENT_TYPE)
                        .to_string(),
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
            created_at: Utc::now(),
            role: role.to_string(),
            state_metadata: None,
            content_type,
            text_payload,
            binary_payload,
            embedding: None,
        };

        self.runtime
            .block_on(self.store.add(std::slice::from_ref(&record)))
            .map_err(to_py_err)?;
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

    fn checkout(&mut self, version_id: u64) -> PyResult<()> {
        self.runtime
            .block_on(self.store.checkout(version_id))
            .map_err(to_py_err)?;
        self.run_id = new_run_id();
        Ok(())
    }
}

fn new_run_id() -> String {
    format!("run-{}-{}", Utc::now().timestamp_micros(), std::process::id())
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
