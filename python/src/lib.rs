use pyo3::prelude::*;
use pyo3::types::PyType;

use lance_context::Context as RustContext;

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyclass]
struct Context {
    inner: RustContext,
}

#[pymethods]
impl Context {
    #[classmethod]
    fn create(_cls: &Bound<'_, PyType>, uri: &str) -> Self {
        Self {
            inner: RustContext::new(uri),
        }
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

    #[pyo3(signature = (role, content, data_type = None))]
    fn add(
        &mut self,
        role: &str,
        content: &Bound<'_, PyAny>,
        data_type: Option<&str>,
    ) -> PyResult<()> {
        let content_str = content.str()?.to_string();
        self.inner.add(role, &content_str, data_type);
        Ok(())
    }

    #[pyo3(signature = (label = None))]
    fn snapshot(&mut self, label: Option<&str>) -> String {
        self.inner.snapshot(label)
    }

    fn fork(&self, branch_name: &str) -> Self {
        Self {
            inner: self.inner.fork(branch_name),
        }
    }

    fn checkout(&mut self, snapshot_id: &str) {
        self.inner.checkout(snapshot_id);
    }
}

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_class::<Context>()?;
    Ok(())
}
