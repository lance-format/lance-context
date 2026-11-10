//! Python bindings for the embedded automatic document store.

use super::{storage_options_from_dict, to_py_err};
use lance_context_core::{shredding::ShredPolicy, DocumentStore as CoreDocumentStore};
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use serde::Serialize;
use serde_json::Value;
use tokio::runtime::Runtime;

#[pyclass]
pub struct DocumentStore {
    store: CoreDocumentStore,
    runtime: Runtime,
}

fn json_to_py(py: Python<'_>, value: &impl Serialize) -> PyResult<PyObject> {
    let encoded = serde_json::to_string(value).map_err(to_py_err)?;
    Ok(py
        .import("json")?
        .call_method1("loads", (encoded,))?
        .unbind())
}

// Large text should cross into Python once, without JSON escaping and reparsing.
fn document_to_py(py: Python<'_>, value: &Value) -> PyResult<PyObject> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(value) => Ok(PyBool::new(py, *value).to_owned().into_any().unbind()),
        Value::String(value) => Ok(PyString::new(py, value).into_any().unbind()),
        Value::Number(number) => {
            if let Some(number) = number.as_i64() {
                return Ok(number.into_pyobject(py)?.into_any().unbind());
            }
            if let Some(number) = number.as_u64() {
                return Ok(number.into_pyobject(py)?.into_any().unbind());
            }
            if let Some(float) = number
                .as_f64()
                .filter(|float| serde_json::Number::from_f64(*float).as_ref() == Some(number))
            {
                return Ok(PyFloat::new(py, float).into_any().unbind());
            }
            // Preserve json.loads semantics for residual arbitrary-precision
            // numbers written with additional serde_json features.
            json_to_py(py, value)
        }
        Value::Array(values) => {
            let values = values
                .iter()
                .map(|value| document_to_py(py, value))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyList::new(py, values)?.into_any().unbind())
        }
        Value::Object(values) => {
            let object = PyDict::new(py);
            for (key, value) in values {
                object.set_item(key, document_to_py(py, value)?)?;
            }
            Ok(object.into_any().unbind())
        }
    }
}

// Keep Python integers exact and reject non-JSON values before mutating storage.
// The generic dumps/parser helper can round integers outside the JSON codec's
// 64-bit range, so document inputs use a strict conversion instead.
fn py_any_to_json(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    convert_json(value, 0)
}

fn convert_json(value: &Bound<'_, PyAny>, depth: usize) -> PyResult<Value> {
    if value.is_none() {
        return Ok(Value::Null);
    }
    if value.is_instance_of::<PyBool>() {
        return Ok(Value::Bool(value.extract()?));
    }
    if value.is_instance_of::<PyInt>() {
        return value
            .extract::<i64>()
            .map(Value::from)
            .or_else(|_| value.extract::<u64>().map(Value::from))
            .map_err(|_| {
                PyValueError::new_err("JSON integers must fit signed or unsigned 64 bits")
            });
    }
    if value.is_instance_of::<PyFloat>() {
        return serde_json::Number::from_f64(value.extract()?)
            .map(Value::Number)
            .ok_or_else(|| PyValueError::new_err("JSON numbers must be finite"));
    }
    if value.is_instance_of::<PyString>() {
        return Ok(Value::String(value.extract()?));
    }
    if depth >= 127 {
        return Err(PyValueError::new_err(
            "document exceeds 127 JSON nesting levels",
        ));
    }
    if let Ok(object) = value.downcast::<PyDict>() {
        let mut result = serde_json::Map::new();
        for (key, child) in object.iter() {
            if !key.is_instance_of::<PyString>() {
                return Err(PyTypeError::new_err("JSON object keys must be strings"));
            }
            result.insert(key.extract()?, convert_json(&child, depth + 1)?);
        }
        return Ok(Value::Object(result));
    }
    if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
        return value
            .try_iter()?
            .map(|child| convert_json(&child?, depth + 1))
            .collect::<PyResult<Vec<_>>>()
            .map(Value::Array);
    }
    Err(PyTypeError::new_err("expected a JSON value"))
}

#[pymethods]
impl DocumentStore {
    #[staticmethod]
    #[pyo3(signature = (uri, policy=None, storage_options=None))]
    fn create(
        py: Python<'_>,
        uri: &str,
        policy: Option<&Bound<'_, PyDict>>,
        storage_options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let mut configured = serde_json::to_value(ShredPolicy::default()).map_err(to_py_err)?;
        if let Some(policy) = policy {
            let Value::Object(options) = py_any_to_json(policy.as_any())? else {
                unreachable!()
            };
            configured.as_object_mut().unwrap().extend(options);
        }
        let policy: ShredPolicy = serde_json::from_value(configured)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        policy.validate().map_err(PyValueError::new_err)?;
        let options = storage_options_from_dict(storage_options)?;
        let runtime = Runtime::new().map_err(to_py_err)?;
        let store = py
            .allow_threads(|| runtime.block_on(CoreDocumentStore::create(uri, policy, options)))
            .map_err(to_py_err)?;
        Ok(Self { store, runtime })
    }

    #[staticmethod]
    #[pyo3(signature = (uri, storage_options=None))]
    fn open(
        py: Python<'_>,
        uri: &str,
        storage_options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let options = storage_options_from_dict(storage_options)?;
        let runtime = Runtime::new().map_err(to_py_err)?;
        let store = py
            .allow_threads(|| runtime.block_on(CoreDocumentStore::open(uri, options)))
            .map_err(to_py_err)?;
        Ok(Self { store, runtime })
    }

    fn add(&mut self, py: Python<'_>, documents: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
        let documents = documents
            .try_iter()?
            .map(|document| py_any_to_json(&document?))
            .collect::<PyResult<Vec<_>>>()?;
        py.allow_threads(|| self.runtime.block_on(self.store.add(documents)))
            .map_err(to_py_err)
    }

    fn get(&self, py: Python<'_>, id: &str) -> PyResult<PyObject> {
        let value = py
            .allow_threads(|| self.runtime.block_on(self.store.get(id)))
            .map_err(to_py_err)?
            .ok_or_else(|| PyKeyError::new_err(id.to_owned()))?;
        document_to_py(py, &value)
    }

    fn get_path(&self, py: Python<'_>, id: &str, path: Vec<String>) -> PyResult<PyObject> {
        let value = py
            .allow_threads(|| self.runtime.block_on(self.store.get_path(id, &path)))
            .map_err(to_py_err)?
            .ok_or_else(|| {
                PyKeyError::new_err(format!("missing document or path: {id}, {path:?}"))
            })?;
        document_to_py(py, &value)
    }

    #[pyo3(signature = (path, value, limit=100))]
    fn filter_eq(
        &self,
        py: Python<'_>,
        path: Vec<String>,
        value: &Bound<'_, PyAny>,
        limit: usize,
    ) -> PyResult<Vec<String>> {
        let value = py_any_to_json(value)?;
        py.allow_threads(|| {
            self.runtime
                .block_on(self.store.filter_eq(&path, &value, limit))
        })
        .map_err(to_py_err)
    }

    #[pyo3(signature = (query, limit=10))]
    fn search(&mut self, py: Python<'_>, query: &str, limit: usize) -> PyResult<PyObject> {
        let results = py
            .allow_threads(|| self.runtime.block_on(self.store.search(query, limit)))
            .map_err(to_py_err)?;
        json_to_py(py, &results)
    }

    fn prepare_text_search(&mut self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.runtime.block_on(self.store.prepare_text_search()))
            .map_err(to_py_err)
    }

    fn layout(&self, py: Python<'_>) -> PyResult<PyObject> {
        json_to_py(py, self.store.layout())
    }
    fn version(&self) -> u64 {
        self.store.version()
    }
    fn checkout(&mut self, py: Python<'_>, version: u64) -> PyResult<()> {
        py.allow_threads(|| self.runtime.block_on(self.store.checkout(version)))
            .map_err(to_py_err)
    }
    fn refresh(&mut self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.runtime.block_on(self.store.refresh()))
            .map_err(to_py_err)
    }
}
