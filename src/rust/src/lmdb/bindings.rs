// SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the ASE LMDB reader using PyO3.
//!
//! Converts parsed JSON rows into Python dictionaries, transforming
//! ASE `__ndarray__` markers into actual NumPy arrays.

use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyList, PyString};

use super::reader::{read_lmdb_files_parallel, read_lmdb_rows};

/// Convert a `serde_json::Value` to a Python object.
///
/// Handles the ASE-specific `__ndarray__` marker by creating actual
/// NumPy arrays. Other JSON types map to their Python equivalents.
fn json_to_python(py: Python<'_>, value: &serde_json::Value) -> PyResult<Py<PyAny>> {
    match value {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok((*b).into_pyobject(py)?.to_owned().into_any().unbind()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.into_any().unbind())
            } else if let Some(f) = n.as_f64() {
                Ok(PyFloat::new(py, f).into_any().unbind())
            } else {
                let s = n.to_string();
                Ok(PyString::new(py, &s).into_any().unbind())
            }
        }
        serde_json::Value::String(s) => Ok(PyString::new(py, s).into_any().unbind()),
        serde_json::Value::Array(arr) => {
            let py_list = PyList::empty(py);
            for item in arr {
                py_list.append(json_to_python(py, item)?)?;
            }
            Ok(py_list.into_any().unbind())
        }
        serde_json::Value::Object(map) => {
            // Check for ASE special markers.
            if let Some(ndarray_val) = map.get("__ndarray__") {
                return convert_ndarray(py, ndarray_val);
            }
            if let Some(dt_val) = map.get("__datetime__") {
                if let Some(s) = dt_val.as_str() {
                    return Ok(PyString::new(py, s).into_any().unbind());
                }
            }
            if let Some(c_val) = map.get("__complex__") {
                if let Some(arr) = c_val.as_array() {
                    if arr.len() == 2 {
                        let real = arr[0].as_f64().unwrap_or(0.0);
                        let imag = arr[1].as_f64().unwrap_or(0.0);
                        let builtins = py.import("builtins")?;
                        let result = builtins.call_method1("complex", (real, imag))?;
                        return Ok(result.unbind());
                    }
                }
            }

            // Regular dict.
            let py_dict = PyDict::new(py);
            for (k, v) in map {
                py_dict.set_item(k, json_to_python(py, v)?)?;
            }
            Ok(py_dict.into_any().unbind())
        }
    }
}

/// Convert an ASE `__ndarray__` JSON marker to a NumPy array.
///
/// The marker format is: `[shape, dtype_str, flat_data]`
/// where `shape` is a list of ints, `dtype_str` is e.g. "float64",
/// and `flat_data` is a flat list of numbers.
fn convert_ndarray(py: Python<'_>, value: &serde_json::Value) -> PyResult<Py<PyAny>> {
    let arr = value.as_array().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("__ndarray__ value must be array")
    })?;

    if arr.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "__ndarray__ must have exactly 3 elements: [shape, dtype, data]",
        ));
    }

    let shape: Vec<usize> = arr[0]
        .as_array()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("shape must be array"))?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as usize)
        .collect();

    let dtype_str = arr[1]
        .as_str()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("dtype must be string"))?;

    let flat_data = arr[2]
        .as_array()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("data must be array"))?;

    // Build the shape tuple for numpy.reshape.
    let shape_tuple: Vec<i64> = shape.iter().map(|&s| s as i64).collect();
    let np_mod = py.import("numpy")?;

    // Create the appropriate NumPy array based on dtype.
    match dtype_str {
        "float64" | "float32" => {
            let data: Vec<f64> = flat_data
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0))
                .collect();
            let np_arr = PyArray1::from_vec(py, data);
            let reshaped = np_mod.call_method1("reshape", (&np_arr, shape_tuple))?;
            if dtype_str == "float32" {
                let result = reshaped.call_method1("astype", ("float32",))?;
                Ok(result.unbind())
            } else {
                Ok(reshaped.unbind())
            }
        }
        s if s.starts_with("int") || s.starts_with("uint") => {
            let data: Vec<i64> = flat_data.iter().map(|v| v.as_i64().unwrap_or(0)).collect();
            let np_arr = PyArray1::from_vec(py, data);
            let reshaped = np_mod.call_method1("reshape", (&np_arr, shape_tuple))?;
            let result = reshaped.call_method1("astype", (dtype_str,))?;
            Ok(result.unbind())
        }
        "bool" => {
            let data: Vec<bool> = flat_data
                .iter()
                .map(|v| v.as_bool().unwrap_or(false))
                .collect();
            let py_list = PyList::empty(py);
            for b in &data {
                py_list.append(*b)?;
            }
            let np_arr = np_mod.call_method1("array", (py_list,))?;
            let reshaped = np_mod.call_method1("reshape", (&np_arr, shape_tuple))?;
            Ok(reshaped.unbind())
        }
        _ => {
            // Fallback: use numpy to create array from list with dtype.
            let py_list = PyList::empty(py);
            for v in flat_data {
                py_list.append(json_to_python(py, v)?)?;
            }
            let np_arr = np_mod.call_method1("array", (py_list,))?;
            let typed = np_arr.call_method1("astype", (dtype_str,))?;
            let reshaped = np_mod.call_method1("reshape", (&typed, shape_tuple))?;
            Ok(reshaped.unbind())
        }
    }
}

/// Read all data rows from a single `.aselmdb` file.
///
/// Returns a list of dicts, one per row, sorted by row ID. Each dict
/// contains the parsed ASE row data with `__ndarray__` markers
/// converted to actual NumPy arrays. A synthetic ``"id"`` key is
/// added to each dict.
///
/// Args:
///     path: Path to the `.aselmdb` file
///
/// Returns:
///     List of row dicts with NumPy arrays
#[pyfunction]
#[pyo3(name = "read_lmdb")]
pub fn py_read_lmdb(py: Python<'_>, path: &str) -> PyResult<Py<PyList>> {
    let rows = read_lmdb_rows(path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    let result = PyList::empty(py);
    for row in &rows {
        let py_dict = json_to_python(py, &row.data)?;
        let dict_ref: &Bound<'_, PyDict> = py_dict.bind(py).cast()?;
        dict_ref.set_item("id", row.id)?;
        result.append(py_dict)?;
    }
    Ok(result.unbind())
}

/// Read rows from multiple `.aselmdb` files in parallel.
///
/// Each file is read on a separate Rayon thread. The heavy I/O,
/// decompression, and JSON parsing all run outside the GIL.
/// Returns a list of lists, one inner list per input file path.
///
/// Args:
///     paths: List of paths to `.aselmdb` files
///
/// Returns:
///     List of lists of row dicts
#[pyfunction]
#[pyo3(name = "read_lmdb_parallel")]
pub fn py_read_lmdb_parallel(py: Python<'_>, paths: Vec<String>) -> PyResult<Py<PyList>> {
    // Rayon work happens outside the GIL automatically (no Python
    // references are held during parallel I/O + JSON parsing).
    let all_results = read_lmdb_files_parallel(&paths);

    let outer = PyList::empty(py);
    for result in all_results {
        let rows =
            result.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let inner = PyList::empty(py);
        for row in &rows {
            let py_dict = json_to_python(py, &row.data)?;
            let dict_ref: &Bound<'_, PyDict> = py_dict.bind(py).cast()?;
            dict_ref.set_item("id", row.id)?;
            inner.append(py_dict)?;
        }
        outer.append(inner)?;
    }
    Ok(outer.unbind())
}

/// Register the LMDB submodule.
pub fn register_lmdb_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let lmdb = PyModule::new(py, "lmdb")?;
    lmdb.add_function(wrap_pyfunction!(py_read_lmdb, &lmdb)?)?;
    lmdb.add_function(wrap_pyfunction!(py_read_lmdb_parallel, &lmdb)?)?;
    parent.add_submodule(&lmdb)?;

    // Register in sys.modules so `from _lib.lmdb import ...` works.
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("physicsnemo_curator._lib.lmdb", &lmdb)?;

    Ok(())
}
