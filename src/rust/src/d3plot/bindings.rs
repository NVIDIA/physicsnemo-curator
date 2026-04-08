// SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Python bindings for d3plot post-processing utilities.
//!
//! Exposes three functions:
//!
//! - `parse_k_file(path) → dict[int, float]`
//! - `compute_node_thickness(connectivity, part_ids, part_thickness, actual_part_ids) → NDArray`
//! - `von_mises_from_voigt(stress, n_total) → NDArray`

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::path::Path;

use super::kfile;
use super::stress;
use super::thickness;

/// Parse an LS-DYNA `.k` keyword file for part thickness.
///
/// Args:
///     path: Path to the `.k` file.
///
/// Returns:
///     Dictionary mapping part ID (int) to thickness (float).
#[pyfunction]
#[pyo3(name = "parse_k_file")]
pub fn py_parse_k_file(py: Python<'_>, path: String) -> PyResult<Py<PyDict>> {
    let result = py
        .detach(|| kfile::parse_k_file(Path::new(&path)))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    let dict = PyDict::new(py);
    for (pid, thickness) in &result {
        dict.set_item(*pid, *thickness)?;
    }
    Ok(dict.unbind())
}

/// Compute per-node thickness from element connectivity and part data.
///
/// Args:
///     connectivity: Element connectivity array, shape (E, nodes_per_cell), int64.
///     part_ids: Part index per element, shape (E,), int64.
///     part_thickness: Dictionary mapping actual part ID to thickness.
///     actual_part_ids: Optional array of actual part IDs for index→ID translation.
///
/// Returns:
///     Per-node thickness array, shape (max_node+1,), float64.
#[pyfunction]
#[pyo3(name = "compute_node_thickness")]
pub fn py_compute_node_thickness<'py>(
    py: Python<'py>,
    connectivity: PyReadonlyArray2<'py, i64>,
    part_ids: PyReadonlyArray1<'py, i64>,
    part_thickness: &Bound<'py, PyDict>,
    actual_part_ids: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    // Convert Python dict to HashMap.
    let mut pt_map: HashMap<i64, f64> = HashMap::new();
    for (key, value) in part_thickness.iter() {
        let k: i64 = key.extract()?;
        let v: f64 = value.extract()?;
        pt_map.insert(k, v);
    }

    let conn_shape = connectivity.shape();
    let n_elements = conn_shape[0];
    let nodes_per_cell = conn_shape[1];

    // Get contiguous slices.
    let conn_array = connectivity.as_array();
    let conn_slice: Vec<i64> = conn_array.iter().copied().collect();
    let pids_slice: Vec<i64> = part_ids.as_array().iter().copied().collect();

    let actual_pids_vec: Option<Vec<i64>> = actual_part_ids
        .as_ref()
        .map(|a| a.as_array().iter().copied().collect());

    // Run computation (GIL released).
    let result = py.detach(|| {
        thickness::compute_node_thickness(
            &conn_slice,
            n_elements,
            nodes_per_cell,
            &pids_slice,
            &pt_map,
            actual_pids_vec.as_deref(),
        )
    });

    Ok(PyArray1::from_vec(py, result))
}

/// Compute von Mises stress from Voigt-notation stress tensor.
///
/// Args:
///     stress: Flattened stress array with last dimension 6, float64.
///         Shape is (..., 6) flattened to 1-D.
///     n_total: Total number of stress entries (product of leading dims).
///
/// Returns:
///     Von Mises stress array, shape (n_total,), float64.
#[pyfunction]
#[pyo3(name = "von_mises_from_voigt")]
pub fn py_von_mises_from_voigt<'py>(
    py: Python<'py>,
    stress_arr: PyReadonlyArray1<'py, f64>,
    n_total: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let stress_slice: Vec<f64> = stress_arr.as_array().iter().copied().collect();

    if stress_slice.len() != n_total * 6 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "stress array length {} does not match n_total * 6 = {}",
            stress_slice.len(),
            n_total * 6
        )));
    }

    let result = py.detach(|| stress::von_mises_from_voigt(&stress_slice, n_total));

    Ok(PyArray1::from_vec(py, result))
}

/// Register the d3plot submodule.
pub fn register_d3plot_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let d3plot = PyModule::new(py, "d3plot")?;
    d3plot.add_function(wrap_pyfunction!(py_parse_k_file, &d3plot)?)?;
    d3plot.add_function(wrap_pyfunction!(py_compute_node_thickness, &d3plot)?)?;
    d3plot.add_function(wrap_pyfunction!(py_von_mises_from_voigt, &d3plot)?)?;
    parent.add_submodule(&d3plot)?;

    // Register in sys.modules so `from _lib.d3plot import ...` works.
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("physicsnemo_curator._lib.d3plot", &d3plot)?;

    Ok(())
}
