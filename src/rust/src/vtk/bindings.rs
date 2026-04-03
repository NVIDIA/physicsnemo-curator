// SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for VTK reader using PyO3.

use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use super::mesh::{DataArray, VTKMesh};
use super::reader::{read_vtk_file, read_vtk_files_parallel};

/// Python-exposed VTK mesh class.
#[pyclass(name = "VTKMesh")]
pub struct PyVTKMesh {
    inner: VTKMesh,
}

#[pymethods]
impl PyVTKMesh {
    /// Number of points in the mesh.
    #[getter]
    fn n_points(&self) -> usize {
        self.inner.n_points
    }

    /// Number of cells in the mesh.
    #[getter]
    fn n_cells(&self) -> usize {
        self.inner.n_cells
    }

    /// File format (e.g., "vtu", "vtp").
    #[getter]
    fn format(&self) -> &str {
        &self.inner.format
    }

    /// Point coordinates as a NumPy array of shape (n_points * 3,).
    fn points<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.points)
    }

    /// Cell connectivity as a NumPy array.
    fn connectivity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_slice(py, &self.inner.connectivity)
    }

    /// Cell offsets as a NumPy array.
    fn offsets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_slice(py, &self.inner.offsets)
    }

    /// Cell types as a NumPy array.
    fn types<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        PyArray1::from_slice(py, &self.inner.types)
    }

    /// Point data arrays as a dict of {name: (data, num_components)}.
    fn point_data<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        data_arrays_to_dict(py, &self.inner.point_data)
    }

    /// Cell data arrays as a dict of {name: (data, num_components)}.
    fn cell_data<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        data_arrays_to_dict(py, &self.inner.cell_data)
    }

    fn __repr__(&self) -> String {
        format!(
            "VTKMesh(n_points={}, n_cells={}, format='{}')",
            self.inner.n_points, self.inner.n_cells, self.inner.format
        )
    }
}

/// Convert a HashMap of DataArrays to a Python dict.
fn data_arrays_to_dict<'py>(
    py: Python<'py>,
    arrays: &HashMap<String, DataArray>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (name, arr) in arrays {
        let data = PyArray1::from_slice(py, &arr.data);
        let tuple = (data, arr.num_components);
        dict.set_item(name, tuple)?;
    }
    Ok(dict)
}

/// Read a single VTK file from disk.
///
/// Args:
///     path: Path to the VTK file (.vtu, .vtp, .vtk, .vts, .vtm)
///
/// Returns:
///     VTKMesh object with mesh data accessible as NumPy arrays
#[pyfunction]
#[pyo3(name = "read_vtk")]
pub fn py_read_vtk(path: &str) -> PyResult<PyVTKMesh> {
    let mesh = read_vtk_file(path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    Ok(PyVTKMesh { inner: mesh })
}

/// Read multiple VTK files in parallel using Rayon.
///
/// Args:
///     paths: List of paths to VTK files
///
/// Returns:
///     List of VTKMesh objects (or raises exception on first error)
#[pyfunction]
#[pyo3(name = "read_vtk_parallel")]
pub fn py_read_vtk_parallel(paths: Vec<String>) -> PyResult<Vec<PyVTKMesh>> {
    // Read files in parallel using Rayon
    let results = read_vtk_files_parallel(&paths);

    results
        .into_iter()
        .map(|r| match r {
            Ok(mesh) => Ok(PyVTKMesh { inner: mesh }),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string())),
        })
        .collect()
}

/// Register the VTK submodule.
pub fn register_vtk_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let vtk = PyModule::new(parent.py(), "vtk")?;
    vtk.add_class::<PyVTKMesh>()?;
    vtk.add_function(wrap_pyfunction!(py_read_vtk, &vtk)?)?;
    vtk.add_function(wrap_pyfunction!(py_read_vtk_parallel, &vtk)?)?;
    parent.add_submodule(&vtk)?;
    Ok(())
}
