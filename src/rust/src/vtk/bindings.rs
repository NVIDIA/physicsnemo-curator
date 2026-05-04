// SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for VTK reader using PyO3.

use pyo3::prelude::*;
use std::fs;

use super::mesh::{ArrayFilter, VtkMeshData};
use super::parser::parse_vtk_xml;
use super::reader::read_vtk_files_parallel_raw;

/// Read a single VTK file from disk.
///
/// Args:
///     path: Path to the VTK file (.vtu, .vtp, .vtk, .vts, .vtm)
///     include_arrays: If set, only include these named data arrays
///     exclude_arrays: If set, exclude these named data arrays
///     skip_cells: If True, skip all cell topology and cell data
///
/// Returns:
///     VtkMeshData object with mesh data accessible as NumPy arrays
#[pyfunction]
#[pyo3(
    name = "read_vtk",
    signature = (path, *, include_arrays=None, exclude_arrays=None, skip_cells=false)
)]
pub fn py_read_vtk(
    py: Python<'_>,
    path: &str,
    include_arrays: Option<Vec<String>>,
    exclude_arrays: Option<Vec<String>>,
    skip_cells: bool,
) -> PyResult<VtkMeshData> {
    let raw = fs::read(path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Cannot read {path}: {e}"))
    })?;
    let filter = ArrayFilter::new(include_arrays, exclude_arrays);
    let arrays = parse_vtk_xml(&raw, &filter, skip_cells).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string())
    })?;
    arrays.into_py_mesh(py)
}

/// Read multiple VTK files in parallel using Rayon.
///
/// Args:
///     paths: List of paths to VTK files
///     include_arrays: If set, only include these named data arrays
///     exclude_arrays: If set, exclude these named data arrays
///     skip_cells: If True, skip all cell topology and cell data
///
/// Returns:
///     List of VtkMeshData objects (or raises exception on first error)
#[pyfunction]
#[pyo3(
    name = "read_vtk_parallel",
    signature = (paths, *, include_arrays=None, exclude_arrays=None, skip_cells=false)
)]
pub fn py_read_vtk_parallel(
    py: Python<'_>,
    paths: Vec<String>,
    include_arrays: Option<Vec<String>>,
    exclude_arrays: Option<Vec<String>>,
    skip_cells: bool,
) -> PyResult<Vec<VtkMeshData>> {
    let filter = ArrayFilter::new(include_arrays, exclude_arrays);
    let results = read_vtk_files_parallel_raw(&paths, &filter, skip_cells);

    results
        .into_iter()
        .map(|r| match r {
            Ok(arrays) => arrays.into_py_mesh(py),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string())),
        })
        .collect()
}

/// Read a VTK mesh from an in-memory byte buffer.
///
/// This avoids writing data to a temporary file when the raw bytes are
/// already available (e.g. after concatenating split volume parts).
///
/// Args:
///     data: Raw bytes of the VTK file content
///     include_arrays: If set, only include these named data arrays
///     exclude_arrays: If set, exclude these named data arrays
///     skip_cells: If True, skip all cell topology and cell data
///
/// Returns:
///     VtkMeshData object with mesh data accessible as NumPy arrays
#[pyfunction]
#[pyo3(
    name = "read_vtk_from_bytes",
    signature = (data, *, include_arrays=None, exclude_arrays=None, skip_cells=false)
)]
pub fn py_read_vtk_from_bytes(
    py: Python<'_>,
    data: &[u8],
    include_arrays: Option<Vec<String>>,
    exclude_arrays: Option<Vec<String>>,
    skip_cells: bool,
) -> PyResult<VtkMeshData> {
    let filter = ArrayFilter::new(include_arrays, exclude_arrays);
    let arrays = parse_vtk_xml(data, &filter, skip_cells).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string())
    })?;
    arrays.into_py_mesh(py)
}

/// Register the VTK submodule.
pub fn register_vtk_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let vtk = PyModule::new(py, "vtk")?;
    vtk.add_class::<VtkMeshData>()?;
    vtk.add_function(wrap_pyfunction!(py_read_vtk, &vtk)?)?;
    vtk.add_function(wrap_pyfunction!(py_read_vtk_parallel, &vtk)?)?;
    vtk.add_function(wrap_pyfunction!(py_read_vtk_from_bytes, &vtk)?)?;
    parent.add_submodule(&vtk)?;

    // Register in sys.modules so `from _lib import vtk` works
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("physicsnemo_curator._lib.vtk", &vtk)?;

    Ok(())
}
