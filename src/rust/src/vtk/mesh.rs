// SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! VTK mesh data structures and array filtering.
//!
//! Adapted from <https://github.com/coreyjadams/VtkToMesh> (Apache-2.0).

use std::collections::HashMap;

use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

// ---------------------------------------------------------------------------
// Array include/exclude filter
// ---------------------------------------------------------------------------

/// Filter for selecting which data arrays to read.
///
/// When `include` is set, only arrays whose names appear in the list are kept.
/// When `exclude` is set, arrays whose names appear in the list are skipped.
/// If neither is set, all arrays are kept.
pub struct ArrayFilter {
    include: Option<Vec<String>>,
    exclude: Option<Vec<String>>,
}

impl ArrayFilter {
    /// Create a new array filter.
    pub fn new(include: Option<Vec<String>>, exclude: Option<Vec<String>>) -> Self {
        Self { include, exclude }
    }

    /// Return `true` if an array with the given name should be included.
    pub fn should_include(&self, name: &str) -> bool {
        if let Some(ref inc) = self.include {
            return inc.iter().any(|s| s == name);
        }
        if let Some(ref exc) = self.exclude {
            return !exc.iter().any(|s| s == name);
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Scalar type enum
// ---------------------------------------------------------------------------

/// VTK scalar types (maps to NumPy dtypes).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScalarType {
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
}

impl ScalarType {
    /// Number of bytes per element.
    pub fn byte_size(self) -> usize {
        match self {
            ScalarType::Int8 | ScalarType::UInt8 => 1,
            ScalarType::Int16 | ScalarType::UInt16 => 2,
            ScalarType::Float32 | ScalarType::Int32 | ScalarType::UInt32 => 4,
            ScalarType::Float64 | ScalarType::Int64 | ScalarType::UInt64 => 8,
        }
    }

    /// Parse a VTK type string (e.g. `"Float64"`, `"Int32"`).
    pub fn from_vtk_str(s: &str) -> Option<Self> {
        match s {
            "Float32" => Some(ScalarType::Float32),
            "Float64" => Some(ScalarType::Float64),
            "Int8" => Some(ScalarType::Int8),
            "Int16" => Some(ScalarType::Int16),
            "Int32" => Some(ScalarType::Int32),
            "Int64" => Some(ScalarType::Int64),
            "UInt8" => Some(ScalarType::UInt8),
            "UInt16" => Some(ScalarType::UInt16),
            "UInt32" => Some(ScalarType::UInt32),
            "UInt64" => Some(ScalarType::UInt64),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Decoded array (raw bytes + type tag)
// ---------------------------------------------------------------------------

/// A decoded data array: raw little-endian bytes plus type metadata.
pub struct DecodedArray {
    /// Raw bytes (little-endian, length = num_tuples * num_comp * scalar_size).
    pub data: Vec<u8>,
    /// Element scalar type.
    pub scalar_type: ScalarType,
    /// Number of components per tuple (1 for scalar, 3 for vector, etc.).
    pub num_comp: usize,
}

// ---------------------------------------------------------------------------
// MeshArrays: the parser output
// ---------------------------------------------------------------------------

/// Parsed VTK mesh data (intermediate Rust representation).
pub struct MeshArrays {
    /// Number of points declared in the `<Piece>` element.
    pub n_points: usize,
    /// Number of cells declared in the `<Piece>` element.
    pub n_cells: usize,
    /// Point coordinates.
    pub points: DecodedArray,
    /// Cell connectivity (optional — absent for point clouds).
    pub cells: Option<DecodedArray>,
    /// Cell offsets into the connectivity array.
    pub cell_offsets: Option<DecodedArray>,
    /// VTK cell type IDs.
    pub cell_types: Option<DecodedArray>,
    /// Named point-data arrays.
    pub point_data: HashMap<String, DecodedArray>,
    /// Named cell-data arrays.
    pub cell_data: HashMap<String, DecodedArray>,
}

// ---------------------------------------------------------------------------
// Python-facing return type
// ---------------------------------------------------------------------------

/// Mesh data returned to Python.  All heavy data lives in NumPy arrays.
#[pyclass(get_all)]
pub struct VtkMeshData {
    /// Number of points.
    pub n_points: usize,
    /// Number of cells.
    pub n_cells: usize,
    /// Point coordinates as NumPy array.
    #[pyo3(name = "points")]
    pub points_obj: Py<PyAny>,
    /// Cell connectivity as NumPy array (or None).
    #[pyo3(name = "cells")]
    pub cells_obj: Py<PyAny>,
    /// Cell offsets as NumPy array (or None).
    #[pyo3(name = "cell_offsets")]
    pub cell_offsets_obj: Py<PyAny>,
    /// Cell types as NumPy array (or None).
    #[pyo3(name = "cell_types")]
    pub cell_types_obj: Py<PyAny>,
    /// Point data dict: ``{name: numpy_array}``.
    #[pyo3(name = "point_data")]
    pub point_data_obj: Py<PyAny>,
    /// Cell data dict: ``{name: numpy_array}``.
    #[pyo3(name = "cell_data")]
    pub cell_data_obj: Py<PyAny>,
}

#[pymethods]
impl VtkMeshData {
    fn __repr__(&self) -> String {
        format!(
            "VtkMeshData(n_points={}, n_cells={})",
            self.n_points, self.n_cells
        )
    }
}

// ---------------------------------------------------------------------------
// Native-dtype NumPy array creation (zero-copy reinterpret)
// ---------------------------------------------------------------------------

/// Create a NumPy array with the *native* dtype of the decoded bytes.
///
/// Returns a flat 1-D array when `num_comp <= 1`, or reshapes to
/// `(-1, num_comp)` for multi-component data.
fn native_numpy(py: Python<'_>, arr: &DecodedArray) -> PyResult<Py<PyAny>> {
    let flat: Py<PyAny> = match arr.scalar_type {
        ScalarType::Float32 => {
            let n = arr.data.len() / 4;
            let s = unsafe { std::slice::from_raw_parts(arr.data.as_ptr() as *const f32, n) };
            PyArray1::from_slice(py, s).into_any().unbind()
        }
        ScalarType::Float64 => {
            let n = arr.data.len() / 8;
            let s = unsafe { std::slice::from_raw_parts(arr.data.as_ptr() as *const f64, n) };
            PyArray1::from_slice(py, s).into_any().unbind()
        }
        ScalarType::Int8 => {
            let s = unsafe {
                std::slice::from_raw_parts(arr.data.as_ptr() as *const i8, arr.data.len())
            };
            PyArray1::from_slice(py, s).into_any().unbind()
        }
        ScalarType::Int16 => {
            let n = arr.data.len() / 2;
            let s = unsafe { std::slice::from_raw_parts(arr.data.as_ptr() as *const i16, n) };
            PyArray1::from_slice(py, s).into_any().unbind()
        }
        ScalarType::Int32 => {
            let n = arr.data.len() / 4;
            let s = unsafe { std::slice::from_raw_parts(arr.data.as_ptr() as *const i32, n) };
            PyArray1::from_slice(py, s).into_any().unbind()
        }
        ScalarType::Int64 => {
            let n = arr.data.len() / 8;
            let s = unsafe { std::slice::from_raw_parts(arr.data.as_ptr() as *const i64, n) };
            PyArray1::from_slice(py, s).into_any().unbind()
        }
        ScalarType::UInt8 => PyArray1::from_slice(py, &arr.data).into_any().unbind(),
        ScalarType::UInt16 => {
            let n = arr.data.len() / 2;
            let s = unsafe { std::slice::from_raw_parts(arr.data.as_ptr() as *const u16, n) };
            PyArray1::from_slice(py, s).into_any().unbind()
        }
        ScalarType::UInt32 => {
            let n = arr.data.len() / 4;
            let s = unsafe { std::slice::from_raw_parts(arr.data.as_ptr() as *const u32, n) };
            PyArray1::from_slice(py, s).into_any().unbind()
        }
        ScalarType::UInt64 => {
            let n = arr.data.len() / 8;
            let s = unsafe { std::slice::from_raw_parts(arr.data.as_ptr() as *const u64, n) };
            PyArray1::from_slice(py, s).into_any().unbind()
        }
    };

    // Reshape to (N, num_comp) for multi-component arrays
    if arr.num_comp > 1 {
        let bound = flat.bind(py);
        let reshaped = bound.call_method1("reshape", ((-1i64, arr.num_comp as i64),))?;
        Ok(reshaped.unbind())
    } else {
        Ok(flat)
    }
}

/// Convert a HashMap of DecodedArrays to a Python dict of NumPy arrays.
fn decoded_to_pydict(py: Python<'_>, map: HashMap<String, DecodedArray>) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    for (name, arr) in &map {
        let np_arr = native_numpy(py, arr)?;
        dict.set_item(name, np_arr)?;
    }
    Ok(dict.into_any().unbind())
}

impl MeshArrays {
    /// Convert to the Python-facing `VtkMeshData` with NumPy arrays.
    pub fn into_py_mesh(self, py: Python<'_>) -> PyResult<VtkMeshData> {
        let points_obj = native_numpy(py, &self.points)?;

        let cells_obj = if let Some(ref c) = self.cells {
            native_numpy(py, c)?
        } else {
            py.None()
        };

        let cell_offsets_obj = if let Some(ref co) = self.cell_offsets {
            native_numpy(py, co)?
        } else {
            py.None()
        };

        let cell_types_obj = if let Some(ref ct) = self.cell_types {
            native_numpy(py, ct)?
        } else {
            py.None()
        };

        let point_data_obj = decoded_to_pydict(py, self.point_data)?;
        let cell_data_obj = decoded_to_pydict(py, self.cell_data)?;

        Ok(VtkMeshData {
            n_points: self.n_points,
            n_cells: self.n_cells,
            points_obj,
            cells_obj,
            cell_offsets_obj,
            cell_types_obj,
            point_data_obj,
            cell_data_obj,
        })
    }
}
