// SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Internal mesh data structure for VTK reader.

use std::collections::HashMap;

/// A named data array (point_data or cell_data field).
#[derive(Debug, Clone)]
pub struct DataArray {
    /// Number of components per tuple (1 for scalar, 3 for vector, etc.).
    pub num_components: usize,
    /// Flattened data values (length = num_tuples * num_components).
    pub data: Vec<f64>,
}

/// Parsed VTK mesh data.
#[derive(Debug, Clone)]
pub struct VTKMesh {
    /// Point coordinates, shape (n_points, 3).
    pub points: Vec<f64>,
    /// Number of points.
    pub n_points: usize,
    /// Cell connectivity (flattened).
    pub connectivity: Vec<i64>,
    /// Cell offsets into connectivity array.
    pub offsets: Vec<i64>,
    /// Cell types (VTK cell type IDs).
    pub types: Vec<u8>,
    /// Number of cells.
    pub n_cells: usize,
    /// Point data arrays.
    pub point_data: HashMap<String, DataArray>,
    /// Cell data arrays.
    pub cell_data: HashMap<String, DataArray>,
    /// File format ("vtu" or "vtp").
    pub format: String,
}

impl VTKMesh {
    /// Create an empty mesh.
    pub fn new(format: &str) -> Self {
        VTKMesh {
            points: Vec::new(),
            n_points: 0,
            connectivity: Vec::new(),
            offsets: Vec::new(),
            types: Vec::new(),
            n_cells: 0,
            point_data: HashMap::new(),
            cell_data: HashMap::new(),
            format: format.to_string(),
        }
    }
}
