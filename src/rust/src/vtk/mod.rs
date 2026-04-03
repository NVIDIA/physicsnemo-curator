// SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! VTK file format reader module.
//!
//! Provides multi-threaded reading of VTK XML files (VTU, VTP) with
//! direct conversion to NumPy arrays for Python integration.

pub mod bindings;
pub mod mesh;
pub mod parser;
pub mod reader;

/// VTK cell type constants (subset commonly used in CFD).
/// See: <https://vtk.org/doc/nightly/html/vtkCellType_8h.html>
#[allow(dead_code)]
pub mod cell_types {
    pub const VTK_VERTEX: u8 = 1;
    pub const VTK_POLY_VERTEX: u8 = 2;
    pub const VTK_LINE: u8 = 3;
    pub const VTK_POLY_LINE: u8 = 4;
    pub const VTK_TRIANGLE: u8 = 5;
    pub const VTK_TRIANGLE_STRIP: u8 = 6;
    pub const VTK_POLYGON: u8 = 7;
    pub const VTK_QUAD: u8 = 9;
    pub const VTK_TETRA: u8 = 10;
    pub const VTK_VOXEL: u8 = 11;
    pub const VTK_HEXAHEDRON: u8 = 12;
    pub const VTK_WEDGE: u8 = 13;
    pub const VTK_PYRAMID: u8 = 14;
}
