# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Type stubs for the native Rust extension module."""

import numpy as np
import numpy.typing as npt

def rust_version() -> str:
    """Return the version of the native Rust library."""
    ...

class VtkMeshData:
    """VTK mesh data with NumPy arrays.

    Returned by :func:`read_vtk`, :func:`read_vtk_parallel`, and
    :func:`read_vtk_from_bytes`.  All heavy data is stored as NumPy
    arrays with native dtypes (float32, float64, int32, int64, etc.).

    This class is exposed via the ``_lib.vtk`` submodule.
    """

    n_points: int
    """Number of points in the mesh."""

    n_cells: int
    """Number of cells in the mesh."""

    points: npt.NDArray[np.floating]
    """Point coordinates, shape ``(n_points, 3)``."""

    cells: npt.NDArray[np.integer] | None
    """Cell connectivity indices (flat), or ``None`` if no cells."""

    cell_offsets: npt.NDArray[np.integer] | None
    """Cell offsets into connectivity, or ``None`` if no cells."""

    cell_types: npt.NDArray[np.uint8] | None
    """VTK cell type IDs, or ``None`` if no cells."""

    point_data: dict[str, npt.NDArray[np.generic]]
    """Per-point data arrays: ``{name: array}``."""

    cell_data: dict[str, npt.NDArray[np.generic]]
    """Per-cell data arrays: ``{name: array}``."""

# VTK submodule functions (accessible via _lib.vtk.read_vtk, etc.)
def read_vtk(
    path: str,
    *,
    include_arrays: list[str] | None = None,
    exclude_arrays: list[str] | None = None,
    skip_cells: bool = False,
    skip_point_data: bool = False,
) -> VtkMeshData:
    """Read a single VTK file.

    Parameters
    ----------
    path : str
        Path to the VTK file (.vtu, .vtp, .vtk, .vts, .vtm).
    include_arrays : list[str] | None
        If set, only include these named data arrays.
    exclude_arrays : list[str] | None
        If set, exclude these named data arrays.
    skip_cells : bool
        If ``True``, skip all cell topology and cell data.
    skip_point_data : bool
        If ``True``, skip all point data field arrays.
        Point coordinates are still read.

    Returns
    -------
    VtkMeshData
        Parsed mesh with data accessible as NumPy arrays.

    Note
    ----
    This function is accessible via ``_lib.vtk.read_vtk()``.
    """
    ...

def read_vtk_parallel(
    paths: list[str],
    *,
    include_arrays: list[str] | None = None,
    exclude_arrays: list[str] | None = None,
    skip_cells: bool = False,
    skip_point_data: bool = False,
) -> list[VtkMeshData]:
    """Read multiple VTK files in parallel using Rayon.

    Parameters
    ----------
    paths : list[str]
        List of paths to VTK files.
    include_arrays : list[str] | None
        If set, only include these named data arrays.
    exclude_arrays : list[str] | None
        If set, exclude these named data arrays.
    skip_cells : bool
        If ``True``, skip all cell topology and cell data.
    skip_point_data : bool
        If ``True``, skip all point data field arrays.
        Point coordinates are still read.

    Returns
    -------
    list[VtkMeshData]
        List of parsed meshes.

    Note
    ----
    This function is accessible via ``_lib.vtk.read_vtk_parallel()``.
    """
    ...

def read_vtk_from_bytes(
    data: bytes,
    *,
    include_arrays: list[str] | None = None,
    exclude_arrays: list[str] | None = None,
    skip_cells: bool = False,
    skip_point_data: bool = False,
) -> VtkMeshData:
    """Read a VTK mesh from an in-memory byte buffer.

    This avoids writing data to a temporary file when the raw bytes are
    already available (e.g. after concatenating split volume parts).

    Parameters
    ----------
    data : bytes
        Raw bytes of the VTK file content.
    include_arrays : list[str] | None
        If set, only include these named data arrays.
    exclude_arrays : list[str] | None
        If set, exclude these named data arrays.
    skip_cells : bool
        If ``True``, skip all cell topology and cell data.
    skip_point_data : bool
        If ``True``, skip all point data field arrays.
        Point coordinates are still read.

    Returns
    -------
    VtkMeshData
        Parsed mesh with data accessible as NumPy arrays.

    Note
    ----
    This function is accessible via ``_lib.vtk.read_vtk_from_bytes()``.
    """
    ...

# LMDB submodule functions (accessible via _lib.lmdb.read_lmdb, etc.)
def read_lmdb(path: str) -> list[dict[str, object]]:
    """Read all data rows from a single ``.aselmdb`` file.

    Opens the LMDB environment in read-only mode, decompresses each
    row value (zlib), parses the JSON, and converts ``__ndarray__``
    markers into actual NumPy arrays.  Returns a list of row dicts
    sorted by ascending row ID.

    Parameters
    ----------
    path : str
        Path to the ``.aselmdb`` file.

    Returns
    -------
    list[dict[str, object]]
        List of row dictionaries.  Each dict contains the parsed ASE
        row data with NumPy arrays and a synthetic ``"id"`` key.

    Note
    ----
    This function is accessible via ``_lib.lmdb.read_lmdb()``.
    """
    ...

def read_lmdb_parallel(paths: list[str]) -> list[list[dict[str, object]]]:
    """Read rows from multiple ``.aselmdb`` files in parallel.

    Each file is read on a separate Rayon worker thread.  The heavy
    I/O, decompression, and JSON parsing run outside the GIL.

    Parameters
    ----------
    paths : list[str]
        List of paths to ``.aselmdb`` files.

    Returns
    -------
    list[list[dict[str, object]]]
        Outer list corresponds to input paths (same order); each inner
        list contains the row dicts for that file.

    Note
    ----
    This function is accessible via ``_lib.lmdb.read_lmdb_parallel()``.
    """
    ...

# D3Plot submodule functions (accessible via _lib.d3plot.*, etc.)
def parse_k_file(path: str) -> dict[int, float]:
    """Parse an LS-DYNA ``.k`` keyword file for part thickness.

    Reads the file in Rust, extracts ``*PART`` and ``*SECTION_SHELL``
    definitions, and returns a mapping from part ID to thickness.

    Parameters
    ----------
    path : str
        Path to the ``.k`` file.

    Returns
    -------
    dict[int, float]
        Mapping from part ID to thickness value.

    Note
    ----
    This function is accessible via ``_lib.d3plot.parse_k_file()``.
    """
    ...

def compute_node_thickness(
    connectivity: npt.NDArray[np.int64],
    part_ids: npt.NDArray[np.int64],
    part_thickness: dict[int, float],
    actual_part_ids: npt.NDArray[np.int64] | None = None,
) -> npt.NDArray[np.float64]:
    """Compute per-node thickness from element connectivity.

    Scatter-accumulates element thickness onto mesh nodes and averages
    by incident element count.

    Parameters
    ----------
    connectivity : NDArray[int64]
        Element connectivity, shape ``(E, nodes_per_cell)``.
    part_ids : NDArray[int64]
        Part index per element, shape ``(E,)``.
    part_thickness : dict[int, float]
        Mapping from actual part ID to thickness.
    actual_part_ids : NDArray[int64] | None
        Optional array of actual part IDs for index→ID translation.

    Returns
    -------
    NDArray[float64]
        Per-node thickness, shape ``(max_node+1,)``.

    Note
    ----
    This function is accessible via ``_lib.d3plot.compute_node_thickness()``.
    """
    ...

def von_mises_from_voigt(
    stress: npt.NDArray[np.float64],
    n_total: int,
) -> npt.NDArray[np.float64]:
    """Compute von Mises stress from Voigt-notation stress tensor.

    Parameters
    ----------
    stress : NDArray[float64]
        Flattened stress array, length ``n_total * 6``.
    n_total : int
        Number of stress entries.

    Returns
    -------
    NDArray[float64]
        Von Mises stress, shape ``(n_total,)``.

    Note
    ----
    This function is accessible via ``_lib.d3plot.von_mises_from_voigt()``.
    """
    ...
