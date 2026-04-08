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

class VTKMesh:  # noqa: N801
    """A VTK mesh parsed from a VTU or VTP file.

    This class is exposed via the ``_lib.vtk`` submodule.
    """

    @property
    def n_points(self) -> int:
        """Number of points in the mesh."""
        ...

    @property
    def n_cells(self) -> int:
        """Number of cells in the mesh."""
        ...

    @property
    def format(self) -> str:
        """File format (e.g., 'vtu', 'vtp')."""
        ...

    def points(self) -> npt.NDArray[np.float64]:
        """Point coordinates as a flattened NumPy array (n_points * 3)."""
        ...

    def connectivity(self) -> npt.NDArray[np.int64]:
        """Cell connectivity as a NumPy array."""
        ...

    def offsets(self) -> npt.NDArray[np.int64]:
        """Cell offsets as a NumPy array."""
        ...

    def types(self) -> npt.NDArray[np.uint8]:
        """Cell types as a NumPy array."""
        ...

    def point_data(self) -> dict[str, tuple[npt.NDArray[np.float64], int]]:
        """Point data arrays as {name: (data, num_components)}."""
        ...

    def cell_data(self) -> dict[str, tuple[npt.NDArray[np.float64], int]]:
        """Cell data arrays as {name: (data, num_components)}."""
        ...

# VTK submodule functions (accessible via _lib.vtk.read_vtk, etc.)
def read_vtk(path: str) -> VTKMesh:
    """Read a single VTK file.

    Parameters
    ----------
    path : str
        Path to the VTK file (.vtu, .vtp, .vtk, .vts, .vtm).

    Returns
    -------
    VTKMesh
        Parsed mesh with data accessible as NumPy arrays.

    Note
    ----
    This function is accessible via ``_lib.vtk.read_vtk()``.
    """
    ...

def read_vtk_parallel(paths: list[str]) -> list[VTKMesh]:
    """Read multiple VTK files in parallel using Rayon.

    Parameters
    ----------
    paths : list[str]
        List of paths to VTK files.

    Returns
    -------
    list[VTKMesh]
        List of parsed meshes.

    Note
    ----
    This function is accessible via ``_lib.vtk.read_vtk_parallel()``.
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
