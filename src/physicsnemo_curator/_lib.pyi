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
