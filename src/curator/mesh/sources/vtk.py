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

"""VTK file source for mesh pipelines.

Reads VTK-format files (``.vtk``, ``.vtp``, ``.vtu``, ``.vts``, ``.vtm``)
from a directory and converts each to a :class:`physicsnemo.mesh.Mesh`
using :func:`physicsnemo.mesh.io.from_pyvista`.

The conversion supports multiple manifold dimensions (point clouds, lines,
surfaces, volumes) and two point-source modes (vertices or cell centroids).
See :func:`physicsnemo.mesh.io.from_pyvista` for full details.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, ClassVar, Literal

import pyvista as pv
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.io import from_pyvista

from curator.core.base import Param, Source

if TYPE_CHECKING:
    from collections.abc import Generator

#: File extensions recognised as VTK formats.
_VTK_EXTENSIONS: frozenset[str] = frozenset({".vtk", ".vtp", ".vtu", ".vts", ".vtm"})


class VTKSource(Source[Mesh]):
    """Read VTK files from a directory and yield :class:`~physicsnemo.mesh.Mesh` objects.

    Each VTK file is loaded via :func:`pyvista.read` and converted to a
    :class:`~physicsnemo.mesh.Mesh` using :func:`physicsnemo.mesh.io.from_pyvista`.
    The conversion parameters (``manifold_dim``, ``point_source``, etc.) are
    forwarded directly to ``from_pyvista``.

    Parameters
    ----------
    input_path : str
        Path to a directory containing VTK files, or a single VTK file path.
    file_pattern : str
        Glob pattern used to filter files when *input_path* is a directory.
        Defaults to ``"*"`` which matches all files; only files whose
        extension is a recognised VTK format are kept regardless.
    manifold_dim : int or {"auto"}
        Target manifold dimension passed to ``from_pyvista``:

        - ``"auto"`` (default): detect from cell types.
        - ``0``: point cloud (vertices only, no cells).
        - ``1``: line mesh (edge cells).
        - ``2``: surface mesh (triangulated).
        - ``3``: volume mesh (tetrahedralized).
    point_source : {"vertices", "cell_centroids"}
        Controls what becomes the Mesh points:

        - ``"vertices"`` (default): mesh vertices become points,
          ``point_data`` is preserved.
        - ``"cell_centroids"``: cell centroids become points,
          ``cell_data`` is mapped to ``point_data``.  Only
          ``manifold_dim`` 0 and 1 are valid in this mode.
    warn_on_lost_data : bool
        If *True* (default), emit a warning when the conversion discards
        non-empty data arrays (e.g. cell data lost during dimension
        reduction).

    Examples
    --------
    >>> source = VTKSource(input_path="./cfd_results/")
    >>> len(source)
    42
    >>> mesh_gen = source[0]
    >>> mesh = next(mesh_gen)

    Read volume meshes and tetrahedralize:

    >>> source = VTKSource(input_path="./volumes/", manifold_dim=3)

    Use cell centroids as points (avoids tetrahedralization for CFD):

    >>> source = VTKSource(
    ...     input_path="./cfd/", point_source="cell_centroids"
    ... )
    """

    name: ClassVar[str] = "VTK Reader"
    description: ClassVar[str] = "Read VTK files (.vtk, .vtp, .vtu, .vts, .vtm) and convert to physicsnemo Mesh"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the VTK source.

        Returns
        -------
        list[Param]
            Parameter list including file path, glob pattern, and
            ``from_pyvista`` conversion options.
        """
        return [
            Param(name="input_path", description="Path to VTK file or directory", type=str),
            Param(name="file_pattern", description="Glob pattern for filtering files", type=str, default="*"),
            Param(
                name="manifold_dim",
                description="Target manifold dimension (auto, 0, 1, 2, 3)",
                type=str,
                default="auto",
                choices=["auto", "0", "1", "2", "3"],
            ),
            Param(
                name="point_source",
                description="Point source mode: vertices or cell_centroids",
                type=str,
                default="vertices",
                choices=["vertices", "cell_centroids"],
            ),
            Param(
                name="warn_on_lost_data",
                description="Warn when data arrays are discarded during conversion",
                type=bool,
                default=True,
            ),
        ]

    def __init__(
        self,
        input_path: str,
        file_pattern: str = "*",
        manifold_dim: int | Literal["auto"] = "auto",
        point_source: Literal["vertices", "cell_centroids"] = "vertices",
        warn_on_lost_data: bool = True,
    ) -> None:
        self._root = pathlib.Path(input_path)
        self._pattern = file_pattern
        self._manifold_dim = manifold_dim
        self._point_source = point_source
        self._warn_on_lost_data = warn_on_lost_data
        self._files = self._discover_files()

    # -- Source interface -----------------------------------------------------

    def __len__(self) -> int:
        """Return the number of discovered VTK files."""
        return len(self._files)

    def __getitem__(self, index: int) -> Generator[Mesh]:
        """Read the *index*-th VTK file and yield a Mesh.

        The file is loaded with :func:`pyvista.read` and converted via
        :func:`physicsnemo.mesh.io.from_pyvista` using the conversion
        parameters supplied at construction time.

        Parameters
        ----------
        index : int
            Zero-based file index.

        Yields
        ------
        Mesh
            The converted physicsnemo Mesh.
        """
        path = self._files[index]
        pv_mesh = pv.read(str(path))
        mesh = from_pyvista(
            pv_mesh,
            manifold_dim=self._manifold_dim,
            point_source=self._point_source,
            warn_on_lost_data=self._warn_on_lost_data,
        )
        yield mesh

    # -- Internal helpers ----------------------------------------------------

    def _discover_files(self) -> list[pathlib.Path]:
        """Scan the input path and return sorted VTK file paths.

        Returns
        -------
        list[pathlib.Path]
            Sorted list of VTK file paths discovered.

        Raises
        ------
        FileNotFoundError
            If *input_path* does not exist.
        ValueError
            If no VTK files are found.
        """
        if self._root.is_file():
            if self._root.suffix.lower() in _VTK_EXTENSIONS:
                return [self._root]
            msg = f"File {self._root} is not a recognised VTK format."
            raise ValueError(msg)

        if not self._root.is_dir():
            msg = f"Input path {self._root} does not exist."
            raise FileNotFoundError(msg)

        files = sorted(p for p in self._root.glob(self._pattern) if p.is_file() and p.suffix.lower() in _VTK_EXTENSIONS)

        if not files:
            msg = f"No VTK files found in {self._root} with pattern {self._pattern!r}."
            raise ValueError(msg)

        return files
