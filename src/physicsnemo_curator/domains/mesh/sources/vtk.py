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
from a local directory and converts each to a
:class:`physicsnemo.mesh.Mesh` using :func:`physicsnemo.mesh.io.from_pyvista`.

File discovery uses :func:`pathlib.Path.glob` with an optional pattern,
filtering to recognised VTK extensions.

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

from physicsnemo_curator.core.base import Param, Source

if TYPE_CHECKING:
    from collections.abc import Generator

#: File extensions recognised as VTK formats.
_VTK_EXTENSIONS: frozenset[str] = frozenset({".vtk", ".vtp", ".vtu", ".vts", ".vtm"})

#: Valid backend options for VTK reading.
Backend = Literal["pyvista", "rust"]


class VTKSource(Source[Mesh]):
    """Read local VTK files and yield :class:`~physicsnemo.mesh.Mesh` objects.

    File discovery uses :func:`pathlib.Path.glob` with an optional
    *file_pattern*, filtering to recognised VTK extensions.  Only local
    paths are supported; for remote datasets use a domain-specific source
    such as :class:`~physicsnemo_curator.domains.mesh.sources.drivaerml.DrivAerMLSource`.

    Parameters
    ----------
    input_path : str
        Path to a local directory containing VTK files, or a single VTK
        file.
    file_pattern : str
        Glob pattern for filtering files inside a directory.  Defaults to
        ``"**/*"`` which recursively discovers all VTK files.  Use ``"*"``
        for flat (non-recursive) discovery, or a custom pattern such as
        ``"timestep_*"`` for selective matching.
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
    backend : {"pyvista", "rust"}
        VTK reading backend:

        - ``"pyvista"`` (default): use PyVista for full-featured reading.
        - ``"rust"``: use the native Rust backend for faster reading.
          Note: The Rust backend only supports ASCII VTU/VTP files and
          does not support ``manifold_dim`` or ``point_source`` options.

    Examples
    --------
    Local directory:

    >>> source = VTKSource("./cfd_results/")
    >>> len(source)
    42
    >>> mesh = next(source[0])

    Custom glob pattern:

    >>> source = VTKSource("./data/", file_pattern="timestep_*")

    Using the fast Rust backend:

    >>> source = VTKSource("./cfd_results/", backend="rust")

    Note
    ----
    - VTK format: `VTK File Formats <https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html>`_
    - PyVista: `PyVista documentation <https://docs.pyvista.org/>`_
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
            Param(name="input_path", description="Path to VTK file or directory (local)", type=str),
            Param(name="file_pattern", description="Glob pattern for filtering files", type=str, default="**/*"),
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
            Param(
                name="backend",
                description="VTK reading backend: pyvista (default) or rust (faster)",
                type=str,
                default="pyvista",
                choices=["pyvista", "rust"],
            ),
        ]

    def __init__(
        self,
        input_path: str,
        file_pattern: str = "**/*",
        *,
        manifold_dim: int | Literal["auto"] = "auto",
        point_source: Literal["vertices", "cell_centroids"] = "vertices",
        warn_on_lost_data: bool = True,
        backend: Backend = "pyvista",
    ) -> None:
        self._manifold_dim = manifold_dim
        self._point_source = point_source
        self._warn_on_lost_data = warn_on_lost_data
        self._backend: Backend = backend

        root = pathlib.Path(input_path)

        # Single file
        if root.is_file():
            if root.suffix.lower() not in _VTK_EXTENSIONS:
                msg = f"File {root} does not have a recognised VTK extension {sorted(_VTK_EXTENSIONS)}."
                raise ValueError(msg)
            self._root = root.parent
            self._files: list[pathlib.Path] = [root.resolve()]
            return

        if not root.is_dir():
            msg = f"Path {root} is not a file or directory."
            raise FileNotFoundError(msg)

        self._root = root.resolve()

        # Glob and filter to VTK extensions, then sort for deterministic order
        discovered = sorted(
            p.resolve() for p in root.glob(file_pattern) if p.is_file() and p.suffix.lower() in _VTK_EXTENSIONS
        )
        if not discovered:
            msg = (
                f"No VTK files found in {root} with pattern {file_pattern!r}; "
                f"expected extensions {sorted(_VTK_EXTENSIONS)}."
            )
            raise ValueError(msg)

        self._files = discovered

    # -- Source interface -----------------------------------------------------

    def __len__(self) -> int:
        """Return the number of discovered VTK files."""
        return len(self._files)

    def __getitem__(self, index: int) -> Generator[Mesh]:
        """Read the *index*-th VTK file and yield a Mesh.

        The file is loaded with the configured backend and converted to
        a :class:`~physicsnemo.mesh.Mesh`. When using the PyVista backend,
        the mesh is converted via :func:`physicsnemo.mesh.io.from_pyvista`
        using the conversion parameters supplied at construction time.

        Parameters
        ----------
        index : int
            Zero-based file index into the discovered VTK file list.

        Yields
        ------
        Mesh
            The converted physicsnemo Mesh.
        """
        path = str(self._files[index])
        mesh = self._read_with_rust(path) if self._backend == "rust" else self._read_with_pyvista(path)
        yield mesh

    # -- Path helpers (used by MeshSink for naming) --------------------------

    @property
    def root(self) -> pathlib.Path:
        """Return the root directory of this source.

        Returns
        -------
        pathlib.Path
            The root directory containing the discovered VTK files.
        """
        return self._root

    def relative_path(self, index: int) -> str:
        """Return the path of the *index*-th file relative to the root.

        This is used by sinks (e.g.
        :class:`~physicsnemo_curator.domains.mesh.sinks.mesh_writer.MeshSink`) to
        resolve ``{relpath}`` and ``{stem}`` naming placeholders.

        Parameters
        ----------
        index : int
            Zero-based file index.

        Returns
        -------
        str
            POSIX-style relative path (e.g. ``"subdir/mesh.vtu"``).
        """
        return self._files[index].relative_to(self._root).as_posix()

    # -- Private readers -----------------------------------------------------

    def _read_with_pyvista(self, path: str) -> Mesh:
        """Read VTK file using PyVista backend.

        Parameters
        ----------
        path : str
            Path to the VTK file.

        Returns
        -------
        Mesh
            Converted physicsnemo Mesh.
        """
        pv_mesh = pv.read(path)
        return from_pyvista(
            pv_mesh,
            manifold_dim=self._manifold_dim,
            point_source=self._point_source,
            warn_on_lost_data=self._warn_on_lost_data,
        )

    def _read_with_rust(self, path: str) -> Mesh:
        """Read VTK file using Rust backend.

        The Rust backend returns raw mesh data without the from_pyvista
        conversion. Currently only supports ASCII VTU/VTP files and does
        not support manifold_dim or point_source options.

        Parameters
        ----------
        path : str
            Path to the VTK file.

        Returns
        -------
        Mesh
            Raw mesh with points and point_data from the VTK file.
        """
        import torch
        from tensordict import TensorDict

        from physicsnemo_curator._lib import vtk

        rust_mesh = vtk.read_vtk(path)

        # Convert Rust mesh to physicsnemo Mesh
        points = torch.from_numpy(rust_mesh.points)

        # Build point_data TensorDict from Rust arrays
        point_data_dict = {}
        for name, data in rust_mesh.point_data.items():
            arr = torch.from_numpy(data)
            point_data_dict[name] = arr

        point_data = TensorDict(point_data_dict, batch_size=[rust_mesh.n_points]) if point_data_dict else None

        return Mesh(
            points=points,
            point_data=point_data,
            # Note: cells/faces not converted - Rust backend is for raw data access
        )
