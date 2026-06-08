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

Reads VTK-format files (``.vtk``, ``.vtp``, ``.vtu``, ``.vts``, ``.vtm``,
``.stl``) from a local directory and converts each to a
:class:`physicsnemo.mesh.Mesh` (or, in *domain-mesh mode*, a
:class:`physicsnemo.mesh.domain_mesh.DomainMesh`).

File discovery uses :func:`pathlib.Path.glob` with an optional pattern,
filtering to recognised extensions.

The conversion supports multiple manifold dimensions (point clouds, lines,
surfaces, volumes) and two point-source modes (vertices or cell centroids),
each resolvable **per file** via path-glob rules.  Data arrays can be
filtered at the VTK reader level (include/exclude keyed by path glob) so
unwanted fields are never materialised.  Reading is delegated to the shared
conversion core in
:mod:`physicsnemo_curator.domains.mesh.sources._vtk_convert`, which supports
both PyVista and a fast native Rust backend.
"""

from __future__ import annotations

import fnmatch
import pathlib
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from physicsnemo.mesh import Mesh

from physicsnemo_curator.core.base import Param, Source
from physicsnemo_curator.domains.mesh.sources._key_filter import (
    resolve_arrays,
    resolve_path_value,
    rules_from_config,
)
from physicsnemo_curator.domains.mesh.sources._vtk_convert import Backend, read_vtk_mesh

if TYPE_CHECKING:
    from collections.abc import Generator

    from physicsnemo.mesh.domain_mesh import DomainMesh

#: File extensions recognised as VTK formats.
_VTK_EXTENSIONS: frozenset[str] = frozenset({".vtk", ".vtp", ".vtu", ".vts", ".vtm", ".stl"})


def _coerce_manifold_dim(value: Any) -> int | Literal["auto"]:
    """Normalise a manifold-dim setting to ``int`` or ``"auto"``."""
    if value is None:
        return "auto"
    if isinstance(value, int):
        return value
    s = str(value).strip().lower()
    if s in ("auto", "", "none", "null"):
        return "auto"
    return int(s)


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
    manifold_dim : int, {"auto"}, or list[dict]
        Target manifold dimension passed to the conversion:

        - ``"auto"`` (default): detect from cell types.
        - ``0``: point cloud (vertices only, no cells).
        - ``1``: line mesh (edge cells).
        - ``2``: surface mesh (triangulated).
        - ``3``: volume mesh (tetrahedralized).

        May also be a list of ``{"pattern": glob, "value": ...}`` rules to
        select the dimension **per file** (longest matching pattern wins),
        mirroring per-file-type defaults (e.g. ``volume_*.vtu`` -> 0).
    point_source : {"vertices", "cell_centroids"} or list[dict]
        Controls what becomes the Mesh points (scalar or per-path rules):

        - ``"vertices"`` (default): mesh vertices become points,
          ``point_data`` is preserved.
        - ``"cell_centroids"``: cell centroids become points,
          ``cell_data`` is mapped to ``point_data``.
    warn_on_lost_data : bool
        If *True* (default), emit a warning when the PyVista conversion
        discards non-empty data arrays.
    backend : {"pyvista", "rust"}
        VTK reading backend:

        - ``"pyvista"`` (default): full-featured reading.
        - ``"rust"``: native Rust backend for faster reading of VTU/VTP;
          transparently falls back to PyVista when unsupported.
    key_filters : list[dict] or None
        Per-path data-array filter rules.  Each dict has ``path_pattern``,
        ``mode`` (``"include"`` / ``"exclude"``), and ``keys``.  Applied at
        the reader level so dropped arrays are never materialised.
    volume_pattern : str or None
        Filename glob identifying volume files for *domain-mesh mode*.
    boundary_pattern : str or None
        Filename glob identifying boundary files for *domain-mesh mode*.
        When both *volume_pattern* and *boundary_pattern* are set, files are
        paired by parent directory into a
        :class:`~physicsnemo.mesh.domain_mesh.DomainMesh` per index;
        unpaired files fall back to standalone :class:`Mesh`.
    boundary_name : str
        Key under which the paired boundary mesh is stored in the
        ``DomainMesh`` (default ``"vehicle"``).
    boundary_generator : object or None
        Optional boundary-condition generator applied to each assembled
        ``DomainMesh`` (domain-mesh mode).  See
        :mod:`physicsnemo_curator.domains.mesh.boundaries`.

    Examples
    --------
    Local directory:

    >>> source = VTKSource("./cfd_results/")
    >>> len(source)
    42
    >>> mesh = next(source[0])

    Domain-mesh mode (volume + boundary -> DomainMesh):

    >>> source = VTKSource(  # doctest: +SKIP
    ...     "./dataset/",
    ...     volume_pattern="volume_*.vtu",
    ...     boundary_pattern="boundary_*.vtp",
    ...     manifold_dim=[{"pattern": "**/volume_*", "value": 0}],
    ...     point_source=[{"pattern": "**/volume_*", "value": "cell_centroids"}],
    ... )

    Note
    ----
    - VTK format: `VTK File Formats <https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html>`_
    - PyVista: `PyVista documentation <https://docs.pyvista.org/>`_
    """

    name: ClassVar[str] = "VTK Reader"
    description: ClassVar[str] = (
        "Read VTK files (.vtk, .vtp, .vtu, .vts, .vtm, .stl) and convert to physicsnemo Mesh / DomainMesh"
    )

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the VTK source.

        Returns
        -------
        list[Param]
            Parameter list including file path, glob pattern, conversion
            options, array filters, and domain-mesh pairing patterns.
        """
        return [
            Param(name="input_path", description="Path to VTK file or directory (local)", type=str),
            Param(name="file_pattern", description="Glob pattern for filtering files", type=str, default="**/*"),
            Param(
                name="manifold_dim",
                description="Target manifold dimension (auto, 0, 1, 2, 3), or per-path rule list",
                type=str,
                default="auto",
                choices=["auto", "0", "1", "2", "3"],
            ),
            Param(
                name="point_source",
                description="Point source mode: vertices or cell_centroids (or per-path rule list)",
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
            Param(
                name="key_filters",
                description="Per-path data-array filter rules (path_pattern, mode, keys)",
                type=list,
                default=None,
            ),
            Param(
                name="volume_pattern",
                description="Filename glob for volume files (enables domain-mesh mode with boundary_pattern)",
                type=str,
                default=None,
            ),
            Param(
                name="boundary_pattern",
                description="Filename glob for boundary files (enables domain-mesh mode with volume_pattern)",
                type=str,
                default=None,
            ),
            Param(
                name="boundary_name",
                description="Boundary key for the paired DomainMesh boundary",
                type=str,
                default="vehicle",
            ),
        ]

    def __init__(
        self,
        input_path: str,
        file_pattern: str = "**/*",
        *,
        manifold_dim: int | Literal["auto"] | list[dict] = "auto",
        point_source: Literal["vertices", "cell_centroids"] | list[dict] = "vertices",
        warn_on_lost_data: bool = True,
        backend: Backend = "pyvista",
        key_filters: list[dict] | None = None,
        volume_pattern: str | None = None,
        boundary_pattern: str | None = None,
        boundary_name: str = "vehicle",
        boundary_generator: Any = None,
    ) -> None:
        self._manifold_dim = manifold_dim
        self._point_source = point_source
        self._warn_on_lost_data = warn_on_lost_data
        self._backend: Backend = backend
        self._key_filters = rules_from_config(key_filters)
        self._volume_pattern = volume_pattern
        self._boundary_pattern = boundary_pattern
        self._boundary_name = boundary_name
        self._boundary_generator = boundary_generator

        root = pathlib.Path(input_path)

        # Single file
        if root.is_file():
            if root.suffix.lower() not in _VTK_EXTENSIONS:
                msg = f"File {root} does not have a recognised VTK extension {sorted(_VTK_EXTENSIONS)}."
                raise ValueError(msg)
            self._root = root.parent
            discovered: list[pathlib.Path] = [root.resolve()]
        elif root.is_dir():
            self._root = root.resolve()
            discovered = sorted(
                p.resolve() for p in root.glob(file_pattern) if p.is_file() and p.suffix.lower() in _VTK_EXTENSIONS
            )
            if not discovered:
                msg = (
                    f"No VTK files found in {root} with pattern {file_pattern!r}; "
                    f"expected extensions {sorted(_VTK_EXTENSIONS)}."
                )
                raise ValueError(msg)
        else:
            msg = f"Path {root} is not a file or directory."
            raise FileNotFoundError(msg)

        # Build work items: ("single", file) or ("pair", volume, boundary).
        self._pairing = volume_pattern is not None and boundary_pattern is not None
        if self._pairing:
            self._items = self._discover_pairs(discovered, volume_pattern, boundary_pattern)
        else:
            self._items = [("single", f) for f in discovered]

        # Representative path per item (for naming / relative_path).
        self._files: list[pathlib.Path] = [item[1] for item in self._items]

    @staticmethod
    def _discover_pairs(
        files: list[pathlib.Path],
        volume_pattern: str,
        boundary_pattern: str,
    ) -> list[tuple[str, pathlib.Path, pathlib.Path] | tuple[str, pathlib.Path]]:
        """Pair volume + boundary files by parent directory.

        Files matching *volume_pattern* / *boundary_pattern* (by filename)
        are paired one-per-directory; everything else becomes a standalone
        item.

        Returns
        -------
        list
            Work items: ``("pair", volume, boundary)`` or ``("single", file)``,
            ordered with pairs first (by directory) then unpaired files.
        """
        volumes: dict[pathlib.Path, pathlib.Path] = {}
        boundaries: dict[pathlib.Path, pathlib.Path] = {}
        other: list[pathlib.Path] = []

        for f in files:
            if fnmatch.fnmatch(f.name, volume_pattern):
                volumes[f.parent] = f
            elif fnmatch.fnmatch(f.name, boundary_pattern):
                boundaries[f.parent] = f
            else:
                other.append(f)

        items: list[tuple] = []
        for parent in sorted(volumes):
            vol = volumes[parent]
            bnd = boundaries.pop(parent, None)
            if bnd is not None:
                items.append(("pair", vol, bnd))
            else:
                other.append(vol)
        # Boundaries with no matching volume become standalone too.
        other.extend(boundaries.values())

        for f in sorted(other):
            items.append(("single", f))
        return items

    # -- Source interface -----------------------------------------------------

    def __len__(self) -> int:
        """Return the number of discovered items (files or volume/boundary pairs)."""
        return len(self._items)

    def __getitem__(self, index: int) -> Generator[Mesh | DomainMesh]:
        """Read the *index*-th item and yield a Mesh or DomainMesh.

        For standalone items a single :class:`~physicsnemo.mesh.Mesh` is
        yielded.  In domain-mesh mode, a volume/boundary pair is assembled
        into a :class:`~physicsnemo.mesh.domain_mesh.DomainMesh` (and the
        optional ``boundary_generator`` is applied).

        Parameters
        ----------
        index : int
            Zero-based index into the discovered item list.

        Yields
        ------
        Mesh or DomainMesh
            The converted physicsnemo object.
        """
        item = self._items[index]

        if item[0] == "single":
            yield self._read_one(item[1])
            return

        from physicsnemo.mesh.domain_mesh import DomainMesh

        _, volume_path, boundary_path = item  # ty: ignore[invalid-assignment]
        interior = self._read_one(volume_path)
        boundary = self._read_one(boundary_path)
        domain = DomainMesh(interior=interior, boundaries={self._boundary_name: boundary})

        if self._boundary_generator is not None:
            from physicsnemo_curator.domains.mesh.boundaries import inject_boundaries

            domain = inject_boundaries(domain, self._boundary_generator)

        yield domain

    def _read_one(self, path: pathlib.Path) -> Mesh:
        """Read one VTK file into a Mesh using per-path conversion settings.

        Parameters
        ----------
        path : pathlib.Path
            Path to the VTK file.

        Returns
        -------
        Mesh
            The converted mesh.
        """
        path_str = str(path)
        manifold_dim = _coerce_manifold_dim(resolve_path_value(self._manifold_dim, path_str, default="auto"))
        point_source = resolve_path_value(self._point_source, path_str, default="vertices")
        include_arrays, exclude_arrays = resolve_arrays(path_str, self._key_filters)

        return read_vtk_mesh(
            path_str,
            backend=self._backend,
            manifold_dim=manifold_dim,
            point_source=point_source,
            include_arrays=include_arrays,
            exclude_arrays=exclude_arrays,
            warn_on_lost_data=self._warn_on_lost_data,
            tessellate_surface=(manifold_dim == 2),
        )

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
        """Return the representative path of the *index*-th item relative to the root.

        For standalone items this is the file itself; for volume/boundary
        pairs it is the volume file.  Used by sinks (e.g.
        :class:`~physicsnemo_curator.domains.mesh.sinks.mesh_writer.MeshSink`)
        to resolve ``{relpath}`` and ``{stem}`` naming placeholders.

        Parameters
        ----------
        index : int
            Zero-based item index.

        Returns
        -------
        str
            POSIX-style relative path (e.g. ``"subdir/mesh.vtu"``).
        """
        return self._files[index].relative_to(self._root).as_posix()
