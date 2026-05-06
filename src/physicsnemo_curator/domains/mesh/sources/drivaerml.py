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

"""DrivAerML dataset source for mesh pipelines.

Reads the `DrivAerML <https://huggingface.co/datasets/neashton/drivaerml>`_
dataset — 500 parametrically morphed variants of the DrivAer notchback
vehicle with high-fidelity scale-resolving CFD (OpenFOAM v2212).

The dataset provides three mesh types per run:

* **boundary** — surface mesh with flow fields (VTP, ~660 MB each)
* **volume** — volumetric field data (VTU, ~50 GB each, split into parts)
* **slices** — x/y/z-normal slice planes with flow fields (VTP)

File discovery and caching are handled internally using ``fsspec``.
"""

from __future__ import annotations

import logging
import pathlib
import re
import tempfile
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.domain_mesh import DomainMesh

from physicsnemo_curator.core.base import Param, Source
from physicsnemo_curator.core.cache import default_data_cache_dir

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)

#: HuggingFace Hub URL for the DrivAerML dataset.
_DRIVAERML_HF_URL = "hf://datasets/neashton/drivaerml"

#: Mesh type → file template mapping.
_MESH_TEMPLATES: dict[str, str] = {
    "boundary": "boundary_{i}.vtp",
    "volume": "volume_{i}.vtu",
}

#: Valid mesh types for this dataset.
MeshType = Literal["boundary", "volume", "slices", "multi"]

#: Valid backend options for VTK reading.
Backend = Literal["pyvista", "rust"]

#: Valid mesh parts for multi mode.
_VALID_MESH_PARTS: set[str] = {"domain", "stl", "single_solid"}

#: Canonical output name templates per mesh part.
_MESH_NAME_TEMPLATES: dict[str, str] = {
    "domain": "domain_{run_id}",
    "stl": "drivaer_{run_id}.stl",
    "single_solid": "drivaer_{run_id}_single_solid.stl",
}


class DrivAerMLSource(Source[Mesh]):
    """Read meshes from the DrivAerML dataset on HuggingFace Hub.

    Each index maps to one simulation run.  The *mesh_type* parameter
    selects which mesh to load for each run:

    * ``"boundary"`` — surface mesh (VTP) with flow fields
    * ``"volume"`` — volumetric mesh (VTU, reconstructed from split parts)
    * ``"slices"`` — x/y/z-normal slice planes (VTP); yields multiple
      meshes per index
    * ``"multi"`` — yields domain (as :class:`DomainMesh`), stl, and/or
      single_solid meshes per run (all converted to float32).  The domain
      mesh combines volume interior, boundary surface, and global data.

    Parameters
    ----------
    mesh_type : {"boundary", "volume", "slices", "multi"}
        Which mesh to read from each run directory.
    url : str
        Base HuggingFace Hub URL.  Override only for testing.
    storage_options : dict[str, object] | None
        Extra ``fsspec`` keyword arguments (e.g. ``{"token": "hf_..."}``).
    cache_storage : str | None
        Local cache directory.  ``None`` → temporary directory.
    cache : bool
        Persist downloaded files across sessions.
    manifold_dim : int or {"auto"}
        Target manifold dimension for ``from_pyvista`` conversion.
        Only used with the ``"pyvista"`` backend.
    point_source : {"vertices", "cell_centroids"}
        Point source mode for ``from_pyvista`` conversion.
        Only used with the ``"pyvista"`` backend.
    warn_on_lost_data : bool
        Warn when data arrays are discarded during conversion.
        Only used with the ``"pyvista"`` backend.
    mesh_parts : list[str] or None
        Which mesh parts to yield in ``"multi"`` mode.  Valid parts are
        ``"domain"``, ``"stl"``, and ``"single_solid"``.  When ``None``
        (the default), all three parts are yielded.  Ignored for other
        mesh types.
    backend : {"pyvista", "rust"}
        VTK reading backend:

        - ``"pyvista"`` (default): uses PyVista + ``from_pyvista`` for full
          conversion (manifold_dim, point_source options respected).
        - ``"rust"``: uses the native Rust VTK reader for faster I/O.
          Constructs :class:`Mesh` directly from raw arrays.  The
          ``manifold_dim`` and ``point_source`` options are ignored; data
          is returned as-is from the file.

    Examples
    --------
    >>> source = DrivAerMLSource(mesh_type="boundary")
    >>> len(source)
    484
    >>> mesh = next(source[0])

    >>> source = DrivAerMLSource(mesh_type="slices")
    >>> for mesh in source[0]:  # yields multiple slice planes
    ...     print(mesh.n_points)

    Note
    ----
    - Dataset: `neashton/drivaerml <https://huggingface.co/datasets/neashton/drivaerml>`_
    - Paper: `arXiv:2408.11969 <https://arxiv.org/abs/2408.11969>`_
    - License: `CC-BY-SA-4.0 <https://huggingface.co/datasets/neashton/drivaerml/blob/main/LICENSE>`_
    """

    name: ClassVar[str] = "DrivAerML"
    description: ClassVar[str] = "DrivAerML dataset — 500 DrivAer notchback variants with scale-resolving CFD"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the DrivAerML source.

        Returns
        -------
        list[Param]
            Parameter list for CLI configuration.
        """
        return [
            Param(
                name="mesh_type",
                description="Mesh type: boundary (surface), volume (3D field), slices (planes), or multi (all)",
                type=str,
                default="boundary",
                choices=["boundary", "volume", "slices", "multi"],
            ),
            Param(name="url", description="Base HuggingFace Hub URL", type=str, default=_DRIVAERML_HF_URL),
            Param(
                name="cache_storage",
                description="Local cache directory for downloaded files",
                type=str,
                default="",
            ),
            Param(
                name="cache",
                description="Persist downloaded files across sessions (False = temp dir, deleted after read)",
                type=bool,
                default=True,
            ),
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
                name="mesh_parts",
                description="Mesh parts to yield in multi mode (domain, stl, single_solid)",
                type=list,
                default=None,
            ),
            Param(
                name="backend",
                description="VTK reading backend: pyvista (PyVista, default) or rust (native, faster)",
                type=str,
                default="pyvista",
                choices=["pyvista", "rust"],
            ),
        ]

    def __init__(
        self,
        mesh_type: MeshType = "boundary",
        url: str = _DRIVAERML_HF_URL,
        storage_options: dict[str, object] | None = None,
        cache_storage: str | None = None,
        cache: bool = True,
        manifold_dim: int | Literal["auto"] = "auto",
        point_source: Literal["vertices", "cell_centroids"] = "vertices",
        warn_on_lost_data: bool = True,
        mesh_parts: list[str] | None = None,
        backend: Backend = "pyvista",
    ) -> None:
        import fsspec

        self._mesh_type: MeshType = mesh_type
        self._url = url
        self._storage_options = storage_options or {}
        self._cache = cache

        # Resolve cache directory:
        # 1. Explicit cache_storage always wins (implies persistent)
        # 2. cache=True → persistent default under ~/.cache/psnc/data/drivaerml/
        # 3. cache=False → ephemeral temp dir (files deleted after read)
        if cache_storage:
            self._cache_storage = cache_storage
        elif cache:
            self._cache_storage = str(default_data_cache_dir("drivaerml"))
        else:
            self._cache_storage = tempfile.mkdtemp(prefix="curator_drivaerml_")

        self._manifold_dim = manifold_dim
        self._point_source = point_source
        self._warn_on_lost_data = warn_on_lost_data
        self._backend: Backend = backend

        self._fs, self._root_path = fsspec.core.url_to_fs(self._url, **self._storage_options)
        self._protocol = self._fs.protocol if isinstance(self._fs.protocol, str) else self._fs.protocol[0]
        self._run_indices = self._discover_runs()

        if mesh_type == "multi":
            self._mesh_parts = mesh_parts or ["domain", "stl", "single_solid"]
            invalid = set(self._mesh_parts) - _VALID_MESH_PARTS
            if invalid:
                msg = f"Invalid mesh_parts: {sorted(invalid)}. Valid parts: {sorted(_VALID_MESH_PARTS)}"
                raise ValueError(msg)
            self._file_template = ""  # not used for multi mode
        elif mesh_type == "slices":
            self._mesh_parts: list[str] = []
            self._build_slices_data()
        else:
            self._mesh_parts = []
            self._file_template = _MESH_TEMPLATES[mesh_type]

    def _discover_runs(self) -> list[int]:
        """Discover ``run_<i>/`` directories at the base URL.

        Returns
        -------
        list[int]
            Sorted list of integer run indices.

        Raises
        ------
        ValueError
            If no ``run_<i>/`` directories are found.
        """
        run_pattern = re.compile(r"^run_(\d+)$")
        entries = self._fs.ls(self._root_path, detail=False)

        run_indices: list[int] = []
        for entry in entries:
            basename = pathlib.PurePosixPath(entry).name
            m = run_pattern.match(basename)
            if m:
                run_indices.append(int(m.group(1)))

        if not run_indices:
            msg = f"No run_<i>/ directories found at {self._url}."
            raise ValueError(msg)

        return sorted(run_indices)

    def _ensure_local(self, remote_path: str, fs: Any = None, protocol: str | None = None) -> str:
        """Download *remote_path* to the local cache if not already present.

        Parameters
        ----------
        remote_path : str
            Filesystem path (no protocol prefix).
        fs : Any, optional
            fsspec filesystem instance. Defaults to ``self._fs``.
        protocol : str, optional
            Filesystem protocol string. Defaults to ``self._protocol``.

        Returns
        -------
        str
            Local filesystem path.
        """
        fs = fs or self._fs
        protocol = protocol or self._protocol

        if protocol in ("file", ""):
            if not pathlib.Path(remote_path).exists():
                msg = f"Local file not found: {remote_path}"
                raise FileNotFoundError(msg)
            return remote_path

        local_path = pathlib.Path(self._cache_storage) / remote_path.lstrip("/")
        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            fs.get(remote_path, str(local_path))

        return str(local_path)

    def _cleanup_local(self, *paths: str) -> None:
        """Delete local files if caching is disabled.

        When ``self._cache`` is ``False``, removes the given files from
        disk to free space immediately after they have been read into
        memory.  Does nothing when caching is enabled or for local
        (``file://``) sources.

        Parameters
        ----------
        *paths : str
            Local file paths to remove.
        """
        if self._cache or self._protocol in ("file", ""):
            return
        for p in paths:
            pathlib.Path(p).unlink(missing_ok=True)

    # -- Source interface -----------------------------------------------------

    def __len__(self) -> int:
        """Return the number of available runs."""
        return len(self._run_indices)

    def run_id(self, index: int) -> int:
        """Return the dataset run ID for the given source index.

        Parameters
        ----------
        index : int
            Zero-based index into the sorted run list.

        Returns
        -------
        int
            The dataset run ID (e.g. 1, 5, 12).
        """
        return self._run_indices[index]

    def mesh_name(self, index: int, seq: int) -> str:
        """Return the canonical output name for mesh at (index, seq).

        Parameters
        ----------
        index : int
            Source index (which run).
        seq : int
            Sequence number within this index (which part).

        Returns
        -------
        str
            Resolved name like ``"domain_1"`` or ``"drivaer_5.stl"``.

        Raises
        ------
        IndexError
            If *seq* is out of range for the active mesh_parts.
        """
        run_id = self._run_indices[index]
        part = self._mesh_parts[seq]
        return _MESH_NAME_TEMPLATES[part].format(run_id=run_id)

    def __getitem__(self, index: int) -> Generator[Mesh | DomainMesh]:  # type: ignore[override]
        """Read the mesh(es) for the *index*-th run.

        For ``"boundary"`` and ``"volume"`` mesh types, yields a single
        :class:`~physicsnemo.mesh.Mesh`.  For ``"slices"``, yields one
        mesh per slice plane file in the run's ``slices/`` directory.
        For ``"multi"`` mesh type, yields a :class:`~physicsnemo.mesh.DomainMesh`
        followed by STL meshes.

        Parameters
        ----------
        index : int
            Zero-based index into the sorted run list.

        Yields
        ------
        Mesh | DomainMesh
            Converted physicsnemo Mesh or DomainMesh object(s).
        """
        if self._mesh_type == "volume":
            yield from self._read_volume(index)
        elif self._mesh_type == "slices":
            yield from self._read_slices(index)
        elif self._mesh_type == "multi":
            yield from self._read_multi(index)
        else:
            if index < -len(self._run_indices) or index >= len(self._run_indices):
                msg = f"Index {index} out of range for source with {len(self._run_indices)} runs."
                raise IndexError(msg)

            run_id = self._run_indices[index]
            filename = self._file_template.format(i=run_id)
            remote_path = f"{self._root_path}/run_{run_id}/{filename}"
            path = self._ensure_local(remote_path)
            mesh = self._read_vtk(path)
            self._cleanup_local(path)
            yield mesh

    # -- VTK reading ----------------------------------------------------------

    def _read_vtk(self, path: str) -> Mesh:
        """Read a single VTK file and convert to Mesh.

        Uses the configured backend (pyvista or rust). The Rust backend
        falls back to PyVista if it cannot parse the file or returns
        empty data.

        Parameters
        ----------
        path : str
            Local filesystem path to a VTK file.

        Returns
        -------
        Mesh
            Converted mesh.
        """
        if self._backend == "rust":
            from tensordict import TensorDict

            from physicsnemo_curator._lib import vtk

            try:
                rust_mesh = vtk.read_vtk(path)
            except OSError:
                logger.debug("Rust VTK reader does not support %s, falling back to pyvista", path)
                return self._read_with_pyvista(path)

            # If Rust parsed the file but returned no usable data, fall back.
            has_data = (
                bool(rust_mesh.point_data)
                or bool(rust_mesh.cell_data)
                or (rust_mesh.cells is not None and rust_mesh.cells.size > 0)
            )
            if rust_mesh.n_points > 0 and not has_data:
                logger.debug("Rust reader returned empty data for %s, falling back to pyvista", path)
                return self._read_with_pyvista(path)

            n_points = rust_mesh.n_points
            n_cells = rust_mesh.n_cells
            points = torch.from_numpy(rust_mesh.points)

            # Cells from connectivity + offsets
            cells = self._build_cells_from_rust(rust_mesh)

            # Point data
            point_data_dict: dict[str, torch.Tensor] = {}
            for name, data in rust_mesh.point_data.items():
                arr = torch.from_numpy(data)
                point_data_dict[name] = arr

            point_data = TensorDict(point_data_dict, batch_size=[n_points]) if point_data_dict else None  # ty: ignore[invalid-argument-type]  # type: ignore[arg-type]

            # Cell data
            cell_data_dict: dict[str, torch.Tensor] = {}
            for name, data in rust_mesh.cell_data.items():
                arr = torch.from_numpy(data)
                cell_data_dict[name] = arr

            cell_data = TensorDict(cell_data_dict, batch_size=[n_cells]) if cell_data_dict else None  # ty: ignore[invalid-argument-type]  # type: ignore[arg-type]

            return Mesh(
                points=points,
                cells=cells,
                point_data=point_data,
                cell_data=cell_data,
            )

        return self._read_with_pyvista(path)

    def _read_with_pyvista(
        self,
        path: str,
        manifold_dim: int | Literal["auto"] | None = None,
        point_source: Literal["vertices", "cell_centroids"] | None = None,
    ) -> Mesh:
        """Read a VTK file using PyVista and convert via from_pyvista.

        Parameters
        ----------
        path : str
            Local filesystem path to a VTK file.
        manifold_dim : int or {"auto"} or None
            Override manifold_dim for this read. None uses instance default.
        point_source : {"vertices", "cell_centroids"} or None
            Override point_source for this read. None uses instance default.

        Returns
        -------
        Mesh
            Converted mesh with manifold_dim/point_source applied.
        """
        import pyvista as pv
        from physicsnemo.mesh.io import from_pyvista

        pv_mesh = pv.read(path)
        return from_pyvista(
            pv_mesh,
            manifold_dim=manifold_dim if manifold_dim is not None else self._manifold_dim,
            point_source=point_source or self._point_source,
            warn_on_lost_data=self._warn_on_lost_data,
        )

    @staticmethod
    def _build_cells_from_rust(rust_mesh: Any) -> torch.Tensor | None:
        """Build a cells tensor from Rust mesh connectivity and offsets.

        Parameters
        ----------
        rust_mesh : VtkMeshData
            Rust VTK mesh object.

        Returns
        -------
        torch.Tensor or None
            Cells tensor of shape ``(n_cells, nodes_per_cell)`` if
            connectivity is available, else ``None``.
        """
        connectivity = rust_mesh.cells
        offsets = rust_mesh.cell_offsets

        if connectivity is None or offsets is None or connectivity.size == 0 or offsets.size == 0:
            return None

        n_cells = rust_mesh.n_cells
        if n_cells == 0:
            return None

        # Determine nodes per cell from offsets
        if offsets.size > 1:
            nodes_per_cell = int(offsets[1] - offsets[0])
        elif connectivity.size > 0:
            nodes_per_cell = connectivity.size // n_cells
        else:
            return None

        # If mixed cell types, cannot form a uniform (n_cells, npc) tensor
        if connectivity.size != n_cells * nodes_per_cell:
            return None

        cells = torch.from_numpy(connectivity.reshape(n_cells, nodes_per_cell)).to(torch.int64)
        return cells

    @staticmethod
    def _tessellate_polygons(
        connectivity: Any,
        offsets: Any,
        n_cells: int,
    ) -> torch.Tensor:
        """Fan-tessellate mixed polygons into triangles.

        Converts arbitrary polygons (3–N vertices) into triangles using
        fan triangulation from the first vertex of each polygon. A polygon
        with *k* vertices produces *k - 2* triangles.

        Parameters
        ----------
        connectivity : numpy.ndarray
            Flat connectivity array (vertex indices for all cells).
        offsets : numpy.ndarray
            Offset into *connectivity* for each cell (length ``n_cells + 1``
            in VTK convention, or ``n_cells`` with implicit 0 start).
        n_cells : int
            Number of polygons.

        Returns
        -------
        torch.Tensor
            Triangle cells of shape ``(n_triangles, 3)`` with dtype int64.
        """
        import numpy as np

        conn = np.asarray(connectivity, dtype=np.int64)
        offs = np.asarray(offsets, dtype=np.int64)

        # VTK offsets: [end0, end1, ...] (cumulative, 1-based)
        # Convert to start/end pairs
        if offs.size == n_cells:
            # Offsets give the *end* of each cell's connectivity
            starts = np.empty(n_cells, dtype=np.int64)
            starts[0] = 0
            starts[1:] = offs[:-1]
            ends = offs
        else:
            # Offset array has n_cells+1 entries: [0, end0, end1, ...]
            starts = offs[:-1]
            ends = offs[1:]

        # Count vertices per cell and total triangles
        n_verts = ends - starts  # vertices per polygon
        n_tris_per_cell = n_verts - 2  # triangles per polygon
        total_tris = int(n_tris_per_cell.sum())

        # Vectorized fan triangulation:
        # For each polygon, fan from vertex 0: tri_j = (v0, v_{j}, v_{j+1})
        # Build index arrays for all triangles simultaneously.

        # Repeat the start offset for each triangle from that polygon
        # cell_of_tri[t] = which polygon triangle t belongs to
        cell_of_tri = np.repeat(np.arange(n_cells, dtype=np.int64), n_tris_per_cell)

        # local_j[t] = which fan triangle within the polygon (0-based)
        # For polygon with k verts -> tris 0..k-3, local_j = 0,1,...,k-3
        local_j = np.arange(total_tris, dtype=np.int64)
        # Subtract cumulative tris-per-cell to get local index
        cum_tris = np.zeros(n_cells + 1, dtype=np.int64)
        np.cumsum(n_tris_per_cell, out=cum_tris[1:])
        local_j -= cum_tris[cell_of_tri]

        # Start of each polygon's connectivity
        poly_starts = starts[cell_of_tri]

        triangles = np.empty((total_tris, 3), dtype=np.int64)
        triangles[:, 0] = conn[poly_starts]  # fan center (vertex 0)
        triangles[:, 1] = conn[poly_starts + local_j + 1]
        triangles[:, 2] = conn[poly_starts + local_j + 2]

        return torch.from_numpy(triangles)

    @staticmethod
    def _expand_cell_data_for_tessellation(
        cell_data_dict: dict[str, torch.Tensor],
        offsets: Any,
        n_cells: int,
    ) -> dict[str, torch.Tensor]:
        """Repeat cell data entries for tessellated triangles.

        When a polygon with *k* vertices is split into *k - 2* triangles,
        the cell data value must be repeated *k - 2* times.

        Parameters
        ----------
        cell_data_dict : dict
            Mapping of field name to tensor of shape ``(n_cells, ...)``.
        offsets : numpy.ndarray
            Cell offsets (same convention as ``_tessellate_polygons``).
        n_cells : int
            Number of original polygons.

        Returns
        -------
        dict
            Expanded cell data with shape ``(n_triangles, ...)``.
        """
        import numpy as np

        offs = np.asarray(offsets, dtype=np.int64)

        if offs.size == n_cells:
            starts = np.empty(n_cells, dtype=np.int64)
            starts[0] = 0
            starts[1:] = offs[:-1]
            ends = offs
        else:
            starts = offs[:-1]
            ends = offs[1:]

        n_verts = ends - starts
        n_tris_per_cell = (n_verts - 2).astype(np.int64)

        # Build repeat indices: each cell i is repeated n_tris_per_cell[i] times
        repeat_counts = torch.from_numpy(n_tris_per_cell)

        expanded = {}
        for name, tensor in cell_data_dict.items():
            expanded[name] = torch.repeat_interleave(tensor, repeat_counts, dim=0)

        return expanded

    # -- Volume reading -------------------------------------------------------

    def _read_volume(self, index: int) -> Generator[Mesh]:
        """Read a volume VTU that may be split across multiple part files.

        DrivAerML volume files are split as ``volume_{i}.vtu.00.part``,
        ``volume_{i}.vtu.01.part``.  This method concatenates the parts
        in memory before reading.

        Parameters
        ----------
        index : int
            Zero-based index into the sorted run list.

        Yields
        ------
        Mesh
            The reconstructed volume mesh.
        """
        run_id = self._run_indices[index]

        # Try direct VTU first; fall back to concatenating parts.
        direct_remote = f"{self._root_path}/run_{run_id}/volume_{run_id}.vtu"
        try:
            local_path = self._ensure_local(direct_remote)
            mesh = self._read_vtk(local_path)
            self._cleanup_local(local_path)
            yield mesh
            return
        except FileNotFoundError:
            pass

        # Concatenate split parts into a temp file and read.
        volume_path = self._concat_volume_parts_tempfile(run_id)
        try:
            mesh = self._read_vtk(volume_path)
        finally:
            pathlib.Path(volume_path).unlink(missing_ok=True)
        yield mesh

    def _concat_volume_parts_tempfile(self, run_id: int) -> str:
        """Concatenate split volume VTU parts into a temporary file.

        Creates a :func:`tempfile.NamedTemporaryFile` (``delete=False``)
        and streams each part into it.  The caller is responsible for
        deleting the file after use.

        Parameters
        ----------
        run_id : int
            The dataset run ID.

        Returns
        -------
        str
            Path to the temporary concatenated VTU file.

        Raises
        ------
        FileNotFoundError
            If no volume part files are found.
        """
        import tempfile

        cleanup_parts: list[str] = []
        try:
            part_paths: list[str] = []
            for part_idx in range(10):
                part_name = f"volume_{run_id}.vtu.{part_idx:02d}.part"
                remote = f"{self._root_path}/run_{run_id}/{part_name}"
                try:
                    local = self._ensure_local(remote)
                    if not pathlib.Path(local).exists():
                        break
                    cleanup_parts.append(local)
                    part_paths.append(local)
                except (FileNotFoundError, OSError):
                    break

            if not part_paths:
                msg = f"No volume files found for run_{run_id}"
                raise FileNotFoundError(msg)

            tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115
                suffix=".vtu",
                dir=pathlib.Path(part_paths[0]).parent,
                delete=False,
            )
            try:
                for part in part_paths:
                    with pathlib.Path(part).open("rb") as inp:
                        while True:
                            chunk = inp.read(64 * 1024 * 1024)  # 64 MB
                            if not chunk:
                                break
                            tmp.write(chunk)
                tmp.close()
            except BaseException:
                tmp.close()
                pathlib.Path(tmp.name).unlink(missing_ok=True)
                raise

            logger.info(
                "Concatenated %d parts for run_%d into %s",
                len(part_paths),
                run_id,
                tmp.name,
            )
            return tmp.name
        finally:
            self._cleanup_local(*cleanup_parts)

    # -- Slices reading -------------------------------------------------------

    def _build_slices_data(self) -> None:
        """Discover per-run slice files for slices mode.

        Populates ``self._slices_run_data`` — a list of
        ``(fsspec_fs, protocol, list[str])`` tuples, one per run.
        """
        import fsspec

        self._slices_run_data: list[tuple[Any, str, list[str]]] = []
        for run_id in self._run_indices:
            slice_url = f"{self._url}/run_{run_id}/slices"
            fs, root_path = fsspec.core.url_to_fs(slice_url, **self._storage_options)
            protocol = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0]

            glob_expr = f"{root_path}/**"
            all_files = fs.glob(glob_expr)
            files = sorted(
                f for f in all_files if pathlib.PurePosixPath(f).suffix.lower() in {".vtp"} and not fs.isdir(f)
            )
            if not files:
                msg = f"No slice files found at {slice_url}."
                raise ValueError(msg)

            self._slices_run_data.append((fs, protocol, files))

    def _read_slices(self, index: int) -> Generator[Mesh]:
        """Read all slice VTP files for a given run index.

        Parameters
        ----------
        index : int
            Zero-based index into the sorted run list.

        Yields
        ------
        Mesh
            One mesh per slice plane file.
        """
        fs, protocol, files = self._slices_run_data[index]
        for remote_path in files:
            local_path = self._ensure_local(remote_path, fs=fs, protocol=protocol)
            mesh = self._read_vtk(local_path)
            self._cleanup_local(local_path)
            yield mesh

    # -- Multi-mode methods ---------------------------------------------------

    def _read_multi(self, index: int) -> Generator[Mesh | DomainMesh]:
        """Read multiple mesh representations for a given run.

        Parameters
        ----------
        index : int
            Zero-based index into the sorted run list.

        Yields
        ------
        Mesh or DomainMesh
            One mesh per active mesh_part, in order.  The ``"domain"`` part
            yields a :class:`DomainMesh`; others yield plain :class:`Mesh`.
        """
        for part in self._mesh_parts:
            if part == "domain":
                yield from self._read_domain(index)
            elif part == "stl":
                yield from self._read_stl(index)
            elif part == "single_solid":
                yield from self._read_single_solid(index)

    def _read_domain(self, index: int) -> Generator[DomainMesh]:
        """Read domain mesh combining volume interior, boundary surface, and global data.

        Produces a :class:`~physicsnemo.mesh.domain_mesh.DomainMesh` with:

        * **interior** — volume VTU converted to a point-cloud via
          ``cell_centroids`` (no connectivity, cells shape ``[0, 1]``).
        * **boundaries["surface"]** — boundary VTP with cell connectivity
          and cell data fields only (point data on surface meshes is not
          extracted; cell data is computed from vertex data if needed).
        * **global_data** — ``U_inf = [30, 0, 0]`` and ``rho_inf = 1.225``.

        All floating-point data is downcast to float32.

        Parameters
        ----------
        index : int
            Zero-based index into the sorted run list.

        Yields
        ------
        DomainMesh
            Combined domain mesh.
        """
        run_id = self._run_indices[index]

        # --- Interior (volume VTU → point-cloud) ---
        interior = self._read_interior(run_id)
        interior = self._downcast_fp32(interior)

        # --- Boundary surface (VTP → surface mesh with cell data only) ---
        boundary_remote = f"{self._root_path}/run_{run_id}/boundary_{run_id}.vtp"
        boundary_path = self._ensure_local(boundary_remote)
        surface = self._read_surface(boundary_path)
        self._cleanup_local(boundary_path)
        surface = self._downcast_fp32(surface)

        # --- Global data ---
        global_data = {
            "U_inf": torch.tensor([30.0, 0.0, 0.0], dtype=torch.float32),
            "rho_inf": torch.tensor(1.225, dtype=torch.float32),
        }

        # --- Assemble DomainMesh ---
        domain_mesh = DomainMesh(
            interior=interior,
            boundaries={"surface": surface},
            global_data=global_data,
        )
        yield domain_mesh

    def _read_interior(self, run_id: int) -> Mesh:
        """Read a volume VTK file as an interior point-cloud.

        Converts the volume mesh to a point-cloud using cell centroids
        (no connectivity). For the Rust backend, reads raw data and
        computes cell centroids from connectivity arrays. For the PyVista
        backend, uses ``cell_centroids`` point source with ``manifold_dim=0``.

        Parameters
        ----------
        run_id : int
            The dataset run ID.

        Returns
        -------
        Mesh
            Point-cloud mesh (no cell connectivity).
        """
        # Resolve volume file — direct VTU or concatenated parts
        direct_remote = f"{self._root_path}/run_{run_id}/volume_{run_id}.vtu"
        volume_path: str | None = None
        cleanup_paths: list[str] = []
        temp_concat = False
        try:
            volume_path = self._ensure_local(direct_remote)
            cleanup_paths.append(volume_path)
        except (FileNotFoundError, OSError):
            volume_path = self._concat_volume_parts_tempfile(run_id)
            temp_concat = True

        try:
            if self._backend == "rust":
                try:
                    rust_mesh = self._read_rust_interior(volume_path)
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "Rust VTK reader failed for volume run_%d, falling back to pyvista",
                        run_id,
                    )
                else:
                    has_data = (
                        bool(rust_mesh.cell_data)  # ty: ignore[unresolved-attribute]
                        or (rust_mesh.cells is not None and rust_mesh.cells.size > 0)  # ty: ignore[unresolved-attribute]
                    )
                    if has_data:
                        return self._build_centroid_mesh(rust_mesh)
                    logger.debug(
                        "Rust reader returned empty data for volume run_%d, falling back to pyvista",
                        run_id,
                    )

            # PyVista backend (or Rust fallback)
            return self._read_with_pyvista(volume_path, manifold_dim=0, point_source="cell_centroids")
        finally:
            if temp_concat:
                pathlib.Path(volume_path).unlink(missing_ok=True)
            else:
                self._cleanup_local(*cleanup_paths)

    def _read_rust_interior(
        self,
        volume_path: str,
    ) -> object:
        """Read volume VTU via the Rust backend.

        Only reads cell topology and cell data — point data fields are
        skipped since the interior mesh uses cell centroids.

        Parameters
        ----------
        volume_path : str
            Local path to a VTU file.

        Returns
        -------
        VtkMeshData
            The Rust VTK mesh handle.
        """
        from physicsnemo_curator._lib import vtk

        return vtk.read_vtk(volume_path, skip_point_data=True)

    def _build_centroid_mesh(self, rust_mesh: object) -> Mesh:
        """Convert a Rust VtkMeshData into a centroid point-cloud Mesh.

        Computes cell centroids from the cell connectivity and promotes
        cell data to point data (one datum per centroid).

        Parameters
        ----------
        rust_mesh : VtkMeshData
            Parsed VTK data from the Rust reader.

        Returns
        -------
        Mesh
            Point-cloud mesh (``cells=None``).
        """
        from tensordict import TensorDict

        n_cells = rust_mesh.n_cells  # ty: ignore[unresolved-attribute]
        connectivity = rust_mesh.cells  # ty: ignore[unresolved-attribute]

        points_raw = torch.from_numpy(
            rust_mesh.points  # ty: ignore[unresolved-attribute]
        )
        offsets = rust_mesh.cell_offsets  # ty: ignore[unresolved-attribute]

        if (
            connectivity is not None
            and connectivity.size > 0
            and offsets is not None
            and offsets.size > 1
            and n_cells > 0
        ):
            import numpy as np

            # Offsets are cumulative node counts: cell i spans
            # connectivity[starts[i]:starts[i+1]].
            off = np.empty(n_cells + 1, dtype=np.int64)
            off[0] = 0
            off[1:] = offsets

            # Check if all cells have the same node count (uniform mesh)
            nodes_first = int(off[1])
            if connectivity.size == n_cells * nodes_first:
                # Uniform cell type — fast vectorized path
                conn = torch.from_numpy(connectivity.reshape(n_cells, nodes_first)).to(torch.int64)
                cell_points = points_raw[conn]  # (n_cells, nodes_per_cell, 3)
                centroids = cell_points.mean(dim=1)  # (n_cells, 3)
            else:
                # Mixed cell types — vectorized scatter-add approach
                conn_t = torch.from_numpy(connectivity).to(torch.int64)
                # Gather all referenced points
                all_pts = points_raw[conn_t]  # (total_nodes, 3)
                # Build cell index for each node in connectivity
                nodes_per_cell = np.diff(off)  # (n_cells,)
                cell_ids = np.repeat(np.arange(n_cells, dtype=np.int64), nodes_per_cell)
                cell_ids_t = torch.from_numpy(cell_ids).unsqueeze(1).expand(-1, 3)
                # Sum points per cell
                centroids = torch.zeros(n_cells, 3, dtype=points_raw.dtype)
                centroids.scatter_add_(0, cell_ids_t, all_pts)
                # Divide by node count per cell
                npc_t = torch.from_numpy(nodes_per_cell.astype(np.float64)).to(points_raw.dtype)
                centroids /= npc_t.unsqueeze(1)
        else:
            centroids = points_raw

        # Cell data becomes point_data for the centroid point-cloud
        point_data_dict: dict[str, torch.Tensor] = {}
        for name, data in rust_mesh.cell_data.items():  # ty: ignore[unresolved-attribute]
            arr = torch.from_numpy(data)
            point_data_dict[name] = arr

        n_pts = centroids.shape[0]
        point_data = (
            TensorDict(point_data_dict, batch_size=[n_pts])  # ty: ignore[invalid-argument-type]  # type: ignore[arg-type]
            if point_data_dict
            else None
        )

        return Mesh(points=centroids, cells=None, point_data=point_data)

    def _read_surface(self, path: str) -> Mesh:
        """Read a boundary VTP file as a surface mesh with cell data only.

        For surface meshes, only cell data is extracted and processed.
        Point data fields on the surface are not included — if cell-level
        quantities need to be derived from vertex data, the physicsnemo
        mesh APIs should be used downstream.

        Parameters
        ----------
        path : str
            Local path to the boundary VTP file.

        Returns
        -------
        Mesh
            Surface mesh with cell connectivity and cell data.
        """
        if self._backend == "rust":
            from tensordict import TensorDict

            from physicsnemo_curator._lib import vtk

            try:
                rust_mesh = vtk.read_vtk(path, skip_point_data=True)
            except OSError:
                logger.debug("Rust VTK reader does not support %s, falling back to pyvista", path)
                return self._read_surface_pyvista(path)

            has_data = (
                bool(rust_mesh.point_data)
                or bool(rust_mesh.cell_data)
                or (rust_mesh.cells is not None and rust_mesh.cells.size > 0)
            )
            if rust_mesh.n_points > 0 and not has_data:
                logger.debug("Rust reader returned empty data for %s, falling back to pyvista", path)
                return self._read_surface_pyvista(path)

            n_cells = rust_mesh.n_cells
            points = torch.from_numpy(rust_mesh.points)
            cells = self._build_cells_from_rust(rust_mesh)

            # Only extract cell data for surface meshes
            cell_data_dict: dict[str, torch.Tensor] = {}
            for name, data in rust_mesh.cell_data.items():
                arr = torch.from_numpy(data)
                cell_data_dict[name] = arr

            # For mixed cell types, tessellate polygons into triangles
            if cells is None and n_cells > 0:
                connectivity = rust_mesh.cells
                offsets = rust_mesh.cell_offsets
                if connectivity is not None and offsets is not None and connectivity.size > 0:
                    cells = self._tessellate_polygons(connectivity, offsets, n_cells)
                    cell_data_dict = self._expand_cell_data_for_tessellation(cell_data_dict, offsets, n_cells)
                    n_cells = cells.shape[0]
                else:
                    # Fallback: no connectivity available
                    cells = torch.arange(n_cells, dtype=torch.int64).unsqueeze(1)

            cell_data = TensorDict(cell_data_dict, batch_size=[n_cells]) if cell_data_dict else None  # ty: ignore[invalid-argument-type]  # type: ignore[arg-type]

            return Mesh(points=points, cells=cells, point_data=None, cell_data=cell_data)

        return self._read_surface_pyvista(path)

    def _read_surface_pyvista(self, path: str) -> Mesh:
        """Read surface VTP via PyVista, retaining only cell data.

        Parameters
        ----------
        path : str
            Local path to the boundary VTP file.

        Returns
        -------
        Mesh
            Surface mesh with cell connectivity and cell data only.
        """
        import pyvista as pv
        from physicsnemo.mesh.io import from_pyvista

        pv_mesh = pv.read(path)
        mesh = from_pyvista(
            pv_mesh,
            manifold_dim="auto",
            point_source="vertices",
            warn_on_lost_data=self._warn_on_lost_data,
        )
        # Strip point_data — surface meshes should only carry cell data
        mesh.point_data = None
        return mesh

    def _read_stl(self, index: int) -> Generator[Mesh]:
        """Read the STL geometry mesh, downcast to fp32.

        Parameters
        ----------
        index : int
            Zero-based index into the sorted run list.

        Yields
        ------
        Mesh
            STL geometry mesh with triangle connectivity, downcast to fp32.
        """
        run_id = self._run_indices[index]
        filename = f"drivaer_{run_id}.stl"
        remote_path = f"{self._root_path}/run_{run_id}/{filename}"
        local_path = self._ensure_local(remote_path)

        mesh = self._read_vtk(local_path)
        self._cleanup_local(local_path)
        yield self._downcast_fp32(mesh)

    def _read_single_solid(self, index: int) -> Generator[Mesh]:
        """Read STL geometry merged into a single solid, downcast to fp32.

        Parameters
        ----------
        index : int
            Zero-based index into the sorted run list.

        Yields
        ------
        Mesh
            Merged STL geometry mesh, downcast to fp32.
        """
        run_id = self._run_indices[index]
        filename = f"drivaer_{run_id}.stl"
        remote_path = f"{self._root_path}/run_{run_id}/{filename}"
        local_path = self._ensure_local(remote_path)

        if self._backend == "rust":
            # Rust reads STL as a single solid already
            mesh = self._read_vtk(local_path)
        else:
            import pyvista as pv
            from physicsnemo.mesh.io import from_pyvista

            pv_mesh = pv.read(local_path)  # type: ignore[arg-type]
            merged = self._merge_to_single_solid(pv_mesh)  # type: ignore[arg-type]
            mesh = from_pyvista(
                merged,
                manifold_dim="auto",
                point_source="vertices",
                warn_on_lost_data=self._warn_on_lost_data,
            )

        self._cleanup_local(local_path)
        yield self._downcast_fp32(mesh)

    # -- Utilities ------------------------------------------------------------

    @staticmethod
    def _downcast_fp32(mesh: Mesh) -> Mesh:
        """Downcast float64 tensors in a Mesh to float32 in-place.

        Parameters
        ----------
        mesh : Mesh
            Input mesh (modified in-place).

        Returns
        -------
        Mesh
            The same mesh with float64 arrays converted to float32.
        """
        if mesh.points.dtype == torch.float64:
            mesh.points = mesh.points.float()

        if mesh.point_data is not None:
            for key in list(mesh.point_data.keys()):  # noqa: SIM118 - TensorDict needs .keys()
                tensor = mesh.point_data[key]
                if tensor.dtype == torch.float64:
                    mesh.point_data[key] = tensor.float()

        if mesh.cell_data is not None:
            for key in list(mesh.cell_data.keys()):  # noqa: SIM118 - TensorDict needs .keys()
                tensor = mesh.cell_data[key]
                if tensor.dtype == torch.float64:
                    mesh.cell_data[key] = tensor.float()

        return mesh

    @staticmethod
    def _merge_to_single_solid(pv_mesh: Any) -> Any:
        """Merge all blocks/cells into a single contiguous PolyData.

        For a PolyData that is already a single block, this is effectively
        a no-op (returns as-is).  For MultiBlock datasets, concatenates
        all blocks into one PolyData.

        Parameters
        ----------
        pv_mesh : pv.PolyData or pv.MultiBlock
            Input PyVista mesh.

        Returns
        -------
        pv.PolyData
            A single merged PolyData with all cells.
        """
        import pyvista as pv

        if isinstance(pv_mesh, pv.MultiBlock):
            combined = pv_mesh.combine()
            if isinstance(combined, pv.UnstructuredGrid):
                return combined.extract_surface()
            return combined
        # Already a single PolyData — return as-is.
        return pv_mesh
