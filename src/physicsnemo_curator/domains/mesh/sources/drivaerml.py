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

import pyvista as pv
import torch
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.domain_mesh import DomainMesh
from physicsnemo.mesh.io import from_pyvista

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
Backend = Literal["python", "rust"]

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
    manifold_dim : int or {"auto"}
        Target manifold dimension for ``from_pyvista`` conversion.
    point_source : {"vertices", "cell_centroids"}
        Point source mode for ``from_pyvista`` conversion.
    warn_on_lost_data : bool
        Warn when data arrays are discarded during conversion.
    mesh_parts : list[str] or None
        Which mesh parts to yield in ``"multi"`` mode.  Valid parts are
        ``"domain"``, ``"stl"``, and ``"single_solid"``.  When ``None``
        (the default), all three parts are yielded.  Ignored for other
        mesh types.
    backend : {"python", "rust"}
        VTK reading backend:

        - ``"python"`` (default): uses PyVista + ``from_pyvista`` for full
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
                description="VTK reading backend: python (PyVista, default) or rust (native, faster)",
                type=str,
                default="python",
                choices=["python", "rust"],
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
        backend: Backend = "python",
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
            self._file_template = "boundary_{i}.vtp"  # dummy, not used for slices
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

    def _ensure_local(self, remote_path: str) -> str:
        """Download *remote_path* to the local cache if not already present.

        Parameters
        ----------
        remote_path : str
            Filesystem path (no protocol prefix).

        Returns
        -------
        str
            Local filesystem path.
        """
        if self._protocol in ("file", ""):
            return remote_path

        local_path = pathlib.Path(self._cache_storage) / remote_path.lstrip("/")
        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self._fs.get(remote_path, str(local_path))

        return str(local_path)

    def _ensure_local_with_fs(self, fs: Any, protocol: str, remote_path: str) -> str:
        """Download *remote_path* using the given filesystem.

        Parameters
        ----------
        fs : Any
            fsspec filesystem instance.
        protocol : str
            Filesystem protocol string.
        remote_path : str
            Remote path to download.

        Returns
        -------
        str
            Local filesystem path.
        """
        if protocol in ("file", ""):
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

    def __getitem__(self, index: int) -> Generator[Mesh | DomainMesh]:  # type: ignore[override]  # ty: ignore[invalid-method-override]
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

    # -- Internal helpers ----------------------------------------------------

    def _read_vtk(self, path: str) -> Mesh:
        """Read a single VTK file and convert to Mesh using the configured backend.

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
            return self._read_vtk_rust(path)
        return self._read_vtk_python(path)

    def _read_vtk_python(self, path: str) -> Mesh:
        """Read a VTK file using PyVista and convert via from_pyvista.

        Parameters
        ----------
        path : str
            Local filesystem path to a VTK file.

        Returns
        -------
        Mesh
            Converted mesh with manifold_dim/point_source applied.
        """
        pv_mesh = pv.read(path)
        return from_pyvista(
            pv_mesh,
            manifold_dim=self._manifold_dim,
            point_source=self._point_source,
            warn_on_lost_data=self._warn_on_lost_data,
        )

    def _read_vtk_rust(self, path: str) -> Mesh:
        """Read a VTK file using the native Rust backend.

        Constructs a :class:`Mesh` directly from the raw arrays returned
        by the Rust VTK reader, including points, cells (connectivity),
        point_data, and cell_data.

        Falls back to the Python backend if the Rust reader does not
        support the file format (e.g. STL).

        Parameters
        ----------
        path : str
            Local filesystem path to a VTK file.

        Returns
        -------
        Mesh
            Mesh with all available data from the file.
        """
        import numpy as np
        from tensordict import TensorDict

        from physicsnemo_curator._lib import vtk

        try:
            rust_mesh = vtk.read_vtk(path)
        except OSError:
            logger.debug("Rust VTK reader does not support %s, falling back to Python", path)
            return self._read_vtk_python(path)

        # If Rust reader parsed the file but returned no usable data arrays
        # (common with binary VTK formats), fall back to Python for full parse.
        has_data = bool(rust_mesh.point_data()) or bool(rust_mesh.cell_data()) or rust_mesh.connectivity().size > 0
        if rust_mesh.n_points > 0 and not has_data:
            logger.debug("Rust reader returned empty data for %s, falling back to Python", path)
            return self._read_vtk_python(path)

        points = torch.from_numpy(rust_mesh.points()[: rust_mesh.n_points * 3].reshape(-1, 3))

        # Cells from connectivity + offsets
        cells = self._build_cells_from_rust(rust_mesh)

        n_points = rust_mesh.n_points
        n_cells = rust_mesh.n_cells

        # Point data (truncate to n_points * num_components to handle Rust reader buffer overruns)
        point_data_dict: dict[str, torch.Tensor] = {}
        for name, (data, num_components) in rust_mesh.point_data().items():
            arr = torch.from_numpy(np.asarray(data)[: n_points * num_components])
            if num_components > 1:
                arr = arr.reshape(-1, num_components)
            point_data_dict[name] = arr

        point_data = TensorDict(point_data_dict, batch_size=[n_points]) if point_data_dict else None  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

        # Cell data (truncate to n_cells * num_components)
        cell_data_dict: dict[str, torch.Tensor] = {}
        for name, (data, num_components) in rust_mesh.cell_data().items():
            arr = torch.from_numpy(np.asarray(data)[: n_cells * num_components])
            if num_components > 1:
                arr = arr.reshape(-1, num_components)
            cell_data_dict[name] = arr

        cell_data = TensorDict(cell_data_dict, batch_size=[n_cells]) if cell_data_dict else None  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

        return Mesh(
            points=points,
            cells=cells,
            point_data=point_data,
            cell_data=cell_data,
        )

    @staticmethod
    def _build_cells_from_rust(rust_mesh: Any) -> torch.Tensor | None:
        """Build a cells tensor from Rust mesh connectivity and offsets.

        Parameters
        ----------
        rust_mesh : VTKMesh
            Rust VTK mesh object.

        Returns
        -------
        torch.Tensor or None
            Cells tensor of shape ``(n_cells, nodes_per_cell)`` if
            connectivity is available, else ``None``.
        """
        connectivity = rust_mesh.connectivity()
        offsets = rust_mesh.offsets()

        if connectivity.size == 0 or offsets.size == 0:
            return None

        n_cells = rust_mesh.n_cells
        # Determine nodes per cell from offsets (assumes uniform cell type)
        if n_cells > 0 and offsets.size > 1:
            nodes_per_cell = int(offsets[1] - offsets[0])
        elif n_cells > 0 and connectivity.size > 0:
            nodes_per_cell = connectivity.size // n_cells
        else:
            return None

        cells = torch.from_numpy(connectivity.reshape(n_cells, nodes_per_cell)).to(torch.int64)
        return cells

    def _read_volume(self, index: int) -> Generator[Mesh]:
        """Read a volume VTU that may be split across multiple part files.

        DrivAerML volume files are split as ``volume_{i}.vtu.00.part``,
        ``volume_{i}.vtu.01.part``.  This method concatenates the parts
        before reading.

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

        # Concatenate split parts.
        parts: list[str] = []
        for part_idx in range(10):  # DrivAerML uses 2 parts typically
            part_name = f"volume_{run_id}.vtu.{part_idx:02d}.part"
            remote = f"{self._root_path}/run_{run_id}/{part_name}"
            try:
                parts.append(self._ensure_local(remote))
            except (FileNotFoundError, OSError):
                break

        if not parts:
            msg = f"No volume files found for run_{run_id}"
            raise FileNotFoundError(msg)

        # Concatenate parts into a single VTU file.
        concat_path = pathlib.Path(self._cache_storage) / f"volume_{run_id}_concat.vtu"
        if not concat_path.exists():
            with concat_path.open("wb") as out:
                for part in parts:
                    with pathlib.Path(part).open("rb") as inp:
                        out.write(inp.read())
            logger.info("Concatenated %d parts into %s", len(parts), concat_path)

        mesh = self._read_vtk(str(concat_path))
        self._cleanup_local(*parts, str(concat_path))
        yield mesh

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
            local_path = self._ensure_local_with_fs(fs, protocol, remote_path)
            mesh = self._read_vtk(local_path)
            self._cleanup_local(local_path)
            yield mesh

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
        * **boundaries["surface"]** — boundary VTP converted with
          ``vertices`` retaining cell connectivity and cell data fields.
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
        direct_remote = f"{self._root_path}/run_{run_id}/volume_{run_id}.vtu"
        volume_cleanup: list[str] = []
        try:
            volume_path = self._ensure_local(direct_remote)
            volume_cleanup.append(volume_path)
            interior = self._read_interior(volume_path)
        except (FileNotFoundError, OSError):
            volume_path = self._read_volume_parts(run_id)
            volume_cleanup.append(volume_path)
            interior = self._read_interior(volume_path)
        self._cleanup_local(*volume_cleanup)
        interior = self._downcast_fp32(interior)

        # --- Boundary surface (boundary VTP → triangulated mesh) ---
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

    def _read_interior(self, path: str) -> Mesh:
        """Read a volume VTK file as an interior point-cloud.

        For the Python backend, uses ``cell_centroids`` and ``manifold_dim=0``
        to produce a point-cloud without connectivity.  For the Rust backend,
        reads raw data and computes cell centroids from connectivity.

        Parameters
        ----------
        path : str
            Local path to the volume VTU file.

        Returns
        -------
        Mesh
            Point-cloud mesh (no cell connectivity).
        """
        if self._backend == "rust":
            return self._read_interior_rust(path)

        pv_volume = pv.read(path)
        return from_pyvista(
            pv_volume,
            manifold_dim=0,
            point_source="cell_centroids",
            warn_on_lost_data=self._warn_on_lost_data,
        )

    def _read_interior_rust(self, path: str) -> Mesh:
        """Read volume VTU via Rust and compute cell centroids.

        Falls back to the Python backend if the Rust reader does not
        support the file or fails to parse connectivity/data arrays.

        Parameters
        ----------
        path : str
            Local path to the volume VTU file.

        Returns
        -------
        Mesh
            Point-cloud mesh with cell-centroid points and cell_data
            promoted to point_data.
        """
        import numpy as np
        from tensordict import TensorDict

        from physicsnemo_curator._lib import vtk

        try:
            rust_mesh = vtk.read_vtk(path)
        except OSError:
            logger.debug("Rust VTK reader does not support %s, falling back to Python", path)
            pv_volume = pv.read(path)
            return from_pyvista(
                pv_volume,
                manifold_dim=0,
                point_source="cell_centroids",
                warn_on_lost_data=self._warn_on_lost_data,
            )

        points_raw = torch.from_numpy(rust_mesh.points()[: rust_mesh.n_points * 3].reshape(-1, 3))
        connectivity = rust_mesh.connectivity()
        offsets = rust_mesh.offsets()

        n_cells = rust_mesh.n_cells

        # If Rust reader returned empty data arrays, fall back to Python
        if n_cells > 0 and connectivity.size == 0:
            logger.debug("Rust reader returned empty connectivity for %s, falling back to Python", path)
            pv_volume = pv.read(path)
            return from_pyvista(
                pv_volume,
                manifold_dim=0,
                point_source="cell_centroids",
                warn_on_lost_data=self._warn_on_lost_data,
            )

        if connectivity.size > 0 and offsets.size > 1 and n_cells > 0:
            # Compute cell centroids from connectivity
            nodes_per_cell = int(offsets[1] - offsets[0])
            conn = torch.from_numpy(connectivity.reshape(n_cells, nodes_per_cell)).to(torch.int64)
            # Gather points for each cell and average
            cell_points = points_raw[conn]  # (n_cells, nodes_per_cell, 3)
            centroids = cell_points.mean(dim=1)  # (n_cells, 3)
        else:
            # Fallback: treat as point-cloud
            centroids = points_raw

        # Cell data becomes point_data for centroids
        point_data_dict: dict[str, torch.Tensor] = {}
        for name, (data, num_components) in rust_mesh.cell_data().items():
            arr = torch.from_numpy(np.asarray(data)[: n_cells * num_components])
            if num_components > 1:
                arr = arr.reshape(-1, num_components)
            point_data_dict[name] = arr

        # Also include point_data if present (unlikely for cell_centroids mode)
        n_points = rust_mesh.n_points
        for name, (data, num_components) in rust_mesh.point_data().items():
            arr = torch.from_numpy(np.asarray(data)[: n_points * num_components])
            if num_components > 1:
                arr = arr.reshape(-1, num_components)
            point_data_dict[name] = arr

        n_pts = centroids.shape[0]
        point_data = TensorDict(point_data_dict, batch_size=[n_pts]) if point_data_dict else None  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

        return Mesh(
            points=centroids,
            cells=None,
            point_data=point_data,
        )

    def _read_surface(self, path: str) -> Mesh:
        """Read a boundary VTP file as a surface mesh preserving connectivity.

        Parameters
        ----------
        path : str
            Local path to the boundary VTP file.

        Returns
        -------
        Mesh
            Surface mesh with cell connectivity and cell/point data.
        """
        if self._backend == "rust":
            return self._read_vtk_rust(path)

        pv_boundary = pv.read(path)
        return from_pyvista(
            pv_boundary,
            manifold_dim="auto",
            point_source="vertices",
            warn_on_lost_data=self._warn_on_lost_data,
        )

    def _read_stl(self, index: int) -> Generator[Mesh]:
        """Read the STL geometry mesh with vertices preserved, downcast to fp32.

        Parameters
        ----------
        index : int
            Zero-based index into the sorted run list.

        Yields
        ------
        Mesh
            STL geometry mesh with triangle connectivity and downcast to fp32.
        """
        run_id = self._run_indices[index]
        filename = f"drivaer_{run_id}.stl"
        remote_path = f"{self._root_path}/run_{run_id}/{filename}"
        local_path = self._ensure_local(remote_path)

        mesh = self._read_vtk(local_path)
        yield self._downcast_fp32(mesh)

    def _read_single_solid(self, index: int) -> Generator[Mesh]:
        """Read STL geometry merged into a single PolyData, with vertices, fp32.

        Parameters
        ----------
        index : int
            Zero-based index into the sorted run list.

        Yields
        ------
        Mesh
            Merged STL geometry mesh with vertices and downcast to fp32.
        """
        run_id = self._run_indices[index]
        filename = f"drivaer_{run_id}.stl"
        remote_path = f"{self._root_path}/run_{run_id}/{filename}"
        local_path = self._ensure_local(remote_path)

        if self._backend == "rust":
            # Rust backend: read directly (STL is single solid already)
            mesh = self._read_vtk_rust(local_path)
        else:
            pv_mesh = pv.read(local_path)  # type: ignore[arg-type]
            merged = self._merge_to_single_solid(pv_mesh)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            mesh = from_pyvista(
                merged,
                manifold_dim="auto",
                point_source="vertices",
                warn_on_lost_data=self._warn_on_lost_data,
            )
        yield self._downcast_fp32(mesh)

    def _read_volume_parts(self, run_id: int) -> str:
        """Concatenate split volume VTU parts into a single local file.

        Parameters
        ----------
        run_id : int
            The dataset run ID.

        Returns
        -------
        str
            Local path to the concatenated VTU file.

        Raises
        ------
        FileNotFoundError
            If no volume part files are found.
        """
        parts: list[str] = []
        for part_idx in range(10):
            part_name = f"volume_{run_id}.vtu.{part_idx:02d}.part"
            remote = f"{self._root_path}/run_{run_id}/{part_name}"
            try:
                local = self._ensure_local(remote)
                if not pathlib.Path(local).exists():
                    break
                parts.append(local)
            except (FileNotFoundError, OSError):
                break

        if not parts:
            msg = f"No volume files found for run_{run_id}"
            raise FileNotFoundError(msg)

        concat_path = pathlib.Path(self._cache_storage) / f"volume_{run_id}_concat.vtu"
        if not concat_path.exists():
            with concat_path.open("wb") as out:
                for part in parts:
                    with pathlib.Path(part).open("rb") as inp:
                        out.write(inp.read())
            logger.info("Concatenated %d parts into %s", len(parts), concat_path)

        return str(concat_path)

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
    def _merge_to_single_solid(pv_mesh: pv.PolyData | pv.MultiBlock) -> pv.PolyData:
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
        if isinstance(pv_mesh, pv.MultiBlock):
            combined = pv_mesh.combine()
            if isinstance(combined, pv.UnstructuredGrid):
                return combined.extract_surface()
            return combined
        # Already a single PolyData — return as-is.
        return pv_mesh
