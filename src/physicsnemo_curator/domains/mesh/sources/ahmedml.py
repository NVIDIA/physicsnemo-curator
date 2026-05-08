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

"""AhmedML dataset source for mesh pipelines.

Reads the `AhmedML <https://huggingface.co/datasets/neashton/ahmedml>`_
dataset — 500 geometric variations of the Ahmed Car Body with transient
hybrid RANS-LES CFD (OpenFOAM v2212, ~20 M cells per case).

The dataset provides three mesh types per run:

* **boundary** — surface mesh with flow fields (VTP, ~83 MB each)
* **volume** — volumetric field data (VTU, ~5.6 GB each)
* **slices** — x/y/z-normal slice planes with flow fields (VTP)
* **multi** — domain mesh combining interior + boundary + STL with global data

Each run also includes CSV metadata (force/moment coefficients and
geometric parameters) which is attached as ``global_data`` on every
yielded mesh, regardless of mesh type.

File discovery and caching are handled internally using ``fsspec``.
"""

from __future__ import annotations

import csv
import pathlib
import re
import tempfile
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from physicsnemo.mesh import Mesh

from physicsnemo_curator.core.base import Param, Source

if TYPE_CHECKING:
    from collections.abc import Generator

#: HuggingFace Hub URL for the AhmedML dataset.
_AHMEDML_HF_URL = "hf://datasets/neashton/ahmedml"

#: Mesh type → file template mapping.
_MESH_TEMPLATES: dict[str, str] = {
    "boundary": "boundary_{i}.vtp",
    "volume": "volume_{i}.vtu",
}

#: STL file template.
_STL_TEMPLATE = "ahmed_{i}.stl"

#: CSV file templates for global data.
_CSV_TEMPLATES: dict[str, str] = {
    "force_mom": "force_mom_{i}.csv",
    "force_mom_varref": "force_mom_varref_{i}.csv",
    "geo_parameters": "geo_parameters_{i}.csv",
}

#: Valid mesh types for this dataset.
MeshType = Literal["boundary", "volume", "slices", "multi"]

#: Valid mesh parts for multi mode.
MeshPart = Literal["domain", "stl"]

#: Valid backend options for VTK reading.
Backend = Literal["pyvista", "rust"]


class AhmedMLSource(Source[Mesh]):
    """Read meshes from the AhmedML dataset on HuggingFace Hub.

    Each index maps to one simulation run.  The *mesh_type* parameter
    selects which mesh to load for each run:

    * ``"boundary"`` — surface mesh (VTP) with flow fields
    * ``"volume"`` — volumetric mesh (VTU, single file per run)
    * ``"slices"`` — x/y/z-normal slice planes (VTP); yields multiple
      meshes per index
    * ``"multi"`` — yields domain mesh (DomainMesh) and/or STL mesh

    All modes attach CSV metadata (force/moment coefficients and geometric
    parameters) as ``global_data`` on each yielded mesh.

    Parameters
    ----------
    mesh_type : {"boundary", "volume", "slices", "multi"}
        Which mesh to read from each run directory.
    mesh_parts : list[str] | None
        Mesh parts to yield in ``"multi"`` mode.  Valid parts are
        ``"domain"`` and ``"stl"``.  Defaults to ``["domain"]``.
        Ignored for non-multi modes.
    url : str
        Base HuggingFace Hub URL.  Override only for testing.
    storage_options : dict[str, object] | None
        Extra ``fsspec`` keyword arguments (e.g. ``{"token": "hf_..."}``).
    cache_storage : str | None
        Local cache directory.  ``None`` → temporary directory.
    manifold_dim : int or {"auto"}
        Target manifold dimension for ``from_pyvista`` conversion.
        Only used with the ``"pyvista"`` backend.
    point_source : {"vertices", "cell_centroids"}
        Point source mode for ``from_pyvista`` conversion.
        Only used with the ``"pyvista"`` backend.
    warn_on_lost_data : bool
        Warn when data arrays are discarded during conversion.
        Only used with the ``"pyvista"`` backend.
    backend : {"pyvista", "rust"}
        VTK reading backend:

        - ``"pyvista"`` (default): use PyVista for full-featured reading.
        - ``"rust"``: use the native Rust backend for faster reading.
          Note: The Rust backend only supports ASCII VTU/VTP files and
          does not support ``manifold_dim`` or ``point_source`` options.

    Examples
    --------
    >>> source = AhmedMLSource(mesh_type="boundary")
    >>> len(source)
    500
    >>> mesh = next(source[0])
    >>> mesh.global_data["cd"]  # force coefficient from CSV
    tensor([0.2405])

    >>> source = AhmedMLSource(mesh_type="multi", mesh_parts=["domain", "stl"])
    >>> for mesh in source[0]:
    ...     print(type(mesh).__name__)
    DomainMesh
    Mesh

    Using the fast Rust backend:

    >>> source = AhmedMLSource(mesh_type="boundary", backend="rust")

    Note
    ----
    - Dataset: `neashton/ahmedml <https://huggingface.co/datasets/neashton/ahmedml>`_
    - Paper: `arXiv:2407.20801 <https://arxiv.org/abs/2407.20801>`_
    - License: `CC-BY-SA-4.0 <https://huggingface.co/datasets/neashton/ahmedml/blob/main/LICENSE>`_
    """

    name: ClassVar[str] = "AhmedML"
    description: ClassVar[str] = "AhmedML dataset — 500 Ahmed Car Body variants with hybrid RANS-LES CFD"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the AhmedML source.

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
            Param(
                name="mesh_parts",
                description="Mesh parts to yield in multi mode (domain, stl)",
                type=str,
                default="domain",
                choices=["domain", "stl"],
            ),
            Param(name="url", description="Base HuggingFace Hub URL", type=str, default=_AHMEDML_HF_URL),
            Param(
                name="cache_storage",
                description="Local cache directory for downloaded files",
                type=str,
                default="",
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
                name="backend",
                description="VTK reading backend: pyvista (default) or rust (faster)",
                type=str,
                default="pyvista",
                choices=["pyvista", "rust"],
            ),
        ]

    def __init__(
        self,
        mesh_type: MeshType = "multi",
        mesh_parts: list[MeshPart] | None = None,
        url: str = _AHMEDML_HF_URL,
        storage_options: dict[str, object] | None = None,
        cache_storage: str | None = None,
        manifold_dim: int | Literal["auto"] = "auto",
        point_source: Literal["vertices", "cell_centroids"] = "vertices",
        warn_on_lost_data: bool = True,
        backend: Backend = "pyvista",
    ) -> None:
        import fsspec

        from physicsnemo_curator.core.logging import flush_logs, get_logger

        self._log = get_logger(self)
        self._flush_logs = flush_logs
        self._mesh_type: MeshType = mesh_type
        self._mesh_parts: list[MeshPart] = mesh_parts or ["domain"]  # ty: ignore[invalid-assignment]
        self._url = url
        self._storage_options = storage_options or {}
        self._cache_storage = cache_storage or tempfile.mkdtemp(prefix="curator_ahmedml_")
        self._manifold_dim = manifold_dim
        self._point_source = point_source
        self._warn_on_lost_data = warn_on_lost_data
        self._backend: Backend = backend

        self._fs, self._root_path = fsspec.core.url_to_fs(self._url, **self._storage_options)
        self._protocol = self._fs.protocol if isinstance(self._fs.protocol, str) else self._fs.protocol[0]
        self._run_indices = self._discover_runs()

        if mesh_type == "multi":
            self._file_template = ""  # not used for multi mode
        elif mesh_type == "slices":
            self._build_slices_data()
        else:
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
            return remote_path

        local_path = pathlib.Path(self._cache_storage) / remote_path.lstrip("/")
        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            fs.get(remote_path, str(local_path))

        return str(local_path)

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
            Sequence number within this index (ignored for boundary/volume).

        Returns
        -------
        str
            Resolved name like ``"boundary_1"`` or ``"volume_5"``.
        """
        run_id = self._run_indices[index]
        if self._mesh_type == "slices":
            _, _, files = self._slices_run_data[index]
            filename = pathlib.PurePosixPath(files[seq]).stem
            return f"{filename}"
        return self._file_template.format(i=run_id).rsplit(".", 1)[0]

    def __getitem__(self, index: int) -> Generator[Mesh]:  # type: ignore[override]
        """Read the mesh(es) for the *index*-th run.

        For ``"boundary"`` and ``"volume"`` mesh types, yields a single
        :class:`~physicsnemo.mesh.Mesh`.  For ``"slices"``, yields one
        mesh per slice plane file in the run's ``slices/`` directory.
        For ``"multi"``, yields a :class:`~physicsnemo.mesh.DomainMesh`
        and/or STL mesh depending on *mesh_parts*.

        All yielded meshes include ``global_data`` populated from the
        run's CSV files (force coefficients and geometric parameters).

        Parameters
        ----------
        index : int
            Zero-based index into the sorted run list.

        Yields
        ------
        Mesh
            Converted physicsnemo Mesh or DomainMesh object(s).
        """
        if self._mesh_type == "multi":
            yield from self._read_multi(index)
        elif self._mesh_type == "slices":
            yield from self._read_slices(index)
        else:
            if index < -len(self._run_indices) or index >= len(self._run_indices):
                msg = f"Index {index} out of range for source with {len(self._run_indices)} runs."
                raise IndexError(msg)

            run_id = self._run_indices[index]
            filename = self._file_template.format(i=run_id)
            remote_path = f"{self._root_path}/run_{run_id}/{filename}"
            path = self._ensure_local(remote_path)
            mesh = self._read_vtk(path)
            mesh = self._attach_global_data(mesh, run_id)
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
            return self._read_with_rust(path)
        return self._read_with_pyvista(path)

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
        import pyvista as pv
        from physicsnemo.mesh.io import from_pyvista

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
        )

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
            One mesh per slice plane file, with global_data attached.
        """
        run_id = self._run_indices[index]
        fs, protocol, files = self._slices_run_data[index]
        for remote_path in files:
            local_path = self._ensure_local(remote_path, fs=fs, protocol=protocol)
            mesh = self._read_vtk(local_path)
            mesh = self._attach_global_data(mesh, run_id)
            yield mesh

    # -- CSV global data -------------------------------------------------------

    def _read_csv_global_data(self, run_id: int) -> dict[str, Any]:
        """Read CSV metadata files for a run and return as a flat dict of floats.

        Reads ``force_mom_{i}.csv``, ``force_mom_varref_{i}.csv``, and
        ``geo_parameters_{i}.csv`` from the run directory.  Missing files
        are silently skipped.

        Parameters
        ----------
        run_id : int
            Dataset run ID.

        Returns
        -------
        dict[str, float]
            Flat dictionary of all CSV values with column names as keys.
            For ``force_mom_varref``, keys are prefixed with ``varref_``
            to avoid collisions with ``force_mom``.
        """
        import torch

        data: dict[str, Any] = {}

        for csv_key, template in _CSV_TEMPLATES.items():
            filename = template.format(i=run_id)
            remote_path = f"{self._root_path}/run_{run_id}/{filename}"
            try:
                local_path = self._ensure_local(remote_path)
            except (FileNotFoundError, OSError):
                continue

            with pathlib.Path(local_path).open(newline="") as f:
                reader = csv.reader(f)
                headers = [h.strip().replace("-", "_") for h in next(reader)]
                values = [float(v.strip()) for v in next(reader)]

            # Prefix varref columns to avoid collision with force_mom
            prefix = "varref_" if csv_key == "force_mom_varref" else ""
            for col, val in zip(headers, values, strict=True):
                data[f"{prefix}{col}"] = torch.tensor([val], dtype=torch.float32)

        return data

    def _attach_global_data(self, mesh: Mesh, run_id: int) -> Mesh:
        """Attach CSV global data to a mesh as its global_data field.

        Parameters
        ----------
        mesh : Mesh
            Mesh to augment.
        run_id : int
            Dataset run ID for CSV lookup.

        Returns
        -------
        Mesh
            Mesh with ``global_data`` populated from CSV metadata.
        """
        from tensordict import TensorDict

        csv_data = self._read_csv_global_data(run_id)
        if csv_data:
            mesh = Mesh(
                points=mesh.points,
                cells=mesh.cells,
                point_data=mesh.point_data,
                cell_data=mesh.cell_data,
                global_data=TensorDict(csv_data, batch_size=[]),  # ty: ignore[invalid-argument-type]
            )
        return mesh

    # -- Multi mode ------------------------------------------------------------

    def _read_multi(self, index: int) -> Generator[Mesh]:
        """Read multiple mesh representations for a given run.

        Parameters
        ----------
        index : int
            Zero-based index into the sorted run list.

        Yields
        ------
        Mesh or DomainMesh
            One mesh per active mesh_part, in order.  The ``"domain"`` part
            yields a :class:`DomainMesh`; ``"stl"`` yields a plain :class:`Mesh`.
        """
        for part in self._mesh_parts:
            if part == "domain":
                yield from self._read_domain(index)
            elif part == "stl":
                yield from self._read_stl(index)

    def _read_domain(self, index: int) -> Generator[Mesh]:
        """Read domain mesh combining volume interior, boundary surface, and global data.

        Produces a :class:`~physicsnemo.mesh.domain_mesh.DomainMesh` with:

        * **interior** — volume VTU converted to a point-cloud via
          ``cell_centroids`` (no connectivity, cells shape ``[0, 1]``).
        * **boundaries["surface"]** — boundary VTP with cell connectivity
          and flow fields.
        * **global_data** — CSV metadata (force coefficients + geometry params).

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
        import time

        from physicsnemo.mesh.domain_mesh import DomainMesh
        from tensordict import TensorDict

        run_id = self._run_indices[index]
        self._log.info("run_%d: Starting domain read", run_id)
        self._flush_logs()

        # --- Interior (volume VTU → point-cloud) ---
        volume_filename = _MESH_TEMPLATES["volume"].format(i=run_id)
        volume_remote = f"{self._root_path}/run_{run_id}/{volume_filename}"
        self._log.debug("run_%d: Ensuring local volume file", run_id)
        t0 = time.perf_counter()
        volume_path = self._ensure_local(volume_remote)
        self._log.debug("run_%d: Volume file ready (%.2fs)", run_id, time.perf_counter() - t0)

        self._log.info("run_%d: Reading interior VTU", run_id)
        t0 = time.perf_counter()
        interior = self._read_vtk_as_interior(volume_path)
        elapsed = time.perf_counter() - t0
        self._log.info("run_%d: Interior read complete: %d pts (%.2fs)", run_id, interior.n_points, elapsed)

        self._log.debug("run_%d: Downcasting interior to fp32", run_id)
        t0 = time.perf_counter()
        interior = self._downcast_fp32(interior)
        self._log.debug("run_%d: Interior downcast complete (%.2fs)", run_id, time.perf_counter() - t0)

        # --- Boundary surface (VTP) ---
        boundary_filename = _MESH_TEMPLATES["boundary"].format(i=run_id)
        boundary_remote = f"{self._root_path}/run_{run_id}/{boundary_filename}"
        self._log.debug("run_%d: Ensuring local boundary file", run_id)
        t0 = time.perf_counter()
        boundary_path = self._ensure_local(boundary_remote)
        self._log.debug("run_%d: Boundary file ready (%.2fs)", run_id, time.perf_counter() - t0)

        self._log.info("run_%d: Reading boundary VTP", run_id)
        t0 = time.perf_counter()
        surface = self._read_vtk(boundary_path)
        elapsed = time.perf_counter() - t0
        self._log.info("run_%d: Boundary read complete: %d pts (%.2fs)", run_id, surface.n_points, elapsed)

        self._log.debug("run_%d: Downcasting boundary to fp32", run_id)
        t0 = time.perf_counter()
        surface = self._downcast_fp32(surface)
        self._log.debug("run_%d: Boundary downcast complete (%.2fs)", run_id, time.perf_counter() - t0)

        # --- Global data from CSVs ---
        self._log.debug("run_%d: Reading CSV global data", run_id)
        t0 = time.perf_counter()
        csv_data = self._read_csv_global_data(run_id)
        global_data = TensorDict(csv_data, batch_size=[]) if csv_data else None  # ty: ignore[invalid-argument-type]
        self._log.debug("run_%d: CSV read complete (%.2fs)", run_id, time.perf_counter() - t0)

        # --- Assemble DomainMesh ---
        self._log.debug("run_%d: Assembling DomainMesh", run_id)
        t0 = time.perf_counter()
        domain_mesh = DomainMesh(
            interior=interior,
            boundaries={"surface": surface},
            global_data=global_data,
        )
        self._log.debug("run_%d: DomainMesh assembled (%.2fs)", run_id, time.perf_counter() - t0)
        self._log.info("run_%d: Domain read complete", run_id)
        yield domain_mesh

    def _read_stl(self, index: int) -> Generator[Mesh]:
        """Read the STL geometry file for a given run.

        Parameters
        ----------
        index : int
            Zero-based index into the sorted run list.

        Yields
        ------
        Mesh
            STL mesh with global_data from CSVs attached.
        """
        run_id = self._run_indices[index]
        stl_filename = _STL_TEMPLATE.format(i=run_id)
        stl_remote = f"{self._root_path}/run_{run_id}/{stl_filename}"
        stl_path = self._ensure_local(stl_remote)
        mesh = self._read_vtk(stl_path)
        mesh = self._attach_global_data(mesh, run_id)
        yield mesh

    def _read_vtk_as_interior(self, path: str) -> Mesh:
        """Read a volume VTK file as an interior point-cloud.

        Converts the volume mesh to a point-cloud using cell centroids
        (no connectivity).

        Parameters
        ----------
        path : str
            Local path to VTU file.

        Returns
        -------
        Mesh
            Point-cloud mesh (manifold_dim=0) with no cell connectivity.
        """
        if self._backend == "rust":
            return self._read_interior_with_rust(path)
        return self._read_interior_with_pyvista(path)

    def _read_interior_with_pyvista(self, path: str) -> Mesh:
        """Read interior mesh using PyVista backend.

        Parameters
        ----------
        path : str
            Local path to VTU file.

        Returns
        -------
        Mesh
            Point-cloud mesh with cell centroids as points.
        """
        import pyvista as pv
        from physicsnemo.mesh.io import from_pyvista

        pv_mesh = pv.read(path)
        return from_pyvista(
            pv_mesh,
            manifold_dim=0,
            point_source="cell_centroids",
            warn_on_lost_data=self._warn_on_lost_data,
        )

    def _read_interior_with_rust(self, path: str) -> Mesh:
        """Read interior mesh using Rust backend.

        Computes cell centroids from the cell connectivity and uses
        cell_data as point_data on the resulting point-cloud.

        Parameters
        ----------
        path : str
            Local path to VTU file.

        Returns
        -------
        Mesh
            Point-cloud mesh with cell centroids as points and cell_data
            converted to point_data.
        """
        import time

        import numpy as np
        import torch
        from tensordict import TensorDict

        from physicsnemo_curator._lib import vtk

        # Skip point_data since we only need cell_data for the interior point cloud
        self._log.debug("Reading VTU via Rust backend: %s", path)
        t0 = time.perf_counter()
        rust_mesh = vtk.read_vtk(path, skip_point_data=True)
        elapsed = time.perf_counter() - t0
        self._log.debug("Rust VTK parsed: %d pts, %d cells (%.2fs)", rust_mesh.n_points, rust_mesh.n_cells, elapsed)

        # Extract arrays we need and free the rust_mesh reference early
        points = rust_mesh.points  # (n_points, 3)
        cells = rust_mesh.cells  # flat connectivity array
        offsets = rust_mesh.cell_offsets  # cell boundary offsets
        n_cells = rust_mesh.n_cells
        cell_data = rust_mesh.cell_data
        self._log.debug("Cell data fields: %s", list(cell_data.keys()))

        if cells is None or offsets is None:
            msg = f"VTK file {path} has no cell connectivity; cannot compute centroids."
            raise ValueError(msg)

        # Compute centroids for each cell using vectorized operations
        # VTK offsets are cumulative end indices:
        # - Cell 0: connectivity[0:offsets[0]]
        # - Cell i: connectivity[offsets[i-1]:offsets[i]]

        self._log.debug("Computing cell centroids...")
        t0 = time.perf_counter()

        # Build start indices: [0, offsets[0], offsets[1], ..., offsets[n-2]]
        starts = np.zeros(n_cells, dtype=offsets.dtype)
        starts[1:] = offsets[:-1]

        # Compute points per cell
        points_per_cell = offsets - starts
        del offsets  # Free memory

        # Check if all cells have the same number of points (common case)
        if np.all(points_per_cell == points_per_cell[0]):
            # Fast path: uniform cell size - fully vectorized
            pts_per_cell = int(points_per_cell[0])
            self._log.debug("Using fast path (uniform %d pts/cell)", pts_per_cell)
            del points_per_cell, starts  # Free memory

            cell_point_ids = cells.reshape(n_cells, pts_per_cell)
            del cells  # Free memory

            # Compute centroids in float32 directly to save memory
            cell_points = points[cell_point_ids].astype(np.float32, copy=False)
            del cell_point_ids, points  # Free memory

            centroids = cell_points.mean(axis=1)
            del cell_points  # Free memory
        else:
            # Slow path: variable cell sizes - use np.add.reduceat
            self._log.debug("Using slow path (variable cell sizes)")
            # Gather all cell points and sum them per cell
            all_cell_points = points[cells].astype(np.float32, copy=False)
            del cells, points  # Free memory

            # Sum points within each cell using reduceat
            cell_sums = np.add.reduceat(all_cell_points, starts, axis=0)
            del all_cell_points, starts  # Free memory

            # Divide by points per cell to get centroids
            centroids = cell_sums / points_per_cell[:, np.newaxis].astype(np.float32)
            del cell_sums, points_per_cell  # Free memory

        self._log.debug("Centroids computed (%.2fs)", time.perf_counter() - t0)

        # Convert to torch tensors
        self._log.debug("Converting to torch tensors...")
        t0 = time.perf_counter()
        points_tensor = torch.from_numpy(centroids)
        del centroids  # Free numpy array

        # Use cell_data as point_data (one value per centroid)
        point_data_dict = {}
        for name, data in cell_data.items():
            # Convert to float32 if floating point to save memory
            if data.dtype in (np.float64,):
                point_data_dict[name] = torch.from_numpy(data.astype(np.float32, copy=False))
            else:
                point_data_dict[name] = torch.from_numpy(data)
        del cell_data  # Free memory

        point_data = TensorDict(point_data_dict, batch_size=[n_cells]) if point_data_dict else None
        self._log.debug("Tensor conversion complete (%.2fs)", time.perf_counter() - t0)

        return Mesh(
            points=points_tensor,
            point_data=point_data,
        )

    @staticmethod
    def _downcast_fp32(mesh: Mesh) -> Mesh:
        """Downcast all floating-point tensors in a mesh to float32.

        Parameters
        ----------
        mesh : Mesh
            Input mesh.

        Returns
        -------
        Mesh
            Mesh with all float64 data converted to float32.
        """
        import torch

        points = mesh.points.float() if mesh.points.dtype == torch.float64 else mesh.points
        cells = mesh.cells

        point_data = None
        if mesh.point_data is not None:
            pd_dict = {}
            for key in mesh.point_data.keys():  # noqa: SIM118
                t = mesh.point_data.get(key)
                pd_dict[key] = t.float() if t.is_floating_point() and t.dtype == torch.float64 else t
            from tensordict import TensorDict

            point_data = TensorDict(pd_dict, batch_size=mesh.point_data.batch_size)

        cell_data = None
        if mesh.cell_data is not None:
            cd_dict = {}
            for key in mesh.cell_data.keys():  # noqa: SIM118
                t = mesh.cell_data.get(key)
                cd_dict[key] = t.float() if t.is_floating_point() and t.dtype == torch.float64 else t
            from tensordict import TensorDict

            cell_data = TensorDict(cd_dict, batch_size=mesh.cell_data.batch_size)

        return Mesh(
            points=points,
            cells=cells,
            point_data=point_data,
            cell_data=cell_data,
            global_data=mesh.global_data,
        )
