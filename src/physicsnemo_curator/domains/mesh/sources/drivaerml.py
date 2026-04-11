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
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.io import from_pyvista

from physicsnemo_curator.core.base import Param, Source

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
MeshType = Literal["boundary", "volume", "slices"]


class DrivAerMLSource(Source[Mesh]):
    """Read meshes from the DrivAerML dataset on HuggingFace Hub.

    Each index maps to one simulation run.  The *mesh_type* parameter
    selects which mesh to load for each run:

    * ``"boundary"`` — surface mesh (VTP) with flow fields
    * ``"volume"`` — volumetric mesh (VTU, reconstructed from split parts)
    * ``"slices"`` — x/y/z-normal slice planes (VTP); yields multiple
      meshes per index

    Parameters
    ----------
    mesh_type : {"boundary", "volume", "slices"}
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
                description="Mesh type: boundary (surface), volume (3D field), or slices (planes)",
                type=str,
                default="boundary",
                choices=["boundary", "volume", "slices"],
            ),
            Param(name="url", description="Base HuggingFace Hub URL", type=str, default=_DRIVAERML_HF_URL),
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
        ]

    def __init__(
        self,
        mesh_type: MeshType = "boundary",
        url: str = _DRIVAERML_HF_URL,
        storage_options: dict[str, object] | None = None,
        cache_storage: str | None = None,
        manifold_dim: int | Literal["auto"] = "auto",
        point_source: Literal["vertices", "cell_centroids"] = "vertices",
        warn_on_lost_data: bool = True,
    ) -> None:
        import fsspec

        self._mesh_type: MeshType = mesh_type
        self._url = url
        self._storage_options = storage_options or {}
        self._cache_storage = cache_storage or tempfile.mkdtemp(prefix="curator_drivaerml_")
        self._manifold_dim = manifold_dim
        self._point_source = point_source
        self._warn_on_lost_data = warn_on_lost_data

        self._fs, self._root_path = fsspec.core.url_to_fs(self._url, **self._storage_options)
        self._protocol = self._fs.protocol if isinstance(self._fs.protocol, str) else self._fs.protocol[0]
        self._run_indices = self._discover_runs()

        if mesh_type == "slices":
            self._file_template = "boundary_{i}.vtp"  # dummy, not used for slices
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

    # -- Source interface -----------------------------------------------------

    def __len__(self) -> int:
        """Return the number of available runs."""
        return len(self._run_indices)

    def __getitem__(self, index: int) -> Generator[Mesh]:
        """Read the mesh(es) for the *index*-th run.

        For ``"boundary"`` and ``"volume"`` mesh types, yields a single
        :class:`~physicsnemo.mesh.Mesh`.  For ``"slices"``, yields one
        mesh per slice plane file in the run's ``slices/`` directory.

        Parameters
        ----------
        index : int
            Zero-based index into the sorted run list.

        Yields
        ------
        Mesh
            Converted physicsnemo Mesh object(s).
        """
        if self._mesh_type == "volume":
            yield from self._read_volume(index)
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
            yield self._read_vtk(path)

    # -- Internal helpers ----------------------------------------------------

    def _read_vtk(self, path: str) -> Mesh:
        """Read a single VTK file and convert to Mesh.

        Parameters
        ----------
        path : str
            Local filesystem path to a VTK file.

        Returns
        -------
        Mesh
            Converted mesh.
        """
        pv_mesh = pv.read(path)
        return from_pyvista(
            pv_mesh,
            manifold_dim=self._manifold_dim,
            point_source=self._point_source,
            warn_on_lost_data=self._warn_on_lost_data,
        )

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
            yield self._read_vtk(local_path)
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

        yield self._read_vtk(str(concat_path))

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
            yield self._read_vtk(local_path)

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
