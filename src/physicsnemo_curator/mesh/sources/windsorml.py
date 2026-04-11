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

"""WindsorML dataset source for mesh pipelines.

Reads the `WindsorML <https://huggingface.co/datasets/neashton/windsorml>`_
dataset — 355 geometric variants of the Windsor body for automotive
aerodynamics (Volcano Platforms GPU-native solver, ~280-300 M cells).

The dataset provides two mesh types per run:

* **boundary** — surface mesh with flow fields (VTU, ~406 MB each)
* **volume** — volumetric field data (VTU, ~21 GB each)

.. note::
   WindsorML does not provide slice plane meshes (only images).

File discovery and caching are handled internally using ``fsspec``.
"""

from __future__ import annotations

import pathlib
import re
import tempfile
from typing import TYPE_CHECKING, ClassVar, Literal

import pyvista as pv
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.io import from_pyvista

from physicsnemo_curator.core.base import Param, Source

if TYPE_CHECKING:
    from collections.abc import Generator

#: HuggingFace Hub URL for the WindsorML dataset.
_WINDSORML_HF_URL = "hf://datasets/neashton/windsorml"

#: Mesh type → file template mapping.
_MESH_TEMPLATES: dict[str, str] = {
    "boundary": "boundary_{i}.vtu",
    "volume": "volume_{i}.vtu",
}

#: Valid mesh types for this dataset.
MeshType = Literal["boundary", "volume"]


class WindsorMLSource(Source[Mesh]):
    """Read meshes from the WindsorML dataset on HuggingFace Hub.

    Each index maps to one simulation run.  The *mesh_type* parameter
    selects which mesh to load for each run:

    * ``"boundary"`` — surface mesh (VTU) with flow fields
    * ``"volume"`` — volumetric mesh (VTU)

    .. note::
       WindsorML runs are indexed starting from 0 (``run_0`` through
       ``run_354``).  Slice plane meshes are not available in this
       dataset (only images).

    Parameters
    ----------
    mesh_type : {"boundary", "volume"}
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
    >>> source = WindsorMLSource(mesh_type="boundary")
    >>> len(source)
    355
    >>> mesh = next(source[0])

    Note
    ----
    - Dataset: `neashton/windsorml <https://huggingface.co/datasets/neashton/windsorml>`_
    - Paper: `arXiv:2407.19320 <https://arxiv.org/abs/2407.19320>`_
    - License: `CC-BY-SA-4.0 <https://huggingface.co/datasets/neashton/windsorml/blob/main/LICENSE>`_
    """

    name: ClassVar[str] = "WindsorML"
    description: ClassVar[str] = "WindsorML dataset — 355 Windsor body variants with GPU-native CFD"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the WindsorML source.

        Returns
        -------
        list[Param]
            Parameter list for CLI configuration.
        """
        return [
            Param(
                name="mesh_type",
                description="Mesh type: boundary (surface) or volume (3D field)",
                type=str,
                default="boundary",
                choices=["boundary", "volume"],
            ),
            Param(name="url", description="Base HuggingFace Hub URL", type=str, default=_WINDSORML_HF_URL),
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
        url: str = _WINDSORML_HF_URL,
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
        self._cache_storage = cache_storage or tempfile.mkdtemp(prefix="curator_windsorml_")
        self._manifold_dim = manifold_dim
        self._point_source = point_source
        self._warn_on_lost_data = warn_on_lost_data

        self._file_template = _MESH_TEMPLATES[mesh_type]
        self._fs, self._root_path = fsspec.core.url_to_fs(self._url, **self._storage_options)
        self._protocol = self._fs.protocol if isinstance(self._fs.protocol, str) else self._fs.protocol[0]
        self._run_indices = self._discover_runs()

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

    # -- Source interface -----------------------------------------------------

    def __len__(self) -> int:
        """Return the number of available runs."""
        return len(self._run_indices)

    def __getitem__(self, index: int) -> Generator[Mesh]:
        """Read the mesh for the *index*-th run.

        Parameters
        ----------
        index : int
            Zero-based index into the sorted run list.

        Yields
        ------
        Mesh
            The converted physicsnemo Mesh.
        """
        if index < -len(self._run_indices) or index >= len(self._run_indices):
            msg = f"Index {index} out of range for source with {len(self._run_indices)} runs."
            raise IndexError(msg)

        run_id = self._run_indices[index]
        filename = self._file_template.format(i=run_id)
        remote_path = f"{self._root_path}/run_{run_id}/{filename}"
        path = self._ensure_local(remote_path)

        pv_mesh = pv.read(path)
        mesh = from_pyvista(
            pv_mesh,
            manifold_dim=self._manifold_dim,
            point_source=self._point_source,
            warn_on_lost_data=self._warn_on_lost_data,
        )
        yield mesh
