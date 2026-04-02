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

File discovery and caching are delegated to a
:class:`~curator.core.store.RunIndexedFileStore` (for boundary/volume)
or :class:`~curator.core.store.FsspecFileStore` (for slices).
"""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, ClassVar, Literal

import pyvista as pv
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.io import from_pyvista

from physicsnemo_curator.core.base import Param, Source
from physicsnemo_curator.core.store import FsspecFileStore, RunIndexedFileStore

if TYPE_CHECKING:
    from collections.abc import Generator

#: HuggingFace Hub URL for the AhmedML dataset.
_AHMEDML_HF_URL = "hf://datasets/neashton/ahmedml"

#: Mesh type → file template mapping.
_MESH_TEMPLATES: dict[str, str] = {
    "boundary": "boundary_{i}.vtp",
    "volume": "volume_{i}.vtu",
}

#: Valid mesh types for this dataset.
MeshType = Literal["boundary", "volume", "slices"]


class AhmedMLSource(Source[Mesh]):
    """Read meshes from the AhmedML dataset on HuggingFace Hub.

    Each index maps to one simulation run.  The *mesh_type* parameter
    selects which mesh to load for each run:

    * ``"boundary"`` — surface mesh (VTP) with flow fields
    * ``"volume"`` — volumetric mesh (VTU, single file per run)
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
    >>> source = AhmedMLSource(mesh_type="boundary")
    >>> len(source)
    500
    >>> mesh = next(source[0])

    >>> source = AhmedMLSource(mesh_type="slices")
    >>> for mesh in source[0]:
    ...     print(mesh.n_points)
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
                description="Mesh type: boundary (surface), volume (3D field), or slices (planes)",
                type=str,
                default="boundary",
                choices=["boundary", "volume", "slices"],
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
        ]

    def __init__(
        self,
        mesh_type: MeshType = "boundary",
        url: str = _AHMEDML_HF_URL,
        storage_options: dict[str, object] | None = None,
        cache_storage: str | None = None,
        manifold_dim: int | Literal["auto"] = "auto",
        point_source: Literal["vertices", "cell_centroids"] = "vertices",
        warn_on_lost_data: bool = True,
    ) -> None:
        self._mesh_type: MeshType = mesh_type
        self._url = url
        self._storage_options = storage_options or {}
        self._cache_storage = cache_storage or tempfile.mkdtemp(prefix="curator_ahmedml_")
        self._manifold_dim = manifold_dim
        self._point_source = point_source
        self._warn_on_lost_data = warn_on_lost_data

        if mesh_type == "slices":
            self._store = self._build_slices_store()
        else:
            template = _MESH_TEMPLATES[mesh_type]
            self._store = RunIndexedFileStore(
                url=self._url,
                file_template=template,
                storage_options=self._storage_options,
                cache_storage=self._cache_storage,
            )

    # -- Source interface -----------------------------------------------------

    def __len__(self) -> int:
        """Return the number of available runs."""
        return len(self._store)

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
        if self._mesh_type == "slices":
            yield from self._read_slices(index)
        else:
            path = self._store[index]
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
        run_store = self._slices_run_stores[index]
        for i in range(len(run_store)):
            path = run_store[i]
            yield self._read_vtk(path)

    def _build_slices_store(self) -> RunIndexedFileStore:
        """Build the store for slices mode and pre-build per-run stores.

        Returns
        -------
        RunIndexedFileStore
            A store used only for its ``run_indices`` and ``__len__``.
        """
        index_store = RunIndexedFileStore(
            url=self._url,
            file_template="boundary_{i}.vtp",
            storage_options=self._storage_options,
            cache_storage=self._cache_storage,
        )

        self._slices_run_stores: list[FsspecFileStore] = []
        for run_id in index_store.run_indices:
            slice_url = f"{self._url}/run_{run_id}/slices"
            store = FsspecFileStore(
                url=slice_url,
                pattern="**",
                extensions=frozenset({".vtp"}),
                storage_options=self._storage_options,
                cache_storage=self._cache_storage,
            )
            self._slices_run_stores.append(store)

        return index_store
