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

"""WindTunnel-20k dataset source for mesh pipelines.

Reads the `WindTunnel-20k
<https://huggingface.co/datasets/inductiva/windtunnel-20k>`_ dataset —
19,812 OpenFOAM simulations of 1,000 unique automobile-like objects in a
virtual wind tunnel.

The dataset provides one VTK mesh type per simulation:

* **pressure_field** — pressure field interpolated onto the input mesh
  (VTK, surface data with point-data scalars)

Simulations are organised by data split (``train``, ``validation``,
``test``) and identified by unique alphanumeric simulation IDs.  File
discovery is delegated to a :class:`~curator.core.store.FsspecFileStore`.
"""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, ClassVar, Literal

import pyvista as pv
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.io import from_pyvista

from physicsnemo_curator.core.base import Param, Source
from physicsnemo_curator.core.store import FsspecFileStore

if TYPE_CHECKING:
    from collections.abc import Generator

#: HuggingFace Hub URL for the WindTunnel-20k dataset.
_WINDTUNNEL_HF_URL = "hf://datasets/inductiva/windtunnel-20k"

#: Valid data splits.
Split = Literal["train", "validation", "test", "all"]


class WindTunnelSource(Source[Mesh]):
    """Read pressure-field meshes from the WindTunnel-20k dataset.

    Each index maps to one simulation.  The *split* parameter selects
    which data partition to use (``"train"``, ``"validation"``,
    ``"test"``, or ``"all"`` for the combined set).

    Only the ``pressure_field_mesh.vtk`` files are supported (OBJ and
    PLY files in the dataset are not VTK-compatible).

    Parameters
    ----------
    split : {"train", "validation", "test", "all"}
        Data split to use.  ``"all"`` combines all three splits.
    url : str
        Base HuggingFace Hub URL.  Override only for testing.
    storage_options : dict[str, object] | None
        Extra ``fsspec`` keyword arguments.
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
    >>> source = WindTunnelSource(split="train")
    >>> len(source)
    13900
    >>> mesh = next(source[0])

    >>> source = WindTunnelSource(split="all")
    >>> len(source)
    19812

    Note
    ----
    - Dataset: `inductiva/windtunnel-20k <https://huggingface.co/datasets/inductiva/windtunnel-20k>`_
    - License: `CC-BY-4.0 <https://huggingface.co/datasets/inductiva/windtunnel-20k/blob/main/README.md>`_
    """

    name: ClassVar[str] = "WindTunnel-20k"
    description: ClassVar[str] = "WindTunnel-20k dataset — 19,812 OpenFOAM simulations of automobile-like objects"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the WindTunnel source.

        Returns
        -------
        list[Param]
            Parameter list for CLI configuration.
        """
        return [
            Param(
                name="split",
                description="Data split: train, validation, test, or all",
                type=str,
                default="train",
                choices=["train", "validation", "test", "all"],
            ),
            Param(name="url", description="Base HuggingFace Hub URL", type=str, default=_WINDTUNNEL_HF_URL),
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
        split: Split = "train",
        url: str = _WINDTUNNEL_HF_URL,
        storage_options: dict[str, object] | None = None,
        cache_storage: str | None = None,
        manifold_dim: int | Literal["auto"] = "auto",
        point_source: Literal["vertices", "cell_centroids"] = "vertices",
        warn_on_lost_data: bool = True,
    ) -> None:
        self._split: Split = split
        self._url = url
        self._storage_options = storage_options or {}
        self._cache_storage = cache_storage or tempfile.mkdtemp(prefix="curator_windtunnel_")
        self._manifold_dim = manifold_dim
        self._point_source = point_source
        self._warn_on_lost_data = warn_on_lost_data

        self._stores = self._build_stores()

    # -- Source interface -----------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of simulations across selected splits."""
        return sum(len(s) for s in self._stores)

    def __getitem__(self, index: int) -> Generator[Mesh]:
        """Read the pressure-field mesh for the *index*-th simulation.

        Parameters
        ----------
        index : int
            Zero-based index across all selected splits.

        Yields
        ------
        Mesh
            The pressure-field surface mesh.
        """
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            msg = f"Index {index} out of range for source with {len(self)} items."
            raise IndexError(msg)

        # Find which store (split) this index falls into.
        offset = 0
        for store in self._stores:
            if index < offset + len(store):
                path = store[index - offset]
                pv_mesh = pv.read(path)
                mesh = from_pyvista(
                    pv_mesh,
                    manifold_dim=self._manifold_dim,
                    point_source=self._point_source,
                    warn_on_lost_data=self._warn_on_lost_data,
                )
                yield mesh
                return
            offset += len(store)

    # -- Internal helpers ----------------------------------------------------

    def _build_stores(self) -> list[FsspecFileStore]:
        """Build FsspecFileStore objects for each selected split.

        Returns
        -------
        list[FsspecFileStore]
            One store per selected data split.
        """
        splits = ["train", "validation", "test"] if self._split == "all" else [self._split]

        stores: list[FsspecFileStore] = []
        for split in splits:
            split_url = f"{self._url}/data/{split}"
            store = FsspecFileStore(
                url=split_url,
                pattern="**/pressure_field_mesh.vtk",
                extensions=frozenset({".vtk"}),
                storage_options=self._storage_options,
                cache_storage=self._cache_storage,
            )
            stores.append(store)

        return stores
