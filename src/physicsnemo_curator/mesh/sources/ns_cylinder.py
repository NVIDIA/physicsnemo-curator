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

"""Navier-Stokes cylinder dataset source for mesh pipelines.

Reads the `Navier-Stokes Cylinder
<https://huggingface.co/datasets/SISSAmathLab/navier-stokes-cylinder>`_
dataset --- 500 incompressible Navier-Stokes simulations of flow around a
2-D cylinder at varying viscosities.

The dataset provides three Parquet configurations:

* **geometry** --- node coordinates and triangular connectivity (1 row,
  shared across all simulations)
* **parameters** --- viscosity value per simulation (500 rows)
* **snapshots** --- velocity (x, y) and pressure fields at each node
  (500 rows)

Each source index maps to one (parameter, snapshot) pair on the shared
geometry.  The resulting :class:`~physicsnemo.mesh.Mesh` carries:

* ``points`` --- *(n_points, 3)* tensor (z = 0)
* ``cells`` --- *(n_cells, 3)* triangle connectivity
* ``point_data`` --- velocity_x, velocity_y, pressure
* ``global_data`` --- viscosity
"""

from __future__ import annotations

import logging
import tempfile
from typing import TYPE_CHECKING, ClassVar

import fsspec
import numpy as np
import pyarrow.parquet as pq
import torch
from physicsnemo.mesh import Mesh
from tensordict import TensorDict

from physicsnemo_curator.core.base import Param, Source

if TYPE_CHECKING:
    from collections.abc import Generator

    import pyarrow as pa

logger = logging.getLogger(__name__)

#: Default HuggingFace Hub URL for the dataset.
_NS_CYLINDER_HF_URL = "hf://datasets/SISSAmathLab/navier-stokes-cylinder"

#: Parquet file name used by HuggingFace auto-converted datasets.
_PARQUET_FILENAME = "default-00000-of-00001.parquet"


class NavierStokesCylinderSource(Source[Mesh]):
    """Read meshes from the Navier-Stokes Cylinder dataset.

    Each index maps to one simulation snapshot.  The shared triangular
    geometry is loaded once and reused for every item.

    Parameters
    ----------
    url : str
        Root URL (local path or fsspec URL) of the dataset.
    storage_options : dict[str, object] | None
        Extra keyword arguments forwarded to ``fsspec.open``.
    cache_storage : str | None
        Local cache directory for downloaded files.  ``None`` creates a
        temporary directory.

    Examples
    --------
    >>> source = NavierStokesCylinderSource()  # doctest: +SKIP
    >>> len(source)  # doctest: +SKIP
    500
    >>> mesh = next(source[0])  # doctest: +SKIP
    """

    name: ClassVar[str] = "Navier-Stokes Cylinder"
    description: ClassVar[str] = (
        "Navier-Stokes Cylinder dataset --- 500 incompressible flow simulations "
        "around a 2-D cylinder at varying viscosities (Parquet, HuggingFace)"
    )

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the Navier-Stokes Cylinder source.

        Returns
        -------
        list[Param]
            Parameter list for CLI configuration.
        """
        return [
            Param(
                name="url",
                description="Root URL of the dataset (local path or fsspec URL)",
                type=str,
                default=_NS_CYLINDER_HF_URL,
            ),
            Param(
                name="cache_storage",
                description="Local cache directory for downloaded files",
                type=str,
                default="",
            ),
        ]

    def __init__(
        self,
        url: str = _NS_CYLINDER_HF_URL,
        storage_options: dict[str, object] | None = None,
        cache_storage: str | None = None,
    ) -> None:
        self._url = url
        self._storage_options = storage_options or {}
        self._cache_storage = cache_storage or tempfile.mkdtemp(prefix="curator_ns_cylinder_")

        # Lazily loaded caches (populated on first access).
        self._geometry_loaded = False
        self._points: torch.Tensor | None = None
        self._cells: torch.Tensor | None = None
        self._snapshots_table: pa.Table | None = None

        # Eagerly discover length from the parameters table (tiny file).
        self._viscosities: np.ndarray = self._read_parameters()
        self._n_snapshots: int = len(self._viscosities)

    # -- Source interface -----------------------------------------------------

    def __len__(self) -> int:
        """Return the number of simulation snapshots."""
        return self._n_snapshots

    def __getitem__(self, index: int) -> Generator[Mesh]:
        """Read the mesh for the *index*-th snapshot.

        Parameters
        ----------
        index : int
            Zero-based snapshot index (supports negative indexing).

        Yields
        ------
        Mesh
            Triangulated 2-D surface mesh with velocity, pressure, and
            viscosity data.
        """
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            msg = f"Index {index} out of range for source with {len(self)} items."
            raise IndexError(msg)

        # Ensure shared geometry is loaded.
        if not self._geometry_loaded:
            self._load_geometry()

        assert self._points is not None  # noqa: S101
        assert self._cells is not None  # noqa: S101

        # Read the snapshot row for this index.
        snapshot = self._read_snapshot(index)

        # Build point data TensorDict (float32 to match points).
        point_data = TensorDict(
            {
                "velocity_x": torch.from_numpy(snapshot["velocity_x"]).float(),
                "velocity_y": torch.from_numpy(snapshot["velocity_y"]).float(),
                "pressure": torch.from_numpy(snapshot["pressure"]).float(),
            },
            batch_size=[self._points.shape[0]],
        )

        # Build global data TensorDict with viscosity (float32 for consistency).
        viscosity_val = float(self._viscosities[index])
        global_data = TensorDict(
            {"viscosity": torch.tensor([viscosity_val], dtype=torch.float32)},
            batch_size=[],
        )

        mesh = Mesh(
            points=self._points.clone(),
            cells=self._cells.clone(),
            point_data=point_data,
            global_data=global_data,
        )
        yield mesh

    # -- Internal helpers ----------------------------------------------------

    def _open_parquet(self, subpath: str) -> pa.Table:
        """Read a Parquet file from the dataset, local or remote.

        Parameters
        ----------
        subpath : str
            Relative path under the dataset root, e.g.
            ``"geometry/default-00000-of-00001.parquet"``.

        Returns
        -------
        pa.Table
            The Parquet table contents.
        """
        full_path = f"{self._url}/{subpath}"
        logger.debug("Reading Parquet: %s", full_path)
        # Use simplecache for remote URLs to avoid re-downloading.
        open_kwargs: dict[str, object] = dict(self._storage_options)
        if "://" in self._url:
            full_path = f"simplecache::{full_path}"
            open_kwargs["simplecache"] = {"cache_storage": self._cache_storage}
        with fsspec.open(full_path, "rb", **open_kwargs) as f:
            return pq.read_table(f)

    def _load_geometry(self) -> None:
        """Load the shared geometry (node coordinates and connectivity)."""
        table = self._open_parquet(f"geometry/{_PARQUET_FILENAME}")
        row = table.to_pydict()

        coords_x = np.array(row["node_coordinates_x"][0], dtype=np.float64)
        coords_y = np.array(row["node_coordinates_y"][0], dtype=np.float64)
        coords_z = np.zeros_like(coords_x)

        # (n_points, 3) with z=0 for 2-D mesh.
        points_np = np.stack([coords_x, coords_y, coords_z], axis=-1)
        self._points = torch.from_numpy(points_np).float()

        connectivity = row["connectivity"][0]
        cells_np = np.array(connectivity, dtype=np.int64)
        if cells_np.ndim != 2 or cells_np.shape[1] != 3:
            msg = f"Expected triangular connectivity (n_cells, 3), got {cells_np.shape}"
            raise ValueError(msg)
        self._cells = torch.from_numpy(cells_np)

        self._geometry_loaded = True
        logger.info(
            "Loaded geometry: %d points, %d cells",
            self._points.shape[0],
            self._cells.shape[0],
        )

    def _read_parameters(self) -> np.ndarray:
        """Read the viscosity parameters table.

        Returns
        -------
        np.ndarray
            1-D array of viscosity values (one per snapshot).
        """
        table = self._open_parquet(f"parameters/{_PARQUET_FILENAME}")
        return table.column("viscosity").to_numpy()

    def _read_snapshot(self, index: int) -> dict[str, np.ndarray]:
        """Read the field arrays for a single snapshot.

        Parameters
        ----------
        index : int
            Zero-based snapshot index.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of field name to 1-D NumPy array.
        """
        # Cache the full snapshot table on first access (~20 MB for the default
        # 500-snapshot dataset).  This avoids re-reading the Parquet file on
        # every ``__getitem__`` call at the cost of holding the table in memory.
        if self._snapshots_table is None:
            self._snapshots_table = self._open_parquet(f"snapshots/{_PARQUET_FILENAME}")
        row = self._snapshots_table.slice(index, 1).to_pydict()
        return {
            "velocity_x": np.array(row["velocity_x"][0], dtype=np.float64),
            "velocity_y": np.array(row["velocity_y"][0], dtype=np.float64),
            "pressure": np.array(row["pressure"][0], dtype=np.float64),
        }
