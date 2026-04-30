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

"""
Creating a Custom Source
=========================

This example shows how to implement and register a custom
:class:`~physicsnemo_curator.core.base.Source`.

We create a ``CylinderFlowSource`` that reads the `Navier-Stokes
Cylinder <https://huggingface.co/datasets/SISSAmathLab/navier-stokes-cylinder>`_
dataset from HuggingFace Hub using Parquet files.  This demonstrates
the core source contract: indexed access with generator semantics,
lazy loading, and shared geometry caching.

The dataset contains 500 incompressible Navier-Stokes simulations of
flow around a 2-D cylinder at varying viscosities.  Each simulation
has velocity (x, y) and pressure fields on a shared triangular mesh.

.. note::

   Install the mesh extras before running::

       pip install physicsnemo-curator[mesh]
"""

# %%
# Step 1 — Define the Source
# ---------------------------
#
# A source inherits from :class:`~physicsnemo_curator.core.base.Source`
# and implements four things:
#
# 1. ``name`` / ``description`` class variables
# 2. ``params()`` class method (parameter descriptors)
# 3. ``__len__()`` — number of items
# 4. ``__getitem__(index)`` — yield data for a given index

from __future__ import annotations

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


_DEFAULT_URL = "hf://datasets/SISSAmathLab/navier-stokes-cylinder"


class CylinderFlowSource(Source["Mesh"]):
    """Read Navier-Stokes cylinder flow data from HuggingFace Parquet.

    Each pipeline index corresponds to one simulation (viscosity
    parameter).  The underlying geometry (nodes and triangles) is
    shared across all simulations and cached on first access.

    Parameters
    ----------
    url : str
        HuggingFace Hub dataset URL.
    cache_storage : str
        Local directory for caching downloaded Parquet files.
    """

    name: ClassVar[str] = "Cylinder Flow (Custom)"
    description: ClassVar[str] = "Read NS cylinder flow from HF Parquet files"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for this source.

        Returns
        -------
        list[Param]
            Parameters: url (str), cache_storage (str).
        """
        return [
            Param(name="url", description="HuggingFace dataset URL", type=str, default=_DEFAULT_URL),
            Param(name="cache_storage", description="Local cache directory", type=str, default=""),
        ]

    def __init__(self, url: str = _DEFAULT_URL, cache_storage: str = "") -> None:
        self._url = url
        self._cache = cache_storage or tempfile.mkdtemp(prefix="curator_cylinder_")

        # Eagerly load lightweight metadata
        fs = fsspec.filesystem("hf")
        self._fs = fs

        # Read parameter table to determine count
        params_path = f"{url}/parameters/part.0.parquet"
        with fsspec.open(params_path, "rb", hf=fs) as f:
            self._params_table: pa.Table = pq.read_table(f)
        self._count = len(self._params_table)

        # Lazy geometry cache
        self._points: torch.Tensor | None = None
        self._cells: torch.Tensor | None = None

    def _load_geometry(self) -> None:
        """Load shared geometry (nodes + triangles) on first access."""
        if self._points is not None:
            return

        geo_path = f"{self._url}/geometry/part.0.parquet"
        with fsspec.open(geo_path, "rb", hf=self._fs) as f:
            geo_table = pq.read_table(f)

        x = np.array(geo_table.column("x"))
        y = np.array(geo_table.column("y"))
        n_points = len(x)

        self._points = torch.stack(
            [
                torch.from_numpy(x).float(),
                torch.from_numpy(y).float(),
                torch.zeros(n_points),
            ],
            dim=1,
        )

        cells_flat = np.array(geo_table.column("triangles")[0].as_py())
        self._cells = torch.from_numpy(cells_flat.reshape(-1, 3).astype(np.int64))

    def __len__(self) -> int:
        """Return the number of simulations."""
        return self._count

    def __getitem__(self, index: int) -> Generator[Mesh]:
        """Yield a Mesh for the simulation at *index*.

        Parameters
        ----------
        index : int
            Zero-based simulation index.  Negative indices supported.

        Yields
        ------
        Mesh
            Mesh with velocity and pressure fields.

        Raises
        ------
        IndexError
            If index is out of range.
        """
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            msg = f"Index {index} out of range for source with {len(self)} items."
            raise IndexError(msg)

        self._load_geometry()
        assert self._points is not None
        assert self._cells is not None

        # Read snapshot for this index
        snap_path = f"{self._url}/snapshots/part.0.parquet"
        with fsspec.open(snap_path, "rb", hf=self._fs) as f:
            snap_table = pq.read_table(f)

        row = snap_table.slice(index, 1)
        vx = np.array(row.column("velocity_x")[0].as_py(), dtype=np.float32)
        vy = np.array(row.column("velocity_y")[0].as_py(), dtype=np.float32)
        p = np.array(row.column("pressure")[0].as_py(), dtype=np.float32)

        n_points = self._points.shape[0]
        point_data = TensorDict(
            {
                "velocity_x": torch.from_numpy(vx),
                "velocity_y": torch.from_numpy(vy),
                "pressure": torch.from_numpy(p),
            },
            batch_size=[n_points],
        )

        viscosity = float(self._params_table.column("viscosity")[index].as_py())
        global_data = TensorDict(
            {"viscosity": torch.tensor(viscosity)},
            batch_size=[],
        )

        yield Mesh(
            points=self._points,
            cells=self._cells,
            point_data=point_data,
            global_data=global_data,
        )


# %%
# Step 2 — Register the Source (Optional)
# ----------------------------------------
#
# Registration makes the source discoverable via the global registry
# and the interactive CLI.

from physicsnemo_curator.core.registry import registry

registry.register_source("mesh", CylinderFlowSource)

registered = registry.sources("mesh")
print(f"Registered mesh sources: {list(registered.keys())}")
assert "Cylinder Flow (Custom)" in registered

# %%
# Step 3 — Use in a Pipeline
# ---------------------------
#
# The custom source works with any compatible filter and sink.

from physicsnemo_curator.domains.mesh.filters.mean import MeanFilter
from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink
from physicsnemo_curator.run import run_pipeline

source = CylinderFlowSource()
print(f"Simulations available: {len(source)}")

pipeline = source.filter(MeanFilter(output="outputs/extending/cylinder_stats.parquet")).write(
    MeshSink(output_dir="outputs/extending/cylinder_meshes/")
)

results = run_pipeline(
    pipeline,
    n_jobs=1,
    backend="sequential",
    indices=range(3),
    progress=True,
)

print(f"\nProcessed {len(results)} simulations")
for i, paths in enumerate(results):
    print(f"  Simulation {i}: {paths}")

# %%
# Step 4 — Verify Output
# -----------------------
#
# Load a saved mesh and inspect its contents.

mesh = Mesh.load(results[0][0])  # ty: ignore[unresolved-attribute]
print(f"\nLoaded mesh from {results[0][0]}:")
print(f"  Points: {mesh.n_points}")
print(f"  Cells: {mesh.n_cells}")
print(f"  Point fields: {list(mesh.point_data.keys())}")
print(f"  Global fields: {list(mesh.global_data.keys())}")

# %%
# Summary
# -------
#
# To create a custom source:
#
# 1. Subclass :class:`~physicsnemo_curator.core.base.Source` with a
#    type parameter (``Source["Mesh"]``, ``Source["xr.DataArray"]``,
#    etc.)
# 2. Set ``name`` and ``description`` class variables
# 3. Implement ``params()``, ``__len__()``, and ``__getitem__(index)``
# 4. Use **generator semantics** — ``__getitem__`` must ``yield``
# 5. Support negative indexing and raise ``IndexError`` for out-of-bounds
# 6. **Eagerly** load lightweight metadata in ``__init__``
# 7. **Lazily** load heavy data (geometry, fields) in ``__getitem__``
# 8. **Cache** shared data (like geometry) across indices
# 9. Optionally register with ``registry.register_source()``
