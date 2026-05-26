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

"""Creating a Custom Source.

See README.md for a full walkthrough of this example.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch
from physicsnemo.mesh import Mesh
from tensordict import TensorDict

from physicsnemo_curator.core.base import Param, Source

if TYPE_CHECKING:
    from collections.abc import Generator


# Step 1 — Define the Source


class SineFlowSource(Source["Mesh"]):
    """Generate synthetic 2D flow fields with a sinusoidal velocity pattern.

    Each pipeline index corresponds to a different phase offset, producing
    a family of flow fields on a shared triangular grid.  No network access
    or external data files are required.

    Parameters
    ----------
    n_samples : int
        Number of flow snapshots this source provides.
    n_points : int
        Number of mesh vertices (arranged on a regular grid).
    seed : int
        Random seed for mesh geometry jitter.
    """

    name: ClassVar[str] = "Sine Flow (Custom)"
    description: ClassVar[str] = "Generate synthetic sinusoidal flow on a 2D mesh"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for this source.

        Returns
        -------
        list[Param]
            Parameters: n_samples, n_points, seed.
        """
        return [
            Param(name="n_samples", description="Number of flow snapshots", type=int, default=10),
            Param(name="n_points", description="Number of mesh vertices", type=int, default=100),
            Param(name="seed", description="Random seed for geometry", type=int, default=42),
        ]

    def __init__(self, n_samples: int = 10, n_points: int = 100, seed: int = 42) -> None:
        self._n_samples = n_samples
        self._n_points = n_points
        self._seed = seed

        # Build a simple 2D grid with triangulation
        rng = np.random.default_rng(seed)
        side = int(np.ceil(np.sqrt(n_points)))
        xs = np.linspace(0, 1, side, dtype=np.float32)
        ys = np.linspace(0, 1, side, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        pts = np.stack([xx.ravel(), yy.ravel(), np.zeros(side * side, dtype=np.float32)], axis=1)
        # Add small jitter
        pts[:, :2] += rng.standard_normal(pts[:, :2].shape).astype(np.float32) * 0.01
        self._points = torch.from_numpy(pts[:n_points])

        # Simple triangulation: connect adjacent grid points
        cells = []
        for i in range(side - 1):
            for j in range(side - 1):
                v0 = i * side + j
                v1 = v0 + 1
                v2 = v0 + side
                v3 = v2 + 1
                if v3 < n_points:
                    cells.append([v0, v1, v2])
                    cells.append([v1, v3, v2])
        self._cells = torch.tensor(cells, dtype=torch.int64) if cells else torch.zeros((0, 3), dtype=torch.int64)

    def __len__(self) -> int:
        """Return the number of flow snapshots."""
        return self._n_samples

    def __getitem__(self, index: int) -> Generator[Mesh]:
        """Yield a Mesh for the flow snapshot at *index*.

        Parameters
        ----------
        index : int
            Zero-based snapshot index.  Negative indices supported.

        Yields
        ------
        Mesh
            Mesh with velocity_x, velocity_y, and pressure fields.

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

        # Generate sinusoidal flow with phase offset per index
        phase = 2.0 * np.pi * index / self._n_samples
        x_coords = self._points[:, 0].numpy()
        y_coords = self._points[:, 1].numpy()

        vx = np.sin(2 * np.pi * x_coords + phase).astype(np.float32)
        vy = np.cos(2 * np.pi * y_coords + phase).astype(np.float32)
        pressure = (np.sin(np.pi * x_coords) * np.cos(np.pi * y_coords)).astype(np.float32)

        n_points = self._points.shape[0]
        point_data = TensorDict(
            {
                "velocity_x": torch.from_numpy(vx),
                "velocity_y": torch.from_numpy(vy),
                "pressure": torch.from_numpy(pressure),
            },
            batch_size=[n_points],
        )

        global_data = TensorDict(
            {"phase": torch.tensor(phase, dtype=torch.float32)},
            batch_size=[],
        )

        yield Mesh(
            points=self._points,
            cells=self._cells,
            point_data=point_data,
            global_data=global_data,
        )


# Step 2 — Register the Source (Optional)

import physicsnemo_curator.domains.mesh  # noqa: F401 - registers "mesh" submodule
from physicsnemo_curator.core.registry import registry

registry.register_source("mesh", SineFlowSource)

registered = registry.sources("mesh")
print(f"Registered mesh sources: {list(registered.keys())}")
assert "Sine Flow (Custom)" in registered

# Step 3 — Use in a Pipeline

from physicsnemo_curator.domains.mesh.filters.mean import MeanFilter
from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink
from physicsnemo_curator.run import run_pipeline

source = SineFlowSource(n_samples=5, n_points=64)
print(f"Simulations available: {len(source)}")

pipeline = source.filter(MeanFilter(output="output/extending/sine_stats.parquet")).write(
    MeshSink(output_dir="output/extending/sine_meshes/")
)

results = run_pipeline(
    pipeline,
    n_jobs=1,
    backend="sequential",
    indices=range(3),
    use_tui=True,
)

print(f"\nProcessed {len(results)} simulations")
for i, paths in enumerate(results):
    print(f"  Simulation {i}: {paths}")

# Step 4 — Verify Output

mesh = Mesh.load(results[0][0])
print(f"\nLoaded mesh from {results[0][0]}:")
print(f"  Points: {mesh.n_points}")
print(f"  Cells: {mesh.n_cells}")
print(f"  Point fields: {list(mesh.point_data.keys())}")
print(f"  Global fields: {list(mesh.global_data.keys())}")
