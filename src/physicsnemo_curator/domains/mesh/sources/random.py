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

"""Random mesh source for testing and examples.

Generates synthetic :class:`~physicsnemo.mesh.Mesh` objects with random
geometry and scalar/vector fields.  Useful for unit tests, example
pipelines, and quick prototyping without needing real data files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

from physicsnemo_curator.core.base import Param, Source

if TYPE_CHECKING:
    from collections.abc import Generator

    from physicsnemo.mesh import Mesh


class RandomMeshSource(Source["Mesh"]):
    """Generate random tetrahedral meshes with configurable fields.

    Each index yields a single :class:`~physicsnemo.mesh.Mesh` with
    random point coordinates, tetrahedral connectivity, and a set of
    random scalar point-data fields.

    Parameters
    ----------
    n_samples : int
        Number of meshes this source provides (i.e. ``len(source)``).
    n_points : int
        Number of vertices per mesh.
    n_cells : int
        Number of tetrahedral cells per mesh.
    n_fields : int
        Number of random scalar point-data fields to generate.
    n_timesteps : int
        Number of displacement timesteps to generate (produces
        ``displacement_t000``, ``displacement_t001``, ...).
    seed : int
        Base random seed.  Each index uses ``seed + index`` for
        reproducibility.

    Examples
    --------
    >>> from physicsnemo_curator.domains.mesh.sources import RandomMeshSource
    >>> source = RandomMeshSource(n_samples=10, n_points=100, n_cells=50)
    >>> len(source)
    10
    >>> mesh = next(source[0])
    >>> mesh.points.shape
    torch.Size([100, 3])
    """

    name: ClassVar[str] = "Random Mesh"
    description: ClassVar[str] = (
        "Generate random tetrahedral meshes with configurable fields for testing and prototyping"
    )

    @classmethod
    def params(cls) -> list[Param]:
        """Declare configurable parameters."""
        return [
            Param(name="n_samples", description="Number of meshes to generate", type=int, default=10),
            Param(name="n_points", description="Number of vertices per mesh", type=int, default=1000),
            Param(name="n_cells", description="Number of tetrahedral cells per mesh", type=int, default=500),
            Param(name="n_fields", description="Number of random scalar point-data fields", type=int, default=3),
            Param(name="n_timesteps", description="Number of displacement timesteps", type=int, default=5),
            Param(name="seed", description="Base random seed for reproducibility", type=int, default=42),
        ]

    def __init__(
        self,
        n_samples: int = 10,
        n_points: int = 1000,
        n_cells: int = 500,
        n_fields: int = 3,
        n_timesteps: int = 5,
        seed: int = 42,
    ) -> None:
        """Initialize the random mesh source."""
        self._n_samples = n_samples
        self._n_points = n_points
        self._n_cells = n_cells
        self._n_fields = n_fields
        self._n_timesteps = n_timesteps
        self._seed = seed

    def __len__(self) -> int:
        """Return the number of meshes available."""
        return self._n_samples

    def __getitem__(self, index: int) -> Generator[Mesh]:
        """Yield a random mesh for the given index.

        Parameters
        ----------
        index : int
            Zero-based index into the source.

        Yields
        ------
        Mesh
            A randomly generated mesh with point data fields.
        """
        from physicsnemo.mesh import Mesh
        from tensordict import TensorDict

        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self)})")

        rng = np.random.default_rng(self._seed + index)

        # Random point coordinates in [0, 1]^3
        points = torch.from_numpy(rng.random((self._n_points, 3), dtype=np.float32))

        # Random tetrahedral connectivity (4 nodes per cell)
        cells = torch.from_numpy(rng.integers(0, self._n_points, size=(self._n_cells, 4), dtype=np.int64))

        # Build point data fields
        pd: dict[str, torch.Tensor] = {}

        # Displacement timesteps (3D vectors)
        for t in range(self._n_timesteps):
            disp = rng.standard_normal((self._n_points, 3)).astype(np.float32) * 0.01 * (t + 1)
            pd[f"displacement_t{t:04d}"] = torch.from_numpy(disp)

        # Random scalar fields
        for f in range(self._n_fields):
            pd[f"field_{f}"] = torch.from_numpy(rng.standard_normal(self._n_points).astype(np.float32))

        point_data = TensorDict(pd, batch_size=[self._n_points])

        mesh = Mesh(
            points=points,
            cells=cells,
            point_data=point_data,
        )
        yield mesh
