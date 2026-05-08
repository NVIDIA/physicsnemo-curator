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

"""Tests for EdgeComputeFilter."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tensordict import TensorDict

pytestmark = pytest.mark.requires("mesh")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def triangle_mesh():
    """Create a mesh with triangle cells."""
    from physicsnemo.mesh import Mesh

    # 4 points forming 2 triangles
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.5, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    # 2 triangles: [0,1,2] and [1,3,2]
    cells = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int64)

    return Mesh(points=points, cells=cells)


@pytest.fixture
def mesh_with_point_data():
    """Create a mesh with point data."""
    from physicsnemo.mesh import Mesh

    points = torch.randn(5, 3, dtype=torch.float32)
    cells = torch.tensor([[0, 1, 2], [2, 3, 4]], dtype=torch.int64)

    point_data = TensorDict(
        {"displacement_t000": torch.randn(5, 3, dtype=torch.float32)},
        batch_size=[5],
    )

    global_data = TensorDict(
        {"num_timesteps": torch.tensor([1], dtype=torch.int64)},
        batch_size=[],
    )

    return Mesh(
        points=points,
        cells=cells,
        point_data=point_data,
        global_data=global_data,
    )


@pytest.fixture
def mesh_without_cells():
    """Create a mesh without cells (point cloud)."""
    from physicsnemo.mesh import Mesh

    points = torch.randn(10, 3, dtype=torch.float32)
    return Mesh(points=points, cells=None)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestEdgeComputeFilterUnit:
    """Unit tests for EdgeComputeFilter."""

    def test_params_returns_empty(self):
        """params() should return empty list."""
        from physicsnemo_curator.domains.mesh.filters.edge_compute import EdgeComputeFilter

        params = EdgeComputeFilter.params()
        assert params == []

    def test_init(self):
        """EdgeComputeFilter should initialize without error."""
        from physicsnemo_curator.domains.mesh.filters.edge_compute import EdgeComputeFilter

        edge_filter = EdgeComputeFilter()
        assert edge_filter is not None


class TestEdgeComputeFilterIntegration:
    """Integration tests for EdgeComputeFilter."""

    def test_computes_edges(self, triangle_mesh):
        """EdgeComputeFilter should compute edges from cells."""
        from physicsnemo_curator.domains.mesh.filters.edge_compute import EdgeComputeFilter

        edge_filter = EdgeComputeFilter()

        def gen():
            yield triangle_mesh

        results = list(edge_filter(gen()))

        assert len(results) == 1
        mesh = results[0]

        # Check edges are in global_data
        assert mesh.global_data is not None
        assert "edges" in mesh.global_data

        edges = mesh.global_data["edges"]
        assert edges.shape[1] == 2  # Each edge has 2 nodes

        # 2 triangles share edge [1,2], so total unique edges = 5
        # Triangle 1: [0,1], [1,2], [2,0]
        # Triangle 2: [1,3], [3,2], [2,1] -> [1,2] is shared
        # Unique: [0,1], [1,2], [0,2], [1,3], [2,3]
        assert edges.shape[0] == 5

    def test_preserves_existing_data(self, mesh_with_point_data):
        """EdgeComputeFilter should preserve point_data and existing global_data."""
        from physicsnemo_curator.domains.mesh.filters.edge_compute import EdgeComputeFilter

        edge_filter = EdgeComputeFilter()

        def gen():
            yield mesh_with_point_data

        results = list(edge_filter(gen()))
        mesh = results[0]

        # Check point_data preserved
        assert mesh.point_data is not None
        assert "displacement_t000" in mesh.point_data

        # Check existing global_data preserved
        assert "num_timesteps" in mesh.global_data

        # Check edges added
        assert "edges" in mesh.global_data

    def test_handles_mesh_without_cells(self, mesh_without_cells, caplog):
        """EdgeComputeFilter should pass through meshes without cells."""
        from physicsnemo_curator.domains.mesh.filters.edge_compute import EdgeComputeFilter

        edge_filter = EdgeComputeFilter()

        def gen():
            yield mesh_without_cells

        results = list(edge_filter(gen()))

        assert len(results) == 1

        # Should pass through unchanged, with warning
        assert "mesh has no cells" in caplog.text

    def test_edges_are_valid_indices(self, triangle_mesh):
        """Computed edges should contain valid node indices."""
        from physicsnemo_curator.domains.mesh.filters.edge_compute import EdgeComputeFilter

        edge_filter = EdgeComputeFilter()

        def gen():
            yield triangle_mesh

        results = list(edge_filter(gen()))
        mesh = results[0]

        edges = mesh.global_data["edges"].numpy()
        n_points = mesh.n_points

        # All edge indices should be valid
        assert np.all(edges >= 0)  # ty: ignore[unsupported-operator]
        assert np.all(edges < n_points)  # ty: ignore[unsupported-operator]

    def test_multiple_meshes(self, triangle_mesh, mesh_with_point_data):
        """EdgeComputeFilter should handle multiple meshes."""
        from physicsnemo_curator.domains.mesh.filters.edge_compute import EdgeComputeFilter

        edge_filter = EdgeComputeFilter()

        def gen():
            yield triangle_mesh
            yield mesh_with_point_data

        results = list(edge_filter(gen()))

        assert len(results) == 2
        for mesh in results:
            assert "edges" in mesh.global_data
