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

"""Tests for WallNodeFilter.

Tests construct meshes directly with torch tensors to avoid needing
real d3plot files.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tensordict import TensorDict

pytestmark = pytest.mark.requires("mesh")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_crash_mesh(
    n_points: int = 10,
    n_cells: int = 4,
    n_timesteps: int = 3,
    wall_fraction: float = 0.5,
) -> Mesh:  # ty: ignore[unresolved-reference]  # noqa: F821
    """Create a mock crash mesh with displacement fields.

    Parameters
    ----------
    n_points : int
        Number of mesh nodes.
    n_cells : int
        Number of shell elements (quads).
    n_timesteps : int
        Number of timesteps.
    wall_fraction : float
        Fraction of nodes that are "wall" (zero displacement).

    Returns
    -------
    Mesh
        A Mesh with displacement_t* fields.
    """
    from physicsnemo.mesh import Mesh

    rng = np.random.default_rng(42)

    points = torch.from_numpy(rng.uniform(-10, 10, size=(n_points, 3)).astype(np.float64))

    # Build quad cells that reference only structural (non-wall) nodes
    # so that filtering does not collapse all cells.
    n_wall = int(n_points * wall_fraction)
    structural_indices = list(range(n_wall, n_points))
    cells_list = []
    for _ in range(n_cells):
        cell = rng.choice(structural_indices, size=4, replace=False)
        cells_list.append(cell)
    cells = torch.from_numpy(np.array(cells_list, dtype=np.int64))

    # Build displacements: some nodes move a lot, some don't.
    n_wall = int(n_points * wall_fraction)
    pd_dict: dict[str, torch.Tensor] = {}

    for t in range(n_timesteps):
        disp = np.zeros((n_points, 3), dtype=np.float64)
        if t > 0:
            # Structural nodes get large displacement.
            disp[n_wall:, :] = rng.uniform(-10, 10, size=(n_points - n_wall, 3))
            # Wall nodes get tiny displacement.
            disp[:n_wall, :] = rng.uniform(-0.01, 0.01, size=(n_wall, 3))
        pd_dict[f"displacement_t{t:03d}"] = torch.from_numpy(disp)

    pd_dict["thickness"] = torch.ones(n_points, dtype=torch.float32)

    point_data = TensorDict(pd_dict, batch_size=[n_points])  # ty: ignore[invalid-argument-type]

    # Cell data: simple scalar per cell per timestep.
    cd_dict: dict[str, torch.Tensor] = {}
    for t in range(n_timesteps):
        cd_dict[f"stress_vm_t{t:03d}"] = torch.from_numpy(rng.uniform(0, 100, size=(n_cells,)).astype(np.float64))
    cell_data = TensorDict(cd_dict, batch_size=[n_cells])  # ty: ignore[invalid-argument-type]

    global_data = TensorDict(
        {"num_timesteps": torch.tensor([n_timesteps], dtype=torch.int64)},
        batch_size=[],
    )

    return Mesh(
        points=points,
        cells=cells,
        point_data=point_data,
        cell_data=cell_data,
        global_data=global_data,
    )


# ---------------------------------------------------------------------------
# Unit tests — parameter descriptors
# ---------------------------------------------------------------------------


class TestWallNodeFilterUnit:
    """Unit tests for WallNodeFilter metadata and params."""

    def test_params_list(self) -> None:
        """params() should return a non-empty list of Param objects."""
        from physicsnemo_curator.domains.mesh.filters.wall_node import WallNodeFilter

        params = WallNodeFilter.params()
        assert len(params) == 1
        assert params[0].name == "threshold"

    def test_name_and_description(self) -> None:
        """Class should have name and description ClassVars."""
        from physicsnemo_curator.domains.mesh.filters.wall_node import WallNodeFilter

        assert WallNodeFilter.name == "Wall Node Filter"
        assert len(WallNodeFilter.description) > 0

    def test_negative_threshold_raises(self) -> None:
        """Negative threshold should raise ValueError."""
        from physicsnemo_curator.domains.mesh.filters.wall_node import WallNodeFilter

        with pytest.raises(ValueError, match="non-negative"):
            WallNodeFilter(threshold=-1.0)


# ---------------------------------------------------------------------------
# Integration tests — filtering behavior
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestWallNodeFilterBehavior:
    """Tests verifying the filter correctly removes wall nodes."""

    def test_removes_wall_nodes(self) -> None:
        """Filter should reduce node count by removing wall nodes."""
        from physicsnemo_curator.domains.mesh.filters.wall_node import WallNodeFilter

        mesh = _make_crash_mesh(n_points=10, n_cells=4, n_timesteps=3, wall_fraction=0.5)
        original_points = mesh.n_points

        filt = WallNodeFilter(threshold=0.5)

        def gen():
            yield mesh

        result = list(filt(gen()))
        assert len(result) == 1
        # Should have fewer points than original.
        assert result[0].n_points < original_points

    def test_all_structure_nodes_kept_with_zero_threshold(self) -> None:
        """With threshold=0, all non-stationary nodes should be kept."""
        from physicsnemo_curator.domains.mesh.filters.wall_node import WallNodeFilter

        mesh = _make_crash_mesh(n_points=10, n_cells=4, n_timesteps=3, wall_fraction=0.5)

        filt = WallNodeFilter(threshold=0.0)

        def gen():
            yield mesh

        result = list(filt(gen()))
        assert len(result) == 1
        # With threshold=0, only truly stationary nodes (displacement exactly 0) are removed.
        # Since wall nodes have tiny but non-zero displacement, they should be kept.
        assert result[0].n_points == mesh.n_points

    def test_preserves_displacement_fields(self) -> None:
        """Filtered mesh should still have displacement_t* fields."""
        from physicsnemo_curator.domains.mesh.filters.wall_node import WallNodeFilter

        mesh = _make_crash_mesh(n_points=10, n_cells=4, n_timesteps=3, wall_fraction=0.5)

        filt = WallNodeFilter(threshold=0.5)

        def gen():
            yield mesh

        result = list(filt(gen()))
        assert len(result) == 1
        keys = set(result[0].point_data.keys())
        assert "displacement_t000" in keys
        assert "displacement_t001" in keys
        assert "displacement_t002" in keys
        assert "thickness" in keys

    def test_preserves_global_data(self) -> None:
        """Filtered mesh should carry the same global data."""
        from physicsnemo_curator.domains.mesh.filters.wall_node import WallNodeFilter

        mesh = _make_crash_mesh(n_points=10, n_cells=4, n_timesteps=3, wall_fraction=0.5)

        filt = WallNodeFilter(threshold=0.5)

        def gen():
            yield mesh

        result = list(filt(gen()))
        assert result[0].global_data["num_timesteps"].item() == 3

    def test_connectivity_remapped(self) -> None:
        """Cell indices should be remapped to new contiguous node indices."""
        from physicsnemo_curator.domains.mesh.filters.wall_node import WallNodeFilter

        mesh = _make_crash_mesh(n_points=10, n_cells=4, n_timesteps=3, wall_fraction=0.5)

        filt = WallNodeFilter(threshold=0.5)

        def gen():
            yield mesh

        result = list(filt(gen()))
        new_mesh = result[0]
        # All cell indices should be valid for the new point count.
        assert new_mesh.cells.max().item() < new_mesh.n_points
        assert new_mesh.cells.min().item() >= 0

    def test_cell_data_filtered(self) -> None:
        """Cell data should be filtered to match surviving cells."""
        from physicsnemo_curator.domains.mesh.filters.wall_node import WallNodeFilter

        mesh = _make_crash_mesh(n_points=10, n_cells=4, n_timesteps=3, wall_fraction=0.5)

        filt = WallNodeFilter(threshold=0.5)

        def gen():
            yield mesh

        result = list(filt(gen()))
        new_mesh = result[0]
        if new_mesh.cell_data is not None:
            # Cell data batch dim should match number of cells.
            for key in new_mesh.cell_data.keys():  # noqa: SIM118
                assert new_mesh.cell_data[key].shape[0] == new_mesh.n_cells

    def test_no_displacement_fields_passthrough(self) -> None:
        """Without displacement fields, mesh should pass through unchanged."""
        from physicsnemo.mesh import Mesh

        from physicsnemo_curator.domains.mesh.filters.wall_node import WallNodeFilter

        points = torch.rand(5, 3)
        cells = torch.tensor([[0, 1, 2], [2, 3, 4]], dtype=torch.int64)
        point_data = TensorDict({"temperature": torch.rand(5)}, batch_size=[5])
        mesh = Mesh(points=points, cells=cells, point_data=point_data)

        filt = WallNodeFilter(threshold=1.0)

        def gen():
            yield mesh

        result = list(filt(gen()))
        assert len(result) == 1
        assert result[0] is mesh  # Same object — passed through.

    def test_no_point_data_passthrough(self) -> None:
        """Without point_data, mesh should pass through unchanged."""
        from physicsnemo.mesh import Mesh

        from physicsnemo_curator.domains.mesh.filters.wall_node import WallNodeFilter

        points = torch.rand(5, 3)
        cells = torch.tensor([[0, 1, 2], [2, 3, 4]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        filt = WallNodeFilter(threshold=1.0)

        def gen():
            yield mesh

        result = list(filt(gen()))
        assert len(result) == 1
        assert result[0] is mesh


# ---------------------------------------------------------------------------
# Registry test
# ---------------------------------------------------------------------------


class TestWallNodeFilterRegistry:
    """Verify the filter is registered in the mesh registry."""

    def test_filter_registered(self) -> None:
        """WallNodeFilter should appear in the mesh registry."""
        import physicsnemo_curator.domains.mesh  # noqa: F401
        from physicsnemo_curator.core.registry import registry

        filters = registry.filters("mesh")
        assert "Wall Node Filter" in filters
