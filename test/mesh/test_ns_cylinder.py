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

"""Tests for NavierStokesCylinderSource.

Unit tests use local Parquet fixtures to avoid network access.
E2E tests (marked ``slow``) hit the real HuggingFace Hub.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

if TYPE_CHECKING:
    import pathlib


# ---------------------------------------------------------------------------
# Helpers — write mock Parquet files matching the HF dataset schema
# ---------------------------------------------------------------------------


def _write_mock_dataset(root: pathlib.Path, n_points: int = 10, n_cells: int = 12, n_snapshots: int = 3) -> None:
    """Create a minimal mock of the SISSAmathLab/navier-stokes-cylinder layout.

    Parameters
    ----------
    root : pathlib.Path
        Directory to write into (acts as the dataset root).
    n_points : int
        Number of mesh nodes.
    n_cells : int
        Number of triangular cells.
    n_snapshots : int
        Number of parameter/snapshot rows.
    """
    rng = np.random.default_rng(42)

    # -- geometry (1 row) ---------------------------------------------------
    coords_x = rng.uniform(0.0, 22.0, size=n_points).tolist()
    coords_y = rng.uniform(-5.0, 5.0, size=n_points).tolist()
    connectivity = [rng.integers(0, n_points, size=3).tolist() for _ in range(n_cells)]

    geo_table = pa.table(
        {
            "node_coordinates_x": pa.array([coords_x], type=pa.list_(pa.float64())),
            "node_coordinates_y": pa.array([coords_y], type=pa.list_(pa.float64())),
            "connectivity": pa.array([connectivity], type=pa.list_(pa.list_(pa.int32()))),
        }
    )
    geo_dir = root / "geometry"
    geo_dir.mkdir(parents=True)
    pq.write_table(geo_table, geo_dir / "default-00000-of-00001.parquet")

    # -- parameters (n_snapshots rows) --------------------------------------
    viscosities = rng.uniform(1.0, 80.0, size=n_snapshots).tolist()
    param_table = pa.table({"viscosity": pa.array(viscosities, type=pa.float64())})
    param_dir = root / "parameters"
    param_dir.mkdir(parents=True)
    pq.write_table(param_table, param_dir / "default-00000-of-00001.parquet")

    # -- snapshots (n_snapshots rows) ---------------------------------------
    snap_table = pa.table(
        {
            "velocity_x": pa.array(
                [rng.standard_normal(n_points).tolist() for _ in range(n_snapshots)],
                type=pa.list_(pa.float64()),
            ),
            "velocity_y": pa.array(
                [rng.standard_normal(n_points).tolist() for _ in range(n_snapshots)],
                type=pa.list_(pa.float64()),
            ),
            "pressure": pa.array(
                [rng.standard_normal(n_points).tolist() for _ in range(n_snapshots)],
                type=pa.list_(pa.float64()),
            ),
        }
    )
    snap_dir = root / "snapshots"
    snap_dir.mkdir(parents=True)
    pq.write_table(snap_table, snap_dir / "default-00000-of-00001.parquet")


# ---------------------------------------------------------------------------
# Unit tests — parameter descriptors and metadata
# ---------------------------------------------------------------------------


@pytest.mark.requires("mesh")
class TestNavierStokesCylinderSourceUnit:
    """Unit tests for NavierStokesCylinderSource metadata and params."""

    def test_params_list(self) -> None:
        """params() should return a non-empty list of Param objects."""
        from physicsnemo_curator.mesh.sources.ns_cylinder import NavierStokesCylinderSource

        params = NavierStokesCylinderSource.params()
        assert len(params) > 0
        names = [p.name for p in params]
        assert "url" in names
        assert "cache_storage" in names

    def test_name_and_description(self) -> None:
        """Class should have name and description ClassVars."""
        from physicsnemo_curator.mesh.sources.ns_cylinder import NavierStokesCylinderSource

        assert NavierStokesCylinderSource.name == "Navier-Stokes Cylinder"
        assert len(NavierStokesCylinderSource.description) > 0


# ---------------------------------------------------------------------------
# Unit tests — reading from local mock Parquet
# ---------------------------------------------------------------------------


@pytest.mark.requires("mesh")
class TestNavierStokesCylinderSourceLocal:
    """Unit tests using local mock Parquet files."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: pathlib.Path) -> None:
        """Write mock Parquet dataset and build source pointing at it."""
        mock_root = tmp_path / "ns_cylinder"
        _write_mock_dataset(mock_root, n_points=10, n_cells=12, n_snapshots=3)
        self.mock_root = mock_root
        self.tmp_path = tmp_path

    def test_len(self) -> None:
        """Length should equal the number of snapshots."""
        from physicsnemo_curator.mesh.sources.ns_cylinder import NavierStokesCylinderSource

        source = NavierStokesCylinderSource(url=str(self.mock_root))
        assert len(source) == 3

    def test_getitem_returns_mesh(self) -> None:
        """__getitem__ should yield a physicsnemo Mesh."""
        from physicsnemo.mesh import Mesh

        from physicsnemo_curator.mesh.sources.ns_cylinder import NavierStokesCylinderSource

        source = NavierStokesCylinderSource(url=str(self.mock_root))
        mesh = next(source[0])
        assert isinstance(mesh, Mesh)

    def test_mesh_geometry(self) -> None:
        """Mesh should have correct point and cell counts."""
        from physicsnemo_curator.mesh.sources.ns_cylinder import NavierStokesCylinderSource

        source = NavierStokesCylinderSource(url=str(self.mock_root))
        mesh = next(source[0])
        assert mesh.n_points == 10
        assert mesh.n_cells == 12
        assert mesh.n_spatial_dims == 3
        assert mesh.n_manifold_dims == 2

    def test_mesh_has_point_data(self) -> None:
        """Mesh should carry velocity and pressure as point data."""
        from physicsnemo_curator.mesh.sources.ns_cylinder import NavierStokesCylinderSource

        source = NavierStokesCylinderSource(url=str(self.mock_root))
        mesh = next(source[0])
        point_data_keys = set(mesh.point_data.keys())
        assert "velocity_x" in point_data_keys
        assert "velocity_y" in point_data_keys
        assert "pressure" in point_data_keys

    def test_mesh_has_global_data(self) -> None:
        """Mesh should carry viscosity as global data."""
        from physicsnemo_curator.mesh.sources.ns_cylinder import NavierStokesCylinderSource

        source = NavierStokesCylinderSource(url=str(self.mock_root))
        mesh = next(source[0])
        assert "viscosity" in mesh.global_data

    def test_different_indices_different_data(self) -> None:
        """Different snapshot indices should yield different field values."""
        import torch

        from physicsnemo_curator.mesh.sources.ns_cylinder import NavierStokesCylinderSource

        source = NavierStokesCylinderSource(url=str(self.mock_root))
        mesh_0 = next(source[0])
        mesh_1 = next(source[1])
        # Velocity fields should differ between snapshots
        assert not torch.equal(mesh_0.point_data["velocity_x"], mesh_1.point_data["velocity_x"])

    def test_negative_index(self) -> None:
        """Negative indexing should work."""
        from physicsnemo.mesh import Mesh

        from physicsnemo_curator.mesh.sources.ns_cylinder import NavierStokesCylinderSource

        source = NavierStokesCylinderSource(url=str(self.mock_root))
        mesh = next(source[-1])
        assert isinstance(mesh, Mesh)

    def test_out_of_range_raises(self) -> None:
        """Out-of-range index should raise IndexError."""
        from physicsnemo_curator.mesh.sources.ns_cylinder import NavierStokesCylinderSource

        source = NavierStokesCylinderSource(url=str(self.mock_root))
        with pytest.raises(IndexError):
            next(source[99])


# ---------------------------------------------------------------------------
# Unit tests — registry integration
# ---------------------------------------------------------------------------


@pytest.mark.requires("mesh")
class TestNavierStokesCylinderRegistry:
    """Verify the source is registered in the mesh registry."""

    def test_source_registered(self) -> None:
        """NavierStokesCylinderSource should appear in the mesh registry."""
        import physicsnemo_curator.mesh  # noqa: F401
        from physicsnemo_curator.core.registry import registry

        sources = registry.list_sources("mesh")
        source_names = {s.name for s in sources}
        assert "Navier-Stokes Cylinder" in source_names


# ---------------------------------------------------------------------------
# E2E tests — real HuggingFace downloads
# ---------------------------------------------------------------------------


@pytest.mark.requires("mesh")
@pytest.mark.e2e
@pytest.mark.slow
class TestNavierStokesCylinderSourceE2E:
    """End-to-end tests fetching real data from HuggingFace.

    Downloads from ``SISSAmathLab/navier-stokes-cylinder`` and
    verifies the full Parquet -> Mesh pipeline.

    The ``slow`` marker allows deselecting in quick CI runs.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: pathlib.Path) -> None:
        """Build the source with a local cache directory."""
        from physicsnemo_curator.mesh.sources.ns_cylinder import NavierStokesCylinderSource

        self.source = NavierStokesCylinderSource(
            cache_storage=str(tmp_path / "cache"),
        )
        self.tmp_path = tmp_path

    def test_discovers_snapshots(self) -> None:
        """Should discover 500 snapshots."""
        assert len(self.source) == 500

    def test_reads_mesh(self) -> None:
        """Should read the first snapshot as a valid Mesh."""
        from physicsnemo.mesh import Mesh

        mesh = next(self.source[0])
        assert isinstance(mesh, Mesh)
        assert mesh.n_points == 1639
        assert mesh.n_cells == 3091

    def test_mesh_is_2d_surface(self) -> None:
        """Mesh should be a 2D triangulated surface in 3D space."""
        mesh = next(self.source[0])
        assert mesh.n_spatial_dims == 3
        assert mesh.n_manifold_dims == 2

    def test_mesh_has_point_data(self) -> None:
        """Mesh should carry velocity and pressure point data."""
        mesh = next(self.source[0])
        keys = set(mesh.point_data.keys())
        assert "velocity_x" in keys
        assert "velocity_y" in keys
        assert "pressure" in keys

    def test_mesh_has_global_viscosity(self) -> None:
        """Mesh should carry viscosity as global data."""
        mesh = next(self.source[0])
        assert "viscosity" in mesh.global_data
        visc = mesh.global_data["viscosity"]
        assert visc.numel() == 1
        assert visc.item() > 0

    def test_different_snapshots_different_viscosity(self) -> None:
        """Different snapshots should have different viscosity values."""
        mesh_0 = next(self.source[0])
        mesh_1 = next(self.source[1])
        # Very likely different (500 distinct values)
        assert mesh_0.global_data["viscosity"].item() != mesh_1.global_data["viscosity"].item()
