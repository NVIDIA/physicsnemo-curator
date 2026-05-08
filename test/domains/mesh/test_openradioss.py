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

"""Tests for OpenRadiossSource."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.requires("mesh")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_vtk_run(tmp_path: Path):
    """Create a mock run directory with VTK files."""
    import pyvista as pv

    run_dir = tmp_path / "run0001"
    run_dir.mkdir()

    n_points = 10
    n_cells = 4
    n_timesteps = 3

    rng = np.random.default_rng(42)

    # Create consistent mesh structure
    points_base = rng.uniform(-10, 10, size=(n_points, 3)).astype(np.float32)
    cells = rng.integers(0, n_points, size=(n_cells, 3)).astype(np.int64)

    # Build cell array in VTK format: [n_nodes, n0, n1, n2, ...]
    cell_array = []
    for cell in cells:
        cell_array.extend([3, *cell])
    cell_array = np.array(cell_array, dtype=np.int64)
    cell_types = np.full(n_cells, 5, dtype=np.uint8)  # VTK_TRIANGLE = 5

    for t in range(n_timesteps):
        # Simulate motion: add displacement to base positions
        displacement = rng.uniform(-1, 1, size=(n_points, 3)).astype(np.float32) * t * 0.1
        points = points_base + displacement

        grid = pv.UnstructuredGrid(cell_array, cell_types, points)

        # Add optional fields
        grid.point_data["velocity"] = rng.uniform(-1, 1, size=(n_points, 3)).astype(np.float32)
        grid.cell_data["stress"] = rng.uniform(0, 100, size=(n_cells, 6)).astype(np.float32)

        vtk_path = run_dir / f"timestep_{t:03d}.vtk"
        grid.save(str(vtk_path))

    return tmp_path


@pytest.fixture
def empty_dir(tmp_path: Path):
    """Create an empty directory."""
    empty = tmp_path / "empty"
    empty.mkdir()
    return empty


@pytest.fixture
def dir_with_no_runs(tmp_path: Path):
    """Create a directory with no valid run subdirectories."""
    # VTK files in root, not in subdirectories
    import pyvista as pv

    points = np.random.rand(5, 3).astype(np.float32)
    grid = pv.PolyData(points)
    grid.save(str(tmp_path / "orphan.vtk"))

    return tmp_path


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestOpenRadiossSourceUnit:
    """Unit tests for OpenRadiossSource."""

    def test_params(self):
        """params() should return expected descriptors."""
        from physicsnemo_curator.domains.mesh.sources.openradioss import OpenRadiossSource

        params = OpenRadiossSource.params()
        names = [p.name for p in params]
        assert "input_dir" in names
        assert "vtk_glob" in names
        assert "read_stress" in names
        assert "read_velocity" in names

    def test_init_nonexistent_dir(self):
        """OpenRadiossSource should raise FileNotFoundError for nonexistent dir."""
        from physicsnemo_curator.domains.mesh.sources.openradioss import OpenRadiossSource

        with pytest.raises(FileNotFoundError, match="does not exist"):
            OpenRadiossSource(input_dir="/nonexistent/path")

    def test_init_no_runs(self, dir_with_no_runs):
        """OpenRadiossSource should discover zero runs if none exist."""
        from physicsnemo_curator.domains.mesh.sources.openradioss import OpenRadiossSource

        source = OpenRadiossSource(input_dir=str(dir_with_no_runs))
        assert len(source) == 0

    def test_von_mises_from_voigt(self):
        """_von_mises_from_voigt should compute correct stress."""
        from physicsnemo_curator.domains.mesh.sources.openradioss import _von_mises_from_voigt

        # Pure tension in x direction: [100, 0, 0, 0, 0, 0]
        sig = np.array([[100.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        vm = _von_mises_from_voigt(sig)
        np.testing.assert_allclose(vm, [100.0], rtol=1e-5)

        # Hydrostatic: [100, 100, 100, 0, 0, 0] -> vm = 0
        sig = np.array([[100.0, 100.0, 100.0, 0.0, 0.0, 0.0]])
        vm = _von_mises_from_voigt(sig)
        np.testing.assert_allclose(vm, [0.0], atol=1e-5)


class TestOpenRadiossSourceIntegration:
    """Integration tests for OpenRadiossSource."""

    def test_discover_runs(self, mock_vtk_run):
        """OpenRadiossSource should discover run directories."""
        from physicsnemo_curator.domains.mesh.sources.openradioss import OpenRadiossSource

        source = OpenRadiossSource(input_dir=str(mock_vtk_run))
        assert len(source) == 1

    def test_read_mesh(self, mock_vtk_run):
        """OpenRadiossSource should read mesh with displacement fields."""
        from physicsnemo_curator.domains.mesh.sources.openradioss import OpenRadiossSource

        source = OpenRadiossSource(input_dir=str(mock_vtk_run))
        meshes = list(source[0])

        assert len(meshes) == 1
        mesh = meshes[0]

        # Check structure
        assert mesh.points is not None
        assert mesh.cells is not None
        assert mesh.point_data is not None
        assert mesh.global_data is not None

        # Check displacement fields
        assert "displacement_t000" in mesh.point_data
        assert "displacement_t001" in mesh.point_data
        assert "displacement_t002" in mesh.point_data

        # Check thickness (should be zeros for solid elements)
        thickness = mesh.point_data["thickness"].numpy()
        np.testing.assert_array_equal(thickness, np.zeros_like(thickness))

        # Check global data
        assert mesh.global_data["num_timesteps"].item() == 3

    def test_read_velocity(self, mock_vtk_run):
        """OpenRadiossSource should read velocity when requested."""
        from physicsnemo_curator.domains.mesh.sources.openradioss import OpenRadiossSource

        source = OpenRadiossSource(input_dir=str(mock_vtk_run), read_velocity=True)
        meshes = list(source[0])
        mesh = meshes[0]

        assert "velocity_t000" in mesh.point_data

    def test_read_stress(self, mock_vtk_run):
        """OpenRadiossSource should read stress when requested."""
        from physicsnemo_curator.domains.mesh.sources.openradioss import OpenRadiossSource

        source = OpenRadiossSource(input_dir=str(mock_vtk_run), read_stress=True)
        meshes = list(source[0])
        mesh = meshes[0]

        # Stress should be in cell_data as von Mises
        assert mesh.cell_data is not None
        assert "stress_vm_t000" in mesh.cell_data

    def test_displacement_is_relative_to_t0(self, mock_vtk_run):
        """Displacement at t=0 should be approximately zero."""
        from physicsnemo_curator.domains.mesh.sources.openradioss import OpenRadiossSource

        source = OpenRadiossSource(input_dir=str(mock_vtk_run))
        meshes = list(source[0])
        mesh = meshes[0]

        disp_t0 = mesh.point_data["displacement_t000"].numpy()
        np.testing.assert_array_equal(disp_t0, np.zeros_like(disp_t0))

    def test_negative_indexing(self, mock_vtk_run):
        """OpenRadiossSource should support negative indexing."""
        from physicsnemo_curator.domains.mesh.sources.openradioss import OpenRadiossSource

        source = OpenRadiossSource(input_dir=str(mock_vtk_run))
        meshes = list(source[-1])

        assert len(meshes) == 1

    def test_index_out_of_range(self, mock_vtk_run):
        """OpenRadiossSource should raise IndexError for invalid index."""
        from physicsnemo_curator.domains.mesh.sources.openradioss import OpenRadiossSource

        source = OpenRadiossSource(input_dir=str(mock_vtk_run))

        with pytest.raises(IndexError):
            list(source[100])

    def test_run_id(self, mock_vtk_run):
        """run_id should return directory name."""
        from physicsnemo_curator.domains.mesh.sources.openradioss import OpenRadiossSource

        source = OpenRadiossSource(input_dir=str(mock_vtk_run))
        assert source.run_id(0) == "run0001"

    def test_custom_vtk_glob(self, mock_vtk_run):
        """OpenRadiossSource should use custom vtk_glob pattern."""
        from physicsnemo_curator.domains.mesh.sources.openradioss import OpenRadiossSource

        # Use glob that doesn't match our files
        source = OpenRadiossSource(input_dir=str(mock_vtk_run), vtk_glob="nonexistent*.vtk")
        assert len(source) == 0

        # Use glob that matches
        source = OpenRadiossSource(input_dir=str(mock_vtk_run), vtk_glob="timestep_*.vtk")
        assert len(source) == 1
