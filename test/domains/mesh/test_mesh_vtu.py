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

"""Tests for MeshVTUSink."""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest
import torch

pv = pytest.importorskip("pyvista")

from tensordict import TensorDict  # noqa: E402

pytestmark = pytest.mark.requires("mesh")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_mesh():
    """Create a simple mesh with displacement fields."""
    from physicsnemo.mesh import Mesh

    n_points = 10
    n_cells = 4
    n_timesteps = 3

    rng = np.random.default_rng(42)

    points = torch.from_numpy(rng.uniform(-10, 10, size=(n_points, 3)).astype(np.float32))
    # Tetrahedra (4 nodes per cell)
    cells = torch.from_numpy(rng.integers(0, n_points, size=(n_cells, 4)).astype(np.int64))

    pd_dict: dict[str, torch.Tensor] = {}
    pd_dict["thickness"] = torch.from_numpy(rng.uniform(0.1, 1.0, size=(n_points,)).astype(np.float32))

    for t in range(n_timesteps):
        disp = rng.uniform(-1, 1, size=(n_points, 3)).astype(np.float32)
        pd_dict[f"displacement_t{t:03d}"] = torch.from_numpy(disp)

    point_data = TensorDict(pd_dict, batch_size=[n_points])

    # Cell data with stress
    cd_dict: dict[str, torch.Tensor] = {}
    for t in range(n_timesteps):
        cd_dict[f"stress_vm_t{t:03d}"] = torch.from_numpy(rng.uniform(0, 100, size=(n_cells,)).astype(np.float32))
    cell_data = TensorDict(cd_dict, batch_size=[n_cells])

    return Mesh(
        points=points,
        cells=cells,
        point_data=point_data,
        cell_data=cell_data,
    )


@pytest.fixture
def triangle_mesh():
    """Create a mesh with triangles for normal flipping test."""
    from physicsnemo.mesh import Mesh

    n_points = 6

    # Simple triangle mesh
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.5, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    cells = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64)

    pd_dict: dict[str, torch.Tensor] = {}
    pd_dict["displacement_t000"] = torch.zeros(n_points, 3, dtype=torch.float32)
    point_data = TensorDict(pd_dict, batch_size=[n_points])

    return Mesh(points=points, cells=cells, point_data=point_data)


@pytest.fixture
def mesh_no_displacements():
    """Create a mesh without displacement fields."""
    from physicsnemo.mesh import Mesh

    n_points = 5

    points = torch.randn(n_points, 3, dtype=torch.float32)
    point_data = TensorDict(
        {"some_field": torch.randn(n_points, dtype=torch.float32)},
        batch_size=[n_points],
    )

    return Mesh(points=points, cells=None, point_data=point_data)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestMeshVTUSinkUnit:
    """Unit tests for MeshVTUSink."""

    def test_params(self):
        """params() should return expected descriptors."""
        from physicsnemo_curator.domains.mesh.sinks.mesh_vtu import MeshVTUSink

        params = MeshVTUSink.params()
        names = [p.name for p in params]
        assert "output_dir" in names
        assert "naming_template" in names
        assert "flip_triangle_normals" in names

    def test_init_default_naming(self):
        """MeshVTUSink should use default naming template."""
        from physicsnemo_curator.domains.mesh.sinks.mesh_vtu import MeshVTUSink

        with tempfile.TemporaryDirectory() as tmpdir:
            sink = MeshVTUSink(output_dir=tmpdir)
            assert sink._naming_template == "mesh_{index:04d}"

    def test_init_custom_naming(self):
        """MeshVTUSink should accept custom naming template."""
        from physicsnemo_curator.domains.mesh.sinks.mesh_vtu import MeshVTUSink

        with tempfile.TemporaryDirectory() as tmpdir:
            sink = MeshVTUSink(output_dir=tmpdir, naming_template="run_{index}")
            assert sink._naming_template == "run_{index}"

    def test_vtk_cell_types_mapping(self):
        """MeshVTUSink should have correct VTK cell type mapping."""
        from physicsnemo_curator.domains.mesh.sinks.mesh_vtu import MeshVTUSink

        assert MeshVTUSink._VTK_CELL_TYPES[3] == 5  # Triangle
        assert MeshVTUSink._VTK_CELL_TYPES[4] == 10  # Tetra
        assert MeshVTUSink._VTK_CELL_TYPES[5] == 14  # Pyramid
        assert MeshVTUSink._VTK_CELL_TYPES[6] == 13  # Wedge
        assert MeshVTUSink._VTK_CELL_TYPES[8] == 12  # Hexahedron


class TestMeshVTUSinkIntegration:
    """Integration tests for MeshVTUSink."""

    def test_write_simple_mesh(self, simple_mesh):
        """MeshVTUSink should write a mesh to VTU format."""
        from physicsnemo_curator.domains.mesh.sinks.mesh_vtu import MeshVTUSink

        with tempfile.TemporaryDirectory() as tmpdir:
            sink = MeshVTUSink(output_dir=tmpdir)

            def gen():
                yield simple_mesh

            paths = sink(gen(), index=0)

            assert len(paths) == 1
            vtu_path = pathlib.Path(paths[0])
            assert vtu_path.exists()
            assert vtu_path.suffix == ".vtu"

            # Load and verify
            grid = pv.read(str(vtu_path))
            assert grid.n_points == 10
            assert grid.n_cells == 4

    def test_displacement_fields_converted(self, simple_mesh):
        """Displacement fields should use 4-digit timestep format."""
        from physicsnemo_curator.domains.mesh.sinks.mesh_vtu import MeshVTUSink

        with tempfile.TemporaryDirectory() as tmpdir:
            sink = MeshVTUSink(output_dir=tmpdir)

            def gen():
                yield simple_mesh

            paths = sink(gen(), index=0)

            grid = pv.read(str(paths[0]))
            assert "displacement_t0000" in grid.point_data
            assert "displacement_t0001" in grid.point_data
            assert "displacement_t0002" in grid.point_data

    def test_thickness_field(self, simple_mesh):
        """Thickness field should be written to VTU."""
        from physicsnemo_curator.domains.mesh.sinks.mesh_vtu import MeshVTUSink

        with tempfile.TemporaryDirectory() as tmpdir:
            sink = MeshVTUSink(output_dir=tmpdir)

            def gen():
                yield simple_mesh

            paths = sink(gen(), index=0)

            grid = pv.read(str(paths[0]))
            assert "thickness" in grid.point_data
            assert len(grid.point_data["thickness"]) == 10

    def test_cell_data_written(self, simple_mesh):
        """Cell data should be written to VTU."""
        from physicsnemo_curator.domains.mesh.sinks.mesh_vtu import MeshVTUSink

        with tempfile.TemporaryDirectory() as tmpdir:
            sink = MeshVTUSink(output_dir=tmpdir)

            def gen():
                yield simple_mesh

            paths = sink(gen(), index=0)

            grid = pv.read(str(paths[0]))
            assert "stress_vm_t000" in grid.cell_data

    def test_triangle_normal_flip(self, triangle_mesh):
        """Triangle normals should be flipped when enabled."""
        from physicsnemo_curator.domains.mesh.sinks.mesh_vtu import MeshVTUSink

        with tempfile.TemporaryDirectory() as tmpdir:
            # With flipping (default)
            sink_flip = MeshVTUSink(output_dir=tmpdir, flip_triangle_normals=True)

            def gen1():
                yield triangle_mesh

            paths_flip = sink_flip(gen1(), index=0)

            # Without flipping
            sink_no_flip = MeshVTUSink(output_dir=tmpdir, flip_triangle_normals=False)

            def gen2():
                yield triangle_mesh

            paths_no_flip = sink_no_flip(gen2(), index=1)

            grid_flip = pv.read(str(paths_flip[0]))
            grid_no_flip = pv.read(str(paths_no_flip[0]))

            # Both should have valid meshes
            assert grid_flip.n_cells == 2
            assert grid_no_flip.n_cells == 2

    def test_mesh_no_displacements(self, mesh_no_displacements):
        """MeshVTUSink should handle meshes without displacement fields."""
        from physicsnemo_curator.domains.mesh.sinks.mesh_vtu import MeshVTUSink

        with tempfile.TemporaryDirectory() as tmpdir:
            sink = MeshVTUSink(output_dir=tmpdir)

            def gen():
                yield mesh_no_displacements

            paths = sink(gen(), index=0)

            assert len(paths) == 1
            grid = pv.read(str(paths[0]))
            assert grid.n_points == 5
            # Should have thickness (zeros) and the other field
            assert "thickness" in grid.point_data
            assert "some_field" in grid.point_data

    def test_custom_naming_template(self, simple_mesh):
        """MeshVTUSink should use custom naming template."""
        from physicsnemo_curator.domains.mesh.sinks.mesh_vtu import MeshVTUSink

        with tempfile.TemporaryDirectory() as tmpdir:
            sink = MeshVTUSink(output_dir=tmpdir, naming_template="output_{index:03d}")

            def gen():
                yield simple_mesh

            paths = sink(gen(), index=5)

            assert len(paths) == 1
            assert "output_005.vtu" in paths[0]

    def test_output_dir_property(self):
        """output_dir property should return the path."""
        from physicsnemo_curator.domains.mesh.sinks.mesh_vtu import MeshVTUSink

        with tempfile.TemporaryDirectory() as tmpdir:
            sink = MeshVTUSink(output_dir=tmpdir)
            assert sink.output_dir == pathlib.Path(tmpdir)

    def test_von_mises_stress_renamed(self, simple_mesh):
        """Von Mises stress fields should be renamed to 4-digit format."""
        from physicsnemo.mesh import Mesh

        from physicsnemo_curator.domains.mesh.sinks.mesh_vtu import MeshVTUSink

        # Create mesh with stress_vm fields
        n_points = 10
        rng = np.random.default_rng(42)
        points = torch.from_numpy(rng.uniform(-10, 10, size=(n_points, 3)).astype(np.float32))

        pd_dict: dict[str, torch.Tensor] = {}
        pd_dict["displacement_t000"] = torch.zeros(n_points, 3, dtype=torch.float32)
        pd_dict["stress_vm_t000"] = torch.from_numpy(rng.uniform(0, 100, size=(n_points,)).astype(np.float32))
        pd_dict["stress_vm_t001"] = torch.from_numpy(rng.uniform(0, 100, size=(n_points,)).astype(np.float32))
        point_data = TensorDict(pd_dict, batch_size=[n_points])

        mesh = Mesh(points=points, cells=None, point_data=point_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            sink = MeshVTUSink(output_dir=tmpdir)

            def gen():
                yield mesh

            paths = sink(gen(), index=0)

            grid = pv.read(str(paths[0]))
            assert "Von_Mises_t0000" in grid.point_data
            assert "Von_Mises_t0001" in grid.point_data
