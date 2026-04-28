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

"""Tests for DrivAerML multi-mesh mode and MeshSink extended naming."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import numpy as np
import pytest
import pyvista as pv
from physicsnemo.mesh.io import from_pyvista

from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink

if TYPE_CHECKING:
    from physicsnemo.mesh import Mesh

pytestmark = pytest.mark.requires("mesh")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeSourceWithRunId:
    """Minimal source stub exposing run_id and mesh_name for MeshSink tests."""

    def __init__(self, run_ids: list[int], mesh_names_map: dict[str, str]):
        self._run_ids = run_ids
        self._mesh_names_map = mesh_names_map  # seq -> name

    def run_id(self, index: int) -> int:
        return self._run_ids[index]

    def mesh_name(self, index: int, seq: int) -> str:
        run_id = self._run_ids[index]
        parts = list(self._mesh_names_map.keys())
        part = parts[seq]
        return self._mesh_names_map[part].format(run_id=run_id)


def _make_simple_mesh() -> Mesh:
    """Create a simple Mesh for sink testing."""
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    cells = np.array([[3, 0, 1, 2], [3, 0, 2, 3]])
    cell_types = np.array([5, 5])
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    grid.point_data["temperature"] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    return from_pyvista(grid, manifold_dim="auto", point_source="vertices")


# ---------------------------------------------------------------------------
# MeshSink extended naming tests
# ---------------------------------------------------------------------------


class TestMeshSinkRunIdPlaceholder:
    """Test {run_id} and {mesh_name} placeholders in MeshSink."""

    def test_run_id_placeholder_resolves(self, tmp_path: pathlib.Path) -> None:
        """MeshSink resolves {run_id} from source.run_id(index)."""
        sink = MeshSink(
            output_dir=str(tmp_path),
            naming_template="run_{run_id}/mesh_{seq}",
        )
        source = FakeSourceWithRunId(
            run_ids=[5, 12],
            mesh_names_map={"domain": "domain_{run_id}"},
        )
        sink.set_source(source)

        meshes = iter([_make_simple_mesh()])
        paths = sink(meshes, index=0)

        assert len(paths) == 1
        assert "run_5/mesh_0" in paths[0]
        assert pathlib.Path(paths[0]).exists()

    def test_mesh_name_placeholder_resolves(self, tmp_path: pathlib.Path) -> None:
        """MeshSink resolves {mesh_name} from source.mesh_name(index, seq)."""
        sink = MeshSink(
            output_dir=str(tmp_path),
            naming_template="run_{run_id}/{mesh_name}",
        )
        source = FakeSourceWithRunId(
            run_ids=[1, 2],
            mesh_names_map={
                "domain": "domain_{run_id}",
                "stl": "drivaer_{run_id}.stl",
            },
        )
        sink.set_source(source)

        mesh1 = _make_simple_mesh()
        mesh2 = _make_simple_mesh()
        meshes = iter([mesh1, mesh2])
        paths = sink(meshes, index=0)

        assert len(paths) == 2
        assert "run_1/domain_1" in paths[0]
        assert "run_1/drivaer_1.stl" in paths[1]

    def test_run_id_without_source_raises(self, tmp_path: pathlib.Path) -> None:
        """Using {run_id} without a compatible source raises ValueError."""
        sink = MeshSink(
            output_dir=str(tmp_path),
            naming_template="run_{run_id}/mesh_{seq}",
        )
        # No source set — calling should fail
        meshes = iter([_make_simple_mesh()])
        with pytest.raises(ValueError, match="run_id"):
            sink(meshes, index=0)

    def test_mesh_name_without_source_raises(self, tmp_path: pathlib.Path) -> None:
        """Using {mesh_name} without a compatible source raises ValueError."""
        sink = MeshSink(
            output_dir=str(tmp_path),
            naming_template="{mesh_name}",
        )
        meshes = iter([_make_simple_mesh()])
        with pytest.raises(ValueError, match="mesh_name"):
            sink(meshes, index=0)

    def test_construction_with_new_placeholders_valid(self) -> None:
        """Construction succeeds when new placeholders are used."""
        # Should not raise
        sink = MeshSink(
            output_dir="/tmp/out",
            naming_template="run_{run_id}/{mesh_name}",
        )
        assert sink is not None


# ---------------------------------------------------------------------------
# DrivAerMLSource multi-mode tests
# ---------------------------------------------------------------------------


class TestDrivAerMLSourceMultiConstruction:
    """Test construction and metadata methods for mesh_type='multi'."""

    @pytest.fixture()
    def local_drivaerml(self, tmp_path: pathlib.Path) -> pathlib.Path:
        """Create a minimal local DrivAerML-like directory structure."""
        for run_id in [1, 5, 12]:
            run_dir = tmp_path / f"run_{run_id}"
            run_dir.mkdir()
            # Create a simple boundary VTP
            sphere = pv.Sphere(radius=1.0, theta_resolution=4, phi_resolution=4)
            sphere.point_data["Pressure"] = np.random.default_rng(run_id).standard_normal(sphere.n_points)
            sphere.cell_data["WallShearStress"] = np.random.default_rng(run_id).standard_normal(sphere.n_cells)
            sphere.save(str(run_dir / f"boundary_{run_id}.vtp"))
            # Create a simple volume VTU
            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
            cells = np.array([[4, 0, 1, 2, 3]])
            cell_types = np.array([10])  # VTK_TETRA
            grid = pv.UnstructuredGrid(cells, cell_types, points)
            grid.point_data["Velocity"] = np.random.default_rng(run_id).standard_normal((4, 3))
            grid.cell_data["Pressure"] = np.array([1.0 * run_id])
            grid.save(str(run_dir / f"volume_{run_id}.vtu"))
        return tmp_path

    def test_multi_mode_construction(self, local_drivaerml: pathlib.Path) -> None:
        """mesh_type='multi' constructs successfully with local data."""
        from physicsnemo_curator.domains.mesh.sources.drivaerml import DrivAerMLSource

        source = DrivAerMLSource(
            mesh_type="multi",
            url=str(local_drivaerml),
        )
        assert len(source) == 3  # 3 runs

    def test_run_id_returns_dataset_ids(self, local_drivaerml: pathlib.Path) -> None:
        """run_id() returns the actual dataset run IDs, not sequential indices."""
        from physicsnemo_curator.domains.mesh.sources.drivaerml import DrivAerMLSource

        source = DrivAerMLSource(
            mesh_type="multi",
            url=str(local_drivaerml),
        )
        assert source.run_id(0) == 1
        assert source.run_id(1) == 5
        assert source.run_id(2) == 12

    def test_mesh_name_returns_canonical_names(self, local_drivaerml: pathlib.Path) -> None:
        """mesh_name() returns canonical filenames with run_id substituted."""
        from physicsnemo_curator.domains.mesh.sources.drivaerml import DrivAerMLSource

        source = DrivAerMLSource(
            mesh_type="multi",
            url=str(local_drivaerml),
            mesh_parts=["domain", "stl", "single_solid"],
        )
        assert source.mesh_name(0, 0) == "domain_1"
        assert source.mesh_name(0, 1) == "drivaer_1.stl"
        assert source.mesh_name(0, 2) == "drivaer_1_single_solid.stl"
        assert source.mesh_name(1, 0) == "domain_5"

    def test_mesh_parts_subset(self, local_drivaerml: pathlib.Path) -> None:
        """mesh_parts subset limits which meshes are yielded."""
        from physicsnemo_curator.domains.mesh.sources.drivaerml import DrivAerMLSource

        source = DrivAerMLSource(
            mesh_type="multi",
            url=str(local_drivaerml),
            mesh_parts=["domain"],
        )
        meshes = list(source[0])
        assert len(meshes) == 1

    def test_invalid_mesh_parts_raises(self, local_drivaerml: pathlib.Path) -> None:
        """Invalid mesh_parts entries raise ValueError."""
        from physicsnemo_curator.domains.mesh.sources.drivaerml import DrivAerMLSource

        with pytest.raises(ValueError, match="Invalid mesh_parts"):
            DrivAerMLSource(
                mesh_type="multi",
                url=str(local_drivaerml),
                mesh_parts=["invalid_part"],
            )

    def test_params_includes_multi(self) -> None:
        """params() should list 'multi' as a mesh_type choice."""
        from physicsnemo_curator.domains.mesh.sources.drivaerml import DrivAerMLSource

        params = DrivAerMLSource.params()
        mesh_type_param = next(p for p in params if p.name == "mesh_type")
        assert "multi" in mesh_type_param.choices


class TestDrivAerMLMultiMeshContent:
    """Test that multi mode yields correctly transformed meshes."""

    @pytest.fixture()
    def local_drivaerml(self, tmp_path: pathlib.Path) -> pathlib.Path:
        """Create a minimal local DrivAerML-like directory structure."""
        for run_id in [1, 5]:
            run_dir = tmp_path / f"run_{run_id}"
            run_dir.mkdir()
            # Boundary VTP with float64 data
            sphere = pv.Sphere(radius=1.0, theta_resolution=4, phi_resolution=4)
            sphere.point_data["Pressure"] = (
                np.random.default_rng(run_id).standard_normal(sphere.n_points).astype(np.float64)
            )
            sphere.cell_data["WallShearStress"] = (
                np.random.default_rng(run_id).standard_normal(sphere.n_cells).astype(np.float64)
            )
            sphere.save(str(run_dir / f"boundary_{run_id}.vtp"))
            # Volume VTU with float64 data
            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
            cells = np.array([[4, 0, 1, 2, 3]])
            cell_types = np.array([10])  # VTK_TETRA
            grid = pv.UnstructuredGrid(cells, cell_types, points)
            grid.point_data["Velocity"] = np.random.default_rng(run_id).standard_normal((4, 3)).astype(np.float64)
            grid.cell_data["Pressure"] = np.array([1.0 * run_id], dtype=np.float64)
            grid.save(str(run_dir / f"volume_{run_id}.vtu"))
        return tmp_path

    def test_multi_yields_three_meshes(self, local_drivaerml: pathlib.Path) -> None:
        """Multi mode yields 3 meshes per index by default."""
        from physicsnemo_curator.domains.mesh.sources.drivaerml import DrivAerMLSource

        source = DrivAerMLSource(mesh_type="multi", url=str(local_drivaerml))
        meshes = list(source[0])
        assert len(meshes) == 3

    def test_domain_mesh_is_fp32(self, local_drivaerml: pathlib.Path) -> None:
        """Domain mesh has float32 points and data."""
        import torch

        from physicsnemo_curator.domains.mesh.sources.drivaerml import DrivAerMLSource

        source = DrivAerMLSource(mesh_type="multi", url=str(local_drivaerml))
        meshes = list(source[0])
        domain = meshes[0]
        # Points should be fp32
        assert domain.points.dtype == torch.float32

    def test_stl_mesh_is_fp32(self, local_drivaerml: pathlib.Path) -> None:
        """STL mesh has float32 points and data."""
        import torch

        from physicsnemo_curator.domains.mesh.sources.drivaerml import DrivAerMLSource

        source = DrivAerMLSource(mesh_type="multi", url=str(local_drivaerml))
        meshes = list(source[0])
        stl = meshes[1]
        assert stl.points.dtype == torch.float32

    def test_single_solid_is_single_polydata(self, local_drivaerml: pathlib.Path) -> None:
        """Single solid mesh should be derived from the boundary file."""
        from physicsnemo_curator.domains.mesh.sources.drivaerml import DrivAerMLSource

        source = DrivAerMLSource(mesh_type="multi", url=str(local_drivaerml))
        meshes = list(source[0])
        single_solid = meshes[2]
        # Should have points (same source file, just merged)
        assert single_solid.n_points > 0

    def test_domain_uses_cell_centroids(self, local_drivaerml: pathlib.Path) -> None:
        """Domain mesh should use cell_centroids point source."""
        from physicsnemo_curator.domains.mesh.sources.drivaerml import DrivAerMLSource

        source = DrivAerMLSource(mesh_type="multi", url=str(local_drivaerml))
        meshes = list(source[0])
        domain = meshes[0]
        # With cell_centroids on a tetrahedron, n_points should equal n_cells (1 cell)
        # The volume has 1 tetrahedron → 1 centroid point
        assert domain.n_points == 1

    def test_mesh_parts_domain_only(self, local_drivaerml: pathlib.Path) -> None:
        """mesh_parts=['domain'] yields only the domain mesh."""
        from physicsnemo_curator.domains.mesh.sources.drivaerml import DrivAerMLSource

        source = DrivAerMLSource(
            mesh_type="multi",
            url=str(local_drivaerml),
            mesh_parts=["domain"],
        )
        meshes = list(source[0])
        assert len(meshes) == 1
