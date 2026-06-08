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

"""Tests for the generalized VTKSource: Rust backend, per-path rules, array
filtering, ``.stl`` support, domain-mesh pairing, and the new generic filters
(clean / point_data_to_cell_data / global_data) and atomic MeshSink writes."""

from __future__ import annotations

import pathlib
import stat

import pytest

pytestmark = pytest.mark.requires("mesh")

import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402
import torch  # noqa: E402
from physicsnemo.mesh import Mesh  # noqa: E402
from physicsnemo.mesh.domain_mesh import DomainMesh  # noqa: E402

from physicsnemo_curator.domains.mesh.filters.clean import CleanFilter  # noqa: E402
from physicsnemo_curator.domains.mesh.filters.global_data import GlobalDataFilter  # noqa: E402
from physicsnemo_curator.domains.mesh.filters.point_data_to_cell import PointDataToCellDataFilter  # noqa: E402
from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink  # noqa: E402
from physicsnemo_curator.domains.mesh.sources.vtk import VTKSource  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_vtu(directory: pathlib.Path, name: str = "test.vtu") -> pathlib.Path:
    """Write a small triangulated VTU with point + cell data."""
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    cells = np.array([[3, 0, 1, 2], [3, 0, 2, 3]])
    cell_types = np.array([5, 5])  # VTK_TRIANGLE
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    grid.point_data["temperature"] = np.array([100.0, 200.0, 300.0, 400.0])
    grid.point_data["pressure"] = np.array([1.0, 2.0, 3.0, 4.0])
    grid.cell_data["velocity"] = np.array([10.0, 20.0])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    grid.save(str(path))
    return path


def _create_vtp(directory: pathlib.Path, name: str = "surface.vtp") -> pathlib.Path:
    """Write a small triangulated VTP surface with cell data."""
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.hstack([[3, 0, 1, 2], [3, 0, 2, 3]])
    poly = pv.PolyData(points, faces)
    poly.cell_data["wss"] = np.array([1.0, 2.0])
    poly.point_data["p"] = np.array([5.0, 6.0, 7.0, 8.0])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    poly.save(str(path))
    return path


def _create_stl(directory: pathlib.Path, name: str = "geom.stl") -> pathlib.Path:
    """Write a small triangulated STL surface."""
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.hstack([[3, 0, 1, 2], [3, 0, 2, 3]])
    poly = pv.PolyData(points, faces)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    poly.save(str(path))
    return path


# ---------------------------------------------------------------------------
# Rust backend
# ---------------------------------------------------------------------------


class TestVTKSourceRustBackend:
    def test_rust_reads_points_cells_data(self, tmp_path):
        _create_vtu(tmp_path / "vtk")
        source = VTKSource(str(tmp_path / "vtk"), backend="rust")
        mesh = next(source[0])
        assert mesh.n_points == 4
        assert mesh.n_cells == 2
        assert "temperature" in mesh.point_data
        assert "velocity" in mesh.cell_data

    def test_rust_matches_pyvista_points(self, tmp_path):
        _create_vtu(tmp_path / "vtk")
        rust_mesh = next(VTKSource(str(tmp_path / "vtk"), backend="rust")[0])
        pv_mesh = next(VTKSource(str(tmp_path / "vtk"), backend="pyvista")[0])
        assert torch.allclose(rust_mesh.points.double(), pv_mesh.points.double())

    def test_rust_cell_centroids(self, tmp_path):
        _create_vtu(tmp_path / "vtk")
        source = VTKSource(str(tmp_path / "vtk"), backend="rust", point_source="cell_centroids")
        mesh = next(source[0])
        # 2 cells -> 2 centroid points, no cells, cell_data promoted to point_data.
        assert mesh.n_points == 2
        assert mesh.n_cells == 0
        assert "velocity" in mesh.point_data

    def test_rust_falls_back_for_stl(self, tmp_path):
        # .stl is not a Rust-supported XML format; must transparently fall back.
        _create_stl(tmp_path / "vtk")
        mesh = next(VTKSource(str(tmp_path / "vtk"), backend="rust")[0])
        assert isinstance(mesh, Mesh)
        assert mesh.n_points == 4


# ---------------------------------------------------------------------------
# Per-path rules + reader-level array filtering
# ---------------------------------------------------------------------------


class TestPerPathAndArrayFilters:
    @pytest.mark.parametrize("backend", ["pyvista", "rust"])
    def test_key_filter_include(self, tmp_path, backend):
        _create_vtu(tmp_path / "vtk")
        source = VTKSource(
            str(tmp_path / "vtk"),
            backend=backend,
            key_filters=[{"path_pattern": "**/*.vtu", "mode": "include", "keys": ["temperature"]}],
        )
        mesh = next(source[0])
        keys = set(mesh.point_data.keys())
        assert "temperature" in keys
        assert "pressure" not in keys

    @pytest.mark.parametrize("backend", ["pyvista", "rust"])
    def test_key_filter_exclude(self, tmp_path, backend):
        _create_vtu(tmp_path / "vtk")
        source = VTKSource(
            str(tmp_path / "vtk"),
            backend=backend,
            key_filters=[{"path_pattern": "**/*.vtu", "mode": "exclude", "keys": ["pressure"]}],
        )
        mesh = next(source[0])
        keys = set(mesh.point_data.keys())
        assert "temperature" in keys
        assert "pressure" not in keys

    def test_per_path_manifold_dim(self, tmp_path):
        vtk_dir = tmp_path / "vtk"
        _create_vtu(vtk_dir, "volume_a.vtu")
        _create_vtu(vtk_dir, "surface_a.vtu")
        source = VTKSource(
            str(vtk_dir),
            manifold_dim=[
                {"pattern": "**/volume_*", "value": 0},
                {"pattern": "**/surface_*", "value": 2},
            ],
        )
        meshes = {pathlib.Path(source._files[i]).name: next(source[i]) for i in range(len(source))}
        assert meshes["volume_a.vtu"].n_manifold_dims == 0
        assert meshes["surface_a.vtu"].n_manifold_dims == 2


# ---------------------------------------------------------------------------
# .stl support
# ---------------------------------------------------------------------------


class TestStlSupport:
    def test_reads_stl(self, tmp_path):
        _create_stl(tmp_path / "vtk")
        source = VTKSource(str(tmp_path / "vtk"))
        assert len(source) == 1
        mesh = next(source[0])
        assert isinstance(mesh, Mesh)
        assert mesh.n_points == 4

    def test_stl_single_file(self, tmp_path):
        stl = _create_stl(tmp_path)
        source = VTKSource(str(stl))
        assert len(source) == 1


# ---------------------------------------------------------------------------
# Domain-mesh pairing
# ---------------------------------------------------------------------------


class TestDomainMeshPairing:
    def test_pairs_volume_and_boundary(self, tmp_path):
        run = tmp_path / "data" / "run_1"
        _create_vtu(run, "volume_1.vtu")
        _create_vtp(run, "boundary_1.vtp")
        source = VTKSource(
            str(tmp_path / "data"),
            volume_pattern="volume_*.vtu",
            boundary_pattern="boundary_*.vtp",
            boundary_name="vehicle",
        )
        assert len(source) == 1
        domain = next(source[0])
        assert isinstance(domain, DomainMesh)
        assert "vehicle" in domain.boundary_names

    def test_unpaired_falls_back_to_mesh(self, tmp_path):
        run = tmp_path / "data" / "run_1"
        _create_vtu(run, "volume_1.vtu")
        _create_vtp(run, "boundary_1.vtp")
        _create_stl(run, "geom.stl")  # unpaired
        source = VTKSource(
            str(tmp_path / "data"),
            volume_pattern="volume_*.vtu",
            boundary_pattern="boundary_*.vtp",
        )
        assert len(source) == 2  # one pair + one standalone
        kinds = sorted(type(next(source[i])).__name__ for i in range(len(source)))
        assert kinds == ["DomainMesh", "Mesh"]


# ---------------------------------------------------------------------------
# New generic filters
# ---------------------------------------------------------------------------


class TestCleanFilter:
    def test_merges_duplicate_points(self):
        # Two coincident points (0 and 2) should merge.
        points = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 1, 0]])
        cells = torch.tensor([[0, 1, 3], [2, 1, 3]])
        mesh = Mesh(points=points, cells=cells)
        out = next(CleanFilter()([mesh].__iter__()))
        assert out.n_points == 3

    def test_point_cloud_passthrough(self):
        mesh = Mesh(points=torch.rand(10, 3))
        out = next(CleanFilter()([mesh].__iter__()))
        assert out.n_points == 10


class TestPointDataToCellDataFilter:
    def test_converts_and_drops_point_data(self):
        points = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells, point_data={"t": torch.tensor([1.0, 2.0, 3.0])})
        out = next(PointDataToCellDataFilter()([mesh].__iter__()))
        assert "t" in out.cell_data
        assert torch.allclose(out.cell_data["t"], torch.tensor([2.0]))  # mean of 1,2,3
        assert out.point_data is None or len(out.point_data.keys()) == 0

    def test_keeps_point_data_when_requested(self):
        points = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells, point_data={"t": torch.tensor([1.0, 2.0, 3.0])})
        out = next(PointDataToCellDataFilter(drop_point_data=False)([mesh].__iter__()))
        assert "t" in out.cell_data
        assert "t" in out.point_data


class TestGlobalDataFilter:
    def test_injects_constants(self):
        mesh = Mesh(points=torch.rand(5, 3))
        out = next(GlobalDataFilter(values={"U_inf": [30.0, 0.0, 0.0], "rho_inf": 1.225})([mesh].__iter__()))
        assert torch.allclose(out.global_data["U_inf"], torch.tensor([30.0, 0.0, 0.0]))
        assert torch.allclose(out.global_data["rho_inf"], torch.tensor(1.225))

    def test_no_overwrite(self):
        mesh = Mesh(points=torch.rand(5, 3), global_data={"rho_inf": torch.tensor(9.9)})
        out = next(GlobalDataFilter(values={"rho_inf": 1.225}, overwrite=False)([mesh].__iter__()))
        assert torch.allclose(out.global_data["rho_inf"], torch.tensor(9.9))

    def test_empty_values_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            GlobalDataFilter(values={})


# ---------------------------------------------------------------------------
# Atomic MeshSink writes
# ---------------------------------------------------------------------------


class TestAtomicMeshSink:
    def test_overwrite_leaves_no_tmp_dirs(self, tmp_path):
        _create_vtu(tmp_path / "vtk", "test.vtu")
        source = VTKSource(str(tmp_path / "vtk"))
        out = tmp_path / "out"
        sink = MeshSink(output_dir=str(out), naming_template="m_{index}")
        sink.set_source(source)
        sink(source[0], 0)
        sink(source[0], 0)  # overwrite
        names = [p.name for p in out.iterdir()]
        assert "m_0.pmsh" in names
        assert not any(n.startswith(".tmp_") for n in names)

    def test_group_readable(self, tmp_path):
        _create_vtu(tmp_path / "vtk", "test.vtu")
        source = VTKSource(str(tmp_path / "vtk"))
        out = tmp_path / "out"
        sink = MeshSink(output_dir=str(out), naming_template="m_{index}", group_readable=True)
        sink.set_source(source)
        paths = sink(source[0], 0)
        mode = pathlib.Path(paths[0]).stat().st_mode
        assert mode & stat.S_IRGRP
