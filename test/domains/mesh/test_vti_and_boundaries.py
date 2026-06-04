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

"""Tests for the structured-grid (.vti) path and the generic boundary-condition
synthesis / injection subsystem."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.requires("mesh")

import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402
import torch  # noqa: E402
from physicsnemo.mesh import Mesh  # noqa: E402
from physicsnemo.mesh.domain_mesh import DomainMesh  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from physicsnemo_curator.domains.mesh.boundaries import (  # noqa: E402
    BoxTunnelBoundaries,
    HemisphereBoundaries,
    inject_boundaries,
)
from physicsnemo_curator.domains.mesh.boundaries import _geometry as geom  # noqa: E402
from physicsnemo_curator.domains.mesh.filters.boundary_injection import BoundaryInjectionFilter  # noqa: E402
from physicsnemo_curator.domains.mesh.sinks.grid_sidecar import GridSidecarSink  # noqa: E402
from physicsnemo_curator.domains.mesh.sources.vti import VTISource  # noqa: E402


def _create_vti(directory, name="grid.vti", dims=(4, 3, 2)):
    """Write a small ImageData with a known scalar (= x + 10y + 100z) and vector."""
    directory.mkdir(parents=True, exist_ok=True)
    img = pv.ImageData(dimensions=dims, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    pts = img.points
    img.point_data["s"] = (pts[:, 0] + 10 * pts[:, 1] + 100 * pts[:, 2]).astype(np.float64)
    img.point_data["v"] = pts.astype(np.float64)
    img.cell_data["c"] = np.arange(img.n_cells, dtype=np.float64)
    path = directory / name
    img.save(str(path))
    return path


# ---------------------------------------------------------------------------
# VTISource
# ---------------------------------------------------------------------------


class TestVTISource:
    def test_batch_sizes(self, tmp_path):
        _create_vti(tmp_path / "g")
        grid = next(VTISource(str(tmp_path / "g"))[0])
        assert tuple(grid["point_data"].batch_size) == (2, 3, 4)  # (nz, ny, nx)
        assert tuple(grid["cell_data"].batch_size) == (1, 2, 3)

    def test_vtk_ordering(self, tmp_path):
        _create_vti(tmp_path / "g")
        grid = next(VTISource(str(tmp_path / "g"))[0])
        s = grid["point_data"]["s"]
        for z in range(2):
            for y in range(3):
                for x in range(4):
                    assert abs(float(s[z, y, x]) - (x + 10 * y + 100 * z)) < 1e-9

    def test_vector_field_shape(self, tmp_path):
        _create_vti(tmp_path / "g")
        grid = next(VTISource(str(tmp_path / "g"))[0])
        assert tuple(grid["point_data"]["v"].shape) == (2, 3, 4, 3)

    def test_grid_metadata(self, tmp_path):
        _create_vti(tmp_path / "g")
        grid = next(VTISource(str(tmp_path / "g"))[0])
        assert grid["grid"]["dimensions"].tolist() == [4, 3, 2]
        assert torch.allclose(grid["grid"]["spacing"], torch.ones(3, dtype=grid["grid"]["spacing"].dtype))

    def test_fp32(self, tmp_path):
        _create_vti(tmp_path / "g")
        grid = next(VTISource(str(tmp_path / "g"), fp32=True)[0])
        assert grid["point_data"]["s"].dtype == torch.float32

    def test_non_vti_raises(self, tmp_path):
        (tmp_path / "x.txt").write_text("nope")
        with pytest.raises(ValueError, match="VTI extension"):
            VTISource(str(tmp_path / "x.txt"))


class TestGridSidecarSink:
    def test_roundtrip(self, tmp_path):
        _create_vti(tmp_path / "g", name="sample.vti")
        source = VTISource(str(tmp_path / "g"))
        out = tmp_path / "out"
        sink = GridSidecarSink(output_dir=str(out))
        sink.set_source(source)
        paths = sink(source[0], 0)
        assert paths[0].endswith("sample.grid")
        loaded = TensorDict.load_memmap(paths[0])
        original = next(source[0])
        assert torch.allclose(loaded["point_data"]["s"], original["point_data"]["s"])
        assert torch.allclose(loaded["cell_data"]["c"], original["cell_data"]["c"])


# ---------------------------------------------------------------------------
# Boundary geometry toolkit
# ---------------------------------------------------------------------------


class TestBoundaryGeometry:
    def test_box_face_inward_normal(self):
        bounds = ((-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0))
        inlet = geom.box_face_patch(bounds=bounds, face="x_min", n_per_side=(3, 3))
        c = inlet.cells[0]
        n = torch.linalg.cross(inlet.points[c[1]] - inlet.points[c[0]], inlet.points[c[2]] - inlet.points[c[0]])
        assert float(n[0]) > 0  # inward at x_min is +x

    def test_box_tunnel_keys(self):
        bounds = ((-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0))
        bt = geom.box_tunnel_boundaries(bounds=bounds, x_bl=0.0, n_per_side=(3, 3))
        assert set(bt.keys()) == {"inlet", "outlet", "slip", "no_slip"}

    def test_box_tunnel_x_bl_out_of_range(self):
        bounds = ((-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0))
        with pytest.raises(ValueError, match="x_bl"):
            geom.box_tunnel_boundaries(bounds=bounds, x_bl=5.0, n_per_side=(3, 3))

    def test_hemisphere_inward_winding(self):
        hemi, equator = geom.build_hemisphere(10.0, n_theta=16, n_phi=8)
        centroids = hemi.cell_centroids
        normals = torch.linalg.cross(
            hemi.points[hemi.cells[:, 1]] - hemi.points[hemi.cells[:, 0]],
            hemi.points[hemi.cells[:, 2]] - hemi.points[hemi.cells[:, 0]],
        )
        # Inward => centroid . normal < 0 everywhere.
        assert bool(((centroids * normals).sum(dim=-1) < 0).all())
        assert equator.shape[0] == 16

    def test_split_by_freestream_partitions(self):
        hemi, _ = geom.build_hemisphere(10.0, n_theta=16, n_phi=8)
        inlet, outlet = geom.split_by_freestream(hemi, torch.tensor([1.0, 0.0, 0.0]))
        assert inlet.n_cells + outlet.n_cells == hemi.n_cells

    def test_constrained_disk_no_holes(self):
        hemi, equator = geom.build_hemisphere(10.0, n_theta=16, n_phi=8)
        disk = geom.constrained_delaunay_disk(equator_xz=hemi.points[equator][:, [0, 2]], silhouette_xzs=[])
        assert disk.n_cells > 0
        # All disk points lie on y=0.
        assert torch.allclose(disk.points[:, 1], torch.zeros(disk.n_points))


# ---------------------------------------------------------------------------
# Generators + injection
# ---------------------------------------------------------------------------


def _toy_domain():
    interior = Mesh(points=torch.rand(50, 3) * torch.tensor([120.0, 44.0, 20.0]) - torch.tensor([40.0, 22.0, 0.0]))
    vehicle = Mesh(
        points=torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        cells=torch.tensor([[0, 1, 2]]),
    )
    return DomainMesh(interior=interior, boundaries={"vehicle": vehicle})


class TestInjection:
    def test_box_tunnel_generator(self):
        dm = _toy_domain()
        gen = BoxTunnelBoundaries(
            x_min=-40, x_max=80, y_min=-22, y_max=22, z_height=20, x_bl=-2.339, n_per_side=(4, 4)
        )
        out = inject_boundaries(dm, gen)
        assert set(out.boundary_names) == {"inlet", "outlet", "slip", "no_slip", "vehicle"}
        assert out.interior.n_points == 50  # interior preserved

    def test_box_tunnel_fixed_z_floor(self):
        dm = _toy_domain()
        gen = BoxTunnelBoundaries(
            x_min=-40, x_max=80, y_min=-22, y_max=22, z_height=20, x_bl=-2.339, n_per_side=(4, 4), z_floor=0.0
        )
        out = inject_boundaries(dm, gen)
        # Floor (no_slip) should sit at z=0.
        assert torch.allclose(out.boundaries["no_slip"].points[:, 2], torch.zeros(out.boundaries["no_slip"].n_points))

    def test_boundary_injection_filter(self):
        dm = _toy_domain()
        gen = BoxTunnelBoundaries(
            x_min=-40, x_max=80, y_min=-22, y_max=22, z_height=20, x_bl=-2.339, n_per_side=(4, 4)
        )
        out = next(BoundaryInjectionFilter(gen)([dm].__iter__()))
        assert "inlet" in out.boundary_names

    def test_filter_passes_through_plain_mesh(self):
        mesh = Mesh(points=torch.rand(5, 3))
        gen = BoxTunnelBoundaries(x_min=-1, x_max=1, y_min=-1, y_max=1, z_height=2, x_bl=0.0)
        out = next(BoundaryInjectionFilter(gen)([mesh].__iter__()))
        assert isinstance(out, Mesh)

    def test_hemisphere_generator(self):
        interior = Mesh(points=torch.rand(200, 3) * torch.tensor([20.0, 10.0, 20.0]) - torch.tensor([10.0, 0.0, 10.0]))
        vehicle = Mesh(
            points=torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 0, 1]]),
            cells=torch.tensor([[0, 1, 2]]),
        )
        dm = DomainMesh(
            interior=interior,
            boundaries={"vehicle": vehicle},
            global_data={"U_inf": torch.tensor([1.0, 0.0, 0.0])},
        )
        gen = HemisphereBoundaries(radius=10.0, n_theta=16, n_phi=8)
        out = inject_boundaries(dm, gen)
        assert set(out.boundary_names) == {"inlet", "outlet", "symmetry", "vehicle"}
