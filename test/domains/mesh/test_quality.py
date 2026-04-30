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

"""Tests for MeshQualityFilter — NaN/Inf detection and cell geometry quality."""

from __future__ import annotations

import json
import math
import pathlib
from typing import TYPE_CHECKING

import pytest

pytestmark = pytest.mark.requires("mesh")

import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from physicsnemo_curator.domains.mesh.filters.quality import (  # noqa: E402
    MeshQualityFilter,
    _aspect_ratios,
    _equiangle_skewness,
    _scaled_jacobian,
    _tet_aspect_ratios,
    _tetrahedron_dihedral_angles,
    _triangle_angles,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from physicsnemo.mesh import Mesh

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_triangle_mesh(
    n_points: int = 6,
    n_cells: int = 2,
    point_data: dict[str, torch.Tensor] | None = None,
) -> Mesh:
    """Create a simple triangular mesh.

    Parameters
    ----------
    n_points : int
        Number of vertices.
    n_cells : int
        Number of triangle cells.
    point_data : dict[str, torch.Tensor] or None
        Optional per-vertex data fields.

    Returns
    -------
    Mesh
        A mesh with triangular cells.
    """
    from physicsnemo.mesh import Mesh

    pts = torch.rand(n_points, 3)
    cells = torch.zeros(n_cells, 3, dtype=torch.int64)
    for i in range(n_cells):
        cells[i] = torch.tensor([i % n_points, (i + 1) % n_points, (i + 2) % n_points])

    td = None
    if point_data:
        td = TensorDict(point_data, batch_size=[n_points])  # ty: ignore[invalid-argument-type]

    return Mesh(points=pts, cells=cells, point_data=td)


def _make_equilateral_mesh() -> Mesh:
    """Create a mesh with a single equilateral triangle.

    Returns
    -------
    Mesh
        A mesh with one perfect equilateral triangle.
    """
    from physicsnemo.mesh import Mesh

    # Equilateral triangle with side length 1
    pts = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, math.sqrt(3.0) / 2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    return Mesh(points=pts, cells=cells)


def _make_degenerate_mesh() -> Mesh:
    """Create a mesh with a degenerate (collinear) triangle.

    Returns
    -------
    Mesh
        A mesh with one degenerate triangle (all points collinear).
    """
    from physicsnemo.mesh import Mesh

    pts = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],  # Collinear
        ],
        dtype=torch.float32,
    )
    cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    return Mesh(points=pts, cells=cells)


def _gen_meshes(*meshes: Mesh) -> Generator[Mesh]:
    """Yield meshes from a sequence.

    Parameters
    ----------
    *meshes : Mesh
        Meshes to yield.

    Yields
    ------
    Mesh
        Each mesh in order.
    """
    yield from meshes


def _make_regular_tet_mesh() -> Mesh:
    """Create a mesh with a single regular tetrahedron.

    Vertices are alternating corners of a unit cube, ordered so that the
    scalar triple product is positive (right-handed orientation).

    Returns
    -------
    Mesh
        A mesh with one regular tetrahedron inscribed in a cube.
    """
    from physicsnemo.mesh import Mesh

    # Regular tetrahedron: alternating vertices of a unit cube,
    # reordered for positive orientation (det > 0).
    pts = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    cells = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    return Mesh(points=pts, cells=cells)


def _make_degenerate_tet_mesh() -> Mesh:
    """Create a mesh with a degenerate (coplanar) tetrahedron.

    Returns
    -------
    Mesh
        A mesh with one degenerate tetrahedron (all points in the XY-plane).
    """
    from physicsnemo.mesh import Mesh

    pts = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],  # Coplanar with the other three
        ],
        dtype=torch.float32,
    )
    cells = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    return Mesh(points=pts, cells=cells)


def _make_inverted_tet_mesh() -> Mesh:
    """Create a mesh with one inverted tetrahedron.

    Swapping two vertices flips the orientation, giving a negative
    signed volume and thus a negative scaled Jacobian.

    Returns
    -------
    Mesh
        A mesh with one inverted tetrahedron.
    """
    from physicsnemo.mesh import Mesh

    # Same regular tet but v1/v2 swapped → negative orientation
    pts = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],  # was v2
            [1.0, 0.0, 1.0],  # was v1
            [0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    cells = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    return Mesh(points=pts, cells=cells)


# ---------------------------------------------------------------------------
# Tests: Geometry helpers
# ---------------------------------------------------------------------------


class TestTriangleAngles:
    """Tests for the _triangle_angles helper function."""

    def test_equilateral_triangle(self) -> None:
        """Equilateral triangle should have three 60-degree angles."""
        pts = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, math.sqrt(3) / 2, 0.0]],
        )
        cells = torch.tensor([[0, 1, 2]])
        angles = _triangle_angles(pts, cells)
        expected = math.pi / 3.0
        assert angles.shape == (1, 3)
        for i in range(3):
            assert abs(angles[0, i].item() - expected) < 1e-5

    def test_right_triangle(self) -> None:
        """Right triangle should have one 90-degree angle."""
        pts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        cells = torch.tensor([[0, 1, 2]])
        angles = _triangle_angles(pts, cells)
        angle_deg = torch.rad2deg(angles).sort(dim=-1).values
        assert abs(angle_deg[0, 0].item() - 45.0) < 1e-3
        assert abs(angle_deg[0, 1].item() - 45.0) < 1e-3
        assert abs(angle_deg[0, 2].item() - 90.0) < 1e-3


class TestAspectRatios:
    """Tests for the _aspect_ratios helper function."""

    def test_equilateral_has_ratio_one(self) -> None:
        """Equilateral triangle has aspect ratio 1.0."""
        pts = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, math.sqrt(3) / 2, 0.0]],
        )
        cells = torch.tensor([[0, 1, 2]])
        ratios = _aspect_ratios(pts, cells)
        assert abs(ratios[0].item() - 1.0) < 1e-5

    def test_elongated_triangle(self) -> None:
        """Elongated triangle should have high aspect ratio."""
        # Very thin triangle: base=10, height=0.01 → shortest edge ≈ 0.01
        pts = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.01, 0.0]])
        cells = torch.tensor([[0, 1, 2]])
        ratios = _aspect_ratios(pts, cells)
        assert ratios[0].item() > 100.0


class TestEquiangleSkewness:
    """Tests for the _equiangle_skewness helper function."""

    def test_equilateral_has_zero_skewness(self) -> None:
        """Equilateral triangle should have skewness near 0."""
        pts = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, math.sqrt(3) / 2, 0.0]],
        )
        cells = torch.tensor([[0, 1, 2]])
        angles = _triangle_angles(pts, cells)
        skew = _equiangle_skewness(angles, math.pi / 3.0)
        assert abs(skew[0].item()) < 1e-5


# ---------------------------------------------------------------------------
# MeshQualityFilter tests
# ---------------------------------------------------------------------------


class TestMeshQualityFilterPassthrough:
    """Test that MeshQualityFilter yields meshes unchanged."""

    def test_yields_mesh_unchanged(self, tmp_path: pathlib.Path) -> None:
        """Mesh should pass through the filter unmodified."""
        mesh = _make_triangle_mesh()
        filt = MeshQualityFilter(output=str(tmp_path / "quality.parquet"))
        result = list(filt(_gen_meshes(mesh)))
        assert len(result) == 1
        assert result[0] is mesh

    def test_multiple_meshes(self, tmp_path: pathlib.Path) -> None:
        """Multiple meshes should all pass through."""
        m1 = _make_triangle_mesh()
        m2 = _make_triangle_mesh(n_points=8, n_cells=3)
        filt = MeshQualityFilter(output=str(tmp_path / "quality.parquet"))
        result = list(filt(_gen_meshes(m1, m2)))
        assert len(result) == 2
        assert result[0] is m1
        assert result[1] is m2


class TestMeshQualityFilterNaN:
    """Test NaN/Inf detection in MeshQualityFilter."""

    def test_clean_data_reports_zero(self, tmp_path: pathlib.Path) -> None:
        """Clean data should report zero NaN/Inf counts."""
        mesh = _make_triangle_mesh(point_data={"temperature": torch.rand(6)})
        filt = MeshQualityFilter(output=str(tmp_path / "q.parquet"), check_geometry=False)
        list(filt(_gen_meshes(mesh)))
        path = filt.flush()

        assert path is not None
        table = pq.read_table(path)
        assert table.num_rows == 1
        assert table["nan_point_data_total"][0].as_py() == 0
        assert table["inf_point_data_total"][0].as_py() == 0

    def test_detects_nan(self, tmp_path: pathlib.Path) -> None:
        """NaN values should be counted correctly."""
        data = torch.rand(6)
        data[2] = float("nan")
        data[4] = float("nan")
        mesh = _make_triangle_mesh(point_data={"temperature": data})
        filt = MeshQualityFilter(output=str(tmp_path / "q.parquet"), check_geometry=False)
        list(filt(_gen_meshes(mesh)))
        path = filt.flush()

        table = pq.read_table(path)
        assert table["nan_point_data_total"][0].as_py() == 2
        details = json.loads(table["nan_field_details"][0].as_py())
        assert "point_data/temperature" in details
        assert details["point_data/temperature"]["nan"] == 2

    def test_detects_inf(self, tmp_path: pathlib.Path) -> None:
        """Inf values should be counted correctly."""
        data = torch.rand(6)
        data[0] = float("inf")
        data[3] = float("-inf")
        mesh = _make_triangle_mesh(point_data={"pressure": data})
        filt = MeshQualityFilter(output=str(tmp_path / "q.parquet"), check_geometry=False)
        list(filt(_gen_meshes(mesh)))
        path = filt.flush()

        table = pq.read_table(path)
        assert table["inf_point_data_total"][0].as_py() == 2

    def test_skips_integer_fields(self, tmp_path: pathlib.Path) -> None:
        """Integer fields should not be checked for NaN/Inf."""
        mesh = _make_triangle_mesh(point_data={"ids": torch.arange(6)})
        filt = MeshQualityFilter(output=str(tmp_path / "q.parquet"), check_geometry=False)
        list(filt(_gen_meshes(mesh)))
        path = filt.flush()

        table = pq.read_table(path)
        assert table["nan_point_data_total"][0].as_py() == 0

    def test_check_nan_disabled(self, tmp_path: pathlib.Path) -> None:
        """When check_nan=False, NaN/Inf should not be checked."""
        data = torch.rand(6)
        data[0] = float("nan")
        mesh = _make_triangle_mesh(point_data={"temperature": data})
        filt = MeshQualityFilter(
            output=str(tmp_path / "q.parquet"),
            check_nan=False,
            check_geometry=False,
        )
        list(filt(_gen_meshes(mesh)))
        path = filt.flush()

        table = pq.read_table(path)
        # nan_point_data_total should not be set (or defaults)
        assert table["nan_point_data_total"][0].as_py() is None


class TestMeshQualityFilterGeometry:
    """Test cell geometry quality metrics."""

    def test_equilateral_triangle(self, tmp_path: pathlib.Path) -> None:
        """Equilateral triangle should have perfect quality metrics."""
        mesh = _make_equilateral_mesh()
        filt = MeshQualityFilter(output=str(tmp_path / "q.parquet"), check_nan=False)
        list(filt(_gen_meshes(mesh)))
        path = filt.flush()

        table = pq.read_table(path)
        assert abs(table["geom_mean_aspect_ratio"][0].as_py() - 1.0) < 0.01
        assert table["geom_mean_skewness"][0].as_py() < 0.01
        assert abs(table["geom_mean_min_angle_deg"][0].as_py() - 60.0) < 0.5
        assert table["geom_n_degenerate_cells"][0].as_py() == 0

    def test_degenerate_triangle(self, tmp_path: pathlib.Path) -> None:
        """Degenerate (collinear) triangle should be flagged."""
        mesh = _make_degenerate_mesh()
        filt = MeshQualityFilter(output=str(tmp_path / "q.parquet"), check_nan=False)
        list(filt(_gen_meshes(mesh)))
        path = filt.flush()

        table = pq.read_table(path)
        assert table["geom_n_degenerate_cells"][0].as_py() >= 1
        assert table["geom_max_skewness"][0].as_py() > 0.9

    def test_check_geometry_disabled(self, tmp_path: pathlib.Path) -> None:
        """When check_geometry=False, geometry metrics should be absent."""
        mesh = _make_equilateral_mesh()
        filt = MeshQualityFilter(
            output=str(tmp_path / "q.parquet"),
            check_nan=False,
            check_geometry=False,
            check_volume=False,
            check_jacobian=False,
        )
        list(filt(_gen_meshes(mesh)))
        path = filt.flush()

        table = pq.read_table(path)
        assert table["geom_min_aspect_ratio"][0].as_py() is None


class TestMeshQualityFilterFlush:
    """Test flush and artifact tracking behavior."""

    def test_flush_returns_none_when_empty(self, tmp_path: pathlib.Path) -> None:
        """Flush with no data should return None."""
        filt = MeshQualityFilter(output=str(tmp_path / "q.parquet"))
        assert filt.flush() is None

    def test_flush_writes_parquet(self, tmp_path: pathlib.Path) -> None:
        """Flush should create a valid Parquet file."""
        mesh = _make_triangle_mesh()
        filt = MeshQualityFilter(output=str(tmp_path / "q.parquet"))
        list(filt(_gen_meshes(mesh)))
        path = filt.flush()

        assert path is not None
        assert pathlib.Path(path).exists()
        table = pq.read_table(path)
        assert table.num_rows == 1

    def test_flush_clears_state(self, tmp_path: pathlib.Path) -> None:
        """Second flush with no new data should return None."""
        mesh = _make_triangle_mesh()
        filt = MeshQualityFilter(output=str(tmp_path / "q.parquet"))
        list(filt(_gen_meshes(mesh)))
        filt.flush()
        assert filt.flush() is None

    def test_artifacts_returns_paths(self, tmp_path: pathlib.Path) -> None:
        """Artifacts should return flush paths and then clear."""
        mesh = _make_triangle_mesh()
        filt = MeshQualityFilter(output=str(tmp_path / "q.parquet"))
        list(filt(_gen_meshes(mesh)))
        filt.flush()

        arts = filt.artifacts()
        assert len(arts) == 1
        assert arts[0].endswith("q.parquet")

        # Second call should be empty
        assert filt.artifacts() == []


class TestMeshQualityFilterMerge:
    """Test merge of parallel worker outputs."""

    def test_merge_concatenates(self, tmp_path: pathlib.Path) -> None:
        """Merge should concatenate rows from multiple files."""
        # Worker 1
        m1 = _make_triangle_mesh()
        f1 = MeshQualityFilter(output=str(tmp_path / "w1.parquet"))
        list(f1(_gen_meshes(m1)))
        p1 = f1.flush()

        # Worker 2
        m2 = _make_triangle_mesh(n_points=8, n_cells=3)
        f2 = MeshQualityFilter(output=str(tmp_path / "w2.parquet"))
        list(f2(_gen_meshes(m2)))
        p2 = f2.flush()

        assert p1 is not None
        assert p2 is not None
        merged_path = MeshQualityFilter.merge([p1, p2], str(tmp_path / "merged.parquet"))
        table = pq.read_table(merged_path)
        assert table.num_rows == 2

    def test_merge_raises_on_empty(self) -> None:
        """Merge with empty list should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            MeshQualityFilter.merge([], "out.parquet")


class TestMeshQualityFilterParams:
    """Test parameter descriptors."""

    def test_params_list(self) -> None:
        """Params should return five parameters."""
        params = MeshQualityFilter.params()
        assert len(params) == 5
        names = {p.name for p in params}
        assert names == {"output", "check_nan", "check_geometry", "check_volume", "check_jacobian"}


# ---------------------------------------------------------------------------
# Tests: Tetrahedron geometry helpers
# ---------------------------------------------------------------------------


class TestTetrahedronDihedralAngles:
    """Tests for the _tetrahedron_dihedral_angles helper function."""

    def test_regular_tet_dihedral_angles(self) -> None:
        """Regular tetrahedron should have six equal dihedral angles of arccos(1/3)."""
        # Regular tet: alternating cube vertices, positive orientation
        pts = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=torch.float64,
        )
        cells = torch.tensor([[0, 1, 2, 3]])
        angles = _tetrahedron_dihedral_angles(pts, cells)
        expected = math.acos(1.0 / 3.0)  # ~70.53°
        assert angles.shape == (1, 6)
        for i in range(6):
            assert abs(angles[0, i].item() - expected) < 1e-4

    def test_degenerate_tet_has_extreme_angles(self) -> None:
        """Degenerate (coplanar) tet should have very high skewness."""
        pts = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.5, 0.0],
            ],
            dtype=torch.float64,
        )
        cells = torch.tensor([[0, 1, 2, 3]])
        angles = _tetrahedron_dihedral_angles(pts, cells)
        # Coplanar tet: dihedral angles become meaningless. The key
        # indicator is equiangle skewness being very high.
        ideal = math.acos(1.0 / 3.0)
        skew = _equiangle_skewness(angles, ideal)
        assert skew[0].item() > 0.1  # significantly distorted vs ideal 0.0


class TestTetAspectRatios:
    """Tests for the _tet_aspect_ratios helper function."""

    def test_regular_tet_has_ratio_one(self) -> None:
        """Regular tetrahedron has aspect ratio 1.0 (all edges equal)."""
        pts = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
        )
        cells = torch.tensor([[0, 1, 2, 3]])
        ratios = _tet_aspect_ratios(pts, cells)
        assert abs(ratios[0].item() - 1.0) < 1e-5

    def test_elongated_tet_has_high_ratio(self) -> None:
        """Elongated tetrahedron should have high aspect ratio."""
        pts = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, 0.01, 0.0],
                [0.0, 0.0, 0.01],
            ],
        )
        cells = torch.tensor([[0, 1, 2, 3]])
        ratios = _tet_aspect_ratios(pts, cells)
        assert ratios[0].item() > 100.0


# ---------------------------------------------------------------------------
# Tests: Tetrahedron geometry in MeshQualityFilter
# ---------------------------------------------------------------------------


class TestMeshQualityFilterTetGeometry:
    """Test that the filter handles tetrahedral cells correctly."""

    def test_regular_tet_good_quality(self, tmp_path: pathlib.Path) -> None:
        """Regular tetrahedron should have good quality metrics."""
        mesh = _make_regular_tet_mesh()
        filt = MeshQualityFilter(
            output=str(tmp_path / "q.parquet"),
            check_nan=False,
            check_volume=False,
            check_jacobian=False,
        )
        list(filt(_gen_meshes(mesh)))
        path = filt.flush()

        table = pq.read_table(path)
        assert abs(table["geom_mean_aspect_ratio"][0].as_py() - 1.0) < 0.01
        assert table["geom_mean_skewness"][0].as_py() < 0.01
        assert table["geom_n_degenerate_cells"][0].as_py() == 0

    def test_degenerate_tet_flagged(self, tmp_path: pathlib.Path) -> None:
        """Degenerate (coplanar) tet should be detected via zero volume and poor Jacobian."""
        mesh = _make_degenerate_tet_mesh()
        filt = MeshQualityFilter(
            output=str(tmp_path / "q.parquet"),
            check_nan=False,
            check_geometry=False,
        )
        list(filt(_gen_meshes(mesh)))
        path = filt.flush()

        table = pq.read_table(path)
        # Degenerate tet has zero volume
        assert table["vol_n_zero"][0].as_py() == 1
        # Degenerate tet has near-zero Jacobian (poor quality)
        assert table["jac_n_poor"][0].as_py() == 1


# ---------------------------------------------------------------------------
# Tests: Cell volume
# ---------------------------------------------------------------------------


class TestMeshQualityFilterVolume:
    """Test cell volume/area statistics in MeshQualityFilter."""

    def test_equilateral_triangle_area(self, tmp_path: pathlib.Path) -> None:
        """Equilateral triangle with side 1 has area sqrt(3)/4."""
        mesh = _make_equilateral_mesh()
        filt = MeshQualityFilter(
            output=str(tmp_path / "q.parquet"),
            check_nan=False,
            check_geometry=False,
            check_jacobian=False,
        )
        list(filt(_gen_meshes(mesh)))
        path = filt.flush()

        table = pq.read_table(path)
        expected_area = math.sqrt(3.0) / 4.0
        assert abs(table["vol_mean"][0].as_py() - expected_area) < 1e-4
        assert table["vol_n_zero"][0].as_py() == 0

    def test_degenerate_triangle_zero_volume(self, tmp_path: pathlib.Path) -> None:
        """Degenerate (collinear) triangle should have zero area."""
        mesh = _make_degenerate_mesh()
        filt = MeshQualityFilter(
            output=str(tmp_path / "q.parquet"),
            check_nan=False,
            check_geometry=False,
            check_jacobian=False,
        )
        list(filt(_gen_meshes(mesh)))
        path = filt.flush()

        table = pq.read_table(path)
        assert table["vol_n_zero"][0].as_py() == 1
        assert table["vol_min"][0].as_py() < 1e-10

    def test_regular_tet_volume(self, tmp_path: pathlib.Path) -> None:
        """Regular tet from alternating cube vertices has known volume."""
        mesh = _make_regular_tet_mesh()
        filt = MeshQualityFilter(
            output=str(tmp_path / "q.parquet"),
            check_nan=False,
            check_geometry=False,
            check_jacobian=False,
        )
        list(filt(_gen_meshes(mesh)))
        path = filt.flush()

        table = pq.read_table(path)
        # Regular tet inscribed in unit cube: edge length = sqrt(2),
        # volume = edge^3 / (6*sqrt(2)) = 2*sqrt(2)/(6*sqrt(2)) = 1/3
        expected_vol = 1.0 / 3.0
        assert abs(table["vol_mean"][0].as_py() - expected_vol) < 1e-4
        assert table["vol_n_zero"][0].as_py() == 0

    def test_volume_ratio(self, tmp_path: pathlib.Path) -> None:
        """Volume ratio should be max/min for multi-cell mesh."""
        mesh = _make_triangle_mesh()
        filt = MeshQualityFilter(
            output=str(tmp_path / "q.parquet"),
            check_nan=False,
            check_geometry=False,
            check_jacobian=False,
        )
        list(filt(_gen_meshes(mesh)))
        path = filt.flush()

        table = pq.read_table(path)
        vol_min = table["vol_min"][0].as_py()
        vol_max = table["vol_max"][0].as_py()
        vol_ratio = table["vol_ratio"][0].as_py()
        # Ratio should approximately equal max/min
        assert abs(vol_ratio - vol_max / (vol_min + 1e-30)) < 1e-3

    def test_check_volume_disabled(self, tmp_path: pathlib.Path) -> None:
        """When check_volume=False, volume metrics should be absent."""
        mesh = _make_equilateral_mesh()
        filt = MeshQualityFilter(
            output=str(tmp_path / "q.parquet"),
            check_nan=False,
            check_geometry=False,
            check_volume=False,
            check_jacobian=False,
        )
        list(filt(_gen_meshes(mesh)))
        path = filt.flush()

        table = pq.read_table(path)
        assert table["vol_min"][0].as_py() is None


# ---------------------------------------------------------------------------
# Tests: Scaled Jacobian
# ---------------------------------------------------------------------------


class TestScaledJacobian:
    """Tests for the _scaled_jacobian helper function."""

    def test_equilateral_triangle_3d(self) -> None:
        """Equilateral triangle in 3-D should have Jacobian = sqrt(3)/2."""
        pts = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, math.sqrt(3.0) / 2.0, 0.0],
            ],
            dtype=torch.float64,
        )
        cells = torch.tensor([[0, 1, 2]])
        jac = _scaled_jacobian(pts, cells)
        expected = math.sqrt(3.0) / 2.0  # ~0.866
        assert abs(jac[0].item() - expected) < 1e-5

    def test_right_triangle_3d(self) -> None:
        """Right triangle in 3-D: Jacobian = sin(90°) = 1.0 at the right-angle vertex."""
        # The scaled Jacobian uses edges from vertex 0, so place the
        # right angle at vertex 0.
        pts = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float64,
        )
        cells = torch.tensor([[0, 1, 2]])
        jac = _scaled_jacobian(pts, cells)
        # Cross product magnitude / (1*1) = 1.0
        assert abs(jac[0].item() - 1.0) < 1e-5

    def test_degenerate_triangle_zero(self) -> None:
        """Degenerate (collinear) triangle should have Jacobian near 0."""
        pts = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        cells = torch.tensor([[0, 1, 2]])
        jac = _scaled_jacobian(pts, cells)
        assert abs(jac[0].item()) < 1e-10

    def test_regular_tet(self) -> None:
        """Regular tetrahedron should have positive Jacobian."""
        pts = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=torch.float64,
        )
        cells = torch.tensor([[0, 1, 2, 3]])
        jac = _scaled_jacobian(pts, cells)
        # Regular tet: det([e1,e2,e3]) = 2, each edge norm = sqrt(2)
        # scaled_jac = 2 / (sqrt(2))^3 = 2 / (2*sqrt(2)) = 1/sqrt(2) ≈ 0.707
        expected = 1.0 / math.sqrt(2.0)
        assert abs(jac[0].item() - expected) < 1e-4

    def test_inverted_tet_negative(self) -> None:
        """Inverted tetrahedron should have negative Jacobian."""
        pts = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],  # swapped
                [1.0, 0.0, 1.0],  # swapped
                [0.0, 1.0, 1.0],
            ],
            dtype=torch.float64,
        )
        cells = torch.tensor([[0, 1, 2, 3]])
        jac = _scaled_jacobian(pts, cells)
        assert jac[0].item() < 0.0

    def test_triangle_2d(self) -> None:
        """Triangle in 2-D: signed Jacobian using 2-D cross product."""
        pts = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0]],
            dtype=torch.float64,
        )
        cells = torch.tensor([[0, 1, 2]])
        jac = _scaled_jacobian(pts, cells)
        expected = math.sqrt(3.0) / 2.0
        assert abs(jac[0].item() - expected) < 1e-5


class TestMeshQualityFilterJacobian:
    """Test scaled Jacobian integration in MeshQualityFilter."""

    def test_equilateral_triangle_jacobian(self, tmp_path: pathlib.Path) -> None:
        """Equilateral triangle should report good Jacobian."""
        mesh = _make_equilateral_mesh()
        filt = MeshQualityFilter(
            output=str(tmp_path / "q.parquet"),
            check_nan=False,
            check_geometry=False,
            check_volume=False,
        )
        list(filt(_gen_meshes(mesh)))
        path = filt.flush()

        table = pq.read_table(path)
        jac_mean = table["jac_mean"][0].as_py()
        assert jac_mean > 0.8  # sqrt(3)/2 ≈ 0.866
        assert table["jac_n_inverted"][0].as_py() == 0
        assert table["jac_n_poor"][0].as_py() == 0

    def test_degenerate_triangle_poor(self, tmp_path: pathlib.Path) -> None:
        """Degenerate triangle should have near-zero Jacobian flagged as poor."""
        mesh = _make_degenerate_mesh()
        filt = MeshQualityFilter(
            output=str(tmp_path / "q.parquet"),
            check_nan=False,
            check_geometry=False,
            check_volume=False,
        )
        list(filt(_gen_meshes(mesh)))
        path = filt.flush()

        table = pq.read_table(path)
        assert table["jac_n_poor"][0].as_py() == 1

    def test_regular_tet_jacobian(self, tmp_path: pathlib.Path) -> None:
        """Regular tet should have positive Jacobian and no poor cells."""
        mesh = _make_regular_tet_mesh()
        filt = MeshQualityFilter(
            output=str(tmp_path / "q.parquet"),
            check_nan=False,
            check_geometry=False,
            check_volume=False,
        )
        list(filt(_gen_meshes(mesh)))
        path = filt.flush()

        table = pq.read_table(path)
        assert table["jac_mean"][0].as_py() > 0.5
        assert table["jac_n_inverted"][0].as_py() == 0
        assert table["jac_n_poor"][0].as_py() == 0

    def test_inverted_tet_jacobian(self, tmp_path: pathlib.Path) -> None:
        """Inverted tet should have negative Jacobian flagged as inverted."""
        mesh = _make_inverted_tet_mesh()
        filt = MeshQualityFilter(
            output=str(tmp_path / "q.parquet"),
            check_nan=False,
            check_geometry=False,
            check_volume=False,
        )
        list(filt(_gen_meshes(mesh)))
        path = filt.flush()

        table = pq.read_table(path)
        assert table["jac_n_inverted"][0].as_py() == 1
        assert table["jac_min"][0].as_py() < 0.0

    def test_check_jacobian_disabled(self, tmp_path: pathlib.Path) -> None:
        """When check_jacobian=False, Jacobian metrics should be absent."""
        mesh = _make_equilateral_mesh()
        filt = MeshQualityFilter(
            output=str(tmp_path / "q.parquet"),
            check_nan=False,
            check_geometry=False,
            check_volume=False,
            check_jacobian=False,
        )
        list(filt(_gen_meshes(mesh)))
        path = filt.flush()

        table = pq.read_table(path)
        assert table["jac_min"][0].as_py() is None
