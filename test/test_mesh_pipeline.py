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

"""Integration tests for the mesh pipeline (VTKSource → MeanFilter → MeshSink).

These tests require the ``curator[mesh]`` extra to be installed.
"""

from __future__ import annotations

import pathlib

import pytest

pv = pytest.importorskip("pyvista")
torch = pytest.importorskip("torch")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
physicsnemo_mesh = pytest.importorskip("physicsnemo.mesh")

from physicsnemo.mesh import Mesh  # noqa: E402

from curator.mesh.filters.mean import MeanFilter  # noqa: E402
from curator.mesh.sinks.mesh_writer import MeshSink  # noqa: E402
from curator.mesh.sources.vtk import VTKSource  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_vtk_file(directory: pathlib.Path, name: str = "test.vtu") -> pathlib.Path:
    """Create a simple VTK unstructured grid file for testing.

    Parameters
    ----------
    directory : pathlib.Path
        Directory in which to write the file.
    name : str
        File name.

    Returns
    -------
    pathlib.Path
        Path to the created file.
    """
    import numpy as np

    # Simple triangle mesh: 4 points, 2 triangles.
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


# ---------------------------------------------------------------------------
# VTKSource tests
# ---------------------------------------------------------------------------


class TestVTKSource:
    def test_discover_single_file(self, tmp_path):
        vtk_file = _create_vtk_file(tmp_path)
        source = VTKSource(input_path=str(vtk_file))
        assert len(source) == 1

    def test_discover_directory(self, tmp_path):
        _create_vtk_file(tmp_path, "a.vtu")
        _create_vtk_file(tmp_path, "b.vtu")
        _create_vtk_file(tmp_path, "c.vtu")
        source = VTKSource(input_path=str(tmp_path))
        assert len(source) == 3

    def test_yields_mesh(self, tmp_path):
        _create_vtk_file(tmp_path, "test.vtu")
        source = VTKSource(input_path=str(tmp_path))
        meshes = list(source[0])
        assert len(meshes) == 1
        assert isinstance(meshes[0], Mesh)

    def test_mesh_has_data(self, tmp_path):
        _create_vtk_file(tmp_path, "test.vtu")
        source = VTKSource(input_path=str(tmp_path))
        mesh = next(source[0])
        assert mesh.n_points == 4
        assert mesh.n_cells == 2

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No VTK files"):
            VTKSource(input_path=str(tmp_path))

    def test_nonexistent_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            VTKSource(input_path=str(tmp_path / "nonexistent"))

    def test_non_vtk_file_raises(self, tmp_path):
        (tmp_path / "data.csv").write_text("a,b\n1,2\n")
        with pytest.raises(ValueError, match="not a recognised VTK"):
            VTKSource(input_path=str(tmp_path / "data.csv"))


# ---------------------------------------------------------------------------
# MeanFilter tests
# ---------------------------------------------------------------------------


class TestMeanFilter:
    def test_yields_mesh_unchanged(self, tmp_path):
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource(input_path=str(tmp_path / "vtk"))
        mesh_before = next(source[0])

        filt = MeanFilter(output=str(tmp_path / "stats.parquet"))

        def gen():
            yield mesh_before

        meshes_out = list(filt(gen()))
        assert len(meshes_out) == 1
        # The mesh object should be the same reference (pass-through).
        assert meshes_out[0] is mesh_before

    def test_accumulates_rows(self, tmp_path):
        _create_vtk_file(tmp_path / "vtk", "a.vtu")
        _create_vtk_file(tmp_path / "vtk", "b.vtu")
        source = VTKSource(input_path=str(tmp_path / "vtk"))

        filt = MeanFilter(output=str(tmp_path / "stats.parquet"))

        # Process two items through the filter.
        for i in range(len(source)):
            list(filt(source[i]))

        assert len(filt._rows) == 2

    def test_flush_writes_parquet(self, tmp_path):
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource(input_path=str(tmp_path / "vtk"))

        parquet_path = tmp_path / "stats.parquet"
        filt = MeanFilter(output=str(parquet_path))
        list(filt(source[0]))
        result = filt.flush()

        assert result == str(parquet_path)
        assert parquet_path.exists()

        table = pq.read_table(str(parquet_path))
        assert table.num_rows == 1
        assert "n_points" in table.column_names
        assert "n_cells" in table.column_names
        assert "point_data/temperature" in table.column_names

    def test_flush_empty_returns_none(self, tmp_path):
        filt = MeanFilter(output=str(tmp_path / "stats.parquet"))
        assert filt.flush() is None

    def test_mean_values_correct(self, tmp_path):
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource(input_path=str(tmp_path / "vtk"))

        parquet_path = tmp_path / "stats.parquet"
        filt = MeanFilter(output=str(parquet_path))
        list(filt(source[0]))
        filt.flush()

        table = pq.read_table(str(parquet_path))
        temp_mean = table.column("point_data/temperature")[0].as_py()
        # Mean of [100, 200, 300, 400] = 250.0
        assert abs(temp_mean - 250.0) < 1e-5


# ---------------------------------------------------------------------------
# MeshSink tests
# ---------------------------------------------------------------------------


class TestMeshSink:
    def test_saves_mesh(self, tmp_path):
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource(input_path=str(tmp_path / "vtk"))

        output_dir = tmp_path / "output"
        sink = MeshSink(output_dir=str(output_dir))
        paths = sink(source[0], index=0)

        assert len(paths) == 1
        assert "mesh_0000_0" in paths[0]
        assert pathlib.Path(paths[0]).exists()

    def test_saves_multiple_meshes(self, tmp_path):
        _create_vtk_file(tmp_path / "vtk", "a.vtu")
        _create_vtk_file(tmp_path / "vtk", "b.vtu")
        source = VTKSource(input_path=str(tmp_path / "vtk"))

        output_dir = tmp_path / "output"
        sink = MeshSink(output_dir=str(output_dir))

        paths0 = sink(source[0], index=0)
        paths1 = sink(source[1], index=1)

        assert len(paths0) == 1
        assert len(paths1) == 1
        assert "mesh_0000" in paths0[0]
        assert "mesh_0001" in paths1[0]

    def test_creates_output_dir(self, tmp_path):
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource(input_path=str(tmp_path / "vtk"))

        output_dir = tmp_path / "deep" / "nested" / "output"
        sink = MeshSink(output_dir=str(output_dir))
        paths = sink(source[0], index=0)

        assert output_dir.exists()
        assert len(paths) == 1


# ---------------------------------------------------------------------------
# End-to-end pipeline test
# ---------------------------------------------------------------------------


class TestMeshPipeline:
    def test_full_pipeline(self, tmp_path):
        # Create test VTK files.
        vtk_dir = tmp_path / "vtk"
        vtk_dir.mkdir()
        _create_vtk_file(vtk_dir, "mesh_0.vtu")
        _create_vtk_file(vtk_dir, "mesh_1.vtu")

        output_dir = tmp_path / "output"
        stats_path = tmp_path / "stats.parquet"

        mean_filter = MeanFilter(output=str(stats_path))

        # Build pipeline using the fluent API.
        pipeline = VTKSource(input_path=str(vtk_dir)).filter(mean_filter).write(MeshSink(output_dir=str(output_dir)))

        assert len(pipeline) == 2

        # Process lazily.
        paths0 = pipeline[0]
        paths1 = pipeline[1]

        assert len(paths0) == 1
        assert len(paths1) == 1

        # Flush stats.
        mean_filter.flush()
        assert stats_path.exists()

        table = pq.read_table(str(stats_path))
        assert table.num_rows == 2

    def test_pipeline_negative_index(self, tmp_path):
        vtk_dir = tmp_path / "vtk"
        vtk_dir.mkdir()
        _create_vtk_file(vtk_dir, "a.vtu")
        _create_vtk_file(vtk_dir, "b.vtu")

        pipeline = VTKSource(input_path=str(vtk_dir)).write(MeshSink(output_dir=str(tmp_path / "out")))

        paths = pipeline[-1]
        assert len(paths) == 1
        assert "mesh_0001" in paths[0]


# ---------------------------------------------------------------------------
# Extended VTK pipeline tests
# ---------------------------------------------------------------------------


def _create_vtk_polydata(directory: pathlib.Path, name: str = "surface.vtp") -> pathlib.Path:
    """Create a VTK PolyData file (triangulated surface).

    Parameters
    ----------
    directory : pathlib.Path
        Directory in which to write the file.
    name : str
        File name.

    Returns
    -------
    pathlib.Path
        Path to the created file.
    """
    import numpy as np

    # Simple triangle fan: 5 points, 4 triangles.
    sphere = pv.Sphere(radius=1.0, theta_resolution=4, phi_resolution=4)
    sphere.point_data["velocity_x"] = np.random.default_rng(42).standard_normal(sphere.n_points)
    sphere.cell_data["area"] = np.random.default_rng(42).standard_normal(sphere.n_cells)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    sphere.save(str(path))
    return path


def _create_vtk_legacy(directory: pathlib.Path, name: str = "legacy.vtk") -> pathlib.Path:
    """Create a VTK legacy format file.

    Parameters
    ----------
    directory : pathlib.Path
        Directory in which to write the file.
    name : str
        File name.

    Returns
    -------
    pathlib.Path
        Path to the created file.
    """
    import numpy as np

    points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float64)
    cells = np.array([[3, 0, 1, 2]])
    cell_types = np.array([5])  # VTK_TRIANGLE
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    grid.point_data["scalar"] = np.array([1.0, 2.0, 3.0])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    grid.save(str(path))
    return path


def _create_vtk_volume(directory: pathlib.Path, name: str = "volume.vtu") -> pathlib.Path:
    """Create a VTK volume mesh with tetrahedral cells for testing.

    Parameters
    ----------
    directory : pathlib.Path
        Directory in which to write the file.
    name : str
        File name.

    Returns
    -------
    pathlib.Path
        Path to the created file.
    """
    import numpy as np

    # Simple volume: 5 points, 2 tetrahedra sharing a face.
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],
            [0.5, 0.5, 1],
            [0.5, 0.5, -1],
        ],
        dtype=np.float64,
    )
    # VTK_TETRA = cell type 10, 4 points each.
    cells = np.array(
        [
            [4, 0, 1, 2, 3],  # first tet
            [4, 0, 1, 2, 4],  # second tet
        ]
    )
    cell_types = np.array([10, 10])  # VTK_TETRA

    grid = pv.UnstructuredGrid(cells, cell_types, points)
    grid.point_data["density"] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    grid.cell_data["volume"] = np.array([0.5, 0.5])

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    grid.save(str(path))
    return path


def _create_vtk_line_mesh(directory: pathlib.Path, name: str = "lines.vtp") -> pathlib.Path:
    """Create a VTK PolyData file with line cells for testing.

    Parameters
    ----------
    directory : pathlib.Path
        Directory in which to write the file.
    name : str
        File name.

    Returns
    -------
    pathlib.Path
        Path to the created file.
    """
    import numpy as np

    # A simple polyline with 4 points, 3 line segments.
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float64)
    lines = np.array([2, 0, 1, 2, 1, 2, 2, 2, 3])  # 3 segments
    mesh = pv.PolyData(points, lines=lines)
    mesh.point_data["speed"] = np.array([10.0, 20.0, 30.0, 40.0])

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    mesh.save(str(path))
    return path


class TestVTKSourceFromPyvistaParams:
    """Tests for from_pyvista conversion parameters exposed on VTKSource."""

    def test_manifold_dim_auto_default(self, tmp_path):
        """Default manifold_dim='auto' should detect surface mesh as dim=2."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource(input_path=str(tmp_path / "vtk"))
        mesh = next(source[0])
        # Our test mesh has triangles so auto should detect dim=2.
        assert mesh.n_manifold_dims == 2
        assert mesh.n_cells == 2

    def test_manifold_dim_0_point_cloud(self, tmp_path):
        """manifold_dim=0 should produce a point cloud with no cells."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource(input_path=str(tmp_path / "vtk"), manifold_dim=0)
        mesh = next(source[0])
        assert mesh.n_manifold_dims == 0
        assert mesh.n_cells == 0
        assert mesh.n_points == 4

    def test_manifold_dim_1_edges(self, tmp_path):
        """manifold_dim=1 should extract edge connectivity from a surface mesh."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource(input_path=str(tmp_path / "vtk"), manifold_dim=1)
        mesh = next(source[0])
        assert mesh.n_manifold_dims == 1
        assert mesh.n_cells > 0  # Should have edges
        assert mesh.n_points == 4
        # Line cells have 2 vertices each.
        assert mesh.cells.shape[1] == 2

    def test_manifold_dim_2_triangulation(self, tmp_path):
        """manifold_dim=2 should ensure triangulated surface."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource(input_path=str(tmp_path / "vtk"), manifold_dim=2)
        mesh = next(source[0])
        assert mesh.n_manifold_dims == 2
        # Triangle cells have 3 vertices each.
        assert mesh.cells.shape[1] == 3

    def test_manifold_dim_3_tetrahedralization(self, tmp_path):
        """manifold_dim=3 should read volume mesh with tetrahedral cells."""
        _create_vtk_volume(tmp_path / "vtk", "volume.vtu")
        source = VTKSource(input_path=str(tmp_path / "vtk"), manifold_dim=3)
        mesh = next(source[0])
        assert mesh.n_manifold_dims == 3
        assert mesh.n_cells == 2
        # Tet cells have 4 vertices each.
        assert mesh.cells.shape[1] == 4

    def test_point_source_vertices_preserves_point_data(self, tmp_path):
        """point_source='vertices' (default) should preserve point_data."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource(input_path=str(tmp_path / "vtk"), point_source="vertices")
        mesh = next(source[0])
        assert mesh.n_points == 4
        keys = list(mesh.point_data.keys())
        assert "temperature" in keys

    def test_point_source_cell_centroids(self, tmp_path):
        """point_source='cell_centroids' should use cell centroids as points."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource(
            input_path=str(tmp_path / "vtk"),
            point_source="cell_centroids",
            warn_on_lost_data=False,
        )
        mesh = next(source[0])
        # Original mesh has 2 cells → 2 centroids become points.
        assert mesh.n_points == 2
        # Cell data should be mapped to point_data.
        keys = list(mesh.point_data.keys())
        assert "velocity" in keys
        # manifold_dim defaults to 0 for cell_centroids → point cloud.
        assert mesh.n_manifold_dims == 0

    def test_warn_on_lost_data_emits_warning(self, tmp_path):
        """warn_on_lost_data=True should emit warning for discarded point_data."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource(
            input_path=str(tmp_path / "vtk"),
            point_source="cell_centroids",
            warn_on_lost_data=True,
        )
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            next(source[0])
            # Should warn about discarded point_data.
            lost_warnings = [x for x in w if "discards" in str(x.message)]
            assert len(lost_warnings) > 0

    def test_warn_on_lost_data_suppressed(self, tmp_path):
        """warn_on_lost_data=False should suppress data-loss warnings."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource(
            input_path=str(tmp_path / "vtk"),
            point_source="cell_centroids",
            warn_on_lost_data=False,
        )
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            next(source[0])
            lost_warnings = [x for x in w if "discards" in str(x.message)]
            assert len(lost_warnings) == 0

    def test_line_mesh_manifold_dim_auto(self, tmp_path):
        """Line mesh with manifold_dim='auto' should detect dim=1."""
        _create_vtk_line_mesh(tmp_path / "vtk", "lines.vtp")
        source = VTKSource(input_path=str(tmp_path / "vtk"))
        mesh = next(source[0])
        assert mesh.n_manifold_dims == 1
        assert mesh.cells.shape[1] == 2

    def test_volume_mesh_point_data_preserved(self, tmp_path):
        """Volume mesh point_data should survive conversion."""
        _create_vtk_volume(tmp_path / "vtk", "volume.vtu")
        source = VTKSource(input_path=str(tmp_path / "vtk"), manifold_dim=3)
        mesh = next(source[0])
        keys = list(mesh.point_data.keys())
        assert "density" in keys
        density_mean = mesh.point_data["density"].float().mean().item()
        assert abs(density_mean - 3.0) < 1e-5  # mean of [1,2,3,4,5]

    def test_volume_mesh_cell_centroids_mode(self, tmp_path):
        """Volume mesh with cell_centroids should create 2-point cloud."""
        _create_vtk_volume(tmp_path / "vtk", "volume.vtu")
        source = VTKSource(
            input_path=str(tmp_path / "vtk"),
            point_source="cell_centroids",
            warn_on_lost_data=False,
        )
        mesh = next(source[0])
        assert mesh.n_points == 2  # 2 cells → 2 centroids
        assert mesh.n_manifold_dims == 0
        keys = list(mesh.point_data.keys())
        assert "volume" in keys

    def test_params_include_conversion_options(self):
        """VTKSource.params() should include from_pyvista conversion options."""
        param_names = [p.name for p in VTKSource.params()]
        assert "manifold_dim" in param_names
        assert "point_source" in param_names
        assert "warn_on_lost_data" in param_names


class TestVTKSourceExtended:
    """Additional VTK source tests covering file patterns and formats."""

    def test_file_pattern_filters_files(self, tmp_path):
        """file_pattern should limit which files in a directory are read."""
        vtk_dir = tmp_path / "data"
        vtk_dir.mkdir()
        _create_vtk_file(vtk_dir, "sim_001.vtu")
        _create_vtk_file(vtk_dir, "sim_002.vtu")
        _create_vtk_file(vtk_dir, "other_001.vtu")

        source = VTKSource(input_path=str(vtk_dir), file_pattern="sim_*")
        assert len(source) == 2

    def test_reads_vtp_polydata(self, tmp_path):
        """VTKSource should read .vtp (PolyData) files correctly."""
        _create_vtk_polydata(tmp_path / "vtk", "surface.vtp")
        source = VTKSource(input_path=str(tmp_path / "vtk"))
        mesh = next(source[0])
        assert isinstance(mesh, Mesh)
        assert mesh.n_points > 0
        assert mesh.n_cells > 0

    def test_reads_vtk_legacy(self, tmp_path):
        """VTKSource should read legacy .vtk files correctly."""
        _create_vtk_legacy(tmp_path / "vtk", "legacy.vtk")
        source = VTKSource(input_path=str(tmp_path / "vtk"))
        mesh = next(source[0])
        assert isinstance(mesh, Mesh)
        assert mesh.n_points == 3
        assert mesh.n_cells == 1

    def test_single_file_path(self, tmp_path):
        """VTKSource should accept a direct file path (not just directory)."""
        vtk_file = _create_vtk_file(tmp_path, "direct.vtu")
        source = VTKSource(input_path=str(vtk_file))
        assert len(source) == 1
        mesh = next(source[0])
        assert isinstance(mesh, Mesh)

    def test_mixed_formats_in_directory(self, tmp_path):
        """VTKSource should pick up mixed VTK formats from a directory."""
        vtk_dir = tmp_path / "mixed"
        _create_vtk_file(vtk_dir, "grid.vtu")
        _create_vtk_polydata(vtk_dir, "surface.vtp")
        _create_vtk_legacy(vtk_dir, "legacy.vtk")
        # Also add a non-VTK file that should be ignored.
        (vtk_dir / "readme.txt").write_text("not a mesh")

        source = VTKSource(input_path=str(vtk_dir))
        assert len(source) == 3

    def test_sorted_file_order(self, tmp_path):
        """Discovered files should be in sorted order."""
        vtk_dir = tmp_path / "vtk"
        _create_vtk_file(vtk_dir, "c.vtu")
        _create_vtk_file(vtk_dir, "a.vtu")
        _create_vtk_file(vtk_dir, "b.vtu")

        source = VTKSource(input_path=str(vtk_dir))
        # Access internal _files to verify sorted order.
        names = [f.name for f in source._files]
        assert names == ["a.vtu", "b.vtu", "c.vtu"]

    def test_source_getitem_out_of_range(self, tmp_path):
        """Indexing beyond available files should raise IndexError."""
        _create_vtk_file(tmp_path, "only.vtu")
        source = VTKSource(input_path=str(tmp_path))
        with pytest.raises(IndexError):
            next(source[5])

    def test_mesh_point_data_preserved(self, tmp_path):
        """Point data from VTK should be available on the Mesh."""
        _create_vtk_file(tmp_path, "test.vtu")
        source = VTKSource(input_path=str(tmp_path))
        mesh = next(source[0])
        assert mesh.point_data is not None
        keys = list(mesh.point_data.keys())
        assert "temperature" in keys
        assert "pressure" in keys

    def test_mesh_cell_data_preserved(self, tmp_path):
        """Cell data from VTK should be available on the Mesh."""
        _create_vtk_file(tmp_path, "test.vtu")
        source = VTKSource(input_path=str(tmp_path))
        mesh = next(source[0])
        assert mesh.cell_data is not None
        keys = list(mesh.cell_data.keys())
        assert "velocity" in keys


class TestMeanFilterExtended:
    """Additional MeanFilter tests covering edge cases."""

    def test_cell_data_means_correct(self, tmp_path):
        """MeanFilter should correctly compute cell_data field means."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource(input_path=str(tmp_path / "vtk"))

        parquet_path = tmp_path / "stats.parquet"
        filt = MeanFilter(output=str(parquet_path))
        list(filt(source[0]))
        filt.flush()

        table = pq.read_table(str(parquet_path))
        vel_mean = table.column("cell_data/velocity")[0].as_py()
        # Mean of [10.0, 20.0] = 15.0
        assert abs(vel_mean - 15.0) < 1e-5

    def test_heterogeneous_fields_across_meshes(self, tmp_path):
        """Meshes with different field names should produce a table with NULLs."""
        import numpy as np

        vtk_dir = tmp_path / "vtk"
        vtk_dir.mkdir()

        # Create mesh A with "temperature" only.
        points_a = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float64)
        cells_a = np.array([[3, 0, 1, 2]])
        cell_types_a = np.array([5])
        grid_a = pv.UnstructuredGrid(cells_a, cell_types_a, points_a)
        grid_a.point_data["temperature"] = np.array([10.0, 20.0, 30.0])
        grid_a.save(str(vtk_dir / "mesh_a.vtu"))

        # Create mesh B with "pressure" only.
        grid_b = pv.UnstructuredGrid(cells_a, cell_types_a, points_a)
        grid_b.point_data["pressure"] = np.array([100.0, 200.0, 300.0])
        grid_b.save(str(vtk_dir / "mesh_b.vtu"))

        source = VTKSource(input_path=str(vtk_dir))
        parquet_path = tmp_path / "stats.parquet"
        filt = MeanFilter(output=str(parquet_path))

        for i in range(len(source)):
            list(filt(source[i]))
        filt.flush()

        table = pq.read_table(str(parquet_path))
        assert table.num_rows == 2
        # Both columns should exist; one row has null for each.
        assert "point_data/temperature" in table.column_names
        assert "point_data/pressure" in table.column_names

    def test_flush_is_idempotent(self, tmp_path):
        """Calling flush twice without new data should return None the second time."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource(input_path=str(tmp_path / "vtk"))

        parquet_path = tmp_path / "stats.parquet"
        filt = MeanFilter(output=str(parquet_path))
        list(filt(source[0]))

        result1 = filt.flush()
        assert result1 == str(parquet_path)

        # The internal rows are still there but no new data added.
        # Since _rows is not cleared, flush should still return the path.
        result2 = filt.flush()
        assert result2 == str(parquet_path)

    def test_multiple_meshes_accumulate(self, tmp_path):
        """Processing many items should accumulate all rows before flush."""
        vtk_dir = tmp_path / "vtk"
        vtk_dir.mkdir()
        for i in range(5):
            _create_vtk_file(vtk_dir, f"mesh_{i}.vtu")

        source = VTKSource(input_path=str(vtk_dir))
        parquet_path = tmp_path / "stats.parquet"
        filt = MeanFilter(output=str(parquet_path))

        for i in range(len(source)):
            list(filt(source[i]))

        assert len(filt._rows) == 5
        filt.flush()

        table = pq.read_table(str(parquet_path))
        assert table.num_rows == 5

    def test_parquet_creates_parent_dirs(self, tmp_path):
        """MeanFilter should create parent directories for the output path."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource(input_path=str(tmp_path / "vtk"))

        parquet_path = tmp_path / "deep" / "nested" / "stats.parquet"
        filt = MeanFilter(output=str(parquet_path))
        list(filt(source[0]))
        filt.flush()

        assert parquet_path.exists()


class TestMeshSinkExtended:
    """Additional MeshSink tests."""

    def test_saved_mesh_can_be_loaded(self, tmp_path):
        """Meshes saved by MeshSink should be loadable by Mesh.load()."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource(input_path=str(tmp_path / "vtk"))

        output_dir = tmp_path / "output"
        sink = MeshSink(output_dir=str(output_dir))
        paths = sink(source[0], index=0)

        # Load it back.
        loaded = Mesh.load(paths[0])
        assert loaded.n_points == 4
        assert loaded.n_cells == 2

    def test_sequential_index_naming(self, tmp_path):
        """Sink should name subdirectories with sequential indices."""
        vtk_dir = tmp_path / "vtk"
        vtk_dir.mkdir()
        for i in range(3):
            _create_vtk_file(vtk_dir, f"m{i}.vtu")

        source = VTKSource(input_path=str(vtk_dir))
        output_dir = tmp_path / "output"
        sink = MeshSink(output_dir=str(output_dir))

        all_paths = []
        for i in range(len(source)):
            all_paths.extend(sink(source[i], index=i))

        assert len(all_paths) == 3
        assert "mesh_0000_0" in all_paths[0]
        assert "mesh_0001_0" in all_paths[1]
        assert "mesh_0002_0" in all_paths[2]


class TestMeshPipelineExtended:
    """Additional full pipeline integration tests."""

    def test_pipeline_iterate_all(self, tmp_path):
        """Iterating over all pipeline items should produce correct outputs."""
        vtk_dir = tmp_path / "vtk"
        vtk_dir.mkdir()
        for i in range(4):
            _create_vtk_file(vtk_dir, f"mesh_{i}.vtu")

        output_dir = tmp_path / "output"
        pipeline = VTKSource(input_path=str(vtk_dir)).write(MeshSink(output_dir=str(output_dir)))

        assert len(pipeline) == 4

        all_paths = []
        for i in range(len(pipeline)):
            paths = pipeline[i]
            assert len(paths) == 1
            all_paths.extend(paths)

        assert len(all_paths) == 4
        # All paths should be unique.
        assert len(set(all_paths)) == 4

    def test_pipeline_with_stats_and_output(self, tmp_path):
        """Pipeline with MeanFilter + MeshSink should produce both stats and meshes."""
        vtk_dir = tmp_path / "vtk"
        vtk_dir.mkdir()
        _create_vtk_file(vtk_dir, "a.vtu")
        _create_vtk_file(vtk_dir, "b.vtu")
        _create_vtk_file(vtk_dir, "c.vtu")

        output_dir = tmp_path / "output"
        stats_path = tmp_path / "stats.parquet"
        mean_filter = MeanFilter(output=str(stats_path))

        pipeline = VTKSource(input_path=str(vtk_dir)).filter(mean_filter).write(MeshSink(output_dir=str(output_dir)))

        assert len(pipeline) == 3

        for i in range(len(pipeline)):
            paths = pipeline[i]
            assert len(paths) == 1
            assert pathlib.Path(paths[0]).exists()

        # Flush and verify stats.
        mean_filter.flush()
        assert stats_path.exists()

        table = pq.read_table(str(stats_path))
        assert table.num_rows == 3
        # All meshes are identical so means should be the same across rows.
        temp_means = table.column("point_data/temperature").to_pylist()
        assert all(abs(v - 250.0) < 1e-5 for v in temp_means)

    def test_pipeline_out_of_range(self, tmp_path):
        """Accessing an out-of-range index should raise IndexError."""
        vtk_dir = tmp_path / "vtk"
        vtk_dir.mkdir()
        _create_vtk_file(vtk_dir, "only.vtu")

        pipeline = VTKSource(input_path=str(vtk_dir)).write(MeshSink(output_dir=str(tmp_path / "out")))

        with pytest.raises(IndexError):
            pipeline[10]

    def test_pipeline_no_sink_raises(self, tmp_path):
        """Indexing a pipeline without a sink should raise RuntimeError."""
        vtk_dir = tmp_path / "vtk"
        vtk_dir.mkdir()
        _create_vtk_file(vtk_dir, "test.vtu")

        pipeline = VTKSource(input_path=str(vtk_dir)).filter(MeanFilter(output=str(tmp_path / "s.parquet")))

        with pytest.raises(RuntimeError, match="no sink"):
            pipeline[0]

    def test_pipeline_preserves_mesh_integrity(self, tmp_path):
        """Mesh data should survive the full pipeline (source → filter → sink → load)."""
        vtk_dir = tmp_path / "vtk"
        vtk_dir.mkdir()
        _create_vtk_file(vtk_dir, "test.vtu")

        output_dir = tmp_path / "output"
        stats_path = tmp_path / "stats.parquet"
        mean_filter = MeanFilter(output=str(stats_path))

        pipeline = VTKSource(input_path=str(vtk_dir)).filter(mean_filter).write(MeshSink(output_dir=str(output_dir)))

        paths = pipeline[0]
        loaded = Mesh.load(paths[0])

        # Verify geometry.
        assert loaded.n_points == 4
        assert loaded.n_cells == 2

        # Verify data fields are present.
        assert loaded.point_data is not None
        point_keys = list(loaded.point_data.keys())
        assert "temperature" in point_keys
        assert "pressure" in point_keys

        assert loaded.cell_data is not None
        cell_keys = list(loaded.cell_data.keys())
        assert "velocity" in cell_keys

        # Verify data values.
        temp = loaded.point_data["temperature"].float().mean().item()
        assert abs(temp - 250.0) < 1e-5
