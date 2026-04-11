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

These tests require the ``mesh`` dependency group to be installed.
"""

from __future__ import annotations

import pathlib

import pytest

pytestmark = pytest.mark.requires("mesh")

import pyarrow.parquet as pq  # noqa: E402
import pyvista as pv  # noqa: E402
from physicsnemo.mesh import Mesh  # noqa: E402

from physicsnemo_curator.mesh.filters.mean import MeanFilter  # noqa: E402
from physicsnemo_curator.mesh.sinks.mesh_writer import MeshSink  # noqa: E402
from physicsnemo_curator.mesh.sources.vtk import VTKSource  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mesh_to_device(mesh: Mesh, device: str) -> Mesh:
    """Move a Mesh tensorclass to the given device.

    Parameters
    ----------
    mesh : Mesh
        Source mesh (usually on CPU after VTK import).
    device : str
        Target device string (``"cpu"`` or ``"cuda"``).

    Returns
    -------
    Mesh
        Mesh on the target device.
    """
    return mesh.to(device)


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


@pytest.mark.integration
class TestVTKSource:
    def test_discover_single_file(self, tmp_path):
        vtk_file = _create_vtk_file(tmp_path)
        source = VTKSource(str(vtk_file))
        assert len(source) == 1

    def test_discover_directory(self, tmp_path):
        _create_vtk_file(tmp_path, "a.vtu")
        _create_vtk_file(tmp_path, "b.vtu")
        _create_vtk_file(tmp_path, "c.vtu")
        source = VTKSource(str(tmp_path))
        assert len(source) == 3

    def test_yields_mesh(self, tmp_path):
        _create_vtk_file(tmp_path, "test.vtu")
        source = VTKSource(str(tmp_path))
        meshes = list(source[0])
        assert len(meshes) == 1
        assert isinstance(meshes[0], Mesh)

    def test_mesh_has_data(self, tmp_path):
        _create_vtk_file(tmp_path, "test.vtu")
        source = VTKSource(str(tmp_path))
        mesh = next(source[0])
        assert mesh.n_points == 4
        assert mesh.n_cells == 2

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No VTK files found"):
            VTKSource(str(tmp_path))

    def test_nonexistent_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            VTKSource(str(tmp_path / "nonexistent"))

    def test_non_vtk_file_raises(self, tmp_path):
        (tmp_path / "data.csv").write_text("a,b\n1,2\n")
        with pytest.raises(ValueError, match="does not have a recognised VTK extension"):
            VTKSource(str(tmp_path / "data.csv"))


# ---------------------------------------------------------------------------
# MeanFilter tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMeanFilter:
    def test_yields_mesh_unchanged(self, tmp_path):
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource(str(tmp_path / "vtk"))
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
        source = VTKSource(str(tmp_path / "vtk"))

        filt = MeanFilter(output=str(tmp_path / "stats.parquet"))

        # Process two items through the filter.
        for i in range(len(source)):
            list(filt(source[i]))

        assert len(filt._rows) == 2

    def test_flush_writes_parquet(self, tmp_path):
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource(str(tmp_path / "vtk"))

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
        source = VTKSource(str(tmp_path / "vtk"))

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


@pytest.mark.integration
class TestMeshSink:
    def test_saves_mesh(self, tmp_path):
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource(str(tmp_path / "vtk"))

        output_dir = tmp_path / "output"
        sink = MeshSink(output_dir=str(output_dir))
        paths = sink(source[0], index=0)

        assert len(paths) == 1
        assert "mesh_0000_0" in paths[0]
        assert pathlib.Path(paths[0]).exists()

    def test_saves_multiple_meshes(self, tmp_path):
        _create_vtk_file(tmp_path / "vtk", "a.vtu")
        _create_vtk_file(tmp_path / "vtk", "b.vtu")
        source = VTKSource(str(tmp_path / "vtk"))

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
        source = VTKSource(str(tmp_path / "vtk"))

        output_dir = tmp_path / "deep" / "nested" / "output"
        sink = MeshSink(output_dir=str(output_dir))
        paths = sink(source[0], index=0)

        assert output_dir.exists()
        assert len(paths) == 1

    def test_custom_naming_template(self, tmp_path):
        """Custom naming_template controls the output directory name."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource(str(tmp_path / "vtk"))

        output_dir = tmp_path / "output"
        sink = MeshSink(
            output_dir=str(output_dir),
            naming_template="boundary_{index}.vtp.pmsh",
        )
        paths = sink(source[0], index=0)

        assert len(paths) == 1
        assert paths[0].endswith("boundary_0.vtp.pmsh")
        assert pathlib.Path(paths[0]).exists()

    def test_naming_template_with_seq(self, tmp_path):
        """Sequence placeholder increments across meshes from one source."""
        vtk_dir = tmp_path / "vtk"
        _create_vtk_file(vtk_dir, "a.vtu")
        _create_vtk_file(vtk_dir, "b.vtu")
        source = VTKSource(str(vtk_dir))

        output_dir = tmp_path / "output"
        sink = MeshSink(
            output_dir=str(output_dir),
            naming_template="run_{index:03d}_step_{seq}.pmsh",
        )

        # source[0] yields 1 mesh, source[1] yields 1 mesh
        paths0 = sink(source[0], index=0)
        paths1 = sink(source[1], index=1)

        assert paths0[0].endswith("run_000_step_0.pmsh")
        assert paths1[0].endswith("run_001_step_0.pmsh")

    def test_naming_template_none_uses_default(self, tmp_path):
        """naming_template=None preserves the original mesh_XXXX_Y pattern."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource(str(tmp_path / "vtk"))

        output_dir = tmp_path / "output"
        sink = MeshSink(output_dir=str(output_dir), naming_template=None)
        paths = sink(source[0], index=0)

        assert len(paths) == 1
        assert "mesh_0000_0" in paths[0]

    def test_naming_template_invalid_placeholder(self):
        """Invalid placeholders raise ValueError at construction time."""
        with pytest.raises(ValueError, match="Invalid naming_template"):
            MeshSink(output_dir="/tmp/out", naming_template="mesh_{bad_key}")

    def test_naming_template_in_params(self):
        """naming_template appears in the params list."""
        param_names = [p.name for p in MeshSink.params()]
        assert "naming_template" in param_names


# ---------------------------------------------------------------------------
# End-to-end pipeline test
# ---------------------------------------------------------------------------


@pytest.mark.e2e
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
        pipeline = VTKSource(str(vtk_dir)).filter(mean_filter).write(MeshSink(output_dir=str(output_dir)))

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

        pipeline = VTKSource(str(vtk_dir)).write(MeshSink(output_dir=str(tmp_path / "out")))

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


@pytest.mark.integration
class TestVTKSourceFromPyvistaParams:
    """Tests for from_pyvista conversion parameters exposed on VTKSource."""

    def test_manifold_dim_auto_default(self, tmp_path):
        """Default manifold_dim='auto' should detect surface mesh as dim=2."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource(str(tmp_path / "vtk"))
        mesh = next(source[0])
        # Our test mesh has triangles so auto should detect dim=2.
        assert mesh.n_manifold_dims == 2
        assert mesh.n_cells == 2

    def test_manifold_dim_0_point_cloud(self, tmp_path):
        """manifold_dim=0 should produce a point cloud with no cells."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource(str(tmp_path / "vtk"), manifold_dim=0)
        mesh = next(source[0])
        assert mesh.n_manifold_dims == 0
        assert mesh.n_cells == 0
        assert mesh.n_points == 4

    def test_manifold_dim_1_edges(self, tmp_path):
        """manifold_dim=1 should extract edge connectivity from a surface mesh."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource(str(tmp_path / "vtk"), manifold_dim=1)
        mesh = next(source[0])
        assert mesh.n_manifold_dims == 1
        assert mesh.n_cells > 0  # Should have edges
        assert mesh.n_points == 4
        # Line cells have 2 vertices each.
        assert mesh.cells.shape[1] == 2

    def test_manifold_dim_2_triangulation(self, tmp_path):
        """manifold_dim=2 should ensure triangulated surface."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource(str(tmp_path / "vtk"), manifold_dim=2)
        mesh = next(source[0])
        assert mesh.n_manifold_dims == 2
        # Triangle cells have 3 vertices each.
        assert mesh.cells.shape[1] == 3

    def test_manifold_dim_3_tetrahedralization(self, tmp_path):
        """manifold_dim=3 should read volume mesh with tetrahedral cells."""
        _create_vtk_volume(tmp_path / "vtk", "volume.vtu")
        source = VTKSource(str(tmp_path / "vtk"), manifold_dim=3)
        mesh = next(source[0])
        assert mesh.n_manifold_dims == 3
        assert mesh.n_cells == 2
        # Tet cells have 4 vertices each.
        assert mesh.cells.shape[1] == 4

    def test_point_source_vertices_preserves_point_data(self, tmp_path):
        """point_source='vertices' (default) should preserve point_data."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource(str(tmp_path / "vtk"), point_source="vertices")
        mesh = next(source[0])
        assert mesh.n_points == 4
        keys = list(mesh.point_data.keys())
        assert "temperature" in keys

    def test_point_source_cell_centroids(self, tmp_path):
        """point_source='cell_centroids' should use cell centroids as points."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource(
            str(tmp_path / "vtk"),
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
            str(tmp_path / "vtk"),
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
            str(tmp_path / "vtk"),
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
        source = VTKSource(str(tmp_path / "vtk"))
        mesh = next(source[0])
        assert mesh.n_manifold_dims == 1
        assert mesh.cells.shape[1] == 2

    def test_volume_mesh_point_data_preserved(self, tmp_path):
        """Volume mesh point_data should survive conversion."""
        _create_vtk_volume(tmp_path / "vtk", "volume.vtu")
        source = VTKSource(str(tmp_path / "vtk"), manifold_dim=3)
        mesh = next(source[0])
        keys = list(mesh.point_data.keys())
        assert "density" in keys
        density_mean = mesh.point_data["density"].float().mean().item()
        assert abs(density_mean - 3.0) < 1e-5  # mean of [1,2,3,4,5]

    def test_volume_mesh_cell_centroids_mode(self, tmp_path):
        """Volume mesh with cell_centroids should create 2-point cloud."""
        _create_vtk_volume(tmp_path / "vtk", "volume.vtu")
        source = VTKSource(
            str(tmp_path / "vtk"),
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


@pytest.mark.integration
class TestVTKSourceExtended:
    """Additional VTK source tests covering file patterns and formats."""

    def test_file_pattern_filters_files(self, tmp_path):
        """file_pattern should limit which files in a directory are read."""
        vtk_dir = tmp_path / "data"
        vtk_dir.mkdir()
        _create_vtk_file(vtk_dir, "sim_001.vtu")
        _create_vtk_file(vtk_dir, "sim_002.vtu")
        _create_vtk_file(vtk_dir, "other_001.vtu")

        source = VTKSource(str(vtk_dir), file_pattern="sim_*")
        assert len(source) == 2

    def test_reads_vtp_polydata(self, tmp_path):
        """VTKSource should read .vtp (PolyData) files correctly."""
        _create_vtk_polydata(tmp_path / "vtk", "surface.vtp")
        source = VTKSource(str(tmp_path / "vtk"))
        mesh = next(source[0])
        assert isinstance(mesh, Mesh)
        assert mesh.n_points > 0
        assert mesh.n_cells > 0

    def test_reads_vtk_legacy(self, tmp_path):
        """VTKSource should read legacy .vtk files correctly."""
        _create_vtk_legacy(tmp_path / "vtk", "legacy.vtk")
        source = VTKSource(str(tmp_path / "vtk"))
        mesh = next(source[0])
        assert isinstance(mesh, Mesh)
        assert mesh.n_points == 3
        assert mesh.n_cells == 1

    def test_single_file_path(self, tmp_path):
        """VTKSource should accept a direct file path (not just directory)."""
        vtk_file = _create_vtk_file(tmp_path, "direct.vtu")
        source = VTKSource(str(vtk_file))
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

        source = VTKSource(str(vtk_dir))
        assert len(source) == 3

    def test_sorted_file_order(self, tmp_path):
        """Discovered files should be in sorted order."""
        vtk_dir = tmp_path / "vtk"
        _create_vtk_file(vtk_dir, "c.vtu")
        _create_vtk_file(vtk_dir, "a.vtu")
        _create_vtk_file(vtk_dir, "b.vtu")

        source = VTKSource(str(vtk_dir))
        # Access internal _files to verify sorted order.
        names = [pathlib.Path(source._files[i]).name for i in range(len(source))]
        assert names == ["a.vtu", "b.vtu", "c.vtu"]

    def test_source_getitem_out_of_range(self, tmp_path):
        """Indexing beyond available files should raise IndexError."""
        _create_vtk_file(tmp_path, "only.vtu")
        source = VTKSource(str(tmp_path))
        with pytest.raises(IndexError):
            next(source[5])

    def test_mesh_point_data_preserved(self, tmp_path):
        """Point data from VTK should be available on the Mesh."""
        _create_vtk_file(tmp_path, "test.vtu")
        source = VTKSource(str(tmp_path))
        mesh = next(source[0])
        assert mesh.point_data is not None
        keys = list(mesh.point_data.keys())
        assert "temperature" in keys
        assert "pressure" in keys

    def test_mesh_cell_data_preserved(self, tmp_path):
        """Cell data from VTK should be available on the Mesh."""
        _create_vtk_file(tmp_path, "test.vtu")
        source = VTKSource(str(tmp_path))
        mesh = next(source[0])
        assert mesh.cell_data is not None
        keys = list(mesh.cell_data.keys())
        assert "velocity" in keys


@pytest.mark.integration
class TestMeanFilterExtended:
    """Additional MeanFilter tests covering edge cases."""

    def test_cell_data_means_correct(self, tmp_path):
        """MeanFilter should correctly compute cell_data field means."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource(str(tmp_path / "vtk"))

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

        source = VTKSource(str(vtk_dir))
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
        source = VTKSource(str(tmp_path / "vtk"))

        parquet_path = tmp_path / "stats.parquet"
        filt = MeanFilter(output=str(parquet_path))
        list(filt(source[0]))

        result1 = filt.flush()
        assert result1 == str(parquet_path)

        # flush() clears internal rows after writing, so a second call
        # with no new data should return None.
        result2 = filt.flush()
        assert result2 is None

    def test_multiple_meshes_accumulate(self, tmp_path):
        """Processing many items should accumulate all rows before flush."""
        vtk_dir = tmp_path / "vtk"
        vtk_dir.mkdir()
        for i in range(5):
            _create_vtk_file(vtk_dir, f"mesh_{i}.vtu")

        source = VTKSource(str(vtk_dir))
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
        source = VTKSource(str(tmp_path / "vtk"))

        parquet_path = tmp_path / "deep" / "nested" / "stats.parquet"
        filt = MeanFilter(output=str(parquet_path))
        list(filt(source[0]))
        filt.flush()

        assert parquet_path.exists()


@pytest.mark.integration
class TestMeshSinkExtended:
    """Additional MeshSink tests."""

    def test_saved_mesh_can_be_loaded(self, tmp_path):
        """Meshes saved by MeshSink should be loadable by Mesh.load()."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource(str(tmp_path / "vtk"))

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

        source = VTKSource(str(vtk_dir))
        output_dir = tmp_path / "output"
        sink = MeshSink(output_dir=str(output_dir))

        all_paths = []
        for i in range(len(source)):
            all_paths.extend(sink(source[i], index=i))

        assert len(all_paths) == 3
        assert "mesh_0000_0" in all_paths[0]
        assert "mesh_0001_0" in all_paths[1]
        assert "mesh_0002_0" in all_paths[2]


@pytest.mark.e2e
class TestMeshPipelineExtended:
    """Additional full pipeline integration tests."""

    def test_pipeline_iterate_all(self, tmp_path):
        """Iterating over all pipeline items should produce correct outputs."""
        vtk_dir = tmp_path / "vtk"
        vtk_dir.mkdir()
        for i in range(4):
            _create_vtk_file(vtk_dir, f"mesh_{i}.vtu")

        output_dir = tmp_path / "output"
        pipeline = VTKSource(str(vtk_dir)).write(MeshSink(output_dir=str(output_dir)))

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

        pipeline = VTKSource(str(vtk_dir)).filter(mean_filter).write(MeshSink(output_dir=str(output_dir)))

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

        pipeline = VTKSource(str(vtk_dir)).write(MeshSink(output_dir=str(tmp_path / "out")))

        with pytest.raises(IndexError):
            pipeline[10]

    def test_pipeline_no_sink_raises(self, tmp_path):
        """Indexing a pipeline without a sink should raise RuntimeError."""
        vtk_dir = tmp_path / "vtk"
        vtk_dir.mkdir()
        _create_vtk_file(vtk_dir, "test.vtu")

        pipeline = VTKSource(str(vtk_dir)).filter(MeanFilter(output=str(tmp_path / "s.parquet")))

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

        pipeline = VTKSource(str(vtk_dir)).filter(mean_filter).write(MeshSink(output_dir=str(output_dir)))

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


# ---------------------------------------------------------------------------
# Device-parametrised tests (CPU + CUDA when available)
# ---------------------------------------------------------------------------


@pytest.mark.device
class TestMeshDeviceOps:
    """Tests that exercise Mesh operations on each available device.

    The ``device`` fixture (from ``conftest.py``) yields ``"cpu"`` and
    ``"cuda"`` when a GPU is present.
    """

    def test_mesh_to_device(self, tmp_path, device):
        """Mesh should be movable to the target device."""
        _create_vtk_file(tmp_path, "test.vtu")
        source = VTKSource(str(tmp_path))
        mesh = next(source[0])
        mesh = _mesh_to_device(mesh, device)

        assert mesh.points.device.type == device.split(":")[0]
        assert mesh.cells.device.type == device.split(":")[0]

    def test_point_data_on_device(self, tmp_path, device):
        """Point data tensors should reside on the correct device after move."""
        _create_vtk_file(tmp_path, "test.vtu")
        source = VTKSource(str(tmp_path))
        mesh = _mesh_to_device(next(source[0]), device)

        for key in mesh.point_data.keys():  # noqa: SIM118
            assert mesh.point_data[key].device.type == device.split(":")[0]

    def test_cell_data_on_device(self, tmp_path, device):
        """Cell data tensors should reside on the correct device after move."""
        _create_vtk_file(tmp_path, "test.vtu")
        source = VTKSource(str(tmp_path))
        mesh = _mesh_to_device(next(source[0]), device)

        for key in mesh.cell_data.keys():  # noqa: SIM118
            assert mesh.cell_data[key].device.type == device.split(":")[0]

    def test_mesh_properties_on_device(self, tmp_path, device):
        """Derived properties (n_points, n_cells) should work on any device."""
        _create_vtk_file(tmp_path, "test.vtu")
        source = VTKSource(str(tmp_path))
        mesh = _mesh_to_device(next(source[0]), device)

        assert mesh.n_points == 4
        assert mesh.n_cells == 2
        assert mesh.n_spatial_dims == 3
        assert mesh.n_manifold_dims == 2

    def test_mean_filter_on_device(self, tmp_path, device):
        """MeanFilter should compute correct means from device tensors."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource(str(tmp_path / "vtk"))
        mesh = _mesh_to_device(next(source[0]), device)

        parquet_path = tmp_path / "stats.parquet"
        filt = MeanFilter(output=str(parquet_path))

        def gen():
            yield mesh

        list(filt(gen()))
        filt.flush()

        table = pq.read_table(str(parquet_path))
        temp_mean = table.column("point_data/temperature")[0].as_py()
        assert abs(temp_mean - 250.0) < 1e-5

    def test_sink_from_device_mesh(self, tmp_path, device):
        """MeshSink should handle meshes on any device (saves to CPU)."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource(str(tmp_path / "vtk"))
        mesh = _mesh_to_device(next(source[0]), device)

        output_dir = tmp_path / "output"
        sink = MeshSink(output_dir=str(output_dir))

        def gen():
            yield mesh

        paths = sink(gen(), index=0)
        assert len(paths) == 1
        assert pathlib.Path(paths[0]).exists()

        # Verify the saved mesh can be loaded.
        loaded = Mesh.load(paths[0])
        assert loaded.n_points == 4


# ---------------------------------------------------------------------------
# Remote DrivAerML dataset end-to-end test
# ---------------------------------------------------------------------------

#: Expected cell-data field names present on every DrivAerML slice.
_DRIVAERML_CELL_FIELDS: frozenset[str] = frozenset(
    {
        "CpMeanTrim",
        "CptMeanTrim",
        "magUMeanNormTrim",
        "microDragMeanTrim",
        "nutMeanTrim",
        "pMeanTrim",
        "pPrime2MeanTrim",
        "UMeanTrim",
        "UPrime2MeanTrim",
    }
)

#: URL for the DrivAerML dataset slices on HuggingFace Hub.
_DRIVAERML_URL = "hf://datasets/neashton/drivaerml/run_1/slices"


@pytest.mark.e2e
@pytest.mark.slow
class TestDrivAerMLRemotePipeline:
    """End-to-end tests fetching real DrivAerML VTP slices from HuggingFace.

    These tests download a small subset of the DrivAerML dataset
    (``xNormal_p6*.vtp`` — 3 files, ~7 MB total) and exercise the full
    VTKSource → MeanFilter → MeshSink pipeline against real CFD data.

    The ``slow`` marker allows deselecting them in quick CI runs via
    ``pytest -m 'not slow'``.
    """

    @pytest.fixture(autouse=True)
    def _drivaerml_source(self, tmp_path):
        """Download DrivAerML slices via fsspec, then read locally.

        Uses fsspec to download a small subset of slice VTP
        files, then creates a VTKSource pointing at the local cache.
        """
        import fsspec

        cache_dir = str(tmp_path / "hf_cache")
        fs, root_path = fsspec.core.url_to_fs(_DRIVAERML_URL)
        glob_expr = f"{root_path}/xNormal_p6*"
        all_files = fs.glob(glob_expr)
        files = sorted(f for f in all_files if f.endswith(".vtp"))
        # Force download by caching each file
        for remote_path in files:
            local_path = pathlib.Path(cache_dir) / remote_path.lstrip("/")
            if not local_path.exists():
                local_path.parent.mkdir(parents=True, exist_ok=True)
                fs.get(remote_path, str(local_path))
        # Point VTKSource at the cache directory
        self.source = VTKSource(cache_dir, warn_on_lost_data=False)
        self.tmp_path = tmp_path

    # -- Source tests -------------------------------------------------------

    def test_discovers_three_files(self):
        """The xNormal_p6* pattern should match exactly 3 VTP files."""
        assert len(self.source) == 3

    def test_yields_mesh_objects(self):
        """Each item should yield a valid physicsnemo Mesh."""
        for i in range(len(self.source)):
            mesh = next(self.source[i])
            assert isinstance(mesh, Mesh)

    def test_mesh_geometry_is_2d_surface(self):
        """DrivAerML slices are 2-D triangulated surfaces in 3-D space."""
        mesh = next(self.source[0])
        assert mesh.n_spatial_dims == 3
        assert mesh.n_manifold_dims == 2
        # Triangulated → 3 vertices per cell.
        assert mesh.cells.shape[1] == 3

    def test_mesh_has_expected_cell_data_fields(self):
        """Every mesh must carry the canonical set of DrivAerML cell fields."""
        for i in range(len(self.source)):
            mesh = next(self.source[i])
            actual_keys = set(mesh.cell_data.keys())  # noqa: SIM118
            assert actual_keys >= _DRIVAERML_CELL_FIELDS, (
                f"Mesh {i} missing fields: {_DRIVAERML_CELL_FIELDS - actual_keys}"
            )

    def test_cell_data_shapes_are_consistent(self):
        """Scalar fields should be (n_cells,) and vector fields (n_cells, d)."""
        mesh = next(self.source[0])
        n_cells = mesh.n_cells
        for key in mesh.cell_data.keys():  # noqa: SIM118
            tensor = mesh.cell_data[key]
            assert tensor.shape[0] == n_cells, f"{key}: first dim should be n_cells"
            if key == "UMeanTrim":
                assert tensor.shape == (n_cells, 3)
            elif key == "UPrime2MeanTrim":
                assert tensor.shape == (n_cells, 6)
            else:
                assert tensor.ndim == 1, f"{key}: expected scalar (1-D), got shape {tensor.shape}"

    def test_no_nans_in_cell_data(self):
        """Cell-data tensors should be free of NaN values."""
        import torch

        mesh = next(self.source[0])
        for key in mesh.cell_data.keys():  # noqa: SIM118
            tensor = mesh.cell_data[key].float()
            assert not torch.isnan(tensor).any(), f"NaN found in {key}"

    def test_slices_have_non_trivial_geometry(self):
        """Each slice should have a meaningful number of points and cells."""
        for i in range(len(self.source)):
            mesh = next(self.source[i])
            assert mesh.n_points > 100, f"Mesh {i}: too few points ({mesh.n_points})"
            assert mesh.n_cells > 100, f"Mesh {i}: too few cells ({mesh.n_cells})"

    # -- Cross-slice consistency -------------------------------------------

    def test_all_slices_share_same_fields(self):
        """All downloaded slices must have identical cell-data field sets."""
        field_sets = []
        for i in range(len(self.source)):
            mesh = next(self.source[i])
            field_sets.append(set(mesh.cell_data.keys()))  # noqa: SIM118
        for j in range(1, len(field_sets)):
            assert field_sets[j] == field_sets[0], (
                f"Slice {j} fields differ from slice 0: "
                f"extra={field_sets[j] - field_sets[0]}, "
                f"missing={field_sets[0] - field_sets[j]}"
            )

    def test_all_slices_are_x_normal_planes(self):
        """x-normal slices should have constant x-coordinates (within tolerance)."""
        for i in range(len(self.source)):
            mesh = next(self.source[i])
            x_coords = mesh.points[:, 0]
            # All points should share the same x-value (it's a planar slice).
            x_range = x_coords.max() - x_coords.min()
            assert x_range < 0.01, f"Mesh {i}: x-range is {x_range.item():.4f}, expected near-constant"

    def test_slices_at_different_x_positions(self):
        """The 3 x-normal slices should be at distinct x-positions."""
        import torch

        x_positions = []
        for i in range(len(self.source)):
            mesh = next(self.source[i])
            x_positions.append(mesh.points[:, 0].mean().item())
        # They should all be different (p61000, p63000, p65000 → x=6.1, 6.3, 6.5).
        x_positions = torch.tensor(x_positions)
        diffs = torch.cdist(x_positions.unsqueeze(1), x_positions.unsqueeze(1))
        # Off-diagonal elements should be > 0.
        for ii in range(len(x_positions)):
            for jj in range(ii + 1, len(x_positions)):
                assert diffs[ii, jj] > 0.01, f"Slices {ii} and {jj} are at the same x-position"

    # -- Full pipeline test ------------------------------------------------

    def test_full_pipeline_source_to_sink(self):
        """Run the complete VTKSource → MeanFilter → MeshSink pipeline."""
        output_dir = self.tmp_path / "output"
        stats_path = self.tmp_path / "stats.parquet"
        mean_filter = MeanFilter(output=str(stats_path))

        pipeline = self.source.filter(mean_filter).write(MeshSink(output_dir=str(output_dir)))

        assert len(pipeline) == 3

        all_paths: list[str] = []
        for i in range(len(pipeline)):
            paths = pipeline[i]
            assert len(paths) >= 1
            for p in paths:
                assert pathlib.Path(p).exists(), f"Output path {p} does not exist"
            all_paths.extend(paths)

        # Flush stats.
        result = mean_filter.flush()
        assert result == str(stats_path)
        assert stats_path.exists()

        # Verify parquet table.
        table = pq.read_table(str(stats_path))
        assert table.num_rows == 3

        # All cell_data fields should appear as columns.
        for field in _DRIVAERML_CELL_FIELDS:
            col_name = f"cell_data/{field}"
            assert col_name in table.column_names, f"Missing column {col_name}"

        # Verify saved meshes can be loaded back.
        for p in all_paths:
            loaded = Mesh.load(p)
            assert loaded.n_points > 0
            assert loaded.n_cells > 0
            assert loaded.n_manifold_dims == 2

    def test_saved_meshes_match_originals(self):
        """Meshes saved and reloaded should match the original geometry."""
        import torch

        output_dir = self.tmp_path / "roundtrip"
        sink = MeshSink(output_dir=str(output_dir))

        for i in range(len(self.source)):
            original = next(self.source[i])
            paths = sink(iter([original]), index=i)
            loaded = Mesh.load(paths[0])

            assert loaded.n_points == original.n_points
            assert loaded.n_cells == original.n_cells
            assert torch.allclose(loaded.points, original.points, atol=1e-6)
            assert torch.equal(loaded.cells, original.cells)

            # Check cell data values survive round-trip.
            for key in original.cell_data.keys():  # noqa: SIM118
                assert torch.allclose(
                    loaded.cell_data[key].float(),
                    original.cell_data[key].float(),
                    atol=1e-6,
                ), f"cell_data[{key!r}] differs after round-trip for mesh {i}"
