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

"""Tests for new mesh filters: MeshInfoFilter, StatsFilter, PrecisionFilter."""

from __future__ import annotations

import json
import pathlib

import pytest

pytestmark = pytest.mark.requires("mesh")

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import pyvista as pv  # noqa: E402
import torch  # noqa: E402

from physicsnemo_curator.mesh.filters.mean import MeanFilter  # noqa: E402
from physicsnemo_curator.mesh.filters.mesh_info import MeshInfoFilter  # noqa: E402
from physicsnemo_curator.mesh.filters.precision import PrecisionFilter  # noqa: E402
from physicsnemo_curator.mesh.filters.stats import StatsFilter, merge_welford_stats  # noqa: E402
from physicsnemo_curator.mesh.sinks.mesh_writer import MeshSink  # noqa: E402
from physicsnemo_curator.mesh.sources.vtk import VTKSource  # noqa: E402

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


def _create_vtk_with_vector_field(directory: pathlib.Path, name: str = "vector.vtu") -> pathlib.Path:
    """Create a VTK file with vector fields for testing.

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
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    cells = np.array([[3, 0, 1, 2], [3, 0, 2, 3]])
    cell_types = np.array([5, 5])

    grid = pv.UnstructuredGrid(cells, cell_types, points)
    # Scalar field
    grid.point_data["temperature"] = np.array([100.0, 200.0, 300.0, 400.0])
    # Vector field (3 components)
    grid.point_data["velocity"] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    grid.save(str(path))
    return path


# ---------------------------------------------------------------------------
# MeshInfoFilter tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMeshInfoFilter:
    def test_yields_mesh_unchanged(self, tmp_path):
        """MeshInfoFilter should yield the mesh without modification."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource.from_path(str(tmp_path / "vtk"))
        mesh_before = next(source[0])

        filt = MeshInfoFilter()

        def gen():
            yield mesh_before

        meshes_out = list(filt(gen()))
        assert len(meshes_out) == 1
        assert meshes_out[0] is mesh_before

    def test_logs_to_console(self, tmp_path, caplog):
        """MeshInfoFilter should log mesh information."""
        import logging

        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource.from_path(str(tmp_path / "vtk"))

        filt = MeshInfoFilter(log_level="info")

        with caplog.at_level(logging.INFO):
            list(filt(source[0]))

        assert "Mesh 0:" in caplog.text
        assert "4 points" in caplog.text
        assert "2 cells" in caplog.text

    def test_writes_jsonl_file(self, tmp_path):
        """MeshInfoFilter should write JSON-lines output when path provided."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource.from_path(str(tmp_path / "vtk"))

        jsonl_path = tmp_path / "mesh_info.jsonl"
        filt = MeshInfoFilter(output=str(jsonl_path))
        list(filt(source[0]))
        filt.flush()

        assert jsonl_path.exists()
        with jsonl_path.open() as f:
            line = f.readline()
            info = json.loads(line)

        assert info["mesh_index"] == 0
        assert info["n_points"] == 4
        assert info["n_cells"] == 2
        assert "memory_estimate_bytes" in info

    def test_includes_field_info(self, tmp_path):
        """MeshInfoFilter should include field details when include_fields=True."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource.from_path(str(tmp_path / "vtk"))

        jsonl_path = tmp_path / "mesh_info.jsonl"
        filt = MeshInfoFilter(output=str(jsonl_path), include_fields=True)
        list(filt(source[0]))
        filt.flush()

        with jsonl_path.open() as f:
            info = json.loads(f.readline())

        assert "point_data_fields" in info
        assert "cell_data_fields" in info
        assert len(info["point_data_fields"]) == 2  # temperature, pressure
        assert len(info["cell_data_fields"]) == 1  # velocity

        # Check field structure
        temp_field = next(f for f in info["point_data_fields"] if f["name"] == "temperature")
        assert "shape" in temp_field
        assert "dtype" in temp_field
        assert "nbytes" in temp_field

    def test_excludes_field_info(self, tmp_path):
        """MeshInfoFilter should exclude field details when include_fields=False."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource.from_path(str(tmp_path / "vtk"))

        jsonl_path = tmp_path / "mesh_info.jsonl"
        filt = MeshInfoFilter(output=str(jsonl_path), include_fields=False)
        list(filt(source[0]))
        filt.flush()

        with jsonl_path.open() as f:
            info = json.loads(f.readline())

        assert "point_data_fields" not in info
        assert "cell_data_fields" not in info
        assert "n_point_fields" in info
        assert "n_cell_fields" in info

    def test_multiple_meshes_increment_index(self, tmp_path):
        """MeshInfoFilter should increment mesh_index for each mesh."""
        _create_vtk_file(tmp_path / "vtk", "a.vtu")
        _create_vtk_file(tmp_path / "vtk", "b.vtu")
        source = VTKSource.from_path(str(tmp_path / "vtk"))

        jsonl_path = tmp_path / "mesh_info.jsonl"
        filt = MeshInfoFilter(output=str(jsonl_path))

        for i in range(len(source)):
            list(filt(source[i]))
        filt.flush()

        with jsonl_path.open() as f:
            lines = f.readlines()

        assert len(lines) == 2
        assert json.loads(lines[0])["mesh_index"] == 0
        assert json.loads(lines[1])["mesh_index"] == 1

    def test_flush_returns_path(self, tmp_path):
        """MeshInfoFilter.flush() should return the output path."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource.from_path(str(tmp_path / "vtk"))

        jsonl_path = tmp_path / "mesh_info.jsonl"
        filt = MeshInfoFilter(output=str(jsonl_path))
        list(filt(source[0]))

        result = filt.flush()
        assert result == str(jsonl_path)

    def test_flush_without_output_returns_none(self, tmp_path):
        """MeshInfoFilter.flush() should return None when no output path."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource.from_path(str(tmp_path / "vtk"))

        filt = MeshInfoFilter()
        list(filt(source[0]))

        result = filt.flush()
        assert result is None


# ---------------------------------------------------------------------------
# StatsFilter tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestStatsFilter:
    def test_yields_mesh_unchanged(self, tmp_path):
        """StatsFilter should yield the mesh without modification."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource.from_path(str(tmp_path / "vtk"))
        mesh_before = next(source[0])

        filt = StatsFilter(output=str(tmp_path / "stats.parquet"))

        def gen():
            yield mesh_before

        meshes_out = list(filt(gen()))
        assert len(meshes_out) == 1
        assert meshes_out[0] is mesh_before

    def test_writes_parquet_with_stats(self, tmp_path):
        """StatsFilter should write Parquet file with comprehensive statistics."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource.from_path(str(tmp_path / "vtk"))

        parquet_path = tmp_path / "stats.parquet"
        filt = StatsFilter(output=str(parquet_path))
        list(filt(source[0]))
        filt.flush()

        assert parquet_path.exists()
        table = pq.read_table(str(parquet_path))

        # Check required columns exist
        required_cols = [
            "field_key",
            "component",
            "n_spatial",
            "mean",
            "std",
            "var",
            "min",
            "max",
            "median",
            "skewness",
            "kurtosis",
            "welford_n",
            "welford_mean",
            "welford_m2",
        ]
        for col in required_cols:
            assert col in table.column_names, f"Missing column: {col}"

    def test_mean_values_correct(self, tmp_path):
        """StatsFilter should compute correct mean values."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource.from_path(str(tmp_path / "vtk"))

        parquet_path = tmp_path / "stats.parquet"
        filt = StatsFilter(output=str(parquet_path))
        list(filt(source[0]))
        filt.flush()

        table = pq.read_table(str(parquet_path))

        # Find temperature row
        for i in range(table.num_rows):
            if table["field_key"][i].as_py() == "point_data/temperature":
                mean_val = table["mean"][i].as_py()
                # Mean of [100, 200, 300, 400] = 250.0
                assert abs(mean_val - 250.0) < 1e-5
                break
        else:
            pytest.fail("temperature field not found in stats")

    def test_std_values_correct(self, tmp_path):
        """StatsFilter should compute correct standard deviation."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource.from_path(str(tmp_path / "vtk"))

        parquet_path = tmp_path / "stats.parquet"
        filt = StatsFilter(output=str(parquet_path))
        list(filt(source[0]))
        filt.flush()

        table = pq.read_table(str(parquet_path))

        # Find temperature row
        for i in range(table.num_rows):
            if table["field_key"][i].as_py() == "point_data/temperature":
                std_val = table["std"][i].as_py()
                # Population std of [100, 200, 300, 400]
                expected_std = np.std([100.0, 200.0, 300.0, 400.0])
                assert abs(std_val - expected_std) < 1e-5
                break

    def test_min_max_values_correct(self, tmp_path):
        """StatsFilter should compute correct min/max values."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource.from_path(str(tmp_path / "vtk"))

        parquet_path = tmp_path / "stats.parquet"
        filt = StatsFilter(output=str(parquet_path))
        list(filt(source[0]))
        filt.flush()

        table = pq.read_table(str(parquet_path))

        for i in range(table.num_rows):
            if table["field_key"][i].as_py() == "point_data/temperature":
                min_val = table["min"][i].as_py()
                max_val = table["max"][i].as_py()
                assert abs(min_val - 100.0) < 1e-5
                assert abs(max_val - 400.0) < 1e-5
                break

    def test_vector_field_per_component_stats(self, tmp_path):
        """StatsFilter should compute per-component stats for vector fields."""
        _create_vtk_with_vector_field(tmp_path / "vtk", "vector.vtu")
        source = VTKSource.from_path(str(tmp_path / "vtk"))

        parquet_path = tmp_path / "stats.parquet"
        filt = StatsFilter(output=str(parquet_path), per_component=True)
        list(filt(source[0]))
        filt.flush()

        table = pq.read_table(str(parquet_path))

        # Find velocity rows (should have 3 components: 0, 1, 2)
        velocity_rows = []
        for i in range(table.num_rows):
            if table["field_key"][i].as_py() == "point_data/velocity":
                velocity_rows.append(
                    {
                        "component": table["component"][i].as_py(),
                        "mean": table["mean"][i].as_py(),
                    }
                )

        assert len(velocity_rows) == 3
        components = {r["component"] for r in velocity_rows}
        assert components == {0, 1, 2}

    def test_scalar_field_component_is_minus_one(self, tmp_path):
        """StatsFilter should mark scalar fields with component=-1."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource.from_path(str(tmp_path / "vtk"))

        parquet_path = tmp_path / "stats.parquet"
        filt = StatsFilter(output=str(parquet_path))
        list(filt(source[0]))
        filt.flush()

        table = pq.read_table(str(parquet_path))

        for i in range(table.num_rows):
            if table["field_key"][i].as_py() == "point_data/temperature":
                component = table["component"][i].as_py()
                assert component == -1
                break

    def test_flush_empty_returns_none(self, tmp_path):
        """StatsFilter.flush() should return None when no data processed."""
        filt = StatsFilter(output=str(tmp_path / "stats.parquet"))
        assert filt.flush() is None

    def test_welford_state_stored(self, tmp_path):
        """StatsFilter should store Welford accumulator state for merging."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource.from_path(str(tmp_path / "vtk"))

        parquet_path = tmp_path / "stats.parquet"
        filt = StatsFilter(output=str(parquet_path))
        list(filt(source[0]))
        filt.flush()

        table = pq.read_table(str(parquet_path))

        for i in range(table.num_rows):
            if table["field_key"][i].as_py() == "point_data/temperature":
                welford_n = table["welford_n"][i].as_py()
                welford_mean = table["welford_mean"][i].as_py()
                welford_m2 = table["welford_m2"][i].as_py()

                assert welford_n == 4  # 4 points
                assert abs(welford_mean - 250.0) < 1e-5
                assert welford_m2 > 0  # Should have non-zero sum of squared deviations
                break


@pytest.mark.integration
class TestMergeWelfordStats:
    def test_merge_two_parquets(self, tmp_path):
        """merge_welford_stats should correctly merge statistics from two files."""
        _create_vtk_file(tmp_path / "vtk1", "a.vtu")
        _create_vtk_file(tmp_path / "vtk2", "b.vtu")

        # Create two separate stats files
        source1 = VTKSource.from_path(str(tmp_path / "vtk1"))
        filt1 = StatsFilter(output=str(tmp_path / "stats1.parquet"))
        list(filt1(source1[0]))
        filt1.flush()

        source2 = VTKSource.from_path(str(tmp_path / "vtk2"))
        filt2 = StatsFilter(output=str(tmp_path / "stats2.parquet"))
        list(filt2(source2[0]))
        filt2.flush()

        # Merge them
        merged = merge_welford_stats([str(tmp_path / "stats1.parquet"), str(tmp_path / "stats2.parquet")])

        # Find temperature row in merged
        for i in range(merged.num_rows):
            if merged["field_key"][i].as_py() == "point_data/temperature":
                welford_n = merged["welford_n"][i].as_py()
                mean_val = merged["mean"][i].as_py()

                # Should have 8 total points (4 + 4)
                assert welford_n == 8
                # Mean should still be 250.0 (same data in both files)
                assert abs(mean_val - 250.0) < 1e-5
                break


# ---------------------------------------------------------------------------
# MeanFilter.merge tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMeanFilterMerge:
    """Tests for MeanFilter.merge() classmethod."""

    def test_merge_two_parquets(self, tmp_path):
        """MeanFilter.merge should concatenate rows from two files."""
        _create_vtk_file(tmp_path / "vtk1", "a.vtu")
        _create_vtk_file(tmp_path / "vtk2", "b.vtu")

        # Simulate two workers writing separate files.
        source1 = VTKSource.from_path(str(tmp_path / "vtk1"))
        filt1 = MeanFilter(output=str(tmp_path / "means_0.parquet"))
        list(filt1(source1[0]))
        filt1.flush()

        source2 = VTKSource.from_path(str(tmp_path / "vtk2"))
        filt2 = MeanFilter(output=str(tmp_path / "means_1.parquet"))
        list(filt2(source2[0]))
        filt2.flush()

        merged_path = MeanFilter.merge(
            [str(tmp_path / "means_0.parquet"), str(tmp_path / "means_1.parquet")],
            output=str(tmp_path / "merged_means.parquet"),
        )

        table = pq.read_table(merged_path)
        # Two meshes -> two rows
        assert table.num_rows == 2
        assert "n_points" in table.column_names
        assert "point_data/temperature" in table.column_names

    def test_merge_preserves_values(self, tmp_path):
        """Merged values should match the originals."""
        _create_vtk_file(tmp_path / "vtk1", "a.vtu")

        source = VTKSource.from_path(str(tmp_path / "vtk1"))
        filt = MeanFilter(output=str(tmp_path / "means.parquet"))
        list(filt(source[0]))
        filt.flush()

        merged_path = MeanFilter.merge(
            [str(tmp_path / "means.parquet")],
            output=str(tmp_path / "merged.parquet"),
        )

        original = pq.read_table(str(tmp_path / "means.parquet"))
        merged = pq.read_table(merged_path)
        assert original.equals(merged)

    def test_merge_empty_raises(self):
        """MeanFilter.merge should raise ValueError on empty list."""
        with pytest.raises(ValueError, match="non-empty"):
            MeanFilter.merge([], output="out.parquet")

    def test_merge_handles_heterogeneous_columns(self, tmp_path):
        """Merge should handle files with different column sets."""
        # File 1: has temperature and pressure columns
        _create_vtk_file(tmp_path / "vtk1", "a.vtu")
        source1 = VTKSource.from_path(str(tmp_path / "vtk1"))
        filt1 = MeanFilter(output=str(tmp_path / "means_0.parquet"))
        list(filt1(source1[0]))
        filt1.flush()

        # File 2: same structure (both have temperature, pressure, velocity)
        _create_vtk_file(tmp_path / "vtk2", "b.vtu")
        source2 = VTKSource.from_path(str(tmp_path / "vtk2"))
        filt2 = MeanFilter(output=str(tmp_path / "means_1.parquet"))
        list(filt2(source2[0]))
        filt2.flush()

        merged_path = MeanFilter.merge(
            [str(tmp_path / "means_0.parquet"), str(tmp_path / "means_1.parquet")],
            output=str(tmp_path / "merged.parquet"),
        )

        table = pq.read_table(merged_path)
        assert table.num_rows == 2


# ---------------------------------------------------------------------------
# StatsFilter.merge tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestStatsFilterMerge:
    """Tests for StatsFilter.merge() classmethod."""

    def test_merge_two_parquets(self, tmp_path):
        """StatsFilter.merge should produce correct merged statistics."""
        _create_vtk_file(tmp_path / "vtk1", "a.vtu")
        _create_vtk_file(tmp_path / "vtk2", "b.vtu")

        source1 = VTKSource.from_path(str(tmp_path / "vtk1"))
        filt1 = StatsFilter(output=str(tmp_path / "stats_0.parquet"))
        list(filt1(source1[0]))
        filt1.flush()

        source2 = VTKSource.from_path(str(tmp_path / "vtk2"))
        filt2 = StatsFilter(output=str(tmp_path / "stats_1.parquet"))
        list(filt2(source2[0]))
        filt2.flush()

        merged_path = StatsFilter.merge(
            [str(tmp_path / "stats_0.parquet"), str(tmp_path / "stats_1.parquet")],
            output=str(tmp_path / "merged_stats.parquet"),
        )

        table = pq.read_table(merged_path)

        # Find temperature row
        for i in range(table.num_rows):
            if table["field_key"][i].as_py() == "point_data/temperature":
                welford_n = table["welford_n"][i].as_py()
                mean_val = table["mean"][i].as_py()
                assert welford_n == 8  # 4 + 4
                assert abs(mean_val - 250.0) < 1e-5
                break
        else:
            pytest.fail("temperature field not found in merged stats")

    def test_merge_writes_file(self, tmp_path):
        """StatsFilter.merge should write the output file."""
        _create_vtk_file(tmp_path / "vtk", "a.vtu")
        source = VTKSource.from_path(str(tmp_path / "vtk"))
        filt = StatsFilter(output=str(tmp_path / "stats.parquet"))
        list(filt(source[0]))
        filt.flush()

        out = tmp_path / "merged.parquet"
        result = StatsFilter.merge([str(tmp_path / "stats.parquet")], output=str(out))
        assert out.exists()
        assert result == str(out)

    def test_merge_empty_raises(self):
        """StatsFilter.merge should raise ValueError on empty list."""
        with pytest.raises(ValueError, match="non-empty"):
            StatsFilter.merge([], output="out.parquet")


# ---------------------------------------------------------------------------
# PrecisionFilter tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPrecisionFilter:
    def test_converts_float64_to_float32(self, tmp_path):
        """PrecisionFilter should convert float64 fields to float32."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource.from_path(str(tmp_path / "vtk"))
        mesh = next(source[0])

        # Verify original dtype is float64
        assert mesh.point_data["temperature"].dtype == torch.float64

        filt = PrecisionFilter(target_dtype="float32")

        def gen():
            yield mesh

        meshes_out = list(filt(gen()))
        assert len(meshes_out) == 1

        # Verify converted dtype
        assert meshes_out[0].point_data["temperature"].dtype == torch.float32
        assert meshes_out[0].point_data["pressure"].dtype == torch.float32

    def test_converts_points_tensor(self, tmp_path):
        """PrecisionFilter should convert points tensor."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource.from_path(str(tmp_path / "vtk"))
        mesh = next(source[0])

        filt = PrecisionFilter(target_dtype="float32")

        def gen():
            yield mesh

        meshes_out = list(filt(gen()))
        assert meshes_out[0].points.dtype == torch.float32

    def test_converts_cell_data(self, tmp_path):
        """PrecisionFilter should convert cell_data fields."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource.from_path(str(tmp_path / "vtk"))
        mesh = next(source[0])

        filt = PrecisionFilter(target_dtype="float32")

        def gen():
            yield mesh

        meshes_out = list(filt(gen()))
        assert meshes_out[0].cell_data["velocity"].dtype == torch.float32

    def test_preserves_values(self, tmp_path):
        """PrecisionFilter should preserve tensor values (within precision)."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource.from_path(str(tmp_path / "vtk"))
        mesh = next(source[0])

        original_temp = mesh.point_data["temperature"].clone()

        filt = PrecisionFilter(target_dtype="float32")

        def gen():
            yield mesh

        meshes_out = list(filt(gen()))
        converted_temp = meshes_out[0].point_data["temperature"]

        # Values should be close (within float32 precision)
        assert torch.allclose(converted_temp.double(), original_temp, atol=1e-6)

    def test_float16_conversion(self, tmp_path):
        """PrecisionFilter should support float16 conversion."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource.from_path(str(tmp_path / "vtk"))
        mesh = next(source[0])

        filt = PrecisionFilter(target_dtype="float16")

        def gen():
            yield mesh

        meshes_out = list(filt(gen()))
        assert meshes_out[0].point_data["temperature"].dtype == torch.float16

    def test_invalid_dtype_raises(self):
        """PrecisionFilter should raise ValueError for invalid dtype."""
        with pytest.raises(ValueError, match="Unsupported target_dtype"):
            PrecisionFilter(target_dtype="int32")

    def test_modifies_mesh_in_place(self, tmp_path):
        """PrecisionFilter should modify the mesh in place."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource.from_path(str(tmp_path / "vtk"))
        mesh = next(source[0])

        filt = PrecisionFilter(target_dtype="float32")

        def gen():
            yield mesh

        meshes_out = list(filt(gen()))

        # Should be the same object
        assert meshes_out[0] is mesh
        # Original mesh should be modified
        assert mesh.point_data["temperature"].dtype == torch.float32

    def test_skips_non_float_tensors(self, tmp_path):
        """PrecisionFilter should skip non-floating-point tensors."""
        _create_vtk_file(tmp_path / "vtk", "test.vtu")
        source = VTKSource.from_path(str(tmp_path / "vtk"))
        mesh = next(source[0])

        # cells tensor should be integer type
        original_cells_dtype = mesh.cells.dtype

        filt = PrecisionFilter(target_dtype="float32")

        def gen():
            yield mesh

        meshes_out = list(filt(gen()))

        # cells should remain unchanged
        assert meshes_out[0].cells.dtype == original_cells_dtype


# ---------------------------------------------------------------------------
# Pipeline integration tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestNewFiltersPipeline:
    def test_chained_filters_pipeline(self, tmp_path):
        """Test pipeline with multiple new filters chained together."""
        vtk_dir = tmp_path / "vtk"
        vtk_dir.mkdir()
        _create_vtk_file(vtk_dir, "mesh_0.vtu")
        _create_vtk_file(vtk_dir, "mesh_1.vtu")

        output_dir = tmp_path / "output"
        info_path = tmp_path / "info.jsonl"
        stats_path = tmp_path / "stats.parquet"

        info_filter = MeshInfoFilter(output=str(info_path))
        precision_filter = PrecisionFilter(target_dtype="float32")
        stats_filter = StatsFilter(output=str(stats_path))

        pipeline = (
            VTKSource.from_path(str(vtk_dir))
            .filter(info_filter)
            .filter(precision_filter)
            .filter(stats_filter)
            .write(MeshSink(output_dir=str(output_dir)))
        )

        assert len(pipeline) == 2

        for i in range(len(pipeline)):
            paths = pipeline[i]
            assert len(paths) == 1
            assert pathlib.Path(paths[0]).exists()

        # Flush filters
        info_filter.flush()
        stats_filter.flush()

        # Verify all outputs
        assert info_path.exists()
        assert stats_path.exists()

        # Check info file
        with info_path.open() as f:
            lines = f.readlines()
        assert len(lines) == 2

        # Check stats file
        table = pq.read_table(str(stats_path))
        assert table.num_rows > 0

    def test_precision_before_save_reduces_size(self, tmp_path):
        """Using PrecisionFilter before saving should reduce file size."""
        vtk_dir = tmp_path / "vtk"
        _create_vtk_file(vtk_dir, "test.vtu")

        # Save without precision filter
        output_dir_fp64 = tmp_path / "output_fp64"
        pipeline_fp64 = VTKSource.from_path(str(vtk_dir)).write(MeshSink(output_dir=str(output_dir_fp64)))
        paths_fp64 = pipeline_fp64[0]

        # Save with precision filter
        output_dir_fp32 = tmp_path / "output_fp32"
        pipeline_fp32 = (
            VTKSource.from_path(str(vtk_dir))
            .filter(PrecisionFilter(target_dtype="float32"))
            .write(MeshSink(output_dir=str(output_dir_fp32)))
        )
        paths_fp32 = pipeline_fp32[0]

        # Compare sizes (fp32 should be smaller or equal)
        def get_dir_size(path):
            total = 0
            for f in pathlib.Path(path).rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
            return total

        size_fp64 = get_dir_size(paths_fp64[0])
        size_fp32 = get_dir_size(paths_fp32[0])

        assert size_fp32 <= size_fp64
