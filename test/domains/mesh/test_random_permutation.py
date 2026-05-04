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

"""Tests for RandomPermutationFilter."""

from __future__ import annotations

import pathlib

import pytest

pytestmark = pytest.mark.requires("mesh")

import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402
import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from physicsnemo_curator.domains.mesh.filters.random_permutation import (  # noqa: E402
    RandomPermutationFilter,
)

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
    # 4 points, 2 triangles
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


def _make_mesh() -> object:
    """Build a small Mesh object directly (no file I/O).

    Returns
    -------
    Mesh
        A mesh with 4 points, 2 triangles, point_data and cell_data.
    """
    from physicsnemo.mesh import Mesh

    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
    point_data = TensorDict(
        {
            "temperature": torch.tensor([100.0, 200.0, 300.0, 400.0], dtype=torch.float64),
            "pressure": torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64),
        },
        batch_size=[4],
    )
    cell_data = TensorDict(
        {
            "velocity": torch.tensor([10.0, 20.0], dtype=torch.float64),
        },
        batch_size=[2],
    )
    return Mesh(
        points=points,
        cells=cells,
        point_data=point_data,
        cell_data=cell_data,
        global_data=TensorDict({"run_id": torch.tensor(42)}, batch_size=[]),
    )


# ---------------------------------------------------------------------------
# Unit tests (metadata)
# ---------------------------------------------------------------------------


class TestRandomPermutationFilterUnit:
    """Metadata and parameter tests."""

    def test_params_list(self) -> None:
        """Filter should declare a seed parameter."""
        params = RandomPermutationFilter.params()
        assert len(params) == 1
        assert params[0].name == "seed"
        assert params[0].type is int
        # seed is required (no default)
        assert params[0].required

    def test_name_and_description(self) -> None:
        """Filter should expose name and description."""
        assert isinstance(RandomPermutationFilter.name, str)
        assert len(RandomPermutationFilter.name) > 0
        assert isinstance(RandomPermutationFilter.description, str)
        assert len(RandomPermutationFilter.description) > 0


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRandomPermutationFilterIntegration:
    """Tests against in-memory meshes."""

    def test_yields_same_mesh_object(self) -> None:
        """Filter should modify the mesh in place and yield the same object."""
        mesh = _make_mesh()
        filt = RandomPermutationFilter(seed=42)

        def gen():
            yield mesh

        meshes_out = list(filt(gen()))
        assert len(meshes_out) == 1
        assert meshes_out[0] is mesh

    def test_preserves_point_set(self) -> None:
        """After shuffling, the set of point coordinates must be the same."""
        mesh = _make_mesh()
        original_points = mesh.points.clone()

        filt = RandomPermutationFilter(seed=99)

        def gen():
            yield mesh

        list(filt(gen()))

        # Same set of rows, possibly in different order
        assert mesh.points.shape == original_points.shape
        sorted_original, _ = original_points.sort(dim=0)
        sorted_new, _ = mesh.points.sort(dim=0)
        assert torch.allclose(sorted_original, sorted_new)

    def test_preserves_point_data_values(self) -> None:
        """Point data values must be preserved (same multiset)."""
        mesh = _make_mesh()
        original_temp = mesh.point_data["temperature"].clone()
        original_pres = mesh.point_data["pressure"].clone()

        filt = RandomPermutationFilter(seed=99)

        def gen():
            yield mesh

        list(filt(gen()))

        new_temp = mesh.point_data["temperature"]
        new_pres = mesh.point_data["pressure"]

        # Same multiset
        assert torch.allclose(original_temp.sort()[0], new_temp.sort()[0])
        assert torch.allclose(original_pres.sort()[0], new_pres.sort()[0])

    def test_preserves_cell_data_values(self) -> None:
        """Cell data values must be preserved (same multiset)."""
        mesh = _make_mesh()
        original_vel = mesh.cell_data["velocity"].clone()

        filt = RandomPermutationFilter(seed=99)

        def gen():
            yield mesh

        list(filt(gen()))

        new_vel = mesh.cell_data["velocity"]
        assert torch.allclose(original_vel.sort()[0], new_vel.sort()[0])

    def test_point_data_consistent_with_points(self) -> None:
        """Point data must follow the same permutation as points."""
        mesh = _make_mesh()
        # Record the correspondence: temperature[i] = (i+1)*100
        # So after shuffling, temp should match points in the same way
        original_points = mesh.points.clone()
        original_temp = mesh.point_data["temperature"].clone()

        filt = RandomPermutationFilter(seed=42)

        def gen():
            yield mesh

        list(filt(gen()))

        # For each new point position, find which original index it came from
        # and verify temperature matches
        for new_idx in range(mesh.points.shape[0]):
            new_pt = mesh.points[new_idx]
            # Find the original index with the same coordinates
            diffs = (original_points - new_pt).abs().sum(dim=1)
            orig_idx = int(diffs.argmin())
            assert torch.allclose(
                mesh.point_data["temperature"][new_idx],
                original_temp[orig_idx],
            )

    def test_cell_connectivity_remapped(self) -> None:
        """Cell connectivity should reference the correct new point indices."""
        mesh = _make_mesh()
        original_points = mesh.points.clone()
        original_cells = mesh.cells.clone()

        filt = RandomPermutationFilter(seed=42)

        def gen():
            yield mesh

        list(filt(gen()))

        # Each cell should reference the same physical points as before
        for cell_idx in range(mesh.cells.shape[0]):
            # Get the physical points for this cell after permutation
            new_cell_pts = mesh.points[mesh.cells[cell_idx]]
            old_cell_pts = original_points[original_cells[cell_idx]]

            # Sort rows to compare sets (cell may be in different order)
            new_sorted, _ = new_cell_pts.sort(dim=0)
            old_sorted, _ = old_cell_pts.sort(dim=0)
            assert torch.allclose(new_sorted, old_sorted)

    def test_cell_data_consistent_with_cells(self) -> None:
        """Cell data must follow the same permutation as cell rows."""
        mesh = _make_mesh()
        # Record original cell-to-velocity mapping using physical point
        # positions so we can identify cells after remapping.
        original_points = mesh.points.clone()
        original_cells = mesh.cells.clone()
        original_vel = mesh.cell_data["velocity"].clone()

        # Build a mapping: frozenset of point coordinates -> velocity
        orig_cell_to_vel: dict[tuple[tuple[float, ...], ...], float] = {}
        for ci in range(original_cells.shape[0]):
            pts = tuple(tuple(original_points[idx].tolist()) for idx in original_cells[ci].tolist())
            key = tuple(sorted(pts))
            orig_cell_to_vel[key] = original_vel[ci].item()

        filt = RandomPermutationFilter(seed=42)

        def gen():
            yield mesh

        list(filt(gen()))

        # Verify each cell's velocity still corresponds to the same
        # physical cell (identified by point coordinates)
        for ci in range(mesh.cells.shape[0]):
            pts = tuple(tuple(mesh.points[idx].tolist()) for idx in mesh.cells[ci].tolist())
            key = tuple(sorted(pts))
            assert key in orig_cell_to_vel
            assert mesh.cell_data["velocity"][ci].item() == pytest.approx(orig_cell_to_vel[key])

    def test_global_data_unchanged(self) -> None:
        """Global data must not be modified."""
        mesh = _make_mesh()
        original_run_id = mesh.global_data["run_id"].clone()

        filt = RandomPermutationFilter(seed=42)

        def gen():
            yield mesh

        list(filt(gen()))

        assert torch.equal(mesh.global_data["run_id"], original_run_id)

    def test_reproducible_with_same_seed(self) -> None:
        """Same seed should produce identical permutations."""
        mesh1 = _make_mesh()
        mesh2 = _make_mesh()

        filt1 = RandomPermutationFilter(seed=42)
        filt2 = RandomPermutationFilter(seed=42)

        def gen1():
            yield mesh1

        def gen2():
            yield mesh2

        list(filt1(gen1()))
        list(filt2(gen2()))

        assert torch.equal(mesh1.points, mesh2.points)
        assert torch.equal(mesh1.cells, mesh2.cells)
        assert torch.equal(
            mesh1.point_data["temperature"],
            mesh2.point_data["temperature"],
        )

    def test_different_seed_gives_different_result(self) -> None:
        """Different seeds should (very likely) produce different orderings."""
        mesh1 = _make_mesh()
        mesh2 = _make_mesh()

        filt1 = RandomPermutationFilter(seed=42)
        filt2 = RandomPermutationFilter(seed=9999)

        def gen1():
            yield mesh1

        def gen2():
            yield mesh2

        list(filt1(gen1()))
        list(filt2(gen2()))

        # At least one of these should differ (overwhelmingly likely)
        points_differ = not torch.equal(mesh1.points, mesh2.points)
        cells_differ = not torch.equal(mesh1.cells, mesh2.cells)
        assert points_differ or cells_differ

    def test_counter_increments_per_mesh(self) -> None:
        """Each mesh in a stream should get a different permutation."""
        mesh1 = _make_mesh()
        mesh2 = _make_mesh()

        filt = RandomPermutationFilter(seed=42)

        def gen():
            yield mesh1
            yield mesh2

        results = list(filt(gen()))
        assert len(results) == 2

        # The two meshes should (very likely) have different orderings
        points_differ = not torch.equal(results[0].points, results[1].points)
        cells_differ = not torch.equal(results[0].cells, results[1].cells)
        assert points_differ or cells_differ

    def test_mesh_without_cells(self) -> None:
        """Filter should handle point-cloud meshes (no cells) gracefully."""
        from physicsnemo.mesh import Mesh

        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        point_data = TensorDict(
            {"field": torch.tensor([10.0, 20.0, 30.0])},
            batch_size=[3],
        )
        mesh = Mesh(
            points=points,
            cells=None,
            point_data=point_data,
            cell_data=None,
            global_data=None,
        )

        filt = RandomPermutationFilter(seed=42)

        def gen():
            yield mesh

        result = list(filt(gen()))
        assert len(result) == 1
        # Points still form the same set
        assert torch.allclose(points.sort(dim=0)[0], result[0].points.sort(dim=0)[0])

    def test_mesh_without_point_data(self) -> None:
        """Filter should handle meshes with no point_data."""
        from physicsnemo.mesh import Mesh

        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            dtype=torch.float64,
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        mesh = Mesh(
            points=points,
            cells=cells,
            point_data=None,
            cell_data=None,
            global_data=None,
        )

        filt = RandomPermutationFilter(seed=42)

        def gen():
            yield mesh

        result = list(filt(gen()))
        assert len(result) == 1

    def test_single_point_mesh_unchanged(self) -> None:
        """A mesh with a single point should pass through unchanged."""
        from physicsnemo.mesh import Mesh

        points = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        mesh = Mesh(
            points=points,
            cells=None,
            point_data=None,
            cell_data=None,
            global_data=None,
        )
        original_points = mesh.points.clone()

        filt = RandomPermutationFilter(seed=42)

        def gen():
            yield mesh

        list(filt(gen()))
        assert torch.equal(mesh.points, original_points)


# ---------------------------------------------------------------------------
# VTK-based integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRandomPermutationFilterVTK:
    """Tests that load from VTK files."""

    def test_from_vtk_source(self, tmp_path: pathlib.Path) -> None:
        """Filter should work with meshes loaded via VTKSource."""
        from physicsnemo_curator.domains.mesh.sources.vtk import VTKSource

        _create_vtk_file(tmp_path / "vtk")
        source = VTKSource(str(tmp_path / "vtk"))
        mesh = next(source[0])

        original_points = mesh.points.clone()

        filt = RandomPermutationFilter(seed=42)

        def gen():
            yield mesh

        result = list(filt(gen()))
        assert len(result) == 1

        # Same set of points
        sorted_orig, _ = original_points.sort(dim=0)
        sorted_new, _ = result[0].points.sort(dim=0)
        assert torch.allclose(sorted_orig, sorted_new)


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestRandomPermutationFilterPipeline:
    """End-to-end pipeline tests."""

    def test_chained_in_pipeline(self, tmp_path: pathlib.Path) -> None:
        """Filter should work correctly in a chained pipeline."""
        from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink
        from physicsnemo_curator.domains.mesh.sources.vtk import VTKSource

        vtk_dir = tmp_path / "vtk"
        _create_vtk_file(vtk_dir, "mesh_0.vtu")
        _create_vtk_file(vtk_dir, "mesh_1.vtu")

        output_dir = tmp_path / "output"

        filt = RandomPermutationFilter(seed=42)

        pipeline = VTKSource(str(vtk_dir)).filter(filt).write(MeshSink(output_dir=str(output_dir)))

        assert len(pipeline) == 2

        for i in range(len(pipeline)):
            paths = pipeline[i]
            assert len(paths) == 1
            assert pathlib.Path(paths[0]).exists()

    def test_with_precision_filter(self, tmp_path: pathlib.Path) -> None:
        """Filter should compose correctly with PrecisionFilter."""
        from physicsnemo_curator.domains.mesh.filters.precision import (
            PrecisionFilter,
        )
        from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink
        from physicsnemo_curator.domains.mesh.sources.vtk import VTKSource

        vtk_dir = tmp_path / "vtk"
        _create_vtk_file(vtk_dir, "mesh.vtu")

        output_dir = tmp_path / "output"

        pipeline = (
            VTKSource(str(vtk_dir))
            .filter(RandomPermutationFilter(seed=42))
            .filter(PrecisionFilter(target_dtype="float32"))
            .write(MeshSink(output_dir=str(output_dir)))
        )

        paths = pipeline[0]
        assert len(paths) == 1
        assert pathlib.Path(paths[0]).exists()


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestRandomPermutationFilterRegistry:
    """Test that the filter is registered."""

    def test_filter_registered(self) -> None:
        """RandomPermutationFilter should be discoverable via the registry."""
        from physicsnemo_curator.core.registry import registry

        names = [f.name for f in registry.list_filters("mesh")]
        assert "Random Permutation" in names
