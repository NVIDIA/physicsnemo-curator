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

"""Guard retained Nastran cells against stale or transformed native projections."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import pyvista as pv
import torch
from physicsnemo.mesh import Mesh

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("pyNastran")

from physicsnemo_curator.domains.mesh.sinks.mesh_vtk import MeshVTKSink, read_vtk_data
from physicsnemo_curator.domains.mesh.sources.nastran import NastranSource

from .test_nastran import _deck

pytestmark = [pytest.mark.requires("mesh"), pytest.mark.integration]


@pytest.fixture
def nastran_mesh(tmp_path: Path) -> Mesh:
    path = tmp_path / "model.bdf"
    _deck(path)
    return next(NastranSource(str(path), read_results=False)[0])


@pytest.mark.parametrize("change", ["subset", "duplicate", "connectivity"])
def test_changed_projection_cannot_resurrect_original_cells(tmp_path: Path, nastran_mesh: Mesh, change: str) -> None:
    if change == "subset":
        mesh = nastran_mesh.slice_cells([0])
    elif change == "duplicate":
        # Preserve all parent IDs and the child count while replacing one cell.
        mesh = nastran_mesh.slice_cells([0, 0, 2, 3, 4, 5, 6, 7])
    else:
        mesh = nastran_mesh
        mesh.cells[0, 0] = 1
    sink = MeshVTKSink(str(tmp_path / "original"))
    with pytest.raises(ValueError, match="Projected Nastran cells changed.*use_original_geometry=False"):
        sink(iter([mesh]), 0)
    assert not (tmp_path / "original").exists()
    path = MeshVTKSink(str(tmp_path / "projection"), use_original_geometry=False)(iter([mesh]), 0)[0]
    grid = pv.read(path)
    assert grid.n_cells == mesh.n_cells
    np.testing.assert_array_equal(grid.cell_connectivity.reshape(-1, 4), mesh.cells.numpy())


def test_original_export_allows_point_cell_and_vertex_permutations(tmp_path: Path, nastran_mesh: Mesh) -> None:
    point_order = torch.randperm(nastran_mesh.n_points, generator=torch.Generator().manual_seed(17))
    inverse = torch.argsort(point_order)
    cell_order = torch.arange(nastran_mesh.n_cells - 1, -1, -1)
    mesh = Mesh(
        points=nastran_mesh.points[point_order],
        cells=inverse[nastran_mesh.cells[cell_order]][:, [1, 0, 2, 3]],
        point_data=nastran_mesh.point_data[point_order],
        cell_data=nastran_mesh.cell_data[cell_order],
        global_data=nastran_mesh.global_data,
    )
    path = MeshVTKSink(str(tmp_path / "out"))(iter([mesh]), 0)[0]
    grid = pv.read(path)
    assert grid.celltypes.tolist() == [24]
    np.testing.assert_array_equal(
        grid.point_data["node_id"][grid.cell_connectivity], nastran_mesh.point_data["node_id"].numpy()
    )


@pytest.mark.parametrize("change", ["missing_snapshot", "duplicate_node_id", "missing_node_id"])
def test_original_export_requires_projection_identity(tmp_path: Path, nastran_mesh: Mesh, change: str) -> None:
    if change == "missing_snapshot":
        del nastran_mesh.global_data["nastran", "projection_node_ids"]
    elif change == "duplicate_node_id":
        nastran_mesh.point_data["node_id"][0] = nastran_mesh.point_data["node_id"][1]
    else:
        del nastran_mesh.point_data["node_id"]
    with pytest.raises(ValueError, match="use_original_geometry=False"):
        MeshVTKSink(str(tmp_path / "out"))(iter([nastran_mesh]), 0)


@pytest.mark.parametrize("change", ["values", "dtype", "shape", "nested_field"])
def test_original_export_rejects_changed_cell_fields(tmp_path: Path, nastran_mesh: Mesh, change: str) -> None:
    if change == "values":
        nastran_mesh.cell_data["density"][0] += 1
    elif change == "dtype":
        nastran_mesh.cell_data["density"] = nastran_mesh.cell_data["density"].float()
    elif change == "shape":
        nastran_mesh.cell_data["density"] = nastran_mesh.cell_data["density"].unsqueeze(1)
    else:
        nastran_mesh.cell_data["new", "value"] = torch.ones(nastran_mesh.n_cells)
    with pytest.raises(ValueError, match="Projected cell .* changed.*use_original_geometry=False"):
        MeshVTKSink(str(tmp_path / "out"))(iter([nastran_mesh]), 0)


def test_retained_nested_parent_fields_roundtrip(tmp_path: Path, nastran_mesh: Mesh) -> None:
    parent_data = nastran_mesh.global_data["nastran", "element_data"]
    parent_data["nested", "value"] = torch.tensor([[1.0, float("nan")]], dtype=torch.float64)
    nastran_mesh.cell_data["nested", "value"] = parent_data["nested", "value"][
        nastran_mesh.global_data["nastran", "parent_row"]
    ]
    path = MeshVTKSink(str(tmp_path / "out"))(iter([nastran_mesh]), 0)[0]
    loaded = read_vtk_data(path)
    torch.testing.assert_close(loaded["cell_data"]["nested", "value"], parent_data["nested", "value"], equal_nan=True)


def test_original_geometry_uses_current_coordinates(tmp_path: Path, nastran_mesh: Mesh) -> None:
    nastran_mesh.points = nastran_mesh.points * 2 + 3
    path = MeshVTKSink(str(tmp_path / "out"))(iter([nastran_mesh]), 0)[0]
    np.testing.assert_array_equal(pv.read(path).points, nastran_mesh.points.numpy())
