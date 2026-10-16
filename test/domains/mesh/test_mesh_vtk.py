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

"""Verify volume/surface VTK exports and metadata by reading the actual files."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import pyvista as pv
import torch
from physicsnemo.mesh import Mesh

from physicsnemo_curator.domains.mesh.sinks.mesh_vtk import MeshVTKSink

pytestmark = [pytest.mark.requires("mesh"), pytest.mark.integration]


@pytest.fixture
def modal_mesh() -> Mesh:
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.25, 0.25, 0.25]], dtype=torch.float64
    )
    cells = torch.tensor([[0, 1, 2, 4], [0, 1, 4, 3], [0, 4, 2, 3], [4, 1, 2, 3]])
    return Mesh(
        points=points,
        cells=cells,
        point_data={
            "node_id": torch.arange(5) + 2**40,
            "mode_shape": torch.arange(30, dtype=torch.float64).reshape(5, 2, 3) / 17,
            "mode_rotation": torch.arange(30, dtype=torch.float32).reshape(5, 2, 3),
            "basis": torch.eye(3, dtype=torch.float64).repeat(5, 1, 1),
            "fixed": torch.tensor([True, False, False, False, False]),
        },
        cell_data={"element_id": torch.arange(4) + 2**45, "density": torch.full((4,), 7.85e-9, dtype=torch.float64)},
        global_data={
            "mode_numbers": torch.tensor([1, 7]),
            "frequencies": torch.tensor([0.0001, 123.456789], dtype=torch.float64),
            "subcase_id": torch.tensor(2),
            "constraints": {
                "mpc_terms": torch.tensor([[2**40, 1], [2**40 + 1, 1]]),
                "mpc_coefficient": torch.tensor([1.0, -1.0]),
                "spc": torch.empty(0, 3, dtype=torch.int64),
            },
        },
    )


@pytest.mark.parametrize("file_format", ["vtu", "vtk", "vtp"])
@pytest.mark.parametrize("binary", [True, False])
def test_export_roundtrip(tmp_path: Path, modal_mesh: Mesh, file_format: str, binary: bool) -> None:
    sink = MeshVTKSink(str(tmp_path), file_format=file_format, binary=binary)
    path = sink(iter([modal_mesh]), 4)[0]
    grid = pv.read(path)
    assert Path(path).name == f"mesh_0004_0.{file_format}"
    assert grid.n_cells == 4
    assert grid.n_points == (4 if file_format == "vtp" else 5)
    assert isinstance(grid, pv.PolyData if file_format == "vtp" else pv.UnstructuredGrid)
    assert "mode_shape_m0001" in grid.point_data and "mode_shape_m0007" in grid.point_data
    assert "mode_rotation_m0007" in grid.point_data
    assert "mode_shape" not in grid.point_data
    assert grid.point_data["node_id"].dtype == np.int64
    assert grid.cell_data["element_id"].dtype == np.int64
    assert grid.points.dtype == np.float64
    assert grid.point_data["mode_shape_m0001"].dtype == np.float64
    ids = grid.point_data["node_id"] - 2**40
    np.testing.assert_allclose(
        grid.point_data["mode_shape_m0007"], modal_mesh.point_data["mode_shape"][ids, 1].numpy(), rtol=1e-9
    )
    np.testing.assert_array_equal(grid.cell_data["element_id"], modal_mesh.cell_data["element_id"].numpy())
    np.testing.assert_array_equal(
        grid.field_data["constraints/mpc_terms"], modal_mesh.global_data["constraints", "mpc_terms"].numpy()
    )
    np.testing.assert_array_equal(grid.field_data["frequencies"], modal_mesh.global_data["frequencies"].numpy())
    schema = json.loads(grid.field_data["physicsnemo_schema"][0])
    assert schema["global_data"]["constraints/spc"]["shape"] == [0, 3]
    assert schema["global_data"]["constraints/spc"]["arrays"] == []
    assert schema["point_data"]["mode_shape"]["shape"] == [grid.n_points, 2, 3]
    assert schema["surface_only"] == (file_format == "vtp")
    if file_format == "vtp":
        assert 2**40 + 4 not in grid.point_data["node_id"]
        assert grid.is_all_triangles
    else:
        np.testing.assert_array_equal(grid.points, modal_mesh.points.numpy())
        np.testing.assert_array_equal(grid.celltypes, np.full(4, 10))


def test_source_naming_and_multiple_outputs(tmp_path: Path, modal_mesh: Mesh) -> None:
    source = Mock()
    source.relative_path.return_value = "case_a/model.deck"
    sink = MeshVTKSink(str(tmp_path), naming_template="{relpath}/{stem}_{seq}")
    sink.set_source(source)
    paths = sink(iter([modal_mesh, modal_mesh]), 0)
    assert [Path(p).relative_to(tmp_path).as_posix() for p in paths] == ["case_a/model_0.vtu", "case_a/model_1.vtu"]
    assert sink(iter([]), 0) == []


def test_duplicate_paths_raise(tmp_path: Path, modal_mesh: Mesh) -> None:
    sink = MeshVTKSink(str(tmp_path), naming_template="one")
    with pytest.raises(ValueError, match="duplicate"):
        sink(iter([modal_mesh, modal_mesh]), 0)


def test_atomic_failure_keeps_existing_file(tmp_path: Path, modal_mesh: Mesh, monkeypatch: pytest.MonkeyPatch) -> None:
    sink = MeshVTKSink(str(tmp_path))
    path = sink(iter([modal_mesh]), 0)[0]
    original = Path(path).read_bytes()
    monkeypatch.setattr(pv.UnstructuredGrid, "save", Mock(side_effect=RuntimeError("write failed")))
    with pytest.raises(RuntimeError, match="write failed"):
        sink(iter([modal_mesh]), 0)
    assert Path(path).read_bytes() == original
    assert list(tmp_path.iterdir()) == [Path(path)]


def test_invalid_format_and_modal_shape(tmp_path: Path, modal_mesh: Mesh) -> None:
    with pytest.raises(ValueError, match="file_format"):
        MeshVTKSink(str(tmp_path), file_format="stl")
    modal_mesh.global_data["mode_numbers"] = torch.tensor([1])
    with pytest.raises(ValueError, match="must have shape"):
        MeshVTKSink(str(tmp_path))(iter([modal_mesh]), 0)


def test_no_modal_fields_or_synthetic_arrays(tmp_path: Path) -> None:
    mesh = Mesh(points=torch.zeros(3, 3), cells=torch.tensor([[0, 1, 2]]))
    grid = pv.read(MeshVTKSink(str(tmp_path))(iter([mesh]), 0)[0])
    assert "thickness" not in grid.point_data
    assert grid.n_cells == 1


def test_registry_and_serialization(tmp_path: Path) -> None:
    from physicsnemo_curator import load_pipeline, registry, save_pipeline
    from physicsnemo_curator.domains.mesh.sources.random import RandomMeshSource

    assert MeshVTKSink in registry.list_sinks("mesh")
    pipeline = RandomMeshSource(n_samples=1, n_points=5, n_cells=2).write(MeshVTKSink(str(tmp_path), file_format="vtk"))
    config = tmp_path / "pipeline.json"
    save_pipeline(pipeline, str(config))
    restored = load_pipeline(str(config))
    assert pv.read(restored[0][0]).n_points == 5
