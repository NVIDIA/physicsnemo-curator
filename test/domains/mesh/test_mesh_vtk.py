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

from physicsnemo_curator.domains.mesh.sinks.mesh_vtk import MeshVTKSink, read_vtk_data

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


@pytest.mark.parametrize("template", ["{unknown}", "{}", "{index:invalid}", "{stem.missing}", "{"])
def test_naming_template_validated_eagerly(tmp_path: Path, template: str) -> None:
    with pytest.raises(ValueError, match="Invalid naming_template"):
        MeshVTKSink(str(tmp_path), naming_template=template)


def test_missing_source_naming_is_explicit(tmp_path: Path, modal_mesh: Mesh) -> None:
    sink = MeshVTKSink(str(tmp_path), naming_template="{relpath}/{stem}_{seq}")
    assert sink(iter([]), 0) == []
    with pytest.raises(ValueError, match="relative_path"):
        sink(iter([modal_mesh]), 0)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("template", ["../outside_{index}", "/outside_{index}"])
def test_output_names_remain_within_output_directory(tmp_path: Path, modal_mesh: Mesh, template: str) -> None:
    sink = MeshVTKSink(str(tmp_path / "out"), naming_template=template)
    with pytest.raises(ValueError, match="within output_dir"):
        sink(iter([modal_mesh]), 0)
    assert not (tmp_path / "out").exists()


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


def test_empty_writer_failure_keeps_existing_file(
    tmp_path: Path, modal_mesh: Mesh, monkeypatch: pytest.MonkeyPatch
) -> None:
    sink = MeshVTKSink(str(tmp_path))
    path = Path(sink(iter([modal_mesh]), 0)[0])
    original = path.read_bytes()
    monkeypatch.setattr(pv.UnstructuredGrid, "save", Mock())
    with pytest.raises(OSError, match="empty file"):
        sink(iter([modal_mesh]), 0)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


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


@pytest.mark.parametrize("file_format", ["vtu", "vtk", "vtp"])
@pytest.mark.parametrize("binary", [True, False])
def test_tensor_schema_roundtrip(tmp_path: Path, modal_mesh: Mesh, file_format: str, binary: bool) -> None:
    complex_values = torch.complex(torch.arange(20).reshape(5, 2, 2).float(), torch.ones(5, 2, 2))
    modal_mesh.point_data["nested", "a/~b"] = complex_values.conj()
    modal_mesh.point_data["half"] = torch.arange(5, dtype=torch.float16)
    modal_mesh.point_data["bfloat"] = torch.arange(5, dtype=torch.bfloat16)
    modal_mesh.point_data["empty"] = torch.empty(5, 0, dtype=torch.int32)
    generator = torch.Generator().manual_seed(17)
    modal_mesh.point_data["precision"] = torch.rand(5, 3, dtype=torch.float64, generator=generator)
    modal_mesh.global_data["nested", "scalar"] = torch.tensor(1 + 2j, dtype=torch.complex128).conj()
    modal_mesh.global_data["empty"] = torch.empty(0, 3, dtype=torch.bool)
    modal_mesh.global_data["half"] = torch.tensor(0.25, dtype=torch.float16)
    modal_mesh.global_data["precision"] = torch.rand(3, 3, dtype=torch.float64, generator=generator)
    path = MeshVTKSink(str(tmp_path), file_format=file_format, binary=binary)(iter([modal_mesh]), 0)[0]
    loaded = read_vtk_data(path)
    point_rows = loaded["point_data"]["node_id"] - 2**40
    for association in ("point_data", "cell_data", "global_data"):
        expected = getattr(modal_mesh, association)
        if association == "point_data":
            expected = expected[point_rows]
        assert set(loaded[association].keys(True, True)) == set(expected.keys(True, True))
        for key, value in expected.items(True, True):
            torch.testing.assert_close(loaded[association][key], value, rtol=0, atol=0)


def test_array_name_collisions_are_rejected(tmp_path: Path, modal_mesh: Mesh) -> None:
    modal_mesh.point_data["mode_shape_m0001"] = torch.ones(5, 3)
    with pytest.raises(ValueError, match="collision"):
        MeshVTKSink(str(tmp_path))(iter([modal_mesh]), 0)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("file_format", ["vtu", "vtk", "vtp"])
@pytest.mark.parametrize("binary", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("negative_infinity", [False, True])
def test_nonfinite_complex_components_roundtrip(
    tmp_path: Path,
    modal_mesh: Mesh,
    file_format: str,
    binary: bool,
    dtype: torch.dtype,
    negative_infinity: bool,
) -> None:
    real = torch.tensor([3.0, 4.0, float("inf"), -7.0, float("nan")], dtype=dtype)
    imaginary = torch.tensor([float("inf"), float("nan"), 2.0, -8.0, 0.0], dtype=dtype)
    if negative_infinity:
        real[3] = imaginary[3] = float("-inf")
    values = torch.complex(real, imaginary)
    modal_mesh.point_data["nonfinite"] = values
    modal_mesh.global_data["nonfinite"] = values
    sink = MeshVTKSink(str(tmp_path), file_format=file_format, binary=binary)
    if not binary and (file_format == "vtk" or negative_infinity):
        reason = "NaN or infinity" if file_format == "vtk" else "negative infinity"
        with pytest.raises(ValueError, match=rf"{reason}.*binary=True"):
            sink(iter([modal_mesh]), 0)
        assert not list(tmp_path.iterdir())
        return
    path = sink(iter([modal_mesh]), 0)[0]
    loaded = read_vtk_data(path)
    for association in ("point_data", "global_data"):
        rows = loaded[association]["node_id"] - 2**40 if association == "point_data" else torch.arange(5)
        restored = loaded[association]["nonfinite"]
        # Complex equal_nan=True alone can hide corruption in the other component.
        torch.testing.assert_close(restored.real, real[rows], rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(restored.imag, imaginary[rows], rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("association", ["point_data", "cell_data", "global_data", "points"])
@pytest.mark.parametrize(
    "file_format,value",
    [
        ("vtu", float("-inf")),
        ("vtp", float("-inf")),
        ("vtk", float("-inf")),
        ("vtk", float("inf")),
        ("vtk", float("nan")),
    ],
)
def test_ascii_rejects_unsupported_nonfinite_values_in_all_arrays(
    tmp_path: Path, modal_mesh: Mesh, association: str, file_format: str, value: float
) -> None:
    if association == "points":
        modal_mesh.points[0, 0] = value
    else:
        count = {"point_data": modal_mesh.n_points, "cell_data": modal_mesh.n_cells, "global_data": 1}[association]
        getattr(modal_mesh, association)["nonfinite"] = torch.full((count,), value)
    with pytest.raises(ValueError, match="infinity.*binary=True"):
        MeshVTKSink(str(tmp_path), file_format=file_format, binary=False)(iter([modal_mesh]), 0)
    assert not list(tmp_path.iterdir())


def test_schema_errors_are_explicit(tmp_path: Path, modal_mesh: Mesh) -> None:
    path = MeshVTKSink(str(tmp_path))(iter([modal_mesh]), 0)[0]
    grid = pv.read(path)
    grid.field_data["physicsnemo_schema"] = np.array([json.dumps({"version": 2})])
    grid.save(path)
    with pytest.raises(ValueError, match="schema version"):
        read_vtk_data(path)
    del grid.field_data["physicsnemo_schema"]
    grid.save(path)
    with pytest.raises(ValueError, match="no physicsnemo_schema"):
        read_vtk_data(path)


def test_configuration_properties(tmp_path: Path) -> None:
    sink = MeshVTKSink(str(tmp_path), file_format="vtk", use_original_geometry=False)
    assert sink.output_dir == tmp_path
    assert sink.file_format == "vtk"
    assert not sink.use_original_geometry
    assert {param.name for param in sink.params()} == {
        "output_dir",
        "file_format",
        "naming_template",
        "binary",
        "use_original_geometry",
    }


@pytest.mark.e2e
def test_registry_and_serialization(tmp_path: Path) -> None:
    from physicsnemo_curator import load_pipeline, registry, save_pipeline
    from physicsnemo_curator.domains.mesh.sources.random import RandomMeshSource

    assert MeshVTKSink in registry.list_sinks("mesh")
    pipeline = RandomMeshSource(n_samples=1, n_points=5, n_cells=2).write(MeshVTKSink(str(tmp_path), file_format="vtk"))
    config = tmp_path / "pipeline.json"
    save_pipeline(pipeline, str(config))
    restored = load_pipeline(str(config))
    assert pv.read(restored[0][0]).n_points == 5
