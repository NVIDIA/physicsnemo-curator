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

"""Synthetic BDF/OP2 round trips; no solver, customer files, or network needed."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

pytest.importorskip("pyNastran")

from pyNastran.bdf.bdf import BDF, CaseControlDeck
from pyNastran.op2.op2 import OP2
from pyNastran.op2.tables.oug.oug_eigenvectors import RealEigenvectorArray

from physicsnemo_curator.domains.mesh.sources.nastran import NastranSource

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.requires("mesh"), pytest.mark.integration]

_POINTS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0.5, 0, 0],
        [0.5, 0.5, 0],
        [0, 0.5, 0],
        [0, 0, 0.5],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5],
    ],
    dtype=np.float64,
)
_NIDS = list(range(10, 110, 10))


def _deck(path: Path, *, quadratic: bool = True, coordinate: str = "", encoding: str = "utf-8") -> BDF:
    model = BDF(debug=None)
    model.sol = 103
    model.case_control_deck = CaseControlDeck(["SUBCASE 1", "SPC = 70", "MPC = 90", "SUBCASE 2", "SPC = 8"])
    if coordinate:
        add_coord = getattr(model, "add_cord2" + coordinate)
        add_coord(5, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
    for nid, xyz in reversed(list(zip(_NIDS, _POINTS, strict=True))):
        model.add_grid(nid, xyz, cd=5 if coordinate else 0, ps="1" if nid == 10 else "")
    model.add_mat1(11, 210000.0, None, 0.3, rho=7.85e-9)
    model.add_psolid(22, 11)
    model.add_ctetra(33, 22, _NIDS if quadratic else _NIDS[:4])
    model.add_spc1(7, "123", [10, 20])
    model.add_spc(8, [30], ["2"], [0.25])
    model.add_spcadd(70, [7])
    model.add_mpc(9, [30, 40], ["3", "3"], [1.0, -1.0])
    model.add_mpcadd(90, [9])
    model.add_rbe2(44, 10, "123456", [20, 30], alpha=1.2e-5)
    model.write_bdf(str(path), encoding=encoding)
    return model


def _op2(path: Path, *, node_ids: list[int] | None = None, subcases: tuple[int, ...] = (1,)) -> None:
    ids = node_ids if node_ids is not None else _NIDS[::-1]
    # Deliberately encode values by node ID; reordering errors are observable.
    data = np.array([[[nid + mode * 100 + c for c in range(6)] for nid in ids] for mode in (1, 3)], dtype=np.float32)
    model = OP2(debug=None, mode="msc")
    for sid in subcases:
        table = RealEigenvectorArray.add_modal_case(
            "OUGV1",
            np.array([[nid, 1] for nid in ids], dtype=np.int32),
            data.copy(),
            sid,
            np.array([1, 3], dtype=np.int32),
            np.array([4 * np.pi**2, 36 * np.pi**2], dtype=np.float32),
            np.array([1.0, 3.0], dtype=np.float32),
        )
        # The generic OUG factory defaults to displacements (table code 1).
        table.table_code = 7
        table.tCode = 7000
        model.eigenvectors[sid] = table
    model.write_op2(str(path), nastran_format="msc")


@pytest.fixture
def sample(tmp_path: Path) -> Path:
    _deck(tmp_path / "Input_model.deck")
    _op2(tmp_path / "Output_model.op2")
    return tmp_path


def test_tet10_subdivision_and_id_alignment(sample: Path) -> None:
    mesh = next(NastranSource(str(sample))[0])
    assert mesh.points.shape == (10, 3)
    assert mesh.cells.shape == (8, 4)
    assert set(mesh.cells.flatten().tolist()) == set(range(10))
    torch.testing.assert_close(mesh.points, torch.from_numpy(_POINTS))
    # Eight positively oriented tets exactly fill the parent straight tetra.
    xyz = mesh.points[mesh.cells]
    volumes = torch.linalg.det(xyz[:, 1:] - xyz[:, :1]) / 6
    assert torch.all(volumes > 0)
    assert volumes.sum().item() == pytest.approx(1 / 6)
    torch.testing.assert_close(mesh.point_data["node_id"], torch.tensor(_NIDS))
    assert mesh.point_data["mode_shape"].shape == (10, 2, 3)
    torch.testing.assert_close(mesh.point_data["mode_shape"][:, 0, 0], torch.tensor(_NIDS, dtype=torch.float32) + 100)
    torch.testing.assert_close(
        mesh.point_data["mode_rotation"][:, 1, 2], torch.tensor(_NIDS, dtype=torch.float32) + 305
    )
    assert mesh.global_data["mode_numbers"].tolist() == [1, 3]
    torch.testing.assert_close(mesh.global_data["frequencies"], torch.tensor([1.0, 3.0], dtype=torch.float64))
    assert mesh.cell_data["element_id"].tolist() == [33] * 8
    assert mesh.cell_data["property_id"].tolist() == [22] * 8
    assert mesh.cell_data["material_id"].tolist() == [11] * 8
    assert mesh.cell_data["young_modulus"].tolist() == [210000.0] * 8


def test_constraints_and_subcase_selection(sample: Path) -> None:
    mesh = next(NastranSource(str(sample))[0])
    data = mesh.global_data["constraints"]
    assert data["spc"].tolist() == [[7, 10, 1], [7, 10, 2], [7, 10, 3], [7, 20, 1], [7, 20, 2], [7, 20, 3], [8, 30, 2]]
    assert data["spc_value"][-1].item() == 0.25
    assert data["spcadds"].tolist() == [[70, 7]]
    assert data["mpcadds"].tolist() == [[90, 9]]
    assert data["mpc_set_id"].tolist() == [9]
    assert data["mpc_offsets"].tolist() == [0, 2]
    assert data["mpc_terms"].tolist() == [[30, 3], [40, 3]]
    assert data["mpc_coefficient"].tolist() == [1.0, -1.0]
    assert data["rbe2"].tolist() == [[44, 10, 20, 123456], [44, 10, 30, 123456]]
    assert data["rbe2_alpha"][0].item() == pytest.approx(1.2e-5)
    assert mesh.global_data["spc_id"].item() == 70
    assert mesh.global_data["mpc_id"].item() == 90
    assert mesh.point_data["permanent_spc"][0].tolist() == [True, False, False, False, False, False]


def test_multiple_materials_and_linear_tet(tmp_path: Path) -> None:
    path = tmp_path / "solid.bdf"
    model = _deck(path, quadratic=False)
    model.add_mat1(12, 70000.0, None, 0.33, rho=2.7e-9)
    model.add_psolid(23, 12)
    model.add_ctetra(34, 23, _NIDS)
    model.write_bdf(str(path))
    mesh = next(NastranSource(str(path), read_results=False)[0])
    assert mesh.cells.shape == (9, 4)
    assert mesh.cell_data["material_id"].tolist() == [11] + [12] * 8
    assert mesh.cell_data["density"][1].item() == pytest.approx(2.7e-9)


def test_rectangular_output_coordinates(tmp_path: Path) -> None:
    _deck(tmp_path / "model.bdf", coordinate="r")
    _op2(tmp_path / "model.op2")
    mesh = next(NastranSource(str(tmp_path))[0])
    # Local x is basic +y, local y is basic -x, z is unchanged.
    torch.testing.assert_close(mesh.point_data["mode_shape"][0, 0], torch.tensor([-111.0, 110.0, 112.0]))
    torch.testing.assert_close(mesh.point_data["mode_rotation"][0, 0], torch.tensor([-114.0, 113.0, 115.0]))
    torch.testing.assert_close(
        mesh.point_data["output_coordinate_basis"][0],
        torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64),
    )


@pytest.mark.parametrize("coordinate", ["c", "s"])
def test_unsupported_output_coordinates_raise(tmp_path: Path, coordinate: str) -> None:
    _deck(tmp_path / "model.bdf", coordinate=coordinate)
    _op2(tmp_path / "model.op2")
    with pytest.raises(NotImplementedError, match="rectangular CD"):
        next(NastranSource(str(tmp_path))[0])


@pytest.mark.parametrize("ids", [_NIDS[:-1], _NIDS + [110], _NIDS[:-1] + [10]])
def test_mismatched_or_duplicate_result_ids_raise(sample: Path, ids: list[int]) -> None:
    _op2(sample / "Output_model.op2", node_ids=ids)
    with pytest.raises(ValueError, match="GRID IDs must match"):
        next(NastranSource(str(sample))[0])


def test_subcases_yield_independent_meshes(sample: Path) -> None:
    _op2(sample / "Output_model.op2", subcases=(1, 2))
    first, second = list(NastranSource(str(sample))[0])
    assert [m.global_data["subcase_id"].item() for m in (first, second)] == [1, 2]
    assert second.global_data["spc_id"].item() == 8
    first.points.fill_(999)
    first.point_data["mode_shape"].fill_(0)
    assert second.points.max() == 1
    assert second.point_data["mode_shape"].min() > 0


def test_op2_pairing_is_explicit_on_ambiguity(sample: Path) -> None:
    _op2(sample / "another.op2")
    with pytest.raises(ValueError, match="Ambiguous OP2"):
        next(NastranSource(str(sample))[0])
    assert next(NastranSource(str(sample), op2_pattern="Output_model.op2")[0]).n_points == 10
    (sample / "Output_model.op2").rename(sample / "Input_model.OP2")
    assert next(NastranSource(str(sample))[0]).n_points == 10


def test_template_pairing(sample: Path) -> None:
    (sample / "Output_model.op2").rename(sample / "results_Input_model.op2")
    assert next(NastranSource(str(sample), op2_pattern="results_{stem}.op2")[0]).n_points == 10


def test_missing_op2_and_geometry_only(sample: Path) -> None:
    (sample / "Output_model.op2").unlink()
    with pytest.raises(FileNotFoundError, match="No OP2"):
        next(NastranSource(str(sample))[0])
    with pytest.raises(FileNotFoundError):
        next(NastranSource(str(sample), op2_pattern="missing.op2")[0])
    assert "mode_shape" not in next(NastranSource(str(sample), read_results=False)[0]).point_data


def test_legacy_encoding_with_relative_include(tmp_path: Path) -> None:
    part = tmp_path / "part.dat"
    _deck(part, encoding="cp932")
    lines = part.read_text(encoding="cp932").split("BEGIN BULK", 1)[1]
    part.write_text("$ 日本語\n" + lines, encoding="cp932")
    deck = tmp_path / "model.nas"
    deck.write_text("$ 日本語\nSOL 103\nCEND\nBEGIN BULK\nINCLUDE 'part.dat'\n", encoding="cp932")
    source = NastranSource(str(deck), encoding="cp932", read_results=False)
    assert next(source[0]).n_points == 10


def test_bulk_only_punch_file(tmp_path: Path) -> None:
    path = tmp_path / "model.bdf"
    _deck(path)
    bulk = path.read_text().split("BEGIN BULK", 1)[1]
    path.write_text(bulk)
    assert next(NastranSource(str(path), punch=True, read_results=False)[0]).n_cells == 8
    path.rename(tmp_path / "model.pch")
    assert next(NastranSource(str(tmp_path), read_results=False)[0]).n_cells == 8


def test_input_coordinate_transform(tmp_path: Path) -> None:
    path = tmp_path / "model.bdf"
    model = _deck(path)
    model.add_cord2r(6, [2.0, 3.0, 4.0], [2.0, 3.0, 5.0], [3.0, 3.0, 4.0])
    for node in model.nodes.values():
        node.cp = 6
    model.write_bdf(str(path))
    mesh = next(NastranSource(str(path), read_results=False)[0])
    torch.testing.assert_close(mesh.points, torch.from_numpy(_POINTS + [2.0, 3.0, 4.0]))


def test_signed_eigenvalues_and_solver_cycles(sample: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pyNastran.op2.op2 import read_op2

    result = read_op2(str(sample / "Output_model.op2"), debug=None)
    table = result.eigenvectors[1]
    table.eigns[0] = -1.0e-6
    table.mode_cycles[0] = 0.00015915494
    monkeypatch.setattr("pyNastran.op2.op2.read_op2", lambda *args, **kwargs: result)
    mesh = next(NastranSource(str(sample))[0])
    assert mesh.global_data["eigenvalues"][0].item() == -1.0e-6
    assert mesh.global_data["frequencies"][0].item() == pytest.approx(0.00015915494)


def test_nonmodal_op2_raises(sample: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = OP2(debug=None, mode="msc")
    monkeypatch.setattr("pyNastran.op2.op2.read_op2", lambda *args, **kwargs: model)
    with pytest.raises(ValueError, match="No modal eigenvectors"):
        next(NastranSource(str(sample))[0])


def test_degenerate_geometry_raises(tmp_path: Path) -> None:
    path = tmp_path / "model.bdf"
    model = _deck(path, quadratic=False)
    model.nodes[40].xyz = np.array([0.0, 0.0, 0.0])
    model.write_bdf(str(path))
    with pytest.raises(ValueError, match="degenerate"):
        next(NastranSource(str(path), read_results=False)[0])


def test_unsupported_support_constraint_raises(tmp_path: Path) -> None:
    path = tmp_path / "model.bdf"
    model = _deck(path)
    model.add_suport([10], ["123"])
    model.write_bdf(str(path))
    with pytest.raises(NotImplementedError, match="suport"):
        next(NastranSource(str(path), read_results=False)[0])


@pytest.mark.parametrize("card", ["element", "rigid"])
def test_unsupported_cards_raise(tmp_path: Path, card: str) -> None:
    path = tmp_path / "model.bdf"
    model = _deck(path)
    if card == "element":
        model.add_ctria3(100, 22, _NIDS[:3])
        # Use a valid shell property so cross referencing succeeds.
        model.add_pshell(24, mid1=11, t=0.1)
        model.elements[100].pid = 24
    else:
        model.rigid_elements.clear()
        model.add_rbe3(100, 10, "123", [1.0], ["123"], [[20, 30]])
    model.write_bdf(str(path))
    with pytest.raises(NotImplementedError):
        next(NastranSource(str(path), read_results=False)[0])


def test_discovery_indices_and_registry(sample: Path) -> None:
    from physicsnemo_curator.core.registry import registry

    source = NastranSource(str(sample))
    assert len(source) == 1
    assert source.relative_path(-1) == "Input_model.deck"
    assert source.root == sample
    assert next(source[-1]).n_points == 10
    for index in (1, -2):
        with pytest.raises(IndexError):
            next(source[index])
    assert NastranSource in registry.list_sources("mesh")
    assert {p.name for p in source.params()} == {
        "input_path",
        "file_pattern",
        "op2_pattern",
        "encoding",
        "read_results",
        "punch",
    }


def test_empty_missing_and_filtered_paths(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        NastranSource(str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="No Nastran decks"):
        NastranSource(str(tmp_path))
    child = tmp_path / "run"
    child.mkdir()
    _deck(child / "model.BDF")
    assert len(NastranSource(str(tmp_path), file_pattern="run/*")) == 1


@pytest.mark.e2e
def test_pipeline_serialization_and_memmap(sample: Path, tmp_path: Path) -> None:
    from physicsnemo.mesh import Mesh

    from physicsnemo_curator import load_pipeline, save_pipeline
    from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink

    _op2(sample / "Output_model.op2", subcases=(1, 2))
    source = NastranSource(str(sample))
    sink = MeshSink(output_dir=str(tmp_path / "out"), naming_template="{stem}_{seq}")
    pipeline = source.write(sink)
    config = tmp_path / "pipeline.json"
    save_pipeline(pipeline, str(config))
    restored = load_pipeline(str(config))
    paths = restored[0]
    assert len(paths) == 2
    mesh = Mesh.load(paths[0])
    assert mesh.point_data["mode_shape"].shape == (10, 2, 3)
    assert mesh.global_data["constraints", "mpc_offsets"].tolist() == [0, 2]
    assert mesh.global_data["constraints", "rbe2"].shape == (2, 4)
