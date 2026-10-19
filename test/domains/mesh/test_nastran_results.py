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

"""Regression coverage for OP2 discovery and field association beyond modal tables."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from scipy.sparse import coo_matrix

pytest.importorskip("pyNastran")

from pyNastran.op2.op2 import OP2
from pyNastran.op2.result_objects.eqexin import EQEXIN
from pyNastran.op2.result_objects.gpdt import GPDT
from pyNastran.op2.result_objects.matrix import Matrix
from pyNastran.op2.result_objects.monpnt import MONPNT3
from pyNastran.op2.tables.oug.oug_displacements import RealDisplacementArray

from physicsnemo_curator.domains.mesh.sources._nastran_results import _discover_results, _project_result
from physicsnemo_curator.domains.mesh.sources.nastran import NastranSource, _read_deck, decode_nastran_json

from .test_nastran import _NIDS, _deck, _op2

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.requires("mesh"), pytest.mark.integration]


def _displacements(subcase: int = 1) -> RealDisplacementArray:
    """Create a real transient table with distinct values for every GRID/component."""
    return RealDisplacementArray.add_transient_case(
        "OUGV1",
        np.array([[nid, 1] for nid in _NIDS[::-1]], dtype=np.int32),
        np.arange(120, dtype=np.float32).reshape(2, 10, 6),
        subcase,
        np.array([0.0, 1.0], dtype=np.float32),
    )


def test_singleton_and_psd_tables_are_archived() -> None:
    op2 = OP2(debug=None)
    gpdt = GPDT(np.array([[10, 0, 0, 0]], dtype=np.int32), np.array([[1.25, 2.5, 5.0]]))
    eqexin = EQEXIN(np.array([10]), np.array([6]), np.array([1]))
    psd = np.array([[1.0, 2.5], [3.0, 4.0]], dtype=np.float64)
    op2.op2_results.gpdt = gpdt
    op2.op2_results.eqexin = eqexin
    op2.op2_results.psds.displacements[("test", 1, 10, 1)] = psd
    records, catalog = _discover_results(op2, "*")
    by_name = {record["name"]: record for record in records}
    assert set(by_name) == {"gpdt", "eqexin", "psds.displacements"}
    torch.testing.assert_close(by_name["gpdt"]["archive"]["arrays", "xyz"], torch.from_numpy(gpdt.xyz))
    torch.testing.assert_close(by_name["eqexin"]["archive"]["arrays", "nid"], torch.from_numpy(eqexin.nid))
    torch.testing.assert_close(by_name["psds.displacements"]["archive"]["arrays", "data"], torch.from_numpy(psd))
    assert all(entry["status"] == "raw_table_preserved" for entry in catalog)
    filtered, catalog = _discover_results(op2, "psds.*")
    assert [record["name"] for record in filtered] == ["psds.displacements"]
    assert {entry["table"] for entry in catalog if entry["status"] == "excluded_by_user"} == {"gpdt", "eqexin"}


def test_empty_backend_does_not_invent_results() -> None:
    # An empty OP2 still contains initialized CSTM/PSD container objects.
    assert _discover_results(OP2(debug=None), "*") == ([], [])


def test_sparse_monitor_table_preserves_values_and_coordinates() -> None:
    matrix = Matrix(
        "MONITOR",
        2,
        coo_matrix((np.array([1 + 2j, 3 - 4j]), ([1, 5], [0, 1])), shape=(6, 2)),
    )
    op2 = OP2(debug=None)
    op2.op2_results.monitor3 = MONPNT3(np.array([1.5, 3.0]), matrix)
    records, _ = _discover_results(op2, "monitor*")
    assert len(records) == 1
    archive = records[0]["archive"]
    torch.testing.assert_close(archive["arrays", "frequencies"], torch.tensor([1.5, 3.0], dtype=torch.float64))
    sparse = archive["sparse_arrays", "data"]
    torch.testing.assert_close(sparse["row"], torch.from_numpy(matrix.data.row))
    torch.testing.assert_close(sparse["column"], torch.from_numpy(matrix.data.col))
    torch.testing.assert_close(sparse["data"], torch.from_numpy(matrix.data.data))
    assert decode_nastran_json(archive["metadata"])["sparse_arrays"]["data"]["shape"] == [6, 2]


@pytest.mark.parametrize("file_format", ["pmsh", "vtu"])
def test_singleton_archives_survive_serialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, file_format: str
) -> None:
    from physicsnemo.mesh import Mesh

    from physicsnemo_curator.domains.mesh.sinks.mesh_vtk import MeshVTKSink, read_vtk_data

    path = tmp_path / "model.bdf"
    _deck(path)
    _op2(path.with_suffix(".op2"))
    op2 = OP2(debug=None)
    op2.op2_results.gpdt = GPDT(np.array([[10, 0, 0, 0]], dtype=np.int32), np.array([[1.0, 2.0, 3.0]]))
    matrix = Matrix("MONITOR", 2, coo_matrix(([1 + 2j], ([1], [0])), shape=(6, 1)))
    op2.op2_results.monitor3 = MONPNT3(np.array([1.5]), matrix)
    monkeypatch.setattr("pyNastran.op2.op2.read_op2", lambda *args, **kwargs: op2)
    mesh = next(NastranSource(str(path))[0])
    if file_format == "pmsh":
        mesh.save(str(tmp_path / "saved.pmsh"))
        loaded = Mesh.load(str(tmp_path / "saved.pmsh")).global_data
    else:
        exported = MeshVTKSink(str(tmp_path / "out"))(iter([mesh]), 0)[0]
        loaded = read_vtk_data(exported)["global_data"]
    expected = mesh.global_data["nastran", "results"]
    actual = loaded["nastran", "results"]
    assert set(actual.keys(True, True)) == set(expected.keys(True, True))
    for key, value in expected.items(True, True):
        torch.testing.assert_close(actual[key], value, rtol=0, atol=0, equal_nan=True)


def test_dense_matrix_normalizes_byteorder() -> None:
    matrix = Matrix("KAA", 1, np.array([[1.25, 2.5], [2.5, 5.0]], dtype=">f8"))
    op2 = OP2(debug=None)
    op2.matrices["KAA"] = matrix
    records, _ = _discover_results(op2, "matrices.*")
    torch.testing.assert_close(
        records[0]["archive"]["arrays", "data"], torch.tensor([[1.25, 2.5], [2.5, 5.0]], dtype=torch.float64)
    )


def test_nonmodal_subcases_keep_constraint_selections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "model.bdf"
    _deck(path)
    _op2(path.with_suffix(".op2"))
    op2 = OP2(debug=None)
    op2.displacements[1] = _displacements(1)
    op2.displacements[2] = _displacements(2)
    monkeypatch.setattr("pyNastran.op2.op2.read_op2", lambda *args, **kwargs: op2)
    first, second = list(NastranSource(str(path))[0])
    assert first.global_data["spc_id"].item() == 70
    assert first.global_data["mpc_id"].item() == 90
    assert second.global_data["spc_id"].item() == 8
    assert second.global_data["mpc_id"].item() == 0
    assert "mode_shape" not in first.point_data


def test_projection_provenance_uses_original_ids(tmp_path: Path) -> None:
    path = tmp_path / "model.bdf"
    _deck(path)
    mesh = next(NastranSource(str(path), read_results=False)[0])
    torch.testing.assert_close(
        mesh.global_data["nastran", "projection_node_ids"], mesh.point_data["node_id"][mesh.cells]
    )
    torch.testing.assert_close(mesh.global_data["nastran", "projection_element_ids"], mesh.cell_data["element_id"])


def test_custom_names_cannot_overwrite_ids_or_masks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "model.bdf"
    _deck(path)
    _op2(path.with_suffix(".op2"))
    table = _displacements()
    op2 = OP2(debug=None)
    tables = {name: {1: table} for name in ("node_id", "custom", "custom__valid")}
    monkeypatch.setattr(op2, "get_table_types", lambda: list(tables))
    monkeypatch.setattr(op2, "get_result", tables.__getitem__)
    monkeypatch.setattr("pyNastran.op2.op2.read_op2", lambda *args, **kwargs: op2)
    mesh = next(NastranSource(str(path))[0])
    torch.testing.assert_close(mesh.point_data["node_id"], torch.tensor(_NIDS))
    report = decode_nastran_json(mesh.global_data["nastran", "projection"])
    names = [field for result in report["results"] for field in result["fields"]]
    assert len(names) == len(set(names)) == 6
    assert "node_id" not in names


def test_sort2_tables_remain_raw() -> None:
    table = _displacements()
    # SORT2 cannot use a SORT1-shaped projection just because its IDs happen
    # to match the deck. The original array remains available in the archive.
    table.sort_method = 2
    location, fields, reason = _project_result({"name": "displacements", "result": table}, None, _NIDS, [33])
    assert location is None and fields == {}
    assert "SORT2" in reason


def test_superelement_local_ids_are_not_assumed_global(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "model.bdf"
    _deck(path)
    _op2(path.with_suffix(".op2"))
    op2 = OP2(debug=None)
    result = _displacements()
    result.superelement_adaptivity_index = "SUPERELEMENT 12"
    op2.displacements[1] = result
    monkeypatch.setattr("pyNastran.op2.op2.read_op2", lambda *args, **kwargs: op2)
    mesh = next(NastranSource(str(path))[0])
    assert "displacements" not in mesh.point_data
    archive = next(iter(mesh.global_data["nastran", "results"].values()))
    torch.testing.assert_close(archive["arrays", "data"], torch.from_numpy(result.data))
    report = decode_nastran_json(mesh.global_data["nastran", "projection"])
    assert "superelement/adaptive IDs" in report["results"][0]["description"]


def test_directory_discovery_can_exclude_include_decks(tmp_path: Path) -> None:
    included = tmp_path / "part.dat"
    _deck(included)
    bulk = included.read_text().split("BEGIN BULK", 1)[1]
    included.write_text(bulk)
    main = tmp_path / "Input_model.bdf"
    main.write_text("SOL 101\nCEND\nBEGIN BULK\nINCLUDE 'part.dat'\n")
    assert len(NastranSource(str(tmp_path), read_results=False)) == 2
    source = NastranSource(str(tmp_path), file_pattern="Input_*.bdf", read_results=False)
    assert len(source) == 1
    assert next(source[0]).n_points == 10


def test_missing_backend_has_installation_guidance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pyNastran.bdf.bdf", None)
    with pytest.raises(ImportError, match=r"physicsnemo-curator\[mesh,nastran\].*Python 3.12"):
        _read_deck(tmp_path / "model.bdf", "latin1", False)
