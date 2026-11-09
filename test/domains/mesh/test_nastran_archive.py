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

"""Verify generalized Nastran fields and exact recovery with synthetic inputs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import pyvista as pv
import torch
from physicsnemo.mesh import Mesh
from scipy.sparse import coo_matrix

pytest.importorskip("pyNastran")

from pyNastran.op2.op2 import OP2
from pyNastran.op2.tables.oug.oug_displacements import ComplexDisplacementArray, RealDisplacementArray

from physicsnemo_curator.domains.mesh.sinks.mesh_vtk import MeshVTKSink, read_vtk_data
from physicsnemo_curator.domains.mesh.sources._nastran_metadata import _encode_json
from physicsnemo_curator.domains.mesh.sources._nastran_results import _array_tensor, _discover_results
from physicsnemo_curator.domains.mesh.sources.nastran import (
    NastranSource,
    decode_nastran_json,
    restore_nastran_sources,
)

from .test_nastran import _NIDS, _deck, _op2

pytestmark = [pytest.mark.requires("mesh"), pytest.mark.integration]


def _tables(monkeypatch: pytest.MonkeyPatch, tables: dict, matrices: dict | None = None) -> None:
    backend = SimpleNamespace(
        get_table_types=lambda: list(tables),
        get_result=tables.__getitem__,
        matrices=matrices or {},
    )
    monkeypatch.setattr("pyNastran.op2.op2.read_op2", Mock(return_value=backend))


def _assert_archive_equal(actual, expected) -> None:
    assert set(actual.keys(True, True)) == set(expected.keys(True, True))
    for key, value in expected.items(True, True):
        torch.testing.assert_close(actual[key], value, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("file_format", ["pmsh", "vtu", "vtk", "vtp"])
def test_original_files_and_all_global_fields_roundtrip(tmp_path: Path, file_format: str) -> None:
    deck = tmp_path / "input.bdf"
    model = _deck(deck, encoding="latin1")
    model.materials[11].a = 2.28e-5
    model.materials[11].tref = 293.15
    model.add_param("WTMASS", [1.234])
    model.write_bdf(str(deck), encoding="latin1")
    # Includes must be recovered too, including their original comment bytes.
    included = tmp_path / "parts" / "extra.inc"
    included.parent.mkdir()
    included.write_bytes(b"$ non-UTF8 comment \x82\xa0\nPARAM,POST,-1\n")
    deck.write_bytes(deck.read_bytes() + b"INCLUDE 'parts/extra.inc'\n")
    op2 = tmp_path / "input.op2"
    _op2(op2)
    originals = {p.relative_to(tmp_path).as_posix(): p.read_bytes() for p in (deck, included, op2)}
    mesh = next(NastranSource(str(deck))[0])
    metadata = decode_nastran_json(mesh.global_data["nastran", "deck_metadata"])
    mat = metadata["cards"]["materials"][0]["fields"]
    assert 2.28e-5 in mat and 293.15 in mat
    assert "params" in metadata["cards"]
    assert mesh.n_cells == 8
    assert mesh.cell_data["material_a"][0] == 2.28e-5
    if file_format == "pmsh":
        mesh.save(str(tmp_path / "saved.pmsh"))
        loaded = Mesh.load(str(tmp_path / "saved.pmsh"))
        global_data = loaded.global_data
    else:
        path = MeshVTKSink(str(tmp_path / "vtk"), file_format=file_format)(iter([mesh]), 0)[0]
        grid = pv.read(path)
        if file_format != "vtp":
            assert grid.n_cells == 1
            assert grid.celltypes.tolist() == [24]
            assert grid.cell_connectivity.tolist() == list(range(10))
        global_data = read_vtk_data(path)["global_data"]
    _assert_archive_equal(global_data, mesh.global_data)
    restored = tmp_path / "restored"
    paths = restore_nastran_sources(global_data, str(restored))
    assert {Path(p).relative_to(restored).as_posix(): Path(p).read_bytes() for p in paths} == originals
    # The recovered files still form a readable deck and paired OP2.
    assert next(NastranSource(str(restored / "input.bdf"))[0]).n_cells == 8
    with pytest.raises(FileExistsError):
        restore_nastran_sources(global_data, str(restored))


@pytest.mark.parametrize("kind", ["transient", "complex"])
def test_actual_nonmodal_op2_results(tmp_path: Path, kind: str) -> None:
    _deck(tmp_path / "model.bdf")
    ids = _NIDS[::-1]
    node_gridtype = np.array([[nid, 1] for nid in ids], dtype=np.int32)
    data = np.arange(120, dtype=np.float32).reshape(2, 10, 6)
    times = np.array([0.125, 0.5], dtype=np.float32)
    if kind == "complex":
        data = (data + 1j * (data + 2)).astype(np.complex64)
        result = ComplexDisplacementArray.add_freq_case("OUGV1", node_gridtype, data, 1, times)
    else:
        result = RealDisplacementArray.add_transient_case("OUGV1", node_gridtype, data, 1, times)
    op2 = OP2(debug=None, mode="msc")
    op2.displacements[1] = result
    op2.write_op2(str(tmp_path / "model.op2"), nastran_format="msc")
    mesh = next(NastranSource(str(tmp_path / "model.bdf"))[0])
    expected = torch.from_numpy(data[:, ::-1].transpose(1, 0, 2).copy())
    torch.testing.assert_close(mesh.point_data["displacements"], expected, rtol=0, atol=0)
    assert "mode_shape" not in mesh.point_data
    archive = next(iter(mesh.global_data["nastran", "results"].values()))
    torch.testing.assert_close(archive["arrays", "data"], torch.from_numpy(data), rtol=0, atol=0)
    mesh.save(str(tmp_path / "nonmodal.pmsh"))
    _assert_archive_equal(Mesh.load(str(tmp_path / "nonmodal.pmsh")).global_data, mesh.global_data)
    for extension in ("vtu", "vtk", "vtp"):
        path = MeshVTKSink(str(tmp_path / extension), file_format=extension)(iter([mesh]), 0)[0]
        fields = read_vtk_data(path)
        _assert_archive_equal(fields["global_data"], mesh.global_data)
        if extension != "vtp":
            torch.testing.assert_close(fields["point_data"]["displacements"], expected, rtol=0, atol=0)


def test_sparse_element_results_and_layered_tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "model.bdf"
    model = _deck(path)
    model.add_ctetra(55, 22, _NIDS)
    model.write_bdf(str(path))
    _op2(path.with_suffix(".op2"))
    energy = SimpleNamespace(
        isubcase=1,
        data=np.arange(18, dtype=np.float32).reshape(2, 3, 3),
        element=np.array([[33, 0, 100000000], [55, 33, 100000000]], dtype=np.int32),
        _times=np.array([1, 3], dtype=np.int32),
        custom_components=np.array(["energy", "percent", "density"]),
        get_headers=lambda: ["strain_energy", "percent", "strain_energy_density"],
    )
    layered = SimpleNamespace(
        isubcase=1,
        data=np.arange(24, dtype=np.float64).reshape(2, 4, 3),
        element_node=np.array([[33, 0], [33, 10], [55, 0], [55, 10]]),
    )
    _tables(monkeypatch, {"strain_energy.ctetra_strain_energy": {1: energy}, "stress.custom": {1: layered}})
    mesh = next(NastranSource(str(path))[0])
    field = "strain_energy.ctetra_strain_energy"
    parent = mesh.global_data["nastran", "element_data"]
    assert parent[field + "__valid"].tolist() == [[True, True], [False, True]]
    assert torch.isnan(parent[field][1, 0]).all()
    torch.testing.assert_close(parent[field][0, 1], torch.from_numpy(energy.data[1, 1]))
    assert "stress.custom" not in mesh.cell_data
    catalog = decode_nastran_json(mesh.global_data["nastran", "catalog"])
    for entry in catalog:
        archive = mesh.global_data["nastran", "results", entry["id"]]
        source = energy if entry["table"] == field else layered
        torch.testing.assert_close(archive["arrays", "data"], torch.from_numpy(source.data))
    projection = decode_nastran_json(mesh.global_data["nastran", "projection"])
    assert "unrecognized association" in projection["results"][1]["description"]
    path = MeshVTKSink(str(tmp_path / "out"))(iter([mesh]), 0)[0]
    fields = read_vtk_data(path)
    _assert_archive_equal(fields["global_data"], mesh.global_data)
    torch.testing.assert_close(fields["cell_data"][field], parent[field], equal_nan=True)


@pytest.mark.filterwarnings("error:The VTK reader")
def test_unknown_fields_multiple_keys_and_filters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "model.bdf"
    _deck(path)
    _op2(path.with_suffix(".op2"))
    result = SimpleNamespace(
        isubcase=1,
        data=np.arange(20, dtype=np.int64).reshape(1, 10, 2) + 2**60,
        node_gridtype=np.array([[nid, 1] for nid in _NIDS]),
        extra_numeric=np.array([complex(3, 5)], dtype=np.complex128),
        extra_labels=np.array(["arbitrary", "components"]),
    )
    tables = {"custom_temperature": {(1, 1): result, (1, 2): result}, "other_table": {1: result}}
    sparse = coo_matrix(([1 + 2j, 3 + 4j], ([0, 1], [2, 0])), shape=(4, 4))
    matrices = {"KAA": SimpleNamespace(data=sparse)}
    _tables(monkeypatch, tables, matrices)
    mesh = next(NastranSource(str(path), result_patterns="custom*,matrices.*", preserve_source_files=False)[0])
    assert "source_files" not in mesh.global_data["nastran"]
    names = list(mesh.point_data.keys())
    fields = [key for key in names if key.startswith("custom_temperature__table_") and not key.endswith("__valid")]
    assert len(fields) == 2
    catalog = decode_nastran_json(mesh.global_data["nastran", "catalog"])
    assert next(e for e in catalog if e["table"] == "other_table")["status"] == "excluded_by_user"
    matrix = next(e for e in catalog if e["table"] == "matrices.KAA")
    archive = mesh.global_data["nastran", "results", matrix["id"]]
    torch.testing.assert_close(archive["arrays", "data"], torch.from_numpy(sparse.data))
    selected, report = _discover_results(SimpleNamespace(get_table_types=lambda: [], matrices=matrices), "custom*")
    assert not selected and report[0]["status"] == "excluded_by_user"
    for fmt in ("vtu", "vtk"):
        vtk = MeshVTKSink(str(tmp_path / fmt), file_format=fmt)(iter([mesh]), 0)[0]
        loaded = read_vtk_data(vtk)
        _assert_archive_equal(loaded["global_data"], mesh.global_data)
        for field in fields:
            torch.testing.assert_close(loaded["point_data"][field], mesh.point_data[field], rtol=0, atol=0)


@pytest.mark.parametrize("bad_path", ["../outside", "/outside", "C:/outside", "a\\b", "."])
def test_restore_rejects_paths_and_checksum(tmp_path: Path, bad_path: str) -> None:
    path = tmp_path / "model.bdf"
    _deck(path)
    mesh = next(NastranSource(str(path), read_results=False)[0])
    sources = mesh.global_data["nastran", "source_files"]
    manifest = decode_nastran_json(sources["manifest"])
    manifest[0]["path"] = bad_path
    sources["manifest"] = _encode_json(manifest)
    with pytest.raises(ValueError, match="Unsafe"):
        restore_nastran_sources(mesh, str(tmp_path / "restore"))
    manifest[0]["path"] = "model.bdf"
    manifest[0]["sha256"] = "0" * 64
    sources["manifest"] = _encode_json(manifest)
    with pytest.raises(ValueError, match="checksum"):
        restore_nastran_sources(mesh, str(tmp_path / "restore"))
    assert not (tmp_path / "restore").exists()


def test_matrix_cards_without_raw_fields_preserve_entries(tmp_path: Path) -> None:
    path = tmp_path / "model.bdf"
    model = _deck(path)
    model.add_dmig("KAA", 6, 2, 0, 0, 1, [(10, 1)], [(20, 2)], [2.5])
    model.write_bdf(str(path))
    mesh = next(NastranSource(str(path), read_results=False, preserve_source_files=False)[0])
    metadata = decode_nastran_json(mesh.global_data["nastran", "deck_metadata"])
    matrix = metadata["cards"]["dmig"][0]
    assert matrix["fields_status"] == "raw_fields_unavailable"
    assert matrix["attributes"]["name"] == "KAA"
    assert matrix["attributes"]["GCj"]["values"] == [[10, 1]]
    assert matrix["attributes"]["GCi"]["values"] == [[20, 2]]
    assert matrix["attributes"]["Real"]["values"] == [2.5]
    exported = MeshVTKSink(str(tmp_path / "out"))(iter([mesh]), 0)[0]
    restored = decode_nastran_json(read_vtk_data(exported)["global_data"]["nastran", "deck_metadata"])
    assert restored == metadata


def test_include_symlink_recovers_at_deck_reference(tmp_path: Path) -> None:
    deck = tmp_path / "model.bdf"
    _deck(deck)
    original = tmp_path / "original.inc"
    original.write_bytes(b"PARAM,POST,-1\n")
    alias = tmp_path / "parts" / "alias.inc"
    alias.parent.mkdir()
    try:
        alias.symlink_to(original)
    except (NotImplementedError, OSError):
        pytest.skip("Creating symbolic links is unavailable")
    deck.write_bytes(deck.read_bytes() + b"INCLUDE 'parts/alias.inc'\n")
    mesh = next(NastranSource(str(deck), read_results=False)[0])
    restored = tmp_path / "restored"
    restore_nastran_sources(mesh, str(restored))
    assert (restored / "parts" / "alias.inc").read_bytes() == original.read_bytes()
    assert not (restored / "parts" / "alias.inc").is_symlink()
    assert next(NastranSource(str(restored / "model.bdf"), read_results=False)[0]).n_cells == 8


def test_restore_validates_payload_dtype(tmp_path: Path) -> None:
    path = tmp_path / "model.bdf"
    _deck(path)
    mesh = next(NastranSource(str(path), read_results=False)[0])
    sources = mesh.global_data["nastran", "source_files"]
    manifest = decode_nastran_json(sources["manifest"])
    key = manifest[0]["key"]
    sources[key] = sources[key].to(torch.int64)
    with pytest.raises(ValueError, match="byte vector"):
        restore_nastran_sources(mesh, str(tmp_path / "restore"))
    assert not (tmp_path / "restore").exists()


def test_transformed_projection_requires_explicit_export(tmp_path: Path) -> None:
    path = tmp_path / "model.bdf"
    _deck(path)
    mesh = next(NastranSource(str(path), read_results=False)[0])
    mesh.cell_data["new_feature"] = torch.ones(mesh.n_cells)
    with pytest.raises(ValueError, match="use_original_geometry=False"):
        MeshVTKSink(str(tmp_path / "out"))(iter([mesh]), 0)
    exported = MeshVTKSink(str(tmp_path / "out"), use_original_geometry=False)(iter([mesh]), 0)[0]
    grid = pv.read(exported)
    assert grid.n_cells == 8 and "new_feature" in grid.cell_data


def test_numeric_byteorder() -> None:
    values = np.array([1.25, 2.5], dtype=">f8")
    torch.testing.assert_close(_array_tensor(values), torch.tensor([1.25, 2.5], dtype=torch.float64))


@pytest.mark.parametrize(
    "card,vtk_class",
    [
        ("CHEXA", "vtkHexahedron"),
        ("CHEXA", "vtkQuadraticHexahedron"),
        ("CPENTA", "vtkWedge"),
        ("CPENTA", "vtkQuadraticWedge"),
        ("CPYRAM", "vtkPyramid"),
        ("CPYRAM", "vtkQuadraticPyramid"),
        ("CQUAD4", "vtkQuad"),
        ("CQUAD8", "vtkQuadraticQuad"),
        ("CTRIA3", "vtkTriangle"),
        ("CTRIA6", "vtkQuadraticTriangle"),
    ],
)
def test_common_topologies_retain_original_cells(tmp_path: Path, card: str, vtk_class: str) -> None:
    from pyNastran.bdf.bdf import BDF, CaseControlDeck
    from vtkmodules import vtkCommonDataModel

    model = BDF(debug=None)
    model.sol = 101
    model.case_control_deck = CaseControlDeck(["SUBCASE 1", "DISPLACEMENT=ALL"])
    cell = getattr(vtkCommonDataModel, vtk_class)()
    vtk_points = np.array(cell.GetParametricCoords()).reshape(-1, 3)
    points = vtk_points
    if vtk_class == "vtkQuadraticHexahedron":
        points = points[[*range(12), *range(16, 20), *range(12, 16)]]
    for nid, xyz in enumerate(points, 1):
        model.add_grid(nid, xyz)
    model.add_mat1(1, 210000.0, None, 0.3)
    if card.startswith(("CTRIA", "CQUAD")):
        model.add_pshell(1, mid1=1, t=0.1)
    else:
        model.add_psolid(1, 1)
    getattr(model, "add_" + card.lower())(1, 1, list(range(1, len(points) + 1)))
    deck = tmp_path / "model.bdf"
    model.write_bdf(str(deck))
    mesh = next(NastranSource(str(deck), read_results=False)[0])
    assert mesh.n_points == len(points)
    assert mesh.n_cells >= 1
    exported = MeshVTKSink(str(tmp_path / "out"))(iter([mesh]), 0)[0]
    grid = pv.read(exported)
    assert grid.celltypes.tolist() == [cell.GetCellType()]
    np.testing.assert_array_equal(grid.points[grid.cell_connectivity], vtk_points)


def test_mixed_dimensions_and_volume_types(tmp_path: Path) -> None:
    from vtkmodules.vtkCommonDataModel import vtkHexahedron

    path = tmp_path / "model.bdf"
    model = _deck(path)
    model.add_pshell(23, mid1=11, t=0.1)
    model.add_ctria3(55, 23, _NIDS[:3])
    model.add_prod(24, 11, A=1.0)
    model.add_crod(77, 24, _NIDS[:2])
    points = np.array(vtkHexahedron().GetParametricCoords()).reshape(-1, 3) + 3.0
    ids = list(range(201, 209))
    for nid, point in zip(ids, points, strict=True):
        model.add_grid(nid, point)
    model.add_chexa(66, 22, ids)
    model.write_bdf(str(path))
    meshes = list(NastranSource(str(path), read_results=False)[0])
    assert [mesh.n_manifold_dims for mesh in meshes] == [1, 2, 3]
    volume = meshes[2]
    assert (volume.cell_data["element_id"] == 33).sum() == 8
    assert (volume.cell_data["element_id"] == 66).sum() >= 5
    for mesh in meshes:
        assert mesh.n_points == 18
    exported = MeshVTKSink(str(tmp_path / "out"))(iter(meshes), 0)
    assert pv.read(exported[2]).celltypes.tolist() == [24, 12]


def test_original_cells_follow_point_permutation(tmp_path: Path) -> None:
    path = tmp_path / "model.bdf"
    _deck(path)
    mesh = next(NastranSource(str(path), read_results=False)[0])
    order = torch.arange(mesh.n_points - 1, -1, -1)
    permuted = Mesh(
        points=mesh.points[order],
        cells=order[mesh.cells],
        point_data=mesh.point_data[order],
        cell_data=mesh.cell_data.clone(),
        global_data=mesh.global_data.clone(),
    )
    exported = MeshVTKSink(str(tmp_path / "out"))(iter([permuted]), 0)[0]
    grid = pv.read(exported)
    np.testing.assert_array_equal(grid.point_data["node_id"][grid.cell_connectivity], _NIDS)
