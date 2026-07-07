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

"""Tests for DomainMeshZarrSink."""

from __future__ import annotations

import pytest
import torch

zarr = pytest.importorskip("zarr")

pytestmark = pytest.mark.requires("mesh")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_surface_mesh(n_cells: int = 40, seed: int = 0):
    """Build a small triangulated surface mesh with cell fields."""
    from physicsnemo.mesh import Mesh

    gen = torch.Generator().manual_seed(seed)
    n_points = 3 * n_cells
    return Mesh(
        points=torch.randn(n_points, 3, generator=gen),
        cells=torch.randint(0, n_points, (n_cells, 3), generator=gen, dtype=torch.int64),
        cell_data={
            "pMeanTrim": torch.randn(n_cells, generator=gen),
            "wallShearStressMeanTrim": torch.randn(n_cells, 3, generator=gen),
        },
        global_data={"TimeValue": torch.tensor([1.0])},
    )


@pytest.fixture
def domain_mesh():
    """Build a DomainMesh matching the DrivAerML multi-mode shape."""
    from physicsnemo.mesh import DomainMesh, Mesh

    gen = torch.Generator().manual_seed(7)
    interior = Mesh(
        points=torch.randn(50, 3, generator=gen),
        point_data={"UMean": torch.randn(50, 3, generator=gen)},
    )
    return DomainMesh(
        interior=interior,
        boundaries={"vehicle": _make_surface_mesh()},
        global_data={
            "U_inf": torch.tensor(38.889),
            "rho_inf": torch.tensor(1.205),
            "L_ref": torch.tensor(2.786),
        },
    )


def _write(domain, tmp_path, **kwargs):
    from physicsnemo_curator.domains.mesh.sinks import DomainMeshZarrSink

    sink = DomainMeshZarrSink(output_dir=str(tmp_path), **kwargs)
    return sink(iter([domain]), index=0)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_params_names():
    from physicsnemo_curator.domains.mesh.sinks import DomainMeshZarrSink

    names = [p.name for p in DomainMeshZarrSink.params()]
    assert names == [
        "output_dir",
        "compression_level",
        "chunk_cells",
        "chunk_points",
        "soup_boundaries",
        "naming_template",
    ]


def test_registered():
    from physicsnemo_curator.core.registry import registry
    from physicsnemo_curator.domains.mesh.sinks import DomainMeshZarrSink

    sinks = registry.sinks("mesh")
    assert DomainMeshZarrSink in sinks.values() or DomainMeshZarrSink in sinks


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_domain_mesh_schema(domain_mesh, tmp_path):
    paths = _write(domain_mesh, tmp_path)
    assert len(paths) == 1
    root = zarr.open_group(paths[0], mode="r")

    assert root.attrs["format"] == "physicsnemo-domainmesh-zarr"
    assert root.attrs["schema_version"] == 1
    assert root.attrs["boundary_names"] == ["vehicle"]

    # Case-level metadata at the root only.
    assert set(root["global_data"].array_keys()) == {"U_inf", "rho_inf", "L_ref"}

    interior = root["interior"]
    assert interior.attrs["format"] == "physicsnemo-mesh-zarr"
    assert interior.attrs["n_cells"] == 0
    assert "UMean" in interior["point_data"].array_keys()

    vehicle = root["boundaries/vehicle"]
    assert vehicle.attrs["layout"] == "indexed"
    assert vehicle.attrs["nodes_per_cell"] == 3
    ref = domain_mesh.boundaries["vehicle"]
    assert (vehicle["points"][...] == ref.points.numpy()).all()
    assert (vehicle["cells"][...] == ref.cells.numpy()).all()
    assert (vehicle["cell_data/pMeanTrim"][...] == ref.cell_data["pMeanTrim"].numpy()).all()
    # Boundary keeps its own global_data; case keys are not duplicated in.
    assert set(vehicle["global_data"].array_keys()) == {"TimeValue"}


def test_soup_boundaries(domain_mesh, tmp_path):
    paths = _write(domain_mesh, tmp_path, soup_boundaries=True)
    root = zarr.open_group(paths[0], mode="r")
    vehicle = root["boundaries/vehicle"]
    ref = domain_mesh.boundaries["vehicle"]

    assert vehicle.attrs["layout"] == "soup"
    assert vehicle.attrs["n_points"] == 3 * ref.n_cells
    # Per-cell vertex geometry must be preserved by the soup relayout.
    soup_points = torch.from_numpy(vehicle["points"][...])
    soup_cells = torch.from_numpy(vehicle["cells"][...])
    assert torch.equal(soup_points[soup_cells], ref.points[ref.cells])
    # Interior is never souped.
    assert root["interior"].attrs["n_cells"] == 0


def test_plain_mesh_items_written_as_mesh_schema(tmp_path):
    mesh = _make_surface_mesh()
    paths = _write(mesh, tmp_path)
    root = zarr.open_group(paths[0], mode="r")
    assert root.attrs["format"] == "physicsnemo-mesh-zarr"
    assert "boundaries" not in root


def test_naming_template_and_compression_off(tmp_path):
    from physicsnemo.mesh import DomainMesh, Mesh

    # Smooth (compressible) fields: random noise at toy scale would make
    # the compressed store larger than raw via codec overhead.
    surface = _make_surface_mesh(n_cells=2000)
    surface.cell_data["smooth"] = torch.linspace(0, 1, 2000).repeat_interleave(16).reshape(2000, 16)
    domain_mesh = DomainMesh(
        interior=Mesh(points=torch.zeros(10, 3)),
        boundaries={"vehicle": surface},
        global_data={"U_inf": torch.tensor(38.889)},
    )
    paths_raw = _write(
        domain_mesh, tmp_path / "raw", compression_level=0, naming_template="case_{index}_{seq}"
    )
    assert paths_raw[0].endswith("case_0_0.zarr")
    paths_zstd = _write(domain_mesh, tmp_path / "zstd", compression_level=3)

    def store_bytes(p):
        import pathlib

        return sum(f.stat().st_size for f in pathlib.Path(p).rglob("*") if f.is_file())

    assert store_bytes(paths_zstd[0]) < store_bytes(paths_raw[0])


def test_atomic_write_leaves_no_partial_store(domain_mesh, tmp_path):
    import pathlib

    _write(domain_mesh, tmp_path)
    leftovers = [p for p in pathlib.Path(tmp_path).iterdir() if p.name.startswith(".tmp_")]
    assert leftovers == []


def test_roundtrip_via_physicsnemo_reader(domain_mesh, tmp_path):
    """Cross-repo contract: physicsnemo's ZarrMeshReader reads sink output."""
    try:
        from physicsnemo.datapipes import ZarrMeshReader
    except ImportError:
        pytest.skip("physicsnemo.datapipes.ZarrMeshReader not available")

    _write(domain_mesh, tmp_path, soup_boundaries=True, naming_template="run_{index}")
    reader = ZarrMeshReader(
        tmp_path,
        subpath="boundaries/vehicle",
        merge_global_data_from="../../global_data",
    )
    mesh, _ = reader[0]
    ref = domain_mesh.boundaries["vehicle"]
    assert torch.equal(mesh.points[mesh.cells], ref.points[ref.cells])
    assert torch.equal(mesh.cell_data["pMeanTrim"], ref.cell_data["pMeanTrim"])
    # Read-time merge: boundary's own key + case-level keys, single-sourced.
    assert torch.equal(mesh.global_data["TimeValue"], torch.tensor([1.0]))
    assert float(mesh.global_data["U_inf"]) == pytest.approx(38.889)
