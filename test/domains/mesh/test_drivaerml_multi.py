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

"""Tests for DrivAerML multi-mesh mode and MeshSink extended naming."""

from __future__ import annotations

import pathlib
from collections.abc import Generator
from unittest.mock import patch

import numpy as np
import pytest
import pyvista as pv
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.io import from_pyvista

from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink

pytestmark = pytest.mark.requires("mesh")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeSourceWithRunId:
    """Minimal source stub exposing run_id and mesh_name for MeshSink tests."""

    def __init__(self, run_ids: list[int], mesh_names_map: dict[str, str]):
        self._run_ids = run_ids
        self._mesh_names_map = mesh_names_map  # seq -> name

    def run_id(self, index: int) -> int:
        return self._run_ids[index]

    def mesh_name(self, index: int, seq: int) -> str:
        run_id = self._run_ids[index]
        parts = list(self._mesh_names_map.keys())
        part = parts[seq]
        return self._mesh_names_map[part].format(run_id=run_id)


def _make_simple_mesh() -> Mesh:
    """Create a simple Mesh for sink testing."""
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    cells = np.array([[3, 0, 1, 2], [3, 0, 2, 3]])
    cell_types = np.array([5, 5])
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    grid.point_data["temperature"] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    return from_pyvista(grid, manifold_dim="auto", point_source="vertices")


# ---------------------------------------------------------------------------
# MeshSink extended naming tests
# ---------------------------------------------------------------------------


class TestMeshSinkRunIdPlaceholder:
    """Test {run_id} and {mesh_name} placeholders in MeshSink."""

    def test_run_id_placeholder_resolves(self, tmp_path: pathlib.Path) -> None:
        """MeshSink resolves {run_id} from source.run_id(index)."""
        sink = MeshSink(
            output_dir=str(tmp_path),
            naming_template="run_{run_id}/mesh_{seq}",
        )
        source = FakeSourceWithRunId(
            run_ids=[5, 12],
            mesh_names_map={"domain": "domain_{run_id}"},
        )
        sink.set_source(source)

        meshes = iter([_make_simple_mesh()])
        paths = sink(meshes, index=0)

        assert len(paths) == 1
        assert "run_5/mesh_0" in paths[0]
        assert pathlib.Path(paths[0]).exists()

    def test_mesh_name_placeholder_resolves(self, tmp_path: pathlib.Path) -> None:
        """MeshSink resolves {mesh_name} from source.mesh_name(index, seq)."""
        sink = MeshSink(
            output_dir=str(tmp_path),
            naming_template="run_{run_id}/{mesh_name}",
        )
        source = FakeSourceWithRunId(
            run_ids=[1, 2],
            mesh_names_map={
                "domain": "domain_{run_id}",
                "stl": "drivaer_{run_id}.stl",
            },
        )
        sink.set_source(source)

        mesh1 = _make_simple_mesh()
        mesh2 = _make_simple_mesh()
        meshes = iter([mesh1, mesh2])
        paths = sink(meshes, index=0)

        assert len(paths) == 2
        assert "run_1/domain_1" in paths[0]
        assert "run_1/drivaer_1.stl" in paths[1]

    def test_run_id_without_source_raises(self, tmp_path: pathlib.Path) -> None:
        """Using {run_id} without a compatible source raises ValueError."""
        sink = MeshSink(
            output_dir=str(tmp_path),
            naming_template="run_{run_id}/mesh_{seq}",
        )
        # No source set — calling should fail
        meshes = iter([_make_simple_mesh()])
        with pytest.raises(ValueError, match="run_id"):
            sink(meshes, index=0)

    def test_mesh_name_without_source_raises(self, tmp_path: pathlib.Path) -> None:
        """Using {mesh_name} without a compatible source raises ValueError."""
        sink = MeshSink(
            output_dir=str(tmp_path),
            naming_template="{mesh_name}",
        )
        meshes = iter([_make_simple_mesh()])
        with pytest.raises(ValueError, match="mesh_name"):
            sink(meshes, index=0)

    def test_construction_with_new_placeholders_valid(self) -> None:
        """Construction succeeds when new placeholders are used."""
        # Should not raise
        sink = MeshSink(
            output_dir="/tmp/out",
            naming_template="run_{run_id}/{mesh_name}",
        )
        assert sink is not None
