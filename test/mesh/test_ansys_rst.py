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

"""Tests for AnsysRSTSource.

Unit tests use mock DPF Model objects to avoid requiring an Ansys
installation or license.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest

if TYPE_CHECKING:
    import pathlib

# ---------------------------------------------------------------------------
# Install a fake ``ansys.dpf.core`` module so that the lazy
# ``from ansys.dpf import core as dpf`` inside AnsysRSTSource._read_rst
# succeeds even when ansys-dpf-core is not installed.
# ---------------------------------------------------------------------------

_mock_dpf_core = MagicMock()
_mock_dpf = MagicMock()
_mock_ansys = MagicMock()

_mock_dpf.core = _mock_dpf_core
_mock_ansys.dpf = _mock_dpf

sys.modules.setdefault("ansys", _mock_ansys)
sys.modules.setdefault("ansys.dpf", _mock_dpf)
sys.modules.setdefault("ansys.dpf.core", _mock_dpf_core)


# ---------------------------------------------------------------------------
# Helpers — write mock .rst files and build mock DPF objects
# ---------------------------------------------------------------------------


def _write_mock_rst_dir(
    root: pathlib.Path,
    n_files: int = 3,
) -> None:
    """Create minimal mock ``.rst`` files (empty sentinels).

    Since ``ansys-dpf-core`` reads proprietary binary ``.rst`` files,
    we create empty sentinel files and mock the DPF Model in tests.

    Parameters
    ----------
    root : pathlib.Path
        Directory to write files into.
    n_files : int
        Number of sentinel ``.rst`` files to create.
    """
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        (root / f"thermal_sim_{i:03d}.rst").touch()


class _MockNode:
    """Mock DPF node."""

    def __init__(self, node_id: int) -> None:
        self.id = node_id


class _MockElement:
    """Mock DPF element with node IDs."""

    def __init__(self, node_ids: list[int]) -> None:
        self.node_ids = node_ids


class _MockNodes:
    """Mock DPF nodes collection."""

    def __init__(self, coords: np.ndarray, node_ids: list[int]) -> None:
        self._coords = coords
        self._node_ids = node_ids
        self.n_nodes = len(node_ids)
        # coordinates_field.data returns flat (N, 3)
        self.coordinates_field = MagicMock()
        self.coordinates_field.data = coords.tolist()

    def node_by_index(self, i: int) -> _MockNode:
        """Return mock node at index *i*."""
        return _MockNode(self._node_ids[i])


class _MockElements:
    """Mock DPF elements collection."""

    def __init__(self, conn: list[list[int]]) -> None:
        self._conn = conn
        self.n_elements = len(conn)

    def element_by_index(self, i: int) -> _MockElement:
        """Return mock element at index *i*."""
        return _MockElement(self._conn[i])


class _MockMeshedRegion:
    """Mock DPF meshed region with nodes and elements."""

    def __init__(
        self,
        n_nodes: int = 10,
        n_elements: int = 4,
        dim: int = 3,
    ) -> None:
        rng = np.random.default_rng(42)
        coords = rng.standard_normal((n_nodes, dim))
        node_ids = list(range(1, n_nodes + 1))  # 1-based IDs
        self.nodes = _MockNodes(coords, node_ids)

        # Build simple triangle connectivity using 1-based node IDs.
        conn: list[list[int]] = []
        for i in range(n_elements):
            nids = [(i % n_nodes) + 1, ((i + 1) % n_nodes) + 1, ((i + 2) % n_nodes) + 1]
            conn.append(nids)
        self.elements = _MockElements(conn)


class _MockField:
    """Mock DPF field with data."""

    def __init__(self, data: np.ndarray) -> None:
        self.data = data.tolist()


class _MockFieldsContainer:
    """Mock DPF fields container (list-like)."""

    def __init__(self, fields: list[_MockField]) -> None:
        self._fields = fields

    def __len__(self) -> int:
        """Return number of fields."""
        return len(self._fields)

    def __getitem__(self, index: int) -> _MockField:
        """Return field at index."""
        return self._fields[index]


class _MockResultOp:
    """Mock DPF result operator."""

    def __init__(self, fields_container: _MockFieldsContainer) -> None:
        self.outputs = MagicMock()
        self.outputs.fields_container.return_value = fields_container


def _make_mock_model(
    n_nodes: int = 10,
    n_elements: int = 4,
    available_results: dict[str, np.ndarray] | None = None,
) -> MagicMock:
    """Create a mock DPF Model.

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    n_elements : int
        Number of elements.
    available_results : dict[str, np.ndarray] | None
        Mapping from result name to data array.

    Returns
    -------
    MagicMock
        Mock ``dpf.Model`` instance.
    """
    meshed_region = _MockMeshedRegion(n_nodes=n_nodes, n_elements=n_elements)
    model = MagicMock()
    model.metadata.meshed_region = meshed_region

    available = available_results or {}

    def _make_result_attr(name: str) -> object:
        """Create a callable that returns a mock result op or raises."""
        if name in available:
            data = available[name]
            field = _MockField(data)
            fc = _MockFieldsContainer([field])
            op = _MockResultOp(fc)
            return lambda: op
        else:

            def _raise():
                msg = f"Result '{name}' not available"
                raise AttributeError(msg)

            return _raise

    # Configure model.results to have attributes for available results
    # and raise AttributeError for unavailable ones.
    results_mock = MagicMock()
    for name in [
        "temperature",
        "displacement",
        "heat_flux",
        "stress",
        "elastic_strain",
        "structural_temperature",
        "velocity",
        "acceleration",
    ]:
        setattr(results_mock, name, _make_result_attr(name))
    model.results = results_mock

    return model


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.requires("mesh")
class TestAnsysRSTSourceUnit:
    """Metadata and parameter tests (no data access)."""

    def test_params_list(self) -> None:
        """Params include input_dir and result_types."""
        from physicsnemo.curator.mesh.sources.ansys_rst import AnsysRSTSource

        params = AnsysRSTSource.params()
        assert len(params) > 0
        names = [p.name for p in params]
        assert "input_dir" in names
        assert "result_types" in names

    def test_name_and_description(self) -> None:
        """Name and description ClassVars are set."""
        from physicsnemo.curator.mesh.sources.ansys_rst import AnsysRSTSource

        assert isinstance(AnsysRSTSource.name, str)
        assert len(AnsysRSTSource.name) > 0
        assert isinstance(AnsysRSTSource.description, str)
        assert len(AnsysRSTSource.description) > 0


@pytest.mark.requires("mesh")
class TestAnsysRSTSourceLocal:
    """Tests against mock DPF data."""

    N_NODES = 20
    N_ELEMENTS = 8

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: pathlib.Path) -> None:
        """Create mock .rst directory."""
        self.mock_root = tmp_path / "ansys_data"
        _write_mock_rst_dir(self.mock_root, n_files=3)

    def _configure_dpf_model(
        self,
        available_results: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Set ``_mock_dpf_core.Model`` to return a mock model.

        Parameters
        ----------
        available_results : dict[str, np.ndarray] | None
            Results the mock model should expose.
        """
        _mock_dpf_core.Model.return_value = _make_mock_model(
            n_nodes=self.N_NODES,
            n_elements=self.N_ELEMENTS,
            available_results=available_results,
        )

    def _make_source(
        self,
        result_types: list[str] | None = None,
    ) -> AnsysRSTSource:  # type: ignore[name-defined]  # noqa: F821  # ty: ignore[unresolved-reference]
        """Create an AnsysRSTSource pointing at mock data."""
        from physicsnemo.curator.mesh.sources.ansys_rst import AnsysRSTSource

        return AnsysRSTSource(
            input_dir=str(self.mock_root),
            result_types=result_types,
        )

    def test_len(self) -> None:
        """Source discovers all .rst files."""
        source = self._make_source()
        assert len(source) == 3

    def test_getitem_returns_mesh(self) -> None:
        """Getitem yields a valid Mesh."""
        from physicsnemo.mesh import Mesh

        rng = np.random.default_rng(0)
        self._configure_dpf_model({"temperature": rng.standard_normal(self.N_NODES)})

        source = self._make_source(result_types=["temperature"])
        mesh = next(source[0])

        assert isinstance(mesh, Mesh)
        assert mesh.n_points == self.N_NODES

    def test_point_data_temperature(self) -> None:
        """Temperature field lands in point_data."""
        rng = np.random.default_rng(1)
        self._configure_dpf_model({"temperature": rng.standard_normal(self.N_NODES) + 300.0})

        source = self._make_source(result_types=["temperature"])
        mesh = next(source[0])

        assert "temperature" in list(mesh.point_data.keys())  # noqa: SIM118
        assert mesh.point_data["temperature"].shape == (self.N_NODES,)

    def test_point_data_displacement(self) -> None:
        """Vector displacement field in point_data."""
        rng = np.random.default_rng(2)
        self._configure_dpf_model({"displacement": rng.standard_normal((self.N_NODES, 3))})

        source = self._make_source(result_types=["displacement"])
        mesh = next(source[0])

        assert "displacement" in list(mesh.point_data.keys())  # noqa: SIM118
        assert mesh.point_data["displacement"].shape == (self.N_NODES, 3)

    def test_cell_data_heat_flux(self) -> None:
        """Heat flux lands in cell_data (elemental)."""
        rng = np.random.default_rng(3)
        self._configure_dpf_model({"heat_flux": rng.standard_normal((self.N_ELEMENTS, 3))})

        source = self._make_source(result_types=["heat_flux"])
        mesh = next(source[0])

        assert "heat_flux" in list(mesh.cell_data.keys())  # noqa: SIM118
        assert mesh.cell_data["heat_flux"].shape == (self.N_ELEMENTS, 3)

    def test_cell_data_stress(self) -> None:
        """Stress lands in cell_data (elemental)."""
        rng = np.random.default_rng(4)
        self._configure_dpf_model({"stress": rng.standard_normal((self.N_ELEMENTS, 6))})

        source = self._make_source(result_types=["stress"])
        mesh = next(source[0])

        assert "stress" in list(mesh.cell_data.keys())  # noqa: SIM118
        assert mesh.cell_data["stress"].shape == (self.N_ELEMENTS, 6)

    def test_multiple_results(self) -> None:
        """Multiple results populate both point_data and cell_data."""
        rng = np.random.default_rng(5)
        self._configure_dpf_model(
            {
                "temperature": rng.standard_normal(self.N_NODES) + 300.0,
                "heat_flux": rng.standard_normal((self.N_ELEMENTS, 3)),
            }
        )

        source = self._make_source(result_types=["temperature", "heat_flux"])
        mesh = next(source[0])

        assert "temperature" in list(mesh.point_data.keys())  # noqa: SIM118
        assert "heat_flux" in list(mesh.cell_data.keys())  # noqa: SIM118

    def test_auto_discover_results(self) -> None:
        """When result_types is empty, all available results are extracted."""
        rng = np.random.default_rng(6)
        self._configure_dpf_model(
            {
                "temperature": rng.standard_normal(self.N_NODES),
                "displacement": rng.standard_normal((self.N_NODES, 3)),
            }
        )

        source = self._make_source(result_types=None)
        mesh = next(source[0])

        pd_keys = list(mesh.point_data.keys())  # noqa: SIM118
        assert "temperature" in pd_keys
        assert "displacement" in pd_keys

    def test_global_data(self) -> None:
        """Global data contains num_nodes and num_elements."""
        self._configure_dpf_model({"temperature": np.zeros(self.N_NODES)})

        source = self._make_source(result_types=["temperature"])
        mesh = next(source[0])

        assert mesh.global_data["num_nodes"].item() == self.N_NODES
        assert mesh.global_data["num_elements"].item() == self.N_ELEMENTS

    def test_cells_shape(self) -> None:
        """Cells tensor has correct shape (E, nodes_per_cell)."""
        self._configure_dpf_model({"temperature": np.zeros(self.N_NODES)})

        source = self._make_source(result_types=["temperature"])
        mesh = next(source[0])

        assert mesh.cells.shape[0] == self.N_ELEMENTS
        assert mesh.cells.shape[1] == 3  # triangles in mock

    def test_negative_index(self) -> None:
        """Negative index maps to correct position."""
        self._configure_dpf_model({"temperature": np.ones(self.N_NODES)})

        source = self._make_source(result_types=["temperature"])
        mesh_neg = next(source[-1])
        mesh_pos = next(source[len(source) - 1])

        assert mesh_neg.n_points == mesh_pos.n_points

    def test_index_out_of_bounds(self) -> None:
        """Out-of-range index raises IndexError."""
        source = self._make_source()
        with pytest.raises(IndexError):
            next(source[len(source)])

    def test_nonexistent_dir(self, tmp_path: pathlib.Path) -> None:
        """Non-existent directory raises FileNotFoundError."""
        from physicsnemo.curator.mesh.sources.ansys_rst import AnsysRSTSource

        with pytest.raises(FileNotFoundError):
            AnsysRSTSource(input_dir=str(tmp_path / "nonexistent"))

    def test_no_cell_data_when_only_nodal(self) -> None:
        """When only nodal results exist, cell_data is empty."""
        self._configure_dpf_model({"temperature": np.zeros(self.N_NODES)})

        source = self._make_source(result_types=["temperature"])
        mesh = next(source[0])

        # cell_data will be empty TensorDict (Mesh always creates one)
        assert len(list(mesh.cell_data.keys())) == 0  # noqa: SIM118


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


@pytest.mark.requires("mesh")
class TestAnsysRSTHelpers:
    """Test module-level helper functions."""

    def test_extract_connectivity(self) -> None:
        """Connectivity extraction produces correct 0-based indices."""
        from physicsnemo.curator.mesh.sources.ansys_rst import _extract_connectivity

        region = _MockMeshedRegion(n_nodes=6, n_elements=3)
        conn = _extract_connectivity(region)

        assert conn.shape[0] == 3
        assert conn.shape[1] == 3  # triangles
        assert conn.dtype == np.int64
        # All indices should be 0-based and valid.
        assert conn.min() >= 0
        assert conn.max() < 6

    def test_discover_available_results(self) -> None:
        """Discovery finds results that are configured."""
        from physicsnemo.curator.mesh.sources.ansys_rst import _discover_available_results

        model = _make_mock_model(
            available_results={
                "temperature": np.zeros(10),
                "displacement": np.zeros((10, 3)),
            },
        )
        results = _discover_available_results(model)

        assert "temperature" in results
        assert "displacement" in results
        # Results that were not configured should not appear.
        assert "stress" not in results

    def test_extract_result_field_success(self) -> None:
        """Successful result extraction returns data and location."""
        from physicsnemo.curator.mesh.sources.ansys_rst import _extract_result_field

        model = _make_mock_model(
            available_results={"temperature": np.arange(10, dtype=np.float64)},
        )
        result = _extract_result_field(model, "temperature")

        assert result is not None
        data, location = result
        assert isinstance(data, np.ndarray)
        assert location == "nodal"
        assert len(data) == 10

    def test_extract_result_field_missing(self) -> None:
        """Missing result returns None."""
        from physicsnemo.curator.mesh.sources.ansys_rst import _extract_result_field

        model = _make_mock_model(available_results={})
        result = _extract_result_field(model, "temperature")

        assert result is None


# ---------------------------------------------------------------------------
# Registry test
# ---------------------------------------------------------------------------


@pytest.mark.requires("mesh")
class TestAnsysRSTRegistry:
    """Test that the source is registered."""

    def test_source_registered(self) -> None:
        """AnsysRSTSource should appear in the mesh registry."""
        from physicsnemo.curator.core.registry import registry

        sources = registry.sources("mesh")
        assert "Ansys RST" in sources
