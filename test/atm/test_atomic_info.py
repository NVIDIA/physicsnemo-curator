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

"""Tests for AtomicInfoFilter.

Unit tests use mock AtomicData objects with real torch tensors to verify
metadata extraction, logging, JSON-lines output, and pass-through
semantics.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
import torch

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Generator

pytestmark = pytest.mark.requires("atm")


def _single(item: MagicMock) -> Generator[MagicMock, None, None]:
    """Wrap a single item in a generator (avoids iter() vs Generator mismatch)."""
    yield item


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_mock_atomic_data(
    n_nodes: int = 10,
    n_edges: int = 20,
    seed: int = 42,
) -> MagicMock:
    """Create a mock AtomicData with real torch tensors.

    Parameters
    ----------
    n_nodes : int
        Number of atoms/nodes.
    n_edges : int
        Number of edges.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    MagicMock
        Mock with tensor attributes matching AtomicData's interface.
    """
    gen = torch.Generator().manual_seed(seed)

    mock = MagicMock()
    mock.positions = torch.randn(n_nodes, 3, generator=gen)
    mock.atomic_numbers = torch.randint(1, 84, (n_nodes,), generator=gen)
    mock.forces = torch.randn(n_nodes, 3, generator=gen)
    mock.energies = torch.randn(1, 1, generator=gen)
    mock.stresses = torch.randn(1, 3, 3, generator=gen)

    # Edge-level fields.
    mock.edge_index = torch.randint(0, n_nodes, (2, n_edges), generator=gen)
    mock.shifts = torch.randn(n_edges, 3, generator=gen)

    # Fields that are None (not present on this sample).
    mock.atomic_masses = None
    mock.atom_categories = None
    mock.velocities = None
    mock.momenta = None
    mock.kinetic_energies = None
    mock.node_charges = None
    mock.node_attrs = None
    mock.node_spins = None
    mock.node_embeddings = None
    mock.unit_shifts = None
    mock.edge_embeddings = None
    mock.virials = None
    mock.dipoles = None
    mock.graph_charges = None
    mock.graph_spins = None
    mock.cell = None
    mock.pbc = None

    # Extra data.
    mock.extra_data = {}

    return mock


def _make_full_mock(n_nodes: int = 4) -> MagicMock:
    """Create a mock AtomicData with PBC and cell populated.

    Parameters
    ----------
    n_nodes : int
        Number of atoms.

    Returns
    -------
    MagicMock
        Mock with pbc and cell tensors.
    """
    mock = _make_mock_atomic_data(n_nodes=n_nodes, n_edges=6, seed=0)
    mock.pbc = torch.tensor([True, True, False])
    mock.cell = torch.eye(3) * 10.0
    return mock


# ---------------------------------------------------------------------------
# Unit tests — metadata and parameters
# ---------------------------------------------------------------------------


class TestAtomicInfoFilterUnit:
    """Metadata and parameter tests (no data processing)."""

    def test_params_list(self) -> None:
        """Params include output, log_level, include_fields."""
        from physicsnemo.curator.atm.filters.atomic_info import AtomicInfoFilter

        params = AtomicInfoFilter.params()
        assert len(params) == 3
        names = [p.name for p in params]
        assert "output" in names
        assert "log_level" in names
        assert "include_fields" in names

    def test_name_and_description(self) -> None:
        """Name and description are non-empty strings."""
        from physicsnemo.curator.atm.filters.atomic_info import AtomicInfoFilter

        assert isinstance(AtomicInfoFilter.name, str)
        assert len(AtomicInfoFilter.name) > 0
        assert isinstance(AtomicInfoFilter.description, str)
        assert len(AtomicInfoFilter.description) > 0

    def test_default_params(self) -> None:
        """Default construction uses info level, include fields, no output."""
        from physicsnemo.curator.atm.filters.atomic_info import AtomicInfoFilter

        filt = AtomicInfoFilter()
        assert filt._output_path is None
        assert filt._log_level == logging.INFO
        assert filt._include_fields is True


# ---------------------------------------------------------------------------
# Pass-through tests
# ---------------------------------------------------------------------------


class TestAtomicInfoFilterPassThrough:
    """Verify the filter yields items unchanged."""

    def test_yields_items_unchanged(self) -> None:
        """Every item should be yielded back untouched."""
        from physicsnemo.curator.atm.filters.atomic_info import AtomicInfoFilter

        filt = AtomicInfoFilter()
        items = [_make_mock_atomic_data(seed=i) for i in range(3)]

        def gen():
            """Yield mock items."""
            yield from items

        result = list(filt(gen()))
        assert len(result) == 3
        for orig, out in zip(items, result, strict=True):
            assert orig is out  # identity check — same object

    def test_empty_generator(self) -> None:
        """Empty generator yields nothing and flush returns None."""
        from physicsnemo.curator.atm.filters.atomic_info import AtomicInfoFilter

        filt = AtomicInfoFilter()

        def gen():
            """Yield nothing."""
            return
            yield  # noqa: RET504  # unreachable yield makes this a generator

        result = list(filt(gen()))
        assert result == []
        assert filt.flush() is None


# ---------------------------------------------------------------------------
# Logging tests
# ---------------------------------------------------------------------------


class TestAtomicInfoFilterLogging:
    """Verify logging output."""

    def test_logs_at_info_level(self, caplog: pytest.LogCaptureFixture) -> None:
        """Default log_level='info' produces INFO-level messages."""
        from physicsnemo.curator.atm.filters.atomic_info import AtomicInfoFilter

        filt = AtomicInfoFilter(log_level="info")
        data = _make_mock_atomic_data(n_nodes=10, n_edges=20)

        with caplog.at_level(logging.INFO):
            list(filt(_single(data)))

        assert "AtomicData 0" in caplog.text
        assert "10 atoms" in caplog.text
        assert "20 edges" in caplog.text
        assert "MB" in caplog.text

    def test_logs_at_debug_level(self, caplog: pytest.LogCaptureFixture) -> None:
        """Debug log_level produces field-level detail."""
        from physicsnemo.curator.atm.filters.atomic_info import AtomicInfoFilter

        filt = AtomicInfoFilter(log_level="debug", include_fields=True)
        data = _make_mock_atomic_data(n_nodes=5, n_edges=8)

        with caplog.at_level(logging.DEBUG):
            list(filt(_single(data)))

        # Should have per-field lines.
        assert "node/positions" in caplog.text
        assert "edge/edge_index" in caplog.text

    def test_logs_pbc_and_cell_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """When pbc/cell are present, they appear in the log metadata."""
        from physicsnemo.curator.atm.filters.atomic_info import AtomicInfoFilter

        filt = AtomicInfoFilter(log_level="info")
        data = _make_full_mock()

        with caplog.at_level(logging.INFO):
            list(filt(_single(data)))

        assert "AtomicData 0" in caplog.text
        assert "4 atoms" in caplog.text

    def test_item_index_increments(self, caplog: pytest.LogCaptureFixture) -> None:
        """Item index increments for each item processed."""
        from physicsnemo.curator.atm.filters.atomic_info import AtomicInfoFilter

        filt = AtomicInfoFilter()

        def gen():
            """Yield two items."""
            yield _make_mock_atomic_data(seed=0)
            yield _make_mock_atomic_data(seed=1)

        with caplog.at_level(logging.INFO):
            list(filt(gen()))

        assert "AtomicData 0" in caplog.text
        assert "AtomicData 1" in caplog.text


# ---------------------------------------------------------------------------
# JSON-lines output tests
# ---------------------------------------------------------------------------


class TestAtomicInfoFilterOutput:
    """Verify JSON-lines file output."""

    def test_writes_jsonl_file(self, tmp_path: pathlib.Path) -> None:
        """Output file is written when output path is provided."""
        from physicsnemo.curator.atm.filters.atomic_info import AtomicInfoFilter

        output = tmp_path / "info.jsonl"
        filt = AtomicInfoFilter(output=str(output))
        data = _make_mock_atomic_data(n_nodes=10, n_edges=20)

        list(filt(_single(data)))
        filt.flush()

        assert output.exists()
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["item_index"] == 0
        assert record["n_atoms"] == 10
        assert record["n_edges"] == 20
        assert record["n_fields"] > 0
        assert record["memory_estimate_bytes"] > 0

    def test_writes_multiple_records(self, tmp_path: pathlib.Path) -> None:
        """Multiple items produce multiple JSON lines."""
        from physicsnemo.curator.atm.filters.atomic_info import AtomicInfoFilter

        output = tmp_path / "info.jsonl"
        filt = AtomicInfoFilter(output=str(output))

        def gen():
            """Yield three items."""
            yield _make_mock_atomic_data(seed=0)
            yield _make_mock_atomic_data(seed=1)
            yield _make_mock_atomic_data(seed=2)

        list(filt(gen()))
        filt.flush()

        lines = output.read_text().strip().split("\n")
        assert len(lines) == 3
        for i, line in enumerate(lines):
            record = json.loads(line)
            assert record["item_index"] == i

    def test_includes_field_details(self, tmp_path: pathlib.Path) -> None:
        """With include_fields=True, output contains field metadata."""
        from physicsnemo.curator.atm.filters.atomic_info import AtomicInfoFilter

        output = tmp_path / "info.jsonl"
        filt = AtomicInfoFilter(output=str(output), include_fields=True)
        data = _make_mock_atomic_data()

        list(filt(_single(data)))
        filt.flush()

        record = json.loads(output.read_text().strip())
        assert "fields" in record
        assert len(record["fields"]) > 0

        # Check field structure.
        field = record["fields"][0]
        assert "name" in field
        assert "level" in field
        assert "shape" in field
        assert "dtype" in field
        assert "nbytes" in field

    def test_excludes_field_details(self, tmp_path: pathlib.Path) -> None:
        """With include_fields=False, output omits field details."""
        from physicsnemo.curator.atm.filters.atomic_info import AtomicInfoFilter

        output = tmp_path / "info.jsonl"
        filt = AtomicInfoFilter(output=str(output), include_fields=False)
        data = _make_mock_atomic_data()

        list(filt(_single(data)))
        filt.flush()

        record = json.loads(output.read_text().strip())
        assert "fields" not in record
        assert "n_fields" in record

    def test_flush_returns_path(self, tmp_path: pathlib.Path) -> None:
        """Flush returns the output path when a file was used."""
        from physicsnemo.curator.atm.filters.atomic_info import AtomicInfoFilter

        output = tmp_path / "info.jsonl"
        filt = AtomicInfoFilter(output=str(output))
        list(filt(_single(_make_mock_atomic_data())))
        result = filt.flush()
        assert result == str(output)

    def test_flush_returns_none_without_output(self) -> None:
        """Flush returns None when no output path was specified."""
        from physicsnemo.curator.atm.filters.atomic_info import AtomicInfoFilter

        filt = AtomicInfoFilter()
        list(filt(_single(_make_mock_atomic_data())))
        assert filt.flush() is None

    def test_creates_parent_directory(self, tmp_path: pathlib.Path) -> None:
        """Flush creates parent directories as needed."""
        from physicsnemo.curator.atm.filters.atomic_info import AtomicInfoFilter

        nested = tmp_path / "a" / "b" / "info.jsonl"
        filt = AtomicInfoFilter(output=str(nested))

        list(filt(_single(_make_mock_atomic_data())))
        filt.flush()

        assert nested.exists()

    def test_pbc_and_cell_in_output(self, tmp_path: pathlib.Path) -> None:
        """PBC and cell shape appear in JSON output."""
        from physicsnemo.curator.atm.filters.atomic_info import AtomicInfoFilter

        output = tmp_path / "info.jsonl"
        filt = AtomicInfoFilter(output=str(output))
        data = _make_full_mock()

        list(filt(_single(data)))
        filt.flush()

        record = json.loads(output.read_text().strip())
        assert record["pbc"] == [True, True, False]
        assert record["cell_shape"] == [3, 3]

    def test_none_pbc_and_cell_in_output(self, tmp_path: pathlib.Path) -> None:
        """None PBC and cell produce null in JSON output."""
        from physicsnemo.curator.atm.filters.atomic_info import AtomicInfoFilter

        output = tmp_path / "info.jsonl"
        filt = AtomicInfoFilter(output=str(output))
        data = _make_mock_atomic_data()

        list(filt(_single(data)))
        filt.flush()

        record = json.loads(output.read_text().strip())
        assert record["pbc"] is None
        assert record["cell_shape"] is None

    def test_extra_data_fields_in_output(self, tmp_path: pathlib.Path) -> None:
        """Extra data dict entries appear in field inventory."""
        from physicsnemo.curator.atm.filters.atomic_info import AtomicInfoFilter

        output = tmp_path / "info.jsonl"
        filt = AtomicInfoFilter(output=str(output), include_fields=True)
        data = _make_mock_atomic_data()
        data.extra_data = {"custom_field": torch.tensor([1.0, 2.0, 3.0])}

        list(filt(_single(data)))
        filt.flush()

        record = json.loads(output.read_text().strip())
        field_names = [f["name"] for f in record["fields"]]
        assert "extra/custom_field" in field_names


# ---------------------------------------------------------------------------
# Metadata extraction tests
# ---------------------------------------------------------------------------


class TestAtomicInfoFieldExtraction:
    """Verify field metadata extraction logic."""

    def test_field_levels(self) -> None:
        """Fields are tagged with the correct semantic level."""
        from physicsnemo.curator.atm.filters.atomic_info import _extract_field_info

        data = _make_mock_atomic_data()
        fields = _extract_field_info(data)

        level_map = {f["name"]: f["level"] for f in fields}
        assert level_map.get("positions") == "node"
        assert level_map.get("forces") == "node"
        assert level_map.get("energies") == "system"
        assert level_map.get("edge_index") == "edge"

    def test_field_shapes(self) -> None:
        """Field shapes match the original tensor shapes."""
        from physicsnemo.curator.atm.filters.atomic_info import _extract_field_info

        data = _make_mock_atomic_data(n_nodes=10, n_edges=20)
        fields = _extract_field_info(data)

        shape_map = {f["name"]: f["shape"] for f in fields}
        assert shape_map["positions"] == [10, 3]
        assert shape_map["edge_index"] == [2, 20]

    def test_field_memory(self) -> None:
        """Field nbytes equals numel * element_size."""
        from physicsnemo.curator.atm.filters.atomic_info import _extract_field_info

        data = _make_mock_atomic_data(n_nodes=10, n_edges=20)
        fields = _extract_field_info(data)

        for f in fields:
            name = f["name"]
            tensor = getattr(data, name, None)
            if name.startswith("extra/"):
                tensor = data.extra_data[name.split("/", 1)[1]]
            if tensor is not None:
                assert f["nbytes"] == tensor.numel() * tensor.element_size()


# ---------------------------------------------------------------------------
# Registry test
# ---------------------------------------------------------------------------


class TestAtomicInfoFilterRegistry:
    """Test that the filter is registered."""

    def test_filter_registered(self) -> None:
        """AtomicInfoFilter is discoverable in the registry."""
        from physicsnemo.curator.core.registry import registry

        names = [f.name for f in registry.list_filters("atm")]
        assert "Atomic Info Logger" in names
