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

"""Tests for AtomicStatsFilter.

Unit tests use mock AtomicData objects with real torch tensors to verify
statistics computation, Parquet output, flush/merge, and pass-through
semantics.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pyarrow.parquet as pq
import pytest
import torch

if TYPE_CHECKING:
    import pathlib


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


def _make_scalar_atomic_data(value: float = 5.0, n_nodes: int = 4) -> MagicMock:
    """Create a simple mock with predictable values for exact checks.

    Parameters
    ----------
    value : float
        Constant value for positions.
    n_nodes : int
        Number of atoms.

    Returns
    -------
    MagicMock
        Mock with constant-valued tensor fields.
    """
    mock = MagicMock()
    mock.positions = torch.full((n_nodes, 3), value)
    mock.atomic_numbers = torch.ones(n_nodes, dtype=torch.int64)
    mock.forces = torch.full((n_nodes, 3), -value)
    mock.energies = torch.tensor([[value * 2]])

    # Everything else None.
    for attr in (
        "atomic_masses",
        "atom_categories",
        "velocities",
        "momenta",
        "kinetic_energies",
        "node_charges",
        "node_attrs",
        "node_spins",
        "node_embeddings",
        "edge_index",
        "shifts",
        "unit_shifts",
        "edge_embeddings",
        "stresses",
        "virials",
        "dipoles",
        "graph_charges",
        "graph_spins",
        "cell",
        "pbc",
    ):
        setattr(mock, attr, None)

    mock.extra_data = {}
    return mock


# ---------------------------------------------------------------------------
# Unit tests — metadata and parameters
# ---------------------------------------------------------------------------


@pytest.mark.requires("atm")
class TestAtomicStatsFilterUnit:
    """Metadata and parameter tests (no data processing)."""

    def test_params_list(self) -> None:
        """Params include output."""
        from physicsnemo_curator.domains.atm.filters.stats import AtomicStatsFilter

        params = AtomicStatsFilter.params()
        assert len(params) >= 1
        names = [p.name for p in params]
        assert "output" in names

    def test_name_and_description(self) -> None:
        """Name and description are non-empty strings."""
        from physicsnemo_curator.domains.atm.filters.stats import AtomicStatsFilter

        assert isinstance(AtomicStatsFilter.name, str)
        assert len(AtomicStatsFilter.name) > 0
        assert isinstance(AtomicStatsFilter.description, str)
        assert len(AtomicStatsFilter.description) > 0

    def test_output_path_property(self, tmp_path: pathlib.Path) -> None:
        """The output_path property reflects the constructor argument."""
        from physicsnemo_curator.domains.atm.filters.stats import AtomicStatsFilter

        filt = AtomicStatsFilter(output=str(tmp_path / "stats.parquet"))
        assert filt.output_path == tmp_path / "stats.parquet"


# ---------------------------------------------------------------------------
# Pass-through tests
# ---------------------------------------------------------------------------


@pytest.mark.requires("atm")
class TestAtomicStatsFilterPassThrough:
    """Verify the filter yields items unchanged."""

    def test_yields_items_unchanged(self, tmp_path: pathlib.Path) -> None:
        """Every item should be yielded back untouched."""
        from physicsnemo_curator.domains.atm.filters.stats import AtomicStatsFilter

        filt = AtomicStatsFilter(output=str(tmp_path / "stats.parquet"))
        items = [_make_mock_atomic_data(seed=i) for i in range(3)]

        def gen():
            """Yield mock items."""
            yield from items

        result = list(filt(gen()))
        assert len(result) == 3
        for orig, out in zip(items, result, strict=True):
            assert orig is out  # identity check — same object

    def test_empty_generator(self, tmp_path: pathlib.Path) -> None:
        """Empty generator yields nothing and flush returns None."""
        from physicsnemo_curator.domains.atm.filters.stats import AtomicStatsFilter

        filt = AtomicStatsFilter(output=str(tmp_path / "stats.parquet"))

        def gen():
            """Yield nothing."""
            return
            yield  # noqa: RET504  # unreachable yield makes this a generator

        result = list(filt(gen()))
        assert result == []
        assert filt.flush() is None


# ---------------------------------------------------------------------------
# Statistics computation tests
# ---------------------------------------------------------------------------


@pytest.mark.requires("atm")
class TestAtomicStatsFilterComputation:
    """Verify correctness of computed statistics."""

    def test_flush_writes_parquet(self, tmp_path: pathlib.Path) -> None:
        """Flush writes a Parquet file with the correct schema."""
        from physicsnemo_curator.domains.atm.filters.stats import AtomicStatsFilter

        filt = AtomicStatsFilter(output=str(tmp_path / "stats.parquet"))
        data = _make_mock_atomic_data()

        def gen():
            """Yield one item."""
            yield data

        list(filt(gen()))
        path = filt.flush()

        assert path is not None
        assert (tmp_path / "stats.parquet").exists()

        table = pq.read_table(path)
        expected_cols = {
            "field_key",
            "level",
            "component",
            "n_values",
            "n_components",
            "mean",
            "std",
            "var",
            "min",
            "max",
            "median",
            "abs_mean",
            "abs_max",
            "skewness",
            "kurtosis",
            "welford_n",
            "welford_mean",
            "welford_m2",
            "welford_m3",
            "welford_m4",
            "welford_abs_sum",
        }
        assert set(table.column_names) == expected_cols

    def test_constant_values_zero_std(self, tmp_path: pathlib.Path) -> None:
        """Constant-valued tensor has std=0, skewness=0, kurtosis=0."""
        from physicsnemo_curator.domains.atm.filters.stats import AtomicStatsFilter

        filt = AtomicStatsFilter(output=str(tmp_path / "stats.parquet"))
        data = _make_scalar_atomic_data(value=5.0, n_nodes=4)

        def gen():
            """Yield one item."""
            yield data

        list(filt(gen()))
        path = filt.flush()

        table = pq.read_table(path)
        # Find the positions row (component 0).
        for i in range(table.num_rows):
            if table["field_key"][i].as_py() == "positions" and table["component"][i].as_py() == 0:
                assert table["mean"][i].as_py() == pytest.approx(5.0)
                assert table["std"][i].as_py() == pytest.approx(0.0)
                assert table["var"][i].as_py() == pytest.approx(0.0)
                assert table["min"][i].as_py() == pytest.approx(5.0)
                assert table["max"][i].as_py() == pytest.approx(5.0)
                assert table["skewness"][i].as_py() == pytest.approx(0.0)
                assert table["kurtosis"][i].as_py() == pytest.approx(0.0)
                break
        else:
            pytest.fail("positions component 0 not found in output")

    def test_field_levels_correct(self, tmp_path: pathlib.Path) -> None:
        """Fields are tagged with the correct semantic level."""
        from physicsnemo_curator.domains.atm.filters.stats import AtomicStatsFilter

        filt = AtomicStatsFilter(output=str(tmp_path / "stats.parquet"))
        data = _make_mock_atomic_data()

        def gen():
            """Yield one item."""
            yield data

        list(filt(gen()))
        path = filt.flush()

        table = pq.read_table(path)
        levels = {}
        for i in range(table.num_rows):
            fk = table["field_key"][i].as_py()
            lv = table["level"][i].as_py()
            levels[fk] = lv

        assert levels.get("positions") == "node"
        assert levels.get("forces") == "node"
        assert levels.get("energies") == "system"
        assert levels.get("edge_index") == "edge"

    def test_vector_components(self, tmp_path: pathlib.Path) -> None:
        """Vector fields produce one row per component."""
        from physicsnemo_curator.domains.atm.filters.stats import AtomicStatsFilter

        filt = AtomicStatsFilter(output=str(tmp_path / "stats.parquet"))
        data = _make_mock_atomic_data()

        def gen():
            """Yield one item."""
            yield data

        list(filt(gen()))
        path = filt.flush()

        table = pq.read_table(path)
        positions_rows = [i for i in range(table.num_rows) if table["field_key"][i].as_py() == "positions"]
        # positions is (n, 3) -> 3 component rows
        assert len(positions_rows) == 3
        components = {table["component"][i].as_py() for i in positions_rows}
        assert components == {0, 1, 2}

    def test_known_statistics(self, tmp_path: pathlib.Path) -> None:
        """Verify exact statistics for a known data distribution."""
        from physicsnemo_curator.domains.atm.filters.stats import AtomicStatsFilter

        filt = AtomicStatsFilter(output=str(tmp_path / "stats.parquet"))

        # Build a mock with a known distribution for energies.
        mock = _make_scalar_atomic_data(value=0.0, n_nodes=2)
        # Overwrite energies with known values: [1, 2, 3, 4, 5] -> mean=3
        mock.energies = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])

        def gen():
            """Yield one item."""
            yield mock

        list(filt(gen()))
        path = filt.flush()
        table = pq.read_table(path)

        for i in range(table.num_rows):
            if table["field_key"][i].as_py() == "energies":
                assert table["mean"][i].as_py() == pytest.approx(3.0)
                assert table["min"][i].as_py() == pytest.approx(1.0)
                assert table["max"][i].as_py() == pytest.approx(5.0)
                # population variance = 2.0
                assert table["var"][i].as_py() == pytest.approx(2.0)
                assert table["std"][i].as_py() == pytest.approx(math.sqrt(2.0))
                assert table["welford_n"][i].as_py() == 5
                break
        else:
            pytest.fail("energies not found in output")

    def test_multiple_items_accumulate(self, tmp_path: pathlib.Path) -> None:
        """Multiple items accumulate rows in a single flush."""
        from physicsnemo_curator.domains.atm.filters.stats import AtomicStatsFilter

        filt = AtomicStatsFilter(output=str(tmp_path / "stats.parquet"))

        def gen():
            """Yield two items."""
            yield _make_mock_atomic_data(seed=0)
            yield _make_mock_atomic_data(seed=1)

        list(filt(gen()))
        path = filt.flush()

        table = pq.read_table(path)
        # Each item contributes rows for positions (3 comps), atomic_numbers (1),
        # forces (3), energies (1), stresses (9), edge_index (2), shifts (3) = 22 per item.
        # With 2 items: 44 rows.
        assert table.num_rows > 0
        # Each field_key appears twice (once per item).
        positions_rows = [
            i
            for i in range(table.num_rows)
            if table["field_key"][i].as_py() == "positions" and table["component"][i].as_py() == 0
        ]
        assert len(positions_rows) == 2

    def test_flush_clears_state(self, tmp_path: pathlib.Path) -> None:
        """After flush, internal rows are cleared."""
        from physicsnemo_curator.domains.atm.filters.stats import AtomicStatsFilter

        filt = AtomicStatsFilter(output=str(tmp_path / "stats.parquet"))

        def gen():
            """Yield one item."""
            yield _make_mock_atomic_data()

        list(filt(gen()))
        filt.flush()

        # Second flush with no new data returns None.
        assert filt.flush() is None

    def test_creates_parent_directory(self, tmp_path: pathlib.Path) -> None:
        """Flush creates parent directories as needed."""
        from physicsnemo_curator.domains.atm.filters.stats import AtomicStatsFilter

        nested = tmp_path / "a" / "b" / "stats.parquet"
        filt = AtomicStatsFilter(output=str(nested))

        def gen():
            """Yield one item."""
            yield _make_mock_atomic_data()

        list(filt(gen()))
        path = filt.flush()

        assert path is not None
        assert nested.exists()

    def test_extra_data_fields(self, tmp_path: pathlib.Path) -> None:
        """Extra data dict entries produce stats rows."""
        from physicsnemo_curator.domains.atm.filters.stats import AtomicStatsFilter

        filt = AtomicStatsFilter(output=str(tmp_path / "stats.parquet"))
        data = _make_scalar_atomic_data()
        data.extra_data = {"custom_field": torch.tensor([1.0, 2.0, 3.0])}

        def gen():
            """Yield one item."""
            yield data

        list(filt(gen()))
        path = filt.flush()
        table = pq.read_table(path)

        field_keys = [table["field_key"][i].as_py() for i in range(table.num_rows)]
        assert "extra/custom_field" in field_keys


# ---------------------------------------------------------------------------
# Merge tests
# ---------------------------------------------------------------------------


@pytest.mark.requires("atm")
class TestAtomicStatsFilterMerge:
    """Tests for Welford merge functionality."""

    def test_merge_two_shards(self, tmp_path: pathlib.Path) -> None:
        """Merging two shard files produces correct aggregate statistics."""
        from physicsnemo_curator.domains.atm.filters.stats import AtomicStatsFilter

        # Shard 1: values [1, 2, 3]
        filt1 = AtomicStatsFilter(output=str(tmp_path / "shard1.parquet"))
        mock1 = _make_scalar_atomic_data(value=0.0, n_nodes=1)
        mock1.energies = torch.tensor([[1.0], [2.0], [3.0]])

        def gen1():
            """Yield shard 1."""
            yield mock1

        list(filt1(gen1()))
        filt1.flush()

        # Shard 2: values [4, 5]
        filt2 = AtomicStatsFilter(output=str(tmp_path / "shard2.parquet"))
        mock2 = _make_scalar_atomic_data(value=0.0, n_nodes=1)
        mock2.energies = torch.tensor([[4.0], [5.0]])

        def gen2():
            """Yield shard 2."""
            yield mock2

        list(filt2(gen2()))
        filt2.flush()

        # Merge.
        merged_path = AtomicStatsFilter.merge(
            [str(tmp_path / "shard1.parquet"), str(tmp_path / "shard2.parquet")],
            str(tmp_path / "merged.parquet"),
        )

        table = pq.read_table(merged_path)
        for i in range(table.num_rows):
            if table["field_key"][i].as_py() == "energies":
                assert table["welford_n"][i].as_py() == 5
                assert table["mean"][i].as_py() == pytest.approx(3.0)
                assert table["min"][i].as_py() == pytest.approx(1.0)
                assert table["max"][i].as_py() == pytest.approx(5.0)
                # population var of [1,2,3,4,5] = 2.0
                assert table["var"][i].as_py() == pytest.approx(2.0)
                break
        else:
            pytest.fail("energies not found in merged output")

    def test_merge_empty_raises(self) -> None:
        """Merge with empty paths raises ValueError."""
        from physicsnemo_curator.domains.atm.filters.stats import AtomicStatsFilter

        with pytest.raises(ValueError, match="non-empty"):
            AtomicStatsFilter.merge([], "output.parquet")

    def test_merge_single_shard(self, tmp_path: pathlib.Path) -> None:
        """Merging a single shard is a no-op identity."""
        from physicsnemo_curator.domains.atm.filters.stats import AtomicStatsFilter

        filt = AtomicStatsFilter(output=str(tmp_path / "shard.parquet"))
        mock = _make_scalar_atomic_data(value=3.0, n_nodes=4)

        def gen():
            """Yield one item."""
            yield mock

        list(filt(gen()))
        filt.flush()

        merged_path = AtomicStatsFilter.merge(
            [str(tmp_path / "shard.parquet")],
            str(tmp_path / "merged.parquet"),
        )

        table = pq.read_table(merged_path)
        for i in range(table.num_rows):
            if table["field_key"][i].as_py() == "positions" and table["component"][i].as_py() == 0:
                assert table["mean"][i].as_py() == pytest.approx(3.0)
                break

    def test_merge_welford_stats_function(self, tmp_path: pathlib.Path) -> None:
        """The public merge_welford_stats function returns a table."""
        from physicsnemo_curator.domains.atm.filters.stats import (
            AtomicStatsFilter,
            merge_welford_stats,
        )

        filt = AtomicStatsFilter(output=str(tmp_path / "shard.parquet"))
        mock = _make_scalar_atomic_data(value=1.0, n_nodes=2)

        def gen():
            """Yield one item."""
            yield mock

        list(filt(gen()))
        filt.flush()

        import pyarrow as pa

        result = merge_welford_stats([str(tmp_path / "shard.parquet")])
        assert isinstance(result, pa.Table)
        assert result.num_rows > 0


# ---------------------------------------------------------------------------
# Registry test
# ---------------------------------------------------------------------------


@pytest.mark.requires("atm")
class TestAtomicStatsFilterRegistry:
    """Test that the filter is registered."""

    def test_filter_registered(self) -> None:
        """AtomicStatsFilter is discoverable in the registry."""
        from physicsnemo_curator.core.registry import registry

        names = [f.name for f in registry.list_filters("atm")]
        assert "Atomic Statistics" in names
