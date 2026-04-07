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

"""Tests for ASELMDBSource.

Unit tests use mock ASE LMDB databases to avoid requiring real data.
E2E tests (marked ``slow``) use the val/ dataset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

if TYPE_CHECKING:
    import pathlib


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _write_mock_aselmdb_files(root: pathlib.Path, n_files: int = 3, n_rows: int = 5) -> None:
    """Create mock .aselmdb files for unit testing.

    These are empty files — the actual database reads are mocked at the
    ``ase.db.connect`` level.
    """
    for i in range(n_files):
        (root / f"data{i:04d}.aselmdb").write_bytes(b"mock")


def _write_mock_metadata(root: pathlib.Path, n_atoms: int = 100) -> None:
    """Write a minimal metadata.npz to *root*."""
    np.savez(
        root / "metadata.npz",
        natoms=np.array([n_atoms, n_atoms + 1]),
        data_ids=np.array([0, 1]),
    )


def _make_mock_atoms() -> MagicMock:
    """Create a mock ASE Atoms-like object suitable for AtomicData.from_atoms."""
    atoms = MagicMock()
    atoms.get_atomic_numbers.return_value = np.array([6, 8, 1])
    atoms.get_positions.return_value = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    atoms.get_cell.return_value = np.eye(3) * 10.0
    atoms.get_pbc.return_value = np.array([True, True, True])
    atoms.numbers = np.array([6, 8, 1])
    atoms.positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return atoms


def _make_mock_row() -> MagicMock:
    """Create a mock database row with a .toatoms() method."""
    row = MagicMock()
    row.toatoms.return_value = _make_mock_atoms()
    return row


def _make_mock_db(n_rows: int = 5) -> MagicMock:
    """Create a mock ASE database connection."""
    db = MagicMock()
    db.select.return_value = [_make_mock_row() for _ in range(n_rows)]
    return db


# ---------------------------------------------------------------------------
# Unit tests — metadata and parameters
# ---------------------------------------------------------------------------


@pytest.mark.requires("alch")
class TestASELMDBSourceUnit:
    """Metadata and parameter tests (no data access)."""

    def test_params_list(self) -> None:
        from physicsnemo_curator.alch.sources.aselmdb import ASELMDBSource

        params = ASELMDBSource.params()
        assert len(params) > 0
        names = [p.name for p in params]
        assert "data_dir" in names
        assert "metadata_path" in names

    def test_name_and_description(self) -> None:
        from physicsnemo_curator.alch.sources.aselmdb import ASELMDBSource

        assert isinstance(ASELMDBSource.name, str)
        assert ASELMDBSource.name == "ASE LMDB"
        assert isinstance(ASELMDBSource.description, str)
        assert len(ASELMDBSource.description) > 0


# ---------------------------------------------------------------------------
# Unit tests — local mock data
# ---------------------------------------------------------------------------


@pytest.mark.requires("alch")
class TestASELMDBSourceLocal:
    """Tests against local mock data with mocked ASE database."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: pathlib.Path) -> None:
        self.mock_root = tmp_path / "mock_data"
        self.mock_root.mkdir()
        _write_mock_aselmdb_files(self.mock_root, n_files=3)
        _write_mock_metadata(self.mock_root)

    def test_len(self) -> None:
        from physicsnemo_curator.alch.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.mock_root))
        assert len(source) == 3

    def test_no_aselmdb_files_raises(self, tmp_path: pathlib.Path) -> None:
        from physicsnemo_curator.alch.sources.aselmdb import ASELMDBSource

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No .aselmdb files"):
            ASELMDBSource(data_dir=str(empty_dir))

    def test_files_sorted_lexicographically(self) -> None:
        from physicsnemo_curator.alch.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.mock_root))
        names = [p.name for p in source.db_files]
        assert names == sorted(names)

    def test_metadata_loaded(self) -> None:
        from physicsnemo_curator.alch.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.mock_root))
        assert "natoms" in source.metadata
        assert "data_ids" in source.metadata

    def test_metadata_not_required(self, tmp_path: pathlib.Path) -> None:
        from physicsnemo_curator.alch.sources.aselmdb import ASELMDBSource

        no_meta = tmp_path / "no_meta"
        no_meta.mkdir()
        _write_mock_aselmdb_files(no_meta, n_files=2)
        source = ASELMDBSource(data_dir=str(no_meta))
        assert len(source) == 2
        assert source.metadata == {}

    @patch("physicsnemo_curator.alch.sources.aselmdb.ase.db.connect")
    def test_getitem_yields_atomic_data(self, mock_connect: MagicMock) -> None:
        from nvalchemi.data import AtomicData

        from physicsnemo_curator.alch.sources.aselmdb import ASELMDBSource

        mock_connect.return_value = _make_mock_db(n_rows=3)

        source = ASELMDBSource(data_dir=str(self.mock_root))
        items = list(source[0])
        assert len(items) == 3
        for item in items:
            assert isinstance(item, AtomicData)

    @patch("physicsnemo_curator.alch.sources.aselmdb.ase.db.connect")
    def test_negative_index(self, mock_connect: MagicMock) -> None:
        from physicsnemo_curator.alch.sources.aselmdb import ASELMDBSource

        mock_connect.return_value = _make_mock_db(n_rows=2)

        source = ASELMDBSource(data_dir=str(self.mock_root))
        # source[-1] should be same as source[2] (last of 3 files)
        items_neg = list(source[-1])
        assert len(items_neg) == 2

    def test_index_out_of_bounds(self) -> None:
        from physicsnemo_curator.alch.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.mock_root))
        with pytest.raises(IndexError):
            next(source[len(source)])

    def test_index_out_of_bounds_negative(self) -> None:
        from physicsnemo_curator.alch.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.mock_root))
        with pytest.raises(IndexError):
            next(source[-(len(source) + 1)])

    def test_data_dir_property(self) -> None:
        from physicsnemo_curator.alch.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.mock_root))
        assert source.data_dir == self.mock_root


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


@pytest.mark.requires("alch")
class TestASELMDBSourceRegistry:
    """Test that the source is registered."""

    def test_source_registered(self) -> None:
        import physicsnemo_curator.alch  # noqa: F401
        from physicsnemo_curator.core.registry import registry

        sources = registry.list_sources("alch")
        source_names = {s.name for s in sources}
        assert "ASE LMDB" in source_names


# ---------------------------------------------------------------------------
# E2E tests (require real val/ data)
# ---------------------------------------------------------------------------


@pytest.mark.requires("alch")
@pytest.mark.e2e
@pytest.mark.slow
class TestASELMDBSourceE2E:
    """End-to-end tests against the val/ dataset."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        from physicsnemo_curator.alch.sources.aselmdb import ASELMDBSource

        self.source = ASELMDBSource(data_dir="val/")

    def test_discovers_files(self) -> None:
        assert len(self.source) == 80

    def test_reads_first_item(self) -> None:
        from nvalchemi.data import AtomicData

        item = next(self.source[0])
        assert isinstance(item, AtomicData)
        assert item.num_nodes > 0

    def test_yields_multiple_items(self) -> None:
        items = list(self.source[0])
        assert len(items) > 1
