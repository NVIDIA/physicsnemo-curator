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
E2E tests generate real ASE LMDB databases on the fly via ``_create_real_aselmdb``.
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
    # AtomicData.from_atoms calls get_cell().array.reshape(1,3,3), so the
    # mock must provide a Cell-like object with an .array attribute.
    cell_mock = MagicMock()
    cell_mock.array = np.eye(3) * 10.0
    cell_mock.__array__ = lambda self: np.eye(3) * 10.0
    atoms.get_cell.return_value = cell_mock
    atoms.get_pbc.return_value = np.array([True, True, True])
    atoms.get_masses.return_value = np.array([12.011, 15.999, 1.008])
    atoms.numbers = np.array([6, 8, 1])
    atoms.positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    atoms.info = {}
    atoms.arrays = {
        "numbers": np.array([6, 8, 1]),
        "positions": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    }
    atoms.calc = None
    atoms.constraints = []
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


def _create_real_aselmdb(
    db_dir: pathlib.Path,
    n_files: int = 1,
    n_rows: int = 3,
    *,
    with_calc: bool = False,
) -> None:
    """Create real .aselmdb files with ASE for backend integration tests.

    Parameters
    ----------
    db_dir : pathlib.Path
        Directory to write database files.
    n_files : int
        Number of ``.aselmdb`` files to create.
    n_rows : int
        Rows per file.
    with_calc : bool
        If ``True``, attach energy and forces via SinglePointCalculator.
    """
    from ase import Atoms
    from ase.db import connect

    rng = np.random.default_rng(42)
    db_dir.mkdir(parents=True, exist_ok=True)
    for f_idx in range(n_files):
        db_path = db_dir / f"data{f_idx:04d}.aselmdb"
        db = connect(str(db_path), type="aselmdb")
        for r_idx in range(n_rows):
            n_atoms = 3 + r_idx
            positions = rng.random((n_atoms, 3)) * 10.0
            numbers = rng.choice([1, 6, 8], size=n_atoms)
            atoms = Atoms(
                numbers=numbers,
                positions=positions,
                cell=[10.0, 10.0, 10.0],
                pbc=True,
            )
            if with_calc:
                from ase.calculators.singlepoint import SinglePointCalculator

                forces = rng.random((n_atoms, 3)) - 0.5
                calc = SinglePointCalculator(atoms, energy=-100.0 * r_idx, forces=forces)
                atoms.calc = calc
            db.write(atoms, key_value_pairs={"file_idx": f_idx, "row_idx": r_idx})
        db.close()


# ---------------------------------------------------------------------------
# Unit tests — metadata and parameters
# ---------------------------------------------------------------------------


@pytest.mark.requires("atm")
class TestASELMDBSourceUnit:
    """Metadata and parameter tests (no data access)."""

    def test_params_list(self) -> None:
        from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource

        params = ASELMDBSource.params()
        assert len(params) > 0
        names = [p.name for p in params]
        assert "data_dir" in names
        assert "metadata_path" in names
        assert "backend" in names

    def test_name_and_description(self) -> None:
        from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource

        assert isinstance(ASELMDBSource.name, str)
        assert ASELMDBSource.name == "ASE LMDB"
        assert isinstance(ASELMDBSource.description, str)
        assert len(ASELMDBSource.description) > 0


# ---------------------------------------------------------------------------
# Unit tests — local mock data
# ---------------------------------------------------------------------------


@pytest.mark.requires("atm")
class TestASELMDBSourceLocal:
    """Tests against local mock data with mocked ASE database."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: pathlib.Path) -> None:
        self.mock_root = tmp_path / "mock_data"
        self.mock_root.mkdir()
        _write_mock_aselmdb_files(self.mock_root, n_files=3)
        _write_mock_metadata(self.mock_root)

    def test_len(self) -> None:
        from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.mock_root))
        assert len(source) == 3

    def test_no_aselmdb_files_raises(self, tmp_path: pathlib.Path) -> None:
        from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No .aselmdb files"):
            ASELMDBSource(data_dir=str(empty_dir))

    def test_files_sorted_lexicographically(self) -> None:
        from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.mock_root))
        names = [p.name for p in source.db_files]
        assert names == sorted(names)

    def test_metadata_loaded(self) -> None:
        from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.mock_root))
        assert "natoms" in source.metadata
        assert "data_ids" in source.metadata

    def test_metadata_not_required(self, tmp_path: pathlib.Path) -> None:
        from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource

        no_meta = tmp_path / "no_meta"
        no_meta.mkdir()
        _write_mock_aselmdb_files(no_meta, n_files=2)
        source = ASELMDBSource(data_dir=str(no_meta))
        assert len(source) == 2
        assert source.metadata == {}

    @patch("ase.db.connect")
    def test_getitem_yields_atomic_data(self, mock_connect: MagicMock) -> None:
        from nvalchemi.data import AtomicData

        from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource

        mock_connect.return_value = _make_mock_db(n_rows=3)

        source = ASELMDBSource(data_dir=str(self.mock_root))
        items = list(source[0])
        assert len(items) == 3
        for item in items:
            assert isinstance(item, AtomicData)

    @patch("ase.db.connect")
    def test_negative_index(self, mock_connect: MagicMock) -> None:
        from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource

        mock_connect.return_value = _make_mock_db(n_rows=2)

        source = ASELMDBSource(data_dir=str(self.mock_root))
        # source[-1] should be same as source[2] (last of 3 files)
        items_neg = list(source[-1])
        assert len(items_neg) == 2

    def test_index_out_of_bounds(self) -> None:
        from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.mock_root))
        with pytest.raises(IndexError):
            next(source[len(source)])

    def test_index_out_of_bounds_negative(self) -> None:
        from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.mock_root))
        with pytest.raises(IndexError):
            next(source[-(len(source) + 1)])

    def test_data_dir_property(self) -> None:
        from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.mock_root))
        assert source.data_dir == self.mock_root


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


@pytest.mark.requires("atm")
class TestASELMDBSourceRegistry:
    """Test that the source is registered."""

    def test_source_registered(self) -> None:
        import physicsnemo_curator.atm  # noqa: F401
        from physicsnemo_curator.core.registry import registry

        sources = registry.list_sources("atm")
        source_names = {s.name for s in sources}
        assert "ASE LMDB" in source_names


# ---------------------------------------------------------------------------
# E2E tests (require real val/ data)
# ---------------------------------------------------------------------------


@pytest.mark.requires("atm")
@pytest.mark.e2e
class TestASELMDBSourceE2E:
    """End-to-end tests with real ASE LMDB databases generated on the fly."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: pathlib.Path) -> None:
        self.db_dir = tmp_path / "e2e_data"
        _create_real_aselmdb(self.db_dir, n_files=3, n_rows=5, with_calc=True)

        from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource

        self.source = ASELMDBSource(data_dir=str(self.db_dir))

    def test_discovers_files(self) -> None:
        assert len(self.source) == 3

    def test_reads_first_item(self) -> None:
        from nvalchemi.data import AtomicData

        item = next(self.source[0])
        assert isinstance(item, AtomicData)
        assert item.num_nodes > 0

    def test_yields_multiple_items(self) -> None:
        items = list(self.source[0])
        assert len(items) == 5


# ---------------------------------------------------------------------------
# Backend parameter tests
# ---------------------------------------------------------------------------


@pytest.mark.requires("atm")
class TestASELMDBSourceBackend:
    """Test backend selection and fallback logic."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: pathlib.Path) -> None:
        self.mock_root = tmp_path / "mock_data"
        self.mock_root.mkdir()
        _write_mock_aselmdb_files(self.mock_root, n_files=2)

    def test_default_backend_is_python(self) -> None:
        from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.mock_root))
        assert source.backend == "python"

    def test_explicit_python_backend(self) -> None:
        from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.mock_root), backend="python")
        assert source.backend == "python"

    def test_rust_backend_accepted(self) -> None:
        from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.mock_root), backend="rust")
        assert source.backend == "rust"

    def test_rust_fallback_on_missing_extension(self) -> None:
        import sys
        from types import ModuleType

        from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource

        # Temporarily remove the lmdb submodule so the import check fails.
        saved = sys.modules.pop("physicsnemo_curator._lib.lmdb", None)
        # Replacing with a broken module triggers ImportError on
        # ``from physicsnemo_curator._lib.lmdb import read_lmdb``.
        broken: ModuleType = ModuleType("physicsnemo_curator._lib.lmdb")
        broken.__dict__.clear()  # make it empty so import-from fails
        sys.modules["physicsnemo_curator._lib.lmdb"] = broken
        try:
            source = ASELMDBSource(data_dir=str(self.mock_root), backend="rust")
            assert source.backend == "python"
        finally:
            # Restore
            if saved is not None:
                sys.modules["physicsnemo_curator._lib.lmdb"] = saved
            else:
                sys.modules.pop("physicsnemo_curator._lib.lmdb", None)

    def test_backend_in_params(self) -> None:
        from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource

        params = ASELMDBSource.params()
        backend_param = next(p for p in params if p.name == "backend")
        assert backend_param.default == "python"
        assert backend_param.choices == ["python", "rust"]


# ---------------------------------------------------------------------------
# Backend integration tests (use real ASE LMDB files)
# ---------------------------------------------------------------------------


@pytest.mark.requires("atm")
@pytest.mark.integration
class TestASELMDBSourceRustBackend:
    """Integration tests comparing Rust and Python backend output."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: pathlib.Path) -> None:
        self.db_dir = tmp_path / "dbs"
        _create_real_aselmdb(self.db_dir, n_files=2, n_rows=4, with_calc=True)

    def test_rust_reads_correct_count(self) -> None:
        from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.db_dir), backend="rust")
        items = list(source[0])
        assert len(items) == 4

    def test_rust_yields_atomic_data(self) -> None:
        from nvalchemi.data import AtomicData

        from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.db_dir), backend="rust")
        for ad in source[0]:
            assert isinstance(ad, AtomicData)
            assert ad.atomic_numbers is not None
            assert ad.positions is not None

    def test_rust_captures_energy(self) -> None:
        from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.db_dir), backend="rust")
        ad = next(source[0])
        assert ad.energies is not None
        assert ad.energies.shape == (1, 1)

    def test_rust_captures_forces(self) -> None:
        from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.db_dir), backend="rust")
        ad = next(source[0])
        assert ad.forces is not None
        n_atoms = ad.atomic_numbers.shape[0]
        assert ad.forces.shape == (n_atoms, 3)

    def test_rust_captures_cell_and_pbc(self) -> None:
        from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.db_dir), backend="rust")
        ad = next(source[0])
        assert ad.cell is not None
        assert ad.cell.shape == (1, 3, 3)
        assert ad.pbc is not None
        assert ad.pbc.shape == (1, 3)

    def test_backends_agree_on_positions(self) -> None:
        import torch

        from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource

        src_py = ASELMDBSource(data_dir=str(self.db_dir), backend="python")
        src_rs = ASELMDBSource(data_dir=str(self.db_dir), backend="rust")

        items_py = list(src_py[0])
        items_rs = list(src_rs[0])
        assert len(items_py) == len(items_rs)

        for ad_py, ad_rs in zip(items_py, items_rs, strict=True):
            assert torch.equal(ad_py.atomic_numbers, ad_rs.atomic_numbers)
            assert torch.allclose(ad_py.positions, ad_rs.positions, atol=1e-5)

    def test_backends_agree_on_cell(self) -> None:
        import torch

        from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource

        src_py = ASELMDBSource(data_dir=str(self.db_dir), backend="python")
        src_rs = ASELMDBSource(data_dir=str(self.db_dir), backend="rust")

        ad_py = next(src_py[0])
        ad_rs = next(src_rs[0])

        # Both should have cell (PBC is True)
        assert ad_py.cell is not None
        assert ad_rs.cell is not None
        assert torch.allclose(ad_py.cell, ad_rs.cell, atol=1e-5)

    def test_multiple_files(self) -> None:
        from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.db_dir), backend="rust")
        assert len(source) == 2
        items_0 = list(source[0])
        items_1 = list(source[1])
        assert len(items_0) == 4
        assert len(items_1) == 4


# ---------------------------------------------------------------------------
# _atomic_data_from_row unit tests
# ---------------------------------------------------------------------------


@pytest.mark.requires("atm")
class TestAtomicDataFromRow:
    """Unit tests for the raw dict → AtomicData conversion function."""

    def test_basic_construction(self) -> None:
        import torch

        from physicsnemo_curator.atm.sources.aselmdb import _atomic_data_from_row

        row: dict[str, object] = {
            "numbers": np.array([1, 1, 8], dtype=np.int64),
            "positions": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
            "pbc": np.array([False, False, False]),
            "cell": np.zeros((3, 3), dtype=np.float64),
            "id": 1,
            "unique_id": "abc123",
            "ctime": 0.0,
            "mtime": 0.0,
            "user": "test",
            "data": {},
            "key_value_pairs": {},
        }
        ad = _atomic_data_from_row(row)
        assert torch.equal(ad.atomic_numbers, torch.tensor([1, 1, 8]))
        assert ad.positions.shape == (3, 3)
        assert ad.cell is None  # PBC all False
        assert ad.pbc is None

    def test_energy_float(self) -> None:
        from physicsnemo_curator.atm.sources.aselmdb import _atomic_data_from_row

        row: dict[str, object] = {
            "numbers": np.array([6], dtype=np.int64),
            "positions": np.array([[0, 0, 0]], dtype=np.float64),
            "pbc": np.array([False, False, False]),
            "energy": -42.5,
            "id": 1,
            "unique_id": "a",
            "ctime": 0.0,
            "mtime": 0.0,
            "user": "t",
            "data": {},
            "key_value_pairs": {},
        }
        ad = _atomic_data_from_row(row)
        assert ad.energies is not None
        assert ad.energies.shape == (1, 1)
        assert abs(float(ad.energies.item()) - (-42.5)) < 1e-3

    def test_forces_and_stress(self) -> None:
        from physicsnemo_curator.atm.sources.aselmdb import _atomic_data_from_row

        row: dict[str, object] = {
            "numbers": np.array([1, 8], dtype=np.int64),
            "positions": np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64),
            "pbc": np.array([True, True, True]),
            "cell": np.eye(3, dtype=np.float64) * 10.0,
            "forces": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64),
            "stress": np.array([1, 2, 3, 4, 5, 6], dtype=np.float64),
            "id": 1,
            "unique_id": "a",
            "ctime": 0.0,
            "mtime": 0.0,
            "user": "t",
            "data": {},
            "key_value_pairs": {},
        }
        ad = _atomic_data_from_row(row)
        assert ad.forces is not None
        assert ad.forces.shape == (2, 3)
        assert ad.stresses is not None
        assert ad.stresses.shape == (1, 3, 3)

    def test_voigt_stress_conversion(self) -> None:

        from physicsnemo_curator.atm.sources.aselmdb import _voigt_to_matrix

        voigt = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        mat = _voigt_to_matrix(voigt)
        assert mat.shape == (3, 3)
        # Check symmetry
        assert np.allclose(mat, mat.T)
        # Check specific entries
        assert mat[0, 0] == 1.0  # xx
        assert mat[1, 1] == 2.0  # yy
        assert mat[2, 2] == 3.0  # zz
        assert mat[1, 2] == 4.0  # yz
        assert mat[0, 2] == 5.0  # xz
        assert mat[0, 1] == 6.0  # xy

    def test_missing_numbers_raises(self) -> None:
        from physicsnemo_curator.atm.sources.aselmdb import _atomic_data_from_row

        row: dict[str, object] = {
            "numbers": "not_an_array",
            "positions": np.array([[0, 0, 0]], dtype=np.float64),
            "pbc": np.array([False, False, False]),
            "id": 1,
            "unique_id": "a",
            "ctime": 0.0,
            "mtime": 0.0,
            "user": "t",
            "data": {},
            "key_value_pairs": {},
        }
        with pytest.raises(TypeError, match="Expected 'numbers' to be ndarray"):
            _atomic_data_from_row(row)
