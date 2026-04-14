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

Unit tests use real ASE LMDB databases generated via ``_write_mock_aselmdb_files``.
E2E tests generate real ASE LMDB databases on the fly via ``_create_real_aselmdb``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import pathlib


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _write_mock_aselmdb_files(root: pathlib.Path, n_files: int = 3, n_rows_per_file: int = 5) -> None:
    """Create real .aselmdb files for unit testing.

    Unlike the old approach of writing empty mock files, we now need real
    LMDB files because the source counts rows at construction time.
    """
    from ase import Atoms

    from physicsnemo_curator.domains.atm.sources.aselmdb import (
        _atoms_to_row_dict,
        _write_aselmdb,
    )

    rng = np.random.default_rng(42)
    for i in range(n_files):
        db_path = root / f"data{i:04d}.aselmdb"
        rows: list[dict[str, object]] = []
        for r_idx in range(n_rows_per_file):
            atoms = Atoms(
                numbers=[6, 8],
                positions=rng.random((2, 3)) * 10.0,
                cell=[10.0, 10.0, 10.0],
                pbc=True,
            )
            row = _atoms_to_row_dict(atoms, row_id=r_idx + 1)
            rows.append(row)
        _write_aselmdb(db_path, rows)


def _write_mock_metadata(root: pathlib.Path, n_atoms: int = 100) -> None:
    """Write a minimal metadata.npz to *root*."""
    np.savez(
        root / "metadata.npz",
        natoms=np.array([n_atoms, n_atoms + 1]),
        data_ids=np.array([0, 1]),
    )


def _create_real_aselmdb(
    db_dir: pathlib.Path,
    n_files: int = 1,
    n_rows: int = 3,
    *,
    with_calc: bool = False,
) -> None:
    """Create real .aselmdb files for backend integration tests.

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

    from physicsnemo_curator.domains.atm.sources.aselmdb import (
        _atoms_to_row_dict,
        _write_aselmdb,
    )

    rng = np.random.default_rng(42)
    db_dir.mkdir(parents=True, exist_ok=True)
    for f_idx in range(n_files):
        db_path = db_dir / f"data{f_idx:04d}.aselmdb"
        rows: list[dict[str, object]] = []
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
            row = _atoms_to_row_dict(atoms, row_id=r_idx + 1, key_value_pairs={"file_idx": f_idx, "row_idx": r_idx})
            rows.append(row)
        _write_aselmdb(db_path, rows)


# ---------------------------------------------------------------------------
# Unit tests — metadata and parameters
# ---------------------------------------------------------------------------


@pytest.mark.requires("atm")
class TestASELMDBSourceUnit:
    """Metadata and parameter tests (no data access)."""

    def test_params_list(self) -> None:
        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

        params = ASELMDBSource.params()
        assert len(params) > 0
        names = [p.name for p in params]
        assert "data_dir" in names
        assert "metadata_path" in names
        assert "backend" in names

    def test_name_and_description(self) -> None:
        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

        assert isinstance(ASELMDBSource.name, str)
        assert ASELMDBSource.name == "ASE LMDB"
        assert isinstance(ASELMDBSource.description, str)
        assert len(ASELMDBSource.description) > 0


# ---------------------------------------------------------------------------
# Unit tests — local mock data
# ---------------------------------------------------------------------------


@pytest.mark.requires("atm")
class TestASELMDBSourceLocal:
    """Tests against local data with real LMDB files."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: pathlib.Path) -> None:
        self.mock_root = tmp_path / "mock_data"
        self.mock_root.mkdir()
        # 3 files with 5 rows each = 15 total structures
        _write_mock_aselmdb_files(self.mock_root, n_files=3, n_rows_per_file=5)
        _write_mock_metadata(self.mock_root)

    def test_len(self) -> None:
        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.mock_root))
        # 3 files * 5 rows each = 15 structures
        assert len(source) == 15

    def test_num_files(self) -> None:
        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.mock_root))
        assert source.num_files == 3

    def test_row_counts(self) -> None:
        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.mock_root))
        assert source.row_counts == [5, 5, 5]

    def test_no_aselmdb_files_raises(self, tmp_path: pathlib.Path) -> None:
        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No .aselmdb files"):
            ASELMDBSource(data_dir=str(empty_dir))

    def test_files_sorted_lexicographically(self) -> None:
        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.mock_root))
        names = [p.name for p in source.db_files]
        assert names == sorted(names)

    def test_metadata_loaded(self) -> None:
        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.mock_root))
        assert "natoms" in source.metadata
        assert "data_ids" in source.metadata

    def test_metadata_not_required(self, tmp_path: pathlib.Path) -> None:
        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

        no_meta = tmp_path / "no_meta"
        no_meta.mkdir()
        _write_mock_aselmdb_files(no_meta, n_files=2, n_rows_per_file=3)
        source = ASELMDBSource(data_dir=str(no_meta))
        # 2 files * 3 rows = 6 structures
        assert len(source) == 6
        assert source.metadata == {}

    def test_getitem_yields_single_atomic_data(self) -> None:
        from nvalchemi.data import AtomicData

        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.mock_root))
        # Get structure at index 0
        items = list(source[0])
        assert len(items) == 1
        assert isinstance(items[0], AtomicData)

    def test_getitem_spans_files(self) -> None:
        from nvalchemi.data import AtomicData

        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.mock_root))
        # Index 0-4 are in file 0, 5-9 in file 1, 10-14 in file 2
        # Get structure at index 5 (first in second file)
        items = list(source[5])
        assert len(items) == 1
        assert isinstance(items[0], AtomicData)

        # Get structure at index 14 (last in third file)
        items = list(source[14])
        assert len(items) == 1
        assert isinstance(items[0], AtomicData)

    def test_negative_index(self) -> None:
        from nvalchemi.data import AtomicData

        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.mock_root))
        # source[-1] should be same as source[14] (last of 15 structures)
        items_neg = list(source[-1])
        assert len(items_neg) == 1
        assert isinstance(items_neg[0], AtomicData)

    def test_index_out_of_bounds(self) -> None:
        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.mock_root))
        with pytest.raises(IndexError):
            next(source[len(source)])

    def test_index_out_of_bounds_negative(self) -> None:
        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.mock_root))
        with pytest.raises(IndexError):
            next(source[-(len(source) + 1)])

    def test_data_dir_property(self) -> None:
        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.mock_root))
        assert source.data_dir == self.mock_root

    def test_root_property(self) -> None:
        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.mock_root))
        assert source.root == self.mock_root.resolve()

    def test_relative_path(self) -> None:
        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.mock_root))
        # Index 0 is in file 0
        rel = source.relative_path(0)
        assert rel == "data0000.aselmdb"
        # Index 5 is in file 1
        rel = source.relative_path(5)
        assert rel == "data0001.aselmdb"

    def test_relative_path_nested(self, tmp_path: pathlib.Path) -> None:
        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

        nested_root = tmp_path / "nested"
        nested_root.mkdir()
        sub = nested_root / "split_a"
        sub.mkdir()
        _write_mock_aselmdb_files(sub, n_files=2, n_rows_per_file=3)
        # Rename files to match test expectations
        (sub / "data0000.aselmdb").rename(sub / "run01.aselmdb")
        (sub / "data0001.aselmdb").rename(sub / "run02.aselmdb")

        source = ASELMDBSource(data_dir=str(nested_root), file_pattern="**/*.aselmdb")
        # 2 files * 3 rows = 6 structures
        assert len(source) == 6
        rel = source.relative_path(0)
        assert rel == "split_a/run01.aselmdb"

    def test_file_pattern_flat(self, tmp_path: pathlib.Path) -> None:
        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

        flat_root = tmp_path / "flat"
        flat_root.mkdir()
        _write_mock_aselmdb_files(flat_root, n_files=1, n_rows_per_file=2)
        (flat_root / "data0000.aselmdb").rename(flat_root / "a.aselmdb")

        sub = flat_root / "nested"
        sub.mkdir()
        _write_mock_aselmdb_files(sub, n_files=1, n_rows_per_file=2)
        (sub / "data0000.aselmdb").rename(sub / "b.aselmdb")

        # Default pattern finds both = 4 structures
        source_all = ASELMDBSource(data_dir=str(flat_root))
        assert len(source_all) == 4

        # Flat pattern finds only top-level = 2 structures
        source_flat = ASELMDBSource(data_dir=str(flat_root), file_pattern="*.aselmdb")
        assert len(source_flat) == 2
        assert source_flat.relative_path(0) == "a.aselmdb"

    def test_file_pattern_param(self) -> None:
        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

        params = ASELMDBSource.params()
        names = [p.name for p in params]
        assert "file_pattern" in names


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


@pytest.mark.requires("atm")
class TestASELMDBSourceRegistry:
    """Test that the source is registered."""

    def test_source_registered(self) -> None:
        import physicsnemo_curator.domains.atm  # noqa: F401
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

        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

        self.source = ASELMDBSource(data_dir=str(self.db_dir))

    def test_discovers_files(self) -> None:
        # 3 files with 5 rows each = 15 total structures
        assert len(self.source) == 15
        assert self.source.num_files == 3

    def test_reads_first_item(self) -> None:
        from nvalchemi.data import AtomicData

        item = next(self.source[0])
        assert isinstance(item, AtomicData)
        assert item.num_nodes > 0

    def test_yields_single_item(self) -> None:
        # Each index should yield exactly one structure
        items = list(self.source[0])
        assert len(items) == 1

    def test_iterate_all_structures(self) -> None:
        from nvalchemi.data import AtomicData

        # Iterate all 15 structures
        for i in range(len(self.source)):
            items = list(self.source[i])
            assert len(items) == 1
            assert isinstance(items[0], AtomicData)


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
        _write_mock_aselmdb_files(self.mock_root, n_files=2, n_rows_per_file=3)

    def test_default_backend_is_python(self) -> None:
        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.mock_root))
        assert source.backend == "python"

    def test_explicit_python_backend(self) -> None:
        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.mock_root), backend="python")
        assert source.backend == "python"

    def test_rust_backend_accepted(self) -> None:
        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.mock_root), backend="rust")
        assert source.backend == "rust"

    def test_rust_fallback_on_missing_extension(self) -> None:
        import sys
        from types import ModuleType

        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

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
        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

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
        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.db_dir), backend="rust")
        # 2 files * 4 rows = 8 total structures
        assert len(source) == 8
        assert source.num_files == 2

    def test_rust_yields_single_atomic_data(self) -> None:
        from nvalchemi.data import AtomicData

        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.db_dir), backend="rust")
        items = list(source[0])
        assert len(items) == 1
        assert isinstance(items[0], AtomicData)
        assert items[0].atomic_numbers is not None
        assert items[0].positions is not None

    def test_rust_captures_energy(self) -> None:
        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.db_dir), backend="rust")
        ad = next(source[0])
        assert ad.energies is not None
        assert ad.energies.shape == (1, 1)

    def test_rust_captures_forces(self) -> None:
        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.db_dir), backend="rust")
        ad = next(source[0])
        assert ad.forces is not None
        n_atoms = ad.atomic_numbers.shape[0]
        assert ad.forces.shape == (n_atoms, 3)

    def test_rust_captures_cell_and_pbc(self) -> None:
        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.db_dir), backend="rust")
        ad = next(source[0])
        assert ad.cell is not None
        assert ad.cell.shape == (1, 3, 3)
        assert ad.pbc is not None
        assert ad.pbc.shape == (1, 3)

    def test_backends_agree_on_positions(self) -> None:
        import torch

        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

        src_py = ASELMDBSource(data_dir=str(self.db_dir), backend="python")
        src_rs = ASELMDBSource(data_dir=str(self.db_dir), backend="rust")

        # Compare first few structures
        for i in range(min(4, len(src_py))):
            ad_py = next(src_py[i])
            ad_rs = next(src_rs[i])
            assert torch.equal(ad_py.atomic_numbers, ad_rs.atomic_numbers)
            assert torch.allclose(ad_py.positions, ad_rs.positions, atol=1e-5)

    def test_backends_agree_on_cell(self) -> None:
        import torch

        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

        src_py = ASELMDBSource(data_dir=str(self.db_dir), backend="python")
        src_rs = ASELMDBSource(data_dir=str(self.db_dir), backend="rust")

        ad_py = next(src_py[0])
        ad_rs = next(src_rs[0])

        # Both should have cell (PBC is True)
        assert ad_py.cell is not None
        assert ad_rs.cell is not None
        assert torch.allclose(ad_py.cell, ad_rs.cell, atol=1e-5)

    def test_multiple_files(self) -> None:
        from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

        source = ASELMDBSource(data_dir=str(self.db_dir), backend="rust")
        # 2 files * 4 rows = 8 total structures
        assert len(source) == 8
        assert source.num_files == 2
        # Access structures from both files
        _ = next(source[0])  # file 0, row 0
        _ = next(source[4])  # file 1, row 0


# ---------------------------------------------------------------------------
# _atomic_data_from_row unit tests
# ---------------------------------------------------------------------------


@pytest.mark.requires("atm")
class TestAtomicDataFromRow:
    """Unit tests for the raw dict → AtomicData conversion function."""

    def test_basic_construction(self) -> None:
        import torch

        from physicsnemo_curator.domains.atm.sources.aselmdb import _atomic_data_from_row

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
        from physicsnemo_curator.domains.atm.sources.aselmdb import _atomic_data_from_row

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
        from physicsnemo_curator.domains.atm.sources.aselmdb import _atomic_data_from_row

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

        from physicsnemo_curator.domains.atm.sources.aselmdb import _voigt_to_matrix

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
        from physicsnemo_curator.domains.atm.sources.aselmdb import _atomic_data_from_row

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
