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

"""Consistency tests comparing Rust LMDB reader against Python ASE reader.

The Rust reader (``physicsnemo.curator._lib.lmdb.read_lmdb``) returns raw
row dictionaries with NumPy arrays for ``__ndarray__`` markers.  The Python
ASE path (``ase.db.connect`` → ``row.toatoms()``) deserialises the same
underlying zlib-compressed JSON but exposes it through an ``AtomsRow`` object.

These tests verify that the Rust reader produces byte-identical NumPy arrays
for the core atomic fields (positions, numbers, cell, pbc) and structurally
equivalent metadata, so that downstream ``AtomicData.from_atoms`` receives
consistent inputs regardless of which reader is used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

if TYPE_CHECKING:
    import pathlib

pytestmark = pytest.mark.requires("atm")


# ---------------------------------------------------------------------------
# Fixtures — write real .aselmdb files via ASE
# ---------------------------------------------------------------------------


@pytest.fixture
def single_water_db(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create an .aselmdb with a single water molecule."""
    from ase import Atoms
    from ase.db import connect

    db_path = tmp_path / "water.aselmdb"
    atoms = Atoms("H2O", positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]])
    atoms.cell = [10.0, 10.0, 10.0]
    atoms.pbc = True

    db = connect(str(db_path), type="aselmdb")
    db.write(atoms, key_value_pairs={"label": "water"})
    db.close()
    return db_path


@pytest.fixture
def multi_row_db(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create an .aselmdb with several diverse rows."""
    from ase import Atoms
    from ase.db import connect

    db_path = tmp_path / "multi.aselmdb"
    db = connect(str(db_path), type="aselmdb")

    # Row 1: water molecule (3 atoms, cubic cell, full PBC)
    water = Atoms("H2O", positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]])
    water.cell = [10.0, 10.0, 10.0]
    water.pbc = True
    db.write(water, key_value_pairs={"label": "water"})

    # Row 2: CO molecule (2 atoms, no PBC)
    co = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.13, 0.0, 0.0]])
    db.write(co)

    # Row 3: larger system — FCC aluminium (4 atoms, triclinic cell)
    al = Atoms(
        "Al4",
        positions=[
            [0.0, 0.0, 0.0],
            [2.025, 2.025, 0.0],
            [2.025, 0.0, 2.025],
            [0.0, 2.025, 2.025],
        ],
        cell=[[4.05, 0.0, 0.0], [0.0, 4.05, 0.0], [0.0, 0.0, 4.05]],
        pbc=[True, True, True],
    )
    db.write(al)

    db.close()
    return db_path


def _read_with_ase(db_path: pathlib.Path) -> list[dict[str, object]]:
    """Read all rows from an .aselmdb via the Python ASE reader.

    Returns a list of dicts with the raw AtomsRow fields so we can compare
    against the Rust reader output without going through ``AtomicData``.
    """
    from ase.db import connect

    db = connect(str(db_path), type="aselmdb", readonly=True)
    rows: list[dict[str, object]] = []
    for row in db.select():
        rows.append(
            {
                "id": row.id,
                "positions": row.positions,
                "numbers": row.numbers,
                "cell": row.cell,
                "pbc": row.pbc,
                "unique_id": row.unique_id,
                "user": row.user,
            }
        )
    db.close()
    return rows


def _read_with_rust(db_path: pathlib.Path) -> list[dict[str, object]]:
    """Read all rows from an .aselmdb via the Rust reader."""
    from physicsnemo.curator._lib.lmdb import read_lmdb

    return read_lmdb(str(db_path))


# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------


class TestRustLmdbReader:
    """Smoke tests for the Rust LMDB reader."""

    def test_import(self) -> None:
        """Rust LMDB functions are importable."""
        from physicsnemo.curator._lib.lmdb import read_lmdb, read_lmdb_parallel

        assert callable(read_lmdb)
        assert callable(read_lmdb_parallel)

    def test_read_single_row(self, single_water_db: pathlib.Path) -> None:
        """Rust reader returns one row for a single-row database."""
        rows = _read_with_rust(single_water_db)
        assert len(rows) == 1
        assert rows[0]["id"] == 1

    def test_read_multi_row(self, multi_row_db: pathlib.Path) -> None:
        """Rust reader returns all rows sorted by ID."""
        rows = _read_with_rust(multi_row_db)
        assert len(rows) == 3
        assert [r["id"] for r in rows] == [1, 2, 3]

    def test_parallel_read(self, single_water_db: pathlib.Path, multi_row_db: pathlib.Path) -> None:
        """Parallel reader returns correct results for multiple files."""
        from physicsnemo.curator._lib.lmdb import read_lmdb_parallel

        results = read_lmdb_parallel([str(single_water_db), str(multi_row_db)])
        assert len(results) == 2
        assert len(results[0]) == 1
        assert len(results[1]) == 3

    def test_ndarray_fields_are_numpy(self, single_water_db: pathlib.Path) -> None:
        """Fields that were __ndarray__ in JSON come back as NumPy arrays."""
        rows = _read_with_rust(single_water_db)
        row = rows[0]
        assert isinstance(row["positions"], np.ndarray)
        assert isinstance(row["numbers"], np.ndarray)
        assert isinstance(row["cell"], np.ndarray)
        assert isinstance(row["pbc"], np.ndarray)

    def test_file_not_found(self) -> None:
        """Rust reader raises on missing file."""
        from physicsnemo.curator._lib.lmdb import read_lmdb

        with pytest.raises(IOError):
            read_lmdb("/nonexistent/path.aselmdb")


# ---------------------------------------------------------------------------
# Rust-vs-Python consistency tests
# ---------------------------------------------------------------------------


class TestRustVsPythonConsistency:
    """Verify that Rust and Python readers produce equivalent outputs.

    The Rust reader returns the raw JSON-level dict (``key_value_pairs``
    remains a nested dict), while the Python ASE reader unpacks key-value
    pairs to the top level.  These tests compare the core array fields
    that must be identical for ``AtomicData.from_atoms`` compatibility.
    """

    def test_row_count_matches(self, multi_row_db: pathlib.Path) -> None:
        """Both readers return the same number of rows."""
        ase_rows = _read_with_ase(multi_row_db)
        rust_rows = _read_with_rust(multi_row_db)
        assert len(rust_rows) == len(ase_rows)

    def test_ids_match(self, multi_row_db: pathlib.Path) -> None:
        """Row IDs are identical and in the same order."""
        ase_rows = _read_with_ase(multi_row_db)
        rust_rows = _read_with_rust(multi_row_db)
        assert [r["id"] for r in rust_rows] == [r["id"] for r in ase_rows]

    def test_positions_match(self, multi_row_db: pathlib.Path) -> None:
        """Atomic positions are bit-identical."""
        ase_rows = _read_with_ase(multi_row_db)
        rust_rows = _read_with_rust(multi_row_db)
        for ase_row, rust_row in zip(ase_rows, rust_rows, strict=True):
            np.testing.assert_array_equal(
                rust_row["positions"],
                ase_row["positions"],
                err_msg=f"positions mismatch for id={ase_row['id']}",
            )

    def test_positions_shape(self, multi_row_db: pathlib.Path) -> None:
        """Positions shape is (N, 3) for both readers."""
        ase_rows = _read_with_ase(multi_row_db)
        rust_rows = _read_with_rust(multi_row_db)
        for ase_row, rust_row in zip(ase_rows, rust_rows, strict=True):
            assert rust_row["positions"].shape == ase_row["positions"].shape

    def test_numbers_match(self, multi_row_db: pathlib.Path) -> None:
        """Atomic numbers are identical."""
        ase_rows = _read_with_ase(multi_row_db)
        rust_rows = _read_with_rust(multi_row_db)
        for ase_row, rust_row in zip(ase_rows, rust_rows, strict=True):
            np.testing.assert_array_equal(
                rust_row["numbers"],
                ase_row["numbers"],
                err_msg=f"numbers mismatch for id={ase_row['id']}",
            )

    def test_cell_match(self, multi_row_db: pathlib.Path) -> None:
        """Unit cell arrays are bit-identical."""
        ase_rows = _read_with_ase(multi_row_db)
        rust_rows = _read_with_rust(multi_row_db)
        for ase_row, rust_row in zip(ase_rows, rust_rows, strict=True):
            np.testing.assert_array_equal(
                rust_row["cell"],
                np.array(ase_row["cell"]),
                err_msg=f"cell mismatch for id={ase_row['id']}",
            )

    def test_pbc_match(self, multi_row_db: pathlib.Path) -> None:
        """Periodic boundary conditions are identical."""
        ase_rows = _read_with_ase(multi_row_db)
        rust_rows = _read_with_rust(multi_row_db)
        for ase_row, rust_row in zip(ase_rows, rust_rows, strict=True):
            np.testing.assert_array_equal(
                rust_row["pbc"],
                ase_row["pbc"],
                err_msg=f"pbc mismatch for id={ase_row['id']}",
            )

    def test_unique_id_match(self, multi_row_db: pathlib.Path) -> None:
        """Unique IDs (hex strings) are identical."""
        ase_rows = _read_with_ase(multi_row_db)
        rust_rows = _read_with_rust(multi_row_db)
        for ase_row, rust_row in zip(ase_rows, rust_rows, strict=True):
            assert rust_row["unique_id"] == ase_row["unique_id"]

    def test_dtypes_match(self, multi_row_db: pathlib.Path) -> None:
        """NumPy dtypes for core fields are consistent."""
        ase_rows = _read_with_ase(multi_row_db)
        rust_rows = _read_with_rust(multi_row_db)
        for ase_row, rust_row in zip(ase_rows, rust_rows, strict=True):
            assert rust_row["positions"].dtype == ase_row["positions"].dtype
            assert rust_row["numbers"].dtype == ase_row["numbers"].dtype


class TestRustVsPythonSingleRow:
    """Detailed comparison on a single-row database for simpler debugging."""

    def test_water_positions(self, single_water_db: pathlib.Path) -> None:
        """Water positions match between readers."""
        ase_rows = _read_with_ase(single_water_db)
        rust_rows = _read_with_rust(single_water_db)
        np.testing.assert_array_equal(rust_rows[0]["positions"], ase_rows[0]["positions"])

    def test_water_numbers(self, single_water_db: pathlib.Path) -> None:
        """Water atomic numbers match between readers."""
        ase_rows = _read_with_ase(single_water_db)
        rust_rows = _read_with_rust(single_water_db)
        np.testing.assert_array_equal(rust_rows[0]["numbers"], ase_rows[0]["numbers"])

    def test_water_cell(self, single_water_db: pathlib.Path) -> None:
        """Water unit cell matches between readers."""
        ase_rows = _read_with_ase(single_water_db)
        rust_rows = _read_with_rust(single_water_db)
        np.testing.assert_array_equal(rust_rows[0]["cell"], np.array(ase_rows[0]["cell"]))

    def test_water_pbc(self, single_water_db: pathlib.Path) -> None:
        """Water periodic boundary conditions match between readers."""
        ase_rows = _read_with_ase(single_water_db)
        rust_rows = _read_with_rust(single_water_db)
        np.testing.assert_array_equal(rust_rows[0]["pbc"], ase_rows[0]["pbc"])

    def test_water_key_value_pairs(self, single_water_db: pathlib.Path) -> None:
        """Key-value pairs are preserved (Rust keeps them nested)."""
        rust_rows = _read_with_rust(single_water_db)
        kvp: object = rust_rows[0]["key_value_pairs"]
        assert isinstance(kvp, dict)
        # kvp narrowed to dict[Unknown, Unknown]; use cast for subscript.
        kvp_typed = cast("dict[str, object]", kvp)
        assert kvp_typed["label"] == "water"


# ---------------------------------------------------------------------------
# E2E tests (require real val/ data)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.slow
class TestRustVsPythonE2E:
    """End-to-end consistency against the val/ dataset."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        """Locate the val/ directory."""
        import pathlib

        val_dir = pathlib.Path("val")
        if not val_dir.is_dir():
            pytest.skip("val/ directory not found")
        files = sorted(val_dir.glob("*.aselmdb"))
        if not files:
            pytest.skip("No .aselmdb files in val/")
        self.db_path = files[0]

    def test_first_file_row_count(self) -> None:
        """Both readers report the same number of rows for the first val/ file."""
        ase_rows = _read_with_ase(self.db_path)
        rust_rows = _read_with_rust(self.db_path)
        assert len(rust_rows) == len(ase_rows)

    def test_first_file_positions(self) -> None:
        """Positions are bit-identical for every row in the first val/ file."""
        ase_rows = _read_with_ase(self.db_path)
        rust_rows = _read_with_rust(self.db_path)
        for ase_row, rust_row in zip(ase_rows, rust_rows, strict=True):
            np.testing.assert_array_equal(rust_row["positions"], ase_row["positions"])

    def test_first_file_numbers(self) -> None:
        """Atomic numbers are identical for every row in the first val/ file."""
        ase_rows = _read_with_ase(self.db_path)
        rust_rows = _read_with_rust(self.db_path)
        for ase_row, rust_row in zip(ase_rows, rust_rows, strict=True):
            np.testing.assert_array_equal(rust_row["numbers"], ase_row["numbers"])
