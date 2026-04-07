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

"""ASV benchmarks comparing Rust LMDB reader vs Python ASE reader.

These benchmarks measure:
- Single database read performance (Rust vs ASE)
- Parallel multi-database read performance
- Scaling behaviour with database size (number of rows)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

# ── helpers ──────────────────────────────────────────────────────────────────


def _create_aselmdb(db_path: Path, n_rows: int) -> None:
    """Create an .aselmdb file with *n_rows* water-like molecules.

    Each row contains a 3-atom system (H₂O) with positions, a cubic
    unit cell, periodic boundary conditions, and a ``row_index``
    key-value pair so rows can be identified downstream.
    """
    import numpy as np
    from ase import Atoms
    from ase.db import connect

    db = connect(str(db_path), type="aselmdb")
    rng = np.random.default_rng(42)
    for i in range(n_rows):
        positions = rng.random((3, 3)) * 10.0
        atoms = Atoms("H2O", positions=positions, cell=[10.0, 10.0, 10.0], pbc=True)
        db.write(atoms, key_value_pairs={"row_index": i})
    db.close()


# ── single-file benchmarks ──────────────────────────────────────────────────


class TimeLmdbReaderSingle:
    """Benchmark single-file LMDB reading: Rust vs ASE."""

    params = [10, 100, 500]
    param_names = ["n_rows"]

    def setup(self, n_rows: int) -> None:
        """Create a test .aselmdb database."""
        self._tmpdir = tempfile.mkdtemp()
        self.db_file = str(Path(self._tmpdir) / "bench.aselmdb")
        _create_aselmdb(Path(self.db_file), n_rows)

    def teardown(self, n_rows: int) -> None:
        """Clean up test files."""
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def time_ase(self, n_rows: int) -> None:
        """Benchmark Python ASE reading a single database."""
        from ase.db import connect

        db = connect(self.db_file, type="aselmdb", readonly=True)
        for row in db.select():
            _ = row.toatoms()
        db.close()

    def time_rust(self, n_rows: int) -> None:
        """Benchmark Rust reader reading a single database."""
        from physicsnemo_curator._lib.lmdb import read_lmdb

        _ = read_lmdb(self.db_file)


# ── parallel multi-file benchmarks ──────────────────────────────────────────


class TimeLmdbReaderParallel:
    """Benchmark parallel LMDB reading: Rust parallel vs ASE sequential."""

    params = [1, 4, 8]
    param_names = ["n_files"]

    def setup(self, n_files: int) -> None:
        """Create multiple .aselmdb databases (100 rows each)."""
        self._tmpdir = tempfile.mkdtemp()
        self.db_files: list[str] = []
        for i in range(n_files):
            path = Path(self._tmpdir) / f"bench_{i}.aselmdb"
            _create_aselmdb(path, n_rows=100)
            self.db_files.append(str(path))

    def teardown(self, n_files: int) -> None:
        """Clean up test files."""
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def time_ase_sequential(self, n_files: int) -> None:
        """Benchmark ASE reading multiple databases sequentially."""
        from ase.db import connect

        for f in self.db_files:
            db = connect(f, type="aselmdb", readonly=True)
            for row in db.select():
                _ = row.toatoms()
            db.close()

    def time_rust_parallel(self, n_files: int) -> None:
        """Benchmark Rust reader reading multiple databases in parallel."""
        from physicsnemo_curator._lib.lmdb import read_lmdb_parallel

        _ = read_lmdb_parallel(self.db_files)

    def time_rust_sequential(self, n_files: int) -> None:
        """Benchmark Rust reader reading multiple databases sequentially."""
        from physicsnemo_curator._lib.lmdb import read_lmdb

        for f in self.db_files:
            _ = read_lmdb(f)


# ── memory benchmarks ───────────────────────────────────────────────────────


class MemLmdbReader:
    """Memory footprint of LMDB readers."""

    params = [100, 500]
    param_names = ["n_rows"]

    def setup(self, n_rows: int) -> None:
        """Create a test .aselmdb database for memory benchmarking."""
        self._tmpdir = tempfile.mkdtemp()
        self.db_file = str(Path(self._tmpdir) / "bench.aselmdb")
        _create_aselmdb(Path(self.db_file), n_rows)

    def teardown(self, n_rows: int) -> None:
        """Clean up test files."""
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def mem_rust(self, n_rows: int):
        """Track memory of Rust LMDB reader."""
        from physicsnemo_curator._lib.lmdb import read_lmdb

        return read_lmdb(self.db_file)
