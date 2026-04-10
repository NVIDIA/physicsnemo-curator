# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""ASV benchmarks for the ATM (Atomistic) domain.

Covers end-to-end pipeline performance, Rust-vs-Python LMDB reader comparison,
ASELMDBSource backend comparison, filter throughput, and sink write time.
"""

from __future__ import annotations

from pathlib import Path

from ._helpers import cleanup_temp_dir, create_synthetic_aselmdb, create_temp_dir


# ---------------------------------------------------------------------------
# E2E: Full ATM pipeline
# ---------------------------------------------------------------------------
class TimeATME2E:
    """End-to-end ATM pipeline: ASELMDBSource -> AtomicStatsFilter -> AtomicDataZarrSink."""

    params = [50, 200]
    param_names = ["n_rows"]

    def setup(self, n_rows):
        """Create synthetic ASELMDB and build the pipeline."""
        from physicsnemo_curator.atm.filters.stats import AtomicStatsFilter
        from physicsnemo_curator.atm.sinks.zarr_writer import AtomicDataZarrSink
        from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource
        from physicsnemo_curator.core.base import Pipeline

        self._data_dir = create_temp_dir()
        self._output_dir = create_temp_dir()
        self._stats_dir = create_temp_dir()

        db_path = Path(self._data_dir) / "data0000.aselmdb"
        create_synthetic_aselmdb(db_path, n_rows, with_calc=True)

        source = ASELMDBSource(data_dir=self._data_dir, backend="python")
        stats = AtomicStatsFilter(output=str(Path(self._stats_dir) / "stats.parquet"))
        sink = AtomicDataZarrSink(output_path=str(Path(self._output_dir) / "out.zarr"))

        self.pipeline = Pipeline(
            source=source,
            filters=[stats],  # ty: ignore[invalid-argument-type]
            sink=sink,
            track_metrics=False,
            track_memory=False,
        )

    def time_e2e(self, n_rows):
        """Run the full pipeline for all indices."""
        for i in range(len(self.pipeline)):
            self.pipeline[i]

    def teardown(self, n_rows):
        """Remove temporary directories."""
        cleanup_temp_dir(self._data_dir)
        cleanup_temp_dir(self._output_dir)
        cleanup_temp_dir(self._stats_dir)


class MemATME2E:
    """Peak memory for full ATM pipeline."""

    params = [50, 200]
    param_names = ["n_rows"]

    def setup(self, n_rows):
        """Create synthetic ASELMDB and build the pipeline."""
        from physicsnemo_curator.atm.filters.stats import AtomicStatsFilter
        from physicsnemo_curator.atm.sinks.zarr_writer import AtomicDataZarrSink
        from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource
        from physicsnemo_curator.core.base import Pipeline

        self._data_dir = create_temp_dir()
        self._output_dir = create_temp_dir()
        self._stats_dir = create_temp_dir()

        db_path = Path(self._data_dir) / "data0000.aselmdb"
        create_synthetic_aselmdb(db_path, n_rows, with_calc=True)

        source = ASELMDBSource(data_dir=self._data_dir, backend="python")
        stats = AtomicStatsFilter(output=str(Path(self._stats_dir) / "stats.parquet"))
        sink = AtomicDataZarrSink(output_path=str(Path(self._output_dir) / "out.zarr"))

        self.pipeline = Pipeline(
            source=source,
            filters=[stats],  # ty: ignore[invalid-argument-type]
            sink=sink,
            track_metrics=False,
            track_memory=False,
        )

    def peakmem_e2e(self, n_rows):
        """Run full pipeline, tracking peak RSS."""
        for i in range(len(self.pipeline)):
            self.pipeline[i]

    def teardown(self, n_rows):
        """Remove temporary directories."""
        cleanup_temp_dir(self._data_dir)
        cleanup_temp_dir(self._output_dir)
        cleanup_temp_dir(self._stats_dir)


# ---------------------------------------------------------------------------
# Component: ASELMDBSource backend comparison (Rust vs Python)
# ---------------------------------------------------------------------------
class TimeASELMDBSourceBackend:
    """Compare ASELMDBSource rust vs python backend on single-file reads."""

    params = [50, 200, 500]
    param_names = ["n_rows"]

    def setup(self, n_rows):
        """Create a single ASELMDB file and both source variants."""
        from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource

        self._data_dir = create_temp_dir()
        db_path = Path(self._data_dir) / "data0000.aselmdb"
        create_synthetic_aselmdb(db_path, n_rows, with_calc=True)

        self.source_python = ASELMDBSource(data_dir=self._data_dir, backend="python")
        self.source_rust = ASELMDBSource(data_dir=self._data_dir, backend="rust")

    def time_python(self, n_rows):
        """Read via Python (ASE) backend."""
        for _ in self.source_python[0]:
            pass

    def time_rust(self, n_rows):
        """Read via Rust LMDB backend."""
        for _ in self.source_rust[0]:
            pass

    def teardown(self, n_rows):
        """Remove temporary directory."""
        cleanup_temp_dir(self._data_dir)


# ---------------------------------------------------------------------------
# Component: Raw LMDB reader comparison (Rust vs ASE)
# ---------------------------------------------------------------------------
class TimeLmdbReaderComparison:
    """Compare raw Rust read_lmdb vs ASE db.select for single files."""

    params = [50, 200, 500]
    param_names = ["n_rows"]

    def setup(self, n_rows):
        """Create a single ASELMDB file."""
        self._data_dir = create_temp_dir()
        self.db_file = str(Path(self._data_dir) / "data0000.aselmdb")
        create_synthetic_aselmdb(Path(self.db_file), n_rows, with_calc=True)

    def time_ase(self, n_rows):
        """Read via ASE db.select + toatoms."""
        from ase.db import connect

        db = connect(self.db_file, type="aselmdb")
        for row in db.select():
            row.toatoms()

    def time_rust(self, n_rows):
        """Read via Rust LMDB reader."""
        from physicsnemo_curator._lib.lmdb import read_lmdb

        read_lmdb(self.db_file)

    def teardown(self, n_rows):
        """Remove temporary directory."""
        cleanup_temp_dir(self._data_dir)


class TimeLmdbReaderParallel:
    """Compare Rust parallel LMDB reader vs sequential reads."""

    params = [1, 4, 8]
    param_names = ["n_files"]

    def setup(self, n_files):
        """Create multiple ASELMDB files."""
        self._data_dir = create_temp_dir()
        self.db_files = []
        for i in range(n_files):
            db_path = Path(self._data_dir) / f"data{i:04d}.aselmdb"
            create_synthetic_aselmdb(db_path, 100, with_calc=True, seed=42 + i)
            self.db_files.append(str(db_path))

    def time_rust_parallel(self, n_files):
        """Read all files with Rust parallel reader."""
        from physicsnemo_curator._lib.lmdb import read_lmdb_parallel

        read_lmdb_parallel(self.db_files)

    def time_rust_sequential(self, n_files):
        """Read all files with Rust sequential reader."""
        from physicsnemo_curator._lib.lmdb import read_lmdb

        for f in self.db_files:
            read_lmdb(f)

    def time_ase_sequential(self, n_files):
        """Read all files with ASE sequentially."""
        from ase.db import connect

        for f in self.db_files:
            db = connect(f, type="aselmdb")
            for row in db.select():
                row.toatoms()

    def teardown(self, n_files):
        """Remove temporary directory."""
        cleanup_temp_dir(self._data_dir)


# ---------------------------------------------------------------------------
# AtomicInfoFilter benchmarks
# ---------------------------------------------------------------------------
class TimeAtomicInfoFilter:
    """Time AtomicInfoFilter throughput."""

    params = [50, 200]
    param_names = ["n_rows"]

    def setup(self, n_rows):
        """Create ASELMDB and source + filter."""
        from physicsnemo_curator.atm.filters.atomic_info import AtomicInfoFilter
        from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource

        self._data_dir = create_temp_dir()
        self._info_dir = create_temp_dir()
        db_path = Path(self._data_dir) / "data0000.aselmdb"
        create_synthetic_aselmdb(db_path, n_rows, with_calc=True)

        self.source = ASELMDBSource(data_dir=self._data_dir, backend="python")
        self.filt = AtomicInfoFilter(
            output=str(Path(self._info_dir) / "info.json"),
            log_level="warning",
        )

    def time_filter(self, n_rows):
        """Apply AtomicInfoFilter to all atoms from index 0."""
        stream = self.source[0]
        for _ in self.filt(stream):
            pass

    def teardown(self, n_rows):
        """Remove temporary directories."""
        cleanup_temp_dir(self._data_dir)
        cleanup_temp_dir(self._info_dir)


# ---------------------------------------------------------------------------
# AtomicStatsFilter benchmarks
# ---------------------------------------------------------------------------
class TimeAtomicStatsFilter:
    """Time AtomicStatsFilter throughput."""

    params = [50, 200]
    param_names = ["n_rows"]

    def setup(self, n_rows):
        """Create ASELMDB and source + filter."""
        from physicsnemo_curator.atm.filters.stats import AtomicStatsFilter
        from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource

        self._data_dir = create_temp_dir()
        self._stats_dir = create_temp_dir()
        db_path = Path(self._data_dir) / "data0000.aselmdb"
        create_synthetic_aselmdb(db_path, n_rows, with_calc=True)

        self.source = ASELMDBSource(data_dir=self._data_dir, backend="python")
        self.filt = AtomicStatsFilter(output=str(Path(self._stats_dir) / "stats.parquet"))

    def time_filter(self, n_rows):
        """Apply AtomicStatsFilter to all atoms from index 0."""
        stream = self.source[0]
        for _ in self.filt(stream):
            pass

    def teardown(self, n_rows):
        """Remove temporary directories."""
        cleanup_temp_dir(self._data_dir)
        cleanup_temp_dir(self._stats_dir)


# ---------------------------------------------------------------------------
# AtomicDataZarrSink benchmarks
# ---------------------------------------------------------------------------
class TimeAtomicDataZarrSink:
    """Time AtomicDataZarrSink write."""

    params = [50, 200]
    param_names = ["n_rows"]

    def setup(self, n_rows):
        """Create ASELMDB and source + sink."""
        from physicsnemo_curator.atm.sinks.zarr_writer import AtomicDataZarrSink
        from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource

        self._data_dir = create_temp_dir()
        self._output_dir = create_temp_dir()
        db_path = Path(self._data_dir) / "data0000.aselmdb"
        create_synthetic_aselmdb(db_path, n_rows, with_calc=True)

        self.source = ASELMDBSource(data_dir=self._data_dir, backend="python")
        self.sink = AtomicDataZarrSink(output_path=str(Path(self._output_dir) / "out.zarr"))

    def time_write(self, n_rows):
        """Write all atoms from index 0 via AtomicDataZarrSink."""
        stream = self.source[0]
        self.sink(stream, 0)

    def teardown(self, n_rows):
        """Remove temporary directories."""
        cleanup_temp_dir(self._data_dir)
        cleanup_temp_dir(self._output_dir)


# ---------------------------------------------------------------------------
# Memory: LMDB reader
# ---------------------------------------------------------------------------
class MemLmdbReader:
    """Peak memory for Rust LMDB reader."""

    params = [100, 500]
    param_names = ["n_rows"]

    def setup(self, n_rows):
        """Create ASELMDB file."""
        self._data_dir = create_temp_dir()
        self.db_file = str(Path(self._data_dir) / "data0000.aselmdb")
        create_synthetic_aselmdb(Path(self.db_file), n_rows, with_calc=True)

    def peakmem_rust(self, n_rows):
        """Peak memory for Rust LMDB read."""
        from physicsnemo_curator._lib.lmdb import read_lmdb

        read_lmdb(self.db_file)

    def teardown(self, n_rows):
        """Remove temporary directory."""
        cleanup_temp_dir(self._data_dir)
