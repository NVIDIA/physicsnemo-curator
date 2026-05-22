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

"""Unified pipeline store with SQLite-backed metrics, checkpointing, and provenance.

Provides :class:`PipelineStore`, a single SQLite database that combines
checkpoint tracking (completed/failed indices), per-index and per-stage
wall-clock metrics, and pipeline provenance (config hashing).

Also contains the metrics dataclasses (:class:`StageMetrics`,
:class:`IndexMetrics`, :class:`PipelineMetrics`), the :class:`_TimedGenerator`
timing utility, and provenance helpers for serializing pipeline configuration.

Usage
-----
>>> from physicsnemo_curator.core.pipeline_store import PipelineStore
>>> config = _pipeline_config(pipeline)
>>> chash = _config_hash(config)
>>> store = PipelineStore(db_path=Path("run.db"), pipeline_config=config, config_hash=chash)
>>> store.is_completed(0)  # None — not yet completed
>>> store.record_success(0, ["/out/0.vtk"], wall_time_ns=1_000_000, ...)
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import inspect
import json
import logging
import pathlib
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from physicsnemo_curator.core.base import REQUIRED

if TYPE_CHECKING:
    from collections.abc import Iterator

    from physicsnemo_curator.core.base import Pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metrics dataclasses
# ---------------------------------------------------------------------------


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage (source, one filter, or sink).

    Parameters
    ----------
    name : str
        Human-readable name of the stage (e.g. ``"source"``,
        ``"DoubleFilter"``, ``"sink"``).
    wall_time_ns : int
        Wall-clock time in nanoseconds spent in this stage.
    """

    name: str
    wall_time_ns: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary with ``"name"`` and ``"wall_time_ns"`` keys.
        """
        return {"name": self.name, "wall_time_ns": self.wall_time_ns}


@dataclass
class IndexMetrics:
    """Metrics for one ``__getitem__`` call (one source index).

    Parameters
    ----------
    index : int
        The source index that was processed.
    stages : list[StageMetrics]
        Per-stage timing breakdown.
    wall_time_ns : int
        Total wall-clock time for this index in nanoseconds.
    peak_memory_bytes : int
        Peak Python memory usage during this index (from ``tracemalloc``).
    gpu_memory_bytes : int | None
        Peak GPU memory delta, or ``None`` if GPU tracking was disabled.
    """

    index: int
    stages: list[StageMetrics]
    wall_time_ns: int
    peak_memory_bytes: int
    gpu_memory_bytes: int | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary.

        Returns
        -------
        dict[str, Any]
            Nested dictionary with all metric fields.
        """
        return {
            "index": self.index,
            "stages": [s.to_dict() for s in self.stages],
            "wall_time_ns": self.wall_time_ns,
            "peak_memory_bytes": self.peak_memory_bytes,
            "gpu_memory_bytes": self.gpu_memory_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IndexMetrics:
        """Reconstruct from a dictionary (e.g. deserialized JSON).

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary as produced by :meth:`to_dict`.

        Returns
        -------
        IndexMetrics
            Reconstructed metrics object.
        """
        stages = [StageMetrics(**s) for s in data["stages"]]
        return cls(
            index=data["index"],
            stages=stages,
            wall_time_ns=data["wall_time_ns"],
            peak_memory_bytes=data["peak_memory_bytes"],
            gpu_memory_bytes=data.get("gpu_memory_bytes"),
        )


@dataclass
class PipelineMetrics:
    """Aggregated metrics across all processed indices.

    Parameters
    ----------
    indices : list[IndexMetrics]
        Per-index metrics, one entry per ``__getitem__`` call.
    """

    indices: list[IndexMetrics] = field(default_factory=list)

    @property
    def total_wall_time_ns(self) -> int:
        """Total wall-clock time across all indices (nanoseconds).

        Returns
        -------
        int
            Sum of per-index wall times.
        """
        return sum(m.wall_time_ns for m in self.indices)

    @property
    def mean_index_time_ns(self) -> float:
        """Mean wall-clock time per index (nanoseconds).

        Returns
        -------
        float
            Average per-index time, or ``0.0`` if no indices were processed.
        """
        if not self.indices:
            return 0.0
        return self.total_wall_time_ns / len(self.indices)

    @property
    def total_peak_memory_bytes(self) -> int:
        """Maximum peak memory observed across all indices (bytes).

        Returns
        -------
        int
            Max of per-index peak memory values.
        """
        if not self.indices:
            return 0
        return max(m.peak_memory_bytes for m in self.indices)

    def summary(self) -> dict[str, Any]:
        """Return a summary dictionary for programmatic use.

        Returns
        -------
        dict[str, Any]
            Dictionary with total/mean wall time, peak memory, index count,
            and per-index breakdowns.
        """
        return {
            "num_indices": len(self.indices),
            "total_wall_time_ns": self.total_wall_time_ns,
            "mean_index_time_ns": self.mean_index_time_ns,
            "total_peak_memory_bytes": self.total_peak_memory_bytes,
            "indices": [m.to_dict() for m in self.indices],
        }

    def to_json(self, path: str | pathlib.Path) -> None:
        """Write metrics to a JSON file.

        Parameters
        ----------
        path : str | pathlib.Path
            Output file path.
        """
        data = self.summary()
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2))

    def to_csv(self, path: str | pathlib.Path) -> None:
        """Write per-index metrics to a CSV file.

        Each row represents one index. Stage timings are included as
        separate columns named ``stage_<name>_ns``.

        Parameters
        ----------
        path : str | pathlib.Path
            Output file path.
        """
        if not self.indices:
            pathlib.Path(path).write_text("")
            return

        # Collect all unique stage names across indices (preserving order)
        stage_names: list[str] = []
        seen: set[str] = set()
        for idx_m in self.indices:
            for s in idx_m.stages:
                if s.name not in seen:
                    stage_names.append(s.name)
                    seen.add(s.name)

        fieldnames = [
            "index",
            "wall_time_ns",
            "peak_memory_bytes",
            "gpu_memory_bytes",
        ] + [f"stage_{name}_ns" for name in stage_names]

        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for idx_m in self.indices:
                row: dict[str, Any] = {
                    "index": idx_m.index,
                    "wall_time_ns": idx_m.wall_time_ns,
                    "peak_memory_bytes": idx_m.peak_memory_bytes,
                    "gpu_memory_bytes": idx_m.gpu_memory_bytes if idx_m.gpu_memory_bytes is not None else "",
                }
                stage_map = {s.name: s.wall_time_ns for s in idx_m.stages}
                for sn in stage_names:
                    row[f"stage_{sn}_ns"] = stage_map.get(sn, "")
                writer.writerow(row)

    def to_console(self) -> None:
        """Print a formatted summary table to stdout.

        Outputs a human-readable table showing per-index and aggregate
        metrics. Uses only stdlib formatting (no external dependencies).
        """
        if not self.indices:
            print("No profiling metrics collected.")
            return

        print("\n=== Pipeline Profiling Results ===\n")

        # Summary
        total_ms = self.total_wall_time_ns / 1e6
        mean_ms = self.mean_index_time_ns / 1e6
        peak_mb = self.total_peak_memory_bytes / (1024 * 1024)
        print(f"  Indices processed : {len(self.indices)}")
        print(f"  Total wall time   : {total_ms:,.2f} ms")
        print(f"  Mean per index    : {mean_ms:,.2f} ms")
        print(f"  Peak memory       : {peak_mb:,.2f} MB")

        # Check for GPU
        gpu_indices = [m for m in self.indices if m.gpu_memory_bytes is not None]
        if gpu_indices:
            max_gpu = max(m.gpu_memory_bytes for m in gpu_indices)  # type: ignore[arg-type]
            print(f"  Peak GPU memory   : {max_gpu / (1024 * 1024):,.2f} MB")

        # Per-index table
        print(f"\n{'Index':>7} {'Wall (ms)':>12} {'Memory (MB)':>13} {'GPU (MB)':>10}")
        print("  " + "-" * 46)
        for m in self.indices:
            wall = m.wall_time_ns / 1e6
            mem = m.peak_memory_bytes / (1024 * 1024)
            gpu = f"{m.gpu_memory_bytes / (1024 * 1024):>10.2f}" if m.gpu_memory_bytes is not None else "       N/A"
            print(f"  {m.index:>5} {wall:>12.2f} {mem:>13.2f} {gpu}")

        # Per-stage averages
        if self.indices and self.indices[0].stages:
            print("\n  Stage Averages:")
            stage_totals: dict[str, list[int]] = {}
            for idx_m in self.indices:
                for s in idx_m.stages:
                    stage_totals.setdefault(s.name, []).append(s.wall_time_ns)
            for name, times in stage_totals.items():
                avg_ms = (sum(times) / len(times)) / 1e6
                print(f"    {name:<30s} {avg_ms:>10.2f} ms (avg)")

        print()


# ---------------------------------------------------------------------------
# _TimedGenerator
# ---------------------------------------------------------------------------


class _TimedGenerator[T]:
    """Generator wrapper that accumulates wall-clock time across ``__next__`` calls.

    This is used internally to attribute time to each pipeline stage. The
    wrapper preserves the full iterator protocol.

    Parameters
    ----------
    inner : Iterator[T]
        The generator or iterator to wrap.
    """

    def __init__(self, inner: Iterator[T]) -> None:
        """Initialize with the inner iterator."""
        self._inner = inner
        self._elapsed_ns: int = 0

    @property
    def elapsed_ns(self) -> int:
        """Total nanoseconds spent inside ``__next__`` of the inner iterator.

        Returns
        -------
        int
            Accumulated wall-clock nanoseconds.
        """
        return self._elapsed_ns

    def __iter__(self) -> _TimedGenerator[T]:
        """Return self (iterator protocol)."""
        return self

    def __next__(self) -> T:
        """Delegate to inner iterator, timing the call.

        Returns
        -------
        T
            Next value from the inner iterator.

        Raises
        ------
        StopIteration
            When the inner iterator is exhausted.
        """
        start = time.perf_counter_ns()
        try:
            value = next(self._inner)
        except StopIteration:
            self._elapsed_ns += time.perf_counter_ns() - start
            raise
        self._elapsed_ns += time.perf_counter_ns() - start
        return value


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def _component_config(component: object) -> dict[str, Any]:
    """Serialize a pipeline component (source, filter, or sink) to a dict.

    Parameters
    ----------
    component : object
        A pipeline component with ``name``, ``params()`` classmethod, and
        ``__init__`` arguments.

    Returns
    -------
    dict[str, Any]
        Serialized configuration dictionary.
    """
    cls = type(component)
    config: dict[str, Any] = {
        "class": cls.__name__,
        "module": cls.__module__,
    }
    if hasattr(cls, "name"):
        config["name"] = cls.name
    if hasattr(cls, "description"):
        config["description"] = cls.description

    # Extract current parameter values from __init__ signature
    sig = inspect.signature(cls.__init__)
    params: dict[str, Any] = {}
    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        # Try to read the stored attribute (common convention: _<name>)
        for attr_name in (f"_{pname}", pname):
            if hasattr(component, attr_name):
                val = getattr(component, attr_name)
                params[pname] = _safe_serialize(val)
                break
        else:
            # Fall back to default if available
            if param.default is not inspect.Parameter.empty:
                params[pname] = _safe_serialize(param.default)

    config["params"] = params
    return config


def _safe_serialize(value: object) -> object:
    """Convert a value to a JSON-safe type.

    Parameters
    ----------
    value : object
        Any Python object.

    Returns
    -------
    object
        JSON-safe representation.
    """
    if value is REQUIRED:
        return "<REQUIRED>"
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_serialize(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe_serialize(v) for k, v in value.items()}
    # pathlib.Path, other types -> string
    return str(value)


def _pipeline_config(pipeline: Pipeline[Any]) -> dict[str, Any]:
    """Build the full pipeline configuration dictionary.

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline to serialize.

    Returns
    -------
    dict[str, Any]
        Full pipeline configuration.
    """
    config: dict[str, Any] = {
        "source": _component_config(pipeline.source),
        "filters": [_component_config(f) for f in pipeline.filters],
    }
    if pipeline.sink is not None:
        config["sink"] = _component_config(pipeline.sink)
    return config


def _config_hash(config: dict[str, Any]) -> str:
    """Compute a stable SHA-256 hash of a pipeline configuration.

    Parameters
    ----------
    config : dict[str, Any]
        Pipeline configuration dict.

    Returns
    -------
    str
        Hex-encoded SHA-256 hash.
    """
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Worker identity
# ---------------------------------------------------------------------------

_worker_id_local = threading.local()


def _get_worker_id() -> str:
    """Return a stable worker ID for the current thread.

    Each thread (and by extension each process in multi-process backends)
    gets a unique UUID4 identifier that is stable across multiple calls
    within the same thread.

    Returns
    -------
    str
        UUID4 hex string identifying the current worker thread.
    """
    wid: str | None = getattr(_worker_id_local, "worker_id", None)
    if wid is None:
        wid = uuid.uuid4().hex
        _worker_id_local.worker_id = wid
    return wid


# ---------------------------------------------------------------------------
# SQL schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    config_hash   TEXT    UNIQUE NOT NULL,
    config_json   TEXT    NOT NULL,
    started_at    TEXT    NOT NULL,
    run_dir       TEXT,
    total_indices INTEGER
);

CREATE TABLE IF NOT EXISTS index_results (
    idx               INTEGER NOT NULL,
    run_id            INTEGER NOT NULL,
    status            TEXT    NOT NULL CHECK (status IN ('completed', 'error')),
    output_paths      TEXT,
    completed_at      TEXT    NOT NULL,
    wall_time_ns      INTEGER,
    peak_memory_bytes INTEGER,
    gpu_memory_bytes  INTEGER,
    error             TEXT,
    worker_id         TEXT,
    PRIMARY KEY (idx, run_id),
    FOREIGN KEY (run_id) REFERENCES pipeline_runs (run_id)
);

CREATE TABLE IF NOT EXISTS stage_metrics (
    idx          INTEGER NOT NULL,
    run_id       INTEGER NOT NULL,
    stage_order  INTEGER NOT NULL,
    stage_name   TEXT    NOT NULL,
    wall_time_ns INTEGER NOT NULL,
    PRIMARY KEY (idx, run_id, stage_order),
    FOREIGN KEY (idx, run_id) REFERENCES index_results (idx, run_id)
);

CREATE TABLE IF NOT EXISTS workers (
    worker_id       TEXT    PRIMARY KEY,
    run_id          INTEGER NOT NULL,
    pid             INTEGER NOT NULL,
    hostname        TEXT    NOT NULL,
    started_at      TEXT    NOT NULL,
    last_heartbeat  TEXT    NOT NULL,
    current_index   INTEGER,
    completed_count INTEGER NOT NULL DEFAULT 0,
    invocation_id   TEXT,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs (run_id)
);

CREATE TABLE IF NOT EXISTS output_files (
    path    TEXT    NOT NULL,
    idx     INTEGER NOT NULL,
    run_id  INTEGER NOT NULL,
    seq     INTEGER NOT NULL,
    PRIMARY KEY (path, run_id),
    FOREIGN KEY (idx, run_id) REFERENCES index_results (idx, run_id)
);

CREATE TABLE IF NOT EXISTS filter_artifacts (
    path         TEXT    NOT NULL,
    idx          INTEGER NOT NULL,
    run_id       INTEGER NOT NULL,
    filter_name  TEXT    NOT NULL,
    filter_order INTEGER NOT NULL,
    PRIMARY KEY (path, idx, run_id),
    FOREIGN KEY (idx, run_id) REFERENCES index_results (idx, run_id)
);

CREATE TABLE IF NOT EXISTS logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL,
    timestamp   TEXT    NOT NULL,
    level       INTEGER NOT NULL,
    level_name  TEXT    NOT NULL,
    logger_name TEXT    NOT NULL,
    message     TEXT    NOT NULL,
    worker_id   TEXT,
    idx         INTEGER,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs (run_id)
);
CREATE INDEX IF NOT EXISTS idx_logs_run_timestamp ON logs (run_id, timestamp);
"""


# ---------------------------------------------------------------------------
# PipelineStore
# ---------------------------------------------------------------------------


class PipelineStore:
    """SQLite-backed store combining checkpoint tracking, metrics, provenance, and worker progress.

    Manages a single database with six tables: ``pipeline_runs``,
    ``index_results``, ``stage_metrics``, ``output_files``,
    ``filter_artifacts``, and ``workers``.  Supports
    checkpoint resumption via config hashing, per-index success/error
    recording, aggregated metrics queries, and live worker progress
    tracking.

    Parameters
    ----------
    db_path : pathlib.Path
        Path to the SQLite database file.  Created automatically if it
        does not exist.
    pipeline_config : dict
        Full pipeline configuration dictionary (from :func:`_pipeline_config`).
    config_hash : str
        SHA-256 hex hash of the pipeline configuration.

    Examples
    --------
    >>> config = _pipeline_config(pipeline)
    >>> chash = _config_hash(config)
    >>> store = PipelineStore(Path("run.db"), config, chash)
    >>> store.is_completed(0)  # None if not yet done
    >>> store.record_success(0, ["/out/0.vtk"], 1_000_000, 4096, None, [])
    """

    def __init__(
        self,
        db_path: pathlib.Path,
        pipeline_config: dict,
        config_hash: str,
        *,
        _worker: bool = False,
    ) -> None:
        """Initialize the pipeline store.

        Parameters
        ----------
        db_path : pathlib.Path
            Path to the SQLite database file.
        pipeline_config : dict
            Full pipeline configuration dictionary.
        config_hash : str
            SHA-256 hex hash of the pipeline configuration.
        _worker : bool, optional
            If ``True``, skip schema creation and just look up the
            existing ``run_id``.  Used by child processes that reconnect
            to a database already initialised by the parent.
        """
        self._db_path = db_path
        self._pipeline_config = pipeline_config
        self._config_hash = config_hash
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        if _worker:
            self._attach_to_existing_run()
        else:
            self._init_db()

    @classmethod
    def from_db(cls, db_path: str | pathlib.Path) -> PipelineStore:
        """Open an existing pipeline database in read-only mode.

        This is the entry point for the dashboard and post-hoc analysis
        tools.  It reads the ``pipeline_runs`` table to recover
        ``config_hash`` and ``pipeline_config``, so the caller does not
        need to know them.

        Parameters
        ----------
        db_path : str or pathlib.Path
            Path to an existing ``.db`` file produced by a pipeline run.

        Returns
        -------
        PipelineStore
            A store instance backed by the existing database.

        Raises
        ------
        FileNotFoundError
            If *db_path* does not exist.
        ValueError
            If the database contains no pipeline run records.
        """
        db_path = pathlib.Path(db_path)
        if not db_path.exists():
            msg = f"Database file not found: {db_path}"
            raise FileNotFoundError(msg)

        conn = cls._open_connection(db_path)
        try:
            row = conn.execute(
                "SELECT config_hash, config_json FROM pipeline_runs ORDER BY run_id DESC LIMIT 1",
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            msg = f"No pipeline run records found in {db_path}"
            raise ValueError(msg)

        config_hash, config_json = row
        pipeline_config = json.loads(config_json)
        return cls(db_path, pipeline_config, config_hash)

    @staticmethod
    def _open_connection(db_path: pathlib.Path) -> sqlite3.Connection:
        """Open a WAL-aware connection with retry logic.

        This is used by :meth:`from_db` and other entry points that need
        to open a database file that may have been written with WAL mode
        enabled.  It sets a busy timeout and attempts to read the WAL
        so that any uncommitted WAL data is visible.

        Parameters
        ----------
        db_path : pathlib.Path
            Path to the database file.

        Returns
        -------
        sqlite3.Connection
            A connection with WAL mode and busy timeout configured.
        """
        import time

        conn = sqlite3.connect(str(db_path), timeout=30)
        conn.execute("PRAGMA busy_timeout=30000")

        max_retries = 10
        delay = 0.05
        for attempt in range(max_retries):
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                break
            except sqlite3.OperationalError:
                if attempt == max_retries - 1:
                    with contextlib.suppress(sqlite3.OperationalError):
                        conn.execute("PRAGMA journal_mode=DELETE")
                else:
                    time.sleep(delay)
                    delay = min(delay * 2, 2.0)

        # Force a WAL checkpoint to ensure all committed data is in the
        # main database file, making it readable by other processes.
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute("PRAGMA wal_checkpoint(PASSIVE)")

        return conn

    def _connect(self) -> sqlite3.Connection:
        """Open a WAL-mode connection to the database.

        Retries the ``PRAGMA journal_mode=WAL`` statement with exponential
        backoff to handle concurrent process initialization on Windows, where
        the WAL mode switch requires a brief exclusive lock that ``busy_timeout``
        alone does not reliably cover.

        Returns
        -------
        sqlite3.Connection
            Database connection with WAL journal mode and busy timeout.
        """
        import time

        conn = sqlite3.connect(str(self._db_path), timeout=30)
        conn.execute("PRAGMA busy_timeout=30000")

        max_retries = 10
        delay = 0.05  # 50 ms initial backoff
        for attempt in range(max_retries):
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                break
            except sqlite3.OperationalError:
                if attempt == max_retries - 1:
                    # WAL not supported (e.g. network filesystem) — use DELETE mode
                    with contextlib.suppress(sqlite3.OperationalError):
                        conn.execute("PRAGMA journal_mode=DELETE")
                else:
                    time.sleep(delay)
                    delay = min(delay * 2, 2.0)

        return conn

    def _resilient_write(
        self,
        operation: str,
        func: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a write operation with retry and graceful failure.

        Wraps database write methods so that transient SQLite locking
        errors do not hang or crash the data pipeline.  If the write
        still fails after retries, a warning is logged and ``None`` is
        returned — the pipeline continues without this metric record.

        Catches both :class:`sqlite3.OperationalError` (lock timeouts,
        disk I/O) and :class:`sqlite3.DatabaseError` (malformed DB).

        Parameters
        ----------
        operation : str
            Human-readable name for logging (e.g. ``"record_success"``).
        func : callable
            The actual write function to invoke.
        *args : Any
            Positional arguments forwarded to *func*.
        **kwargs : Any
            Keyword arguments forwarded to *func*.

        Returns
        -------
        Any
            The return value of *func*, or ``None`` on failure.
        """
        max_retries = 3
        delay = 0.1
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
                if attempt == max_retries - 1:
                    logger.warning(
                        "PipelineStore: %s failed after %d retries (%s) — pipeline will continue without this record",
                        operation,
                        max_retries,
                        exc,
                    )
                    return None
                time.sleep(delay)
                delay = min(delay * 2, 1.0)
        return None  # pragma: no cover

    def _init_db(self) -> None:
        """Create schema and register or resume a pipeline run by config hash.

        Uses INSERT OR IGNORE to handle concurrent init from multiple
        threads/processes safely (avoids TOCTOU race on the UNIQUE
        ``config_hash`` column).

        If the existing database is malformed (e.g. from a previous crash
        during WAL checkpoint), the corrupted file is removed and a fresh
        database is created.
        """
        try:
            self._init_db_inner()
        except sqlite3.DatabaseError as exc:
            import warnings

            warnings.warn(
                f"Pipeline metrics database is malformed ({exc}). "
                f"Removing corrupted file and creating a fresh database: {self._db_path}",
                stacklevel=2,
            )
            logger.warning(
                "PipelineStore: existing database is malformed (%s) — "
                "removing corrupted file and creating fresh database: %s",
                exc,
                self._db_path,
            )
            # Remove the corrupted DB and associated WAL/SHM files
            for suffix in ("", "-wal", "-shm"):
                p = pathlib.Path(str(self._db_path) + suffix)
                if p.exists():
                    p.unlink()
            self._init_db_inner()

    def _init_db_inner(self) -> None:
        """Create schema and register the pipeline run (may raise on malformed DB)."""
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA_SQL)

            # Migrate: add new columns to workers table if missing
            cols = {row[1] for row in conn.execute("PRAGMA table_info(workers)").fetchall()}
            if "completed_count" not in cols:
                conn.execute("ALTER TABLE workers ADD COLUMN completed_count INTEGER NOT NULL DEFAULT 0")
            if "invocation_id" not in cols:
                conn.execute("ALTER TABLE workers ADD COLUMN invocation_id TEXT")

            # Migrate: add run_dir column to pipeline_runs if missing
            run_cols = {row[1] for row in conn.execute("PRAGMA table_info(pipeline_runs)").fetchall()}
            if "run_dir" not in run_cols:
                conn.execute("ALTER TABLE pipeline_runs ADD COLUMN run_dir TEXT")
            if "total_indices" not in run_cols:
                conn.execute("ALTER TABLE pipeline_runs ADD COLUMN total_indices INTEGER")

            # Migrate: add worker_id column to index_results if missing
            idx_cols = {row[1] for row in conn.execute("PRAGMA table_info(index_results)").fetchall()}
            if "worker_id" not in idx_cols:
                conn.execute("ALTER TABLE index_results ADD COLUMN worker_id TEXT")

            conn.commit()

            # Atomically insert if not exists, then SELECT to get run_id.
            # This avoids the TOCTOU race where two threads both see no
            # existing row and both try to INSERT.
            now = datetime.now(tz=UTC).isoformat()
            run_dir = str(pathlib.Path.cwd())
            conn.execute(
                "INSERT OR IGNORE INTO pipeline_runs (config_hash, config_json, started_at, run_dir) "
                "VALUES (?, ?, ?, ?)",
                (self._config_hash, json.dumps(self._pipeline_config, sort_keys=True, default=str), now, run_dir),
            )

            # Update run_dir for existing rows that may have NULL
            conn.execute(
                "UPDATE pipeline_runs SET run_dir = ? WHERE config_hash = ? AND run_dir IS NULL",
                (run_dir, self._config_hash),
            )

            row = conn.execute(
                "SELECT run_id FROM pipeline_runs WHERE config_hash = ?",
                (self._config_hash,),
            ).fetchone()
            self._run_id: int = row[0]

            conn.commit()
            logger.info("Pipeline run_id=%d (config hash %s...)", self._run_id, self._config_hash[:12])
        finally:
            conn.close()

    def _attach_to_existing_run(self) -> None:
        """Attach to a database already initialised by the parent process.

        Performs a lightweight read-only lookup of the ``run_id`` without
        running schema creation or migrations.  This avoids the exclusive
        lock that ``executescript`` requires and eliminates contention
        when many worker processes start concurrently.

        Falls back to full :meth:`_init_db` if the lookup fails (e.g.
        the parent hasn't finished init yet).
        """
        max_retries = 5
        delay = 0.2
        for _attempt in range(max_retries):
            try:
                conn = self._connect()
                try:
                    row = conn.execute(
                        "SELECT run_id FROM pipeline_runs WHERE config_hash = ?",
                        (self._config_hash,),
                    ).fetchone()
                    if row is not None:
                        self._run_id = row[0]
                        return
                finally:
                    conn.close()
            except (sqlite3.OperationalError, sqlite3.DatabaseError):
                pass
            time.sleep(delay)
            delay = min(delay * 2, 2.0)

        # Fallback: parent may not have finished yet, do full init
        logger.debug("Worker could not attach to existing run — falling back to full _init_db")
        self._init_db()

    @property
    def run_dir(self) -> str | None:
        """Return the working directory recorded when the pipeline was started.

        Returns
        -------
        str or None
            Absolute path of the CWD at pipeline start, or ``None`` for
            databases created before this column was added.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT run_dir FROM pipeline_runs WHERE run_id = ?",
                (self._run_id,),
            ).fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def resolve_artifact(self, path: str) -> pathlib.Path:
        """Resolve a relative artifact path using the stored run directory.

        Parameters
        ----------
        path : str
            Artifact path (may be relative or absolute).

        Returns
        -------
        pathlib.Path
            Absolute path.  If *path* is already absolute, it is returned
            as-is.  Otherwise it is resolved relative to :attr:`run_dir`
            (falling back to the current working directory if ``run_dir``
            is not available).
        """
        p = pathlib.Path(path)
        if p.is_absolute():
            return p
        base = self.run_dir
        if base:
            return pathlib.Path(base) / p
        return p.resolve()

    def is_completed(self, index: int) -> list[str] | None:
        """Check if an index has been completed successfully.

        Parameters
        ----------
        index : int
            Source index to check.

        Returns
        -------
        list[str] | None
            Cached output paths if completed, ``None`` otherwise.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT output_paths FROM index_results WHERE idx = ? AND run_id = ? AND status = 'completed'",
                (index, self._run_id),
            ).fetchone()
            if row is not None:
                return json.loads(row[0])  # type: ignore[no-any-return]
            return None
        finally:
            conn.close()

    def record_success(
        self,
        index: int,
        output_paths: list[str],
        wall_time_ns: int,
        peak_memory_bytes: int,
        gpu_memory_bytes: int | None,
        stages: list[StageMetrics],
        worker_id: str | None = None,
    ) -> None:
        """Record a successfully completed index with metrics.

        Parameters
        ----------
        index : int
            Source index that completed.
        output_paths : list[str]
            File paths written by the sink.
        wall_time_ns : int
            Total wall-clock time in nanoseconds.
        peak_memory_bytes : int
            Peak memory usage in bytes.
        gpu_memory_bytes : int | None
            Peak GPU memory delta, or ``None``.
        stages : list[StageMetrics]
            Per-stage timing breakdown.
        worker_id : str | None
            ID of the worker that processed this index.
        """
        now = datetime.now(tz=UTC).isoformat()
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO index_results "
                "(idx, run_id, status, output_paths, completed_at, wall_time_ns, "
                "peak_memory_bytes, gpu_memory_bytes, error, worker_id) "
                "VALUES (?, ?, 'completed', ?, ?, ?, ?, ?, NULL, ?)",
                (
                    index,
                    self._run_id,
                    json.dumps(output_paths),
                    now,
                    wall_time_ns,
                    peak_memory_bytes,
                    gpu_memory_bytes,
                    worker_id,
                ),
            )

            # Delete any existing stage_metrics for this index/run before inserting
            conn.execute(
                "DELETE FROM stage_metrics WHERE idx = ? AND run_id = ?",
                (index, self._run_id),
            )
            for order, stage in enumerate(stages):
                conn.execute(
                    "INSERT INTO stage_metrics (idx, run_id, stage_order, stage_name, wall_time_ns) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (index, self._run_id, order, stage.name, stage.wall_time_ns),
                )

            # Populate the normalized output_files table for reverse lookup
            conn.execute(
                "DELETE FROM output_files WHERE idx = ? AND run_id = ?",
                (index, self._run_id),
            )
            for seq, path in enumerate(output_paths):
                conn.execute(
                    "INSERT OR REPLACE INTO output_files (path, idx, run_id, seq) VALUES (?, ?, ?, ?)",
                    (path, index, self._run_id, seq),
                )

            conn.commit()
        finally:
            conn.close()

    def record_error(self, index: int, error: str, wall_time_ns: int, worker_id: str | None = None) -> None:
        """Record a failed index execution.

        Parameters
        ----------
        index : int
            Source index that failed.
        error : str
            Error message.
        wall_time_ns : int
            Wall-clock time before the error in nanoseconds.
        worker_id : str | None
            ID of the worker that processed this index.
        """
        now = datetime.now(tz=UTC).isoformat()
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO index_results "
                "(idx, run_id, status, output_paths, completed_at, wall_time_ns, "
                "peak_memory_bytes, gpu_memory_bytes, error, worker_id) "
                "VALUES (?, ?, 'error', NULL, ?, ?, NULL, NULL, ?, ?)",
                (index, self._run_id, now, wall_time_ns, error, worker_id),
            )
            conn.commit()
        finally:
            conn.close()

    def completed_indices(self) -> set[int]:
        """Return the set of successfully completed indices for this run.

        Returns
        -------
        set[int]
            Indices with ``status='completed'``.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT idx FROM index_results WHERE run_id = ? AND status = 'completed'",
                (self._run_id,),
            ).fetchall()
            return {r[0] for r in rows}
        finally:
            conn.close()

    def failed_indices(self) -> dict[int, str]:
        """Return indices that failed with their error messages.

        Returns
        -------
        dict[int, str]
            Mapping from index to error message string.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT idx, error FROM index_results WHERE run_id = ? AND status = 'error'",
                (self._run_id,),
            ).fetchall()
            return {r[0]: r[1] for r in rows}
        finally:
            conn.close()

    def indices_by_worker(self, worker_id: str) -> dict[str, list[int]]:
        """Return indices processed by a specific worker, grouped by status.

        Parameters
        ----------
        worker_id : str
            The worker ID to query.

        Returns
        -------
        dict[str, list[int]]
            Dictionary with keys 'completed' and 'failed', each containing
            a sorted list of indices processed by this worker.
        """
        conn = self._connect()
        try:
            completed = conn.execute(
                "SELECT idx FROM index_results "
                "WHERE run_id = ? AND worker_id = ? AND status = 'completed' ORDER BY idx",
                (self._run_id, worker_id),
            ).fetchall()
            failed = conn.execute(
                "SELECT idx FROM index_results WHERE run_id = ? AND worker_id = ? AND status = 'error' ORDER BY idx",
                (self._run_id, worker_id),
            ).fetchall()
            return {
                "completed": [r[0] for r in completed],
                "failed": [r[0] for r in failed],
            }
        finally:
            conn.close()

    def remaining_indices(self, total: int) -> list[int]:
        """Return indices not yet completed or failed for this run.

        Parameters
        ----------
        total : int
            Total number of source indices.

        Returns
        -------
        list[int]
            Sorted list of indices still needing processing.
        """
        done = self.completed_indices() | set(self.failed_indices().keys())
        return sorted(i for i in range(total) if i not in done)

    def summary(self, total: int) -> dict[str, Any]:
        """Return a summary of the store state.

        Parameters
        ----------
        total : int
            Total number of source indices.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys: ``total``, ``completed``, ``failed``,
            ``remaining``, ``config_hash``, ``db_path``,
            ``total_elapsed_s``, ``workers``.
        """
        conn = self._connect()
        try:
            completed = conn.execute(
                "SELECT COUNT(*) FROM index_results WHERE run_id = ? AND status = 'completed'",
                (self._run_id,),
            ).fetchone()[0]
            failed = conn.execute(
                "SELECT COUNT(*) FROM index_results WHERE run_id = ? AND status = 'error'",
                (self._run_id,),
            ).fetchone()[0]
            elapsed_row = conn.execute(
                "SELECT COALESCE(SUM(wall_time_ns), 0) FROM index_results WHERE run_id = ? AND status = 'completed'",
                (self._run_id,),
            ).fetchone()
            total_elapsed_ns: int = elapsed_row[0]
            worker_count = conn.execute(
                "SELECT COUNT(*) FROM workers WHERE run_id = ?",
                (self._run_id,),
            ).fetchone()[0]
        finally:
            conn.close()

        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "remaining": total - completed - failed,
            "config_hash": self._config_hash,
            "db_path": str(self._db_path),
            "total_elapsed_s": total_elapsed_ns / 1e9,
            "workers": worker_count,
        }

    def set_total_indices(self, total: int) -> None:
        """Store the total number of source indices for this run.

        Called by the pipeline runner once the source length is known.
        This value is persisted so the dashboard can show accurate
        progress even when the pipeline is still running.

        Parameters
        ----------
        total : int
            Total number of indices the pipeline will process.
        """
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE pipeline_runs SET total_indices = ? WHERE run_id = ?",
                (total, self._run_id),
            )
            conn.commit()
        finally:
            conn.close()

    def get_total_indices(self) -> int | None:
        """Get the total number of source indices for this run.

        Returns
        -------
        int | None
            Total indices if set, otherwise None.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT total_indices FROM pipeline_runs WHERE run_id = ?",
                (self._run_id,),
            ).fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def checkpoint(self) -> None:
        """Force a WAL checkpoint to flush data to the main database file.

        This ensures that all committed writes are transferred from the
        WAL file into the main ``.db`` file, making them visible to
        readers that open the database in a separate process (e.g. the
        dashboard).

        Uses ``PASSIVE`` mode which does not block concurrent readers or
        writers.  It is safe to call at any time.
        """
        conn = self._connect()
        try:
            conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Logging methods
    # -------------------------------------------------------------------------

    def record_logs(
        self,
        logs: list[tuple[str, int, str, str, str, str | None, int | None]],
    ) -> None:
        """Record a batch of log entries.

        Uses a single transaction for efficiency and minimal lock time.
        Each entry is a tuple of:
        ``(timestamp, level, level_name, logger_name, message, worker_id, idx)``

        Parameters
        ----------
        logs : list[tuple]
            List of log entry tuples.
        """
        if not logs:
            return

        conn = self._connect()
        try:
            conn.executemany(
                "INSERT INTO logs "
                "(run_id, timestamp, level, level_name, logger_name, message, worker_id, idx) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [(self._run_id, *entry) for entry in logs],
            )
            conn.commit()
        finally:
            conn.close()

    def get_logs(
        self,
        since_id: int = 0,
        limit: int = 100,
        min_level: int = 0,
    ) -> list[dict[str, Any]]:
        """Retrieve log entries since a given ID.

        Parameters
        ----------
        since_id : int
            Return logs with id > since_id (for polling new entries).
        limit : int
            Maximum number of entries to return.
        min_level : int
            Minimum log level (e.g., 20 for INFO, 10 for DEBUG).

        Returns
        -------
        list[dict]
            Log entries with keys: id, timestamp, level, level_name,
            logger_name, message, worker_id, idx.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, timestamp, level, level_name, logger_name, message, worker_id, idx "
                "FROM logs "
                "WHERE run_id = ? AND id > ? AND level >= ? "
                "ORDER BY id ASC LIMIT ?",
                (self._run_id, since_id, min_level, limit),
            ).fetchall()
            return [
                {
                    "id": row[0],
                    "timestamp": row[1],
                    "level": row[2],
                    "level_name": row[3],
                    "logger_name": row[4],
                    "message": row[5],
                    "worker_id": row[6],
                    "idx": row[7],
                }
                for row in rows
            ]
        finally:
            conn.close()

    def reset(self) -> None:
        """Clear all records for this run and re-register.

        Deletes all index results, stage metrics, and pipeline run metadata
        from the database.  The database file is kept and a fresh run is
        registered.
        """
        conn = self._connect()
        try:
            conn.execute("DELETE FROM filter_artifacts WHERE run_id = ?", (self._run_id,))
            conn.execute("DELETE FROM output_files WHERE run_id = ?", (self._run_id,))
            conn.execute("DELETE FROM stage_metrics WHERE run_id = ?", (self._run_id,))
            conn.execute("DELETE FROM index_results WHERE run_id = ?", (self._run_id,))
            conn.execute("DELETE FROM workers WHERE run_id = ?", (self._run_id,))
            conn.execute("DELETE FROM pipeline_runs WHERE run_id = ?", (self._run_id,))
            conn.commit()
        finally:
            conn.close()

        # Re-initialize to register a fresh run
        self._init_db()
        logger.info("Pipeline store reset — all records cleared for run_id=%d", self._run_id)

    def reset_index(self, index: int) -> None:
        """Remove records for a single index from this run.

        Parameters
        ----------
        index : int
            Source index to remove.
        """
        conn = self._connect()
        try:
            conn.execute(
                "DELETE FROM filter_artifacts WHERE idx = ? AND run_id = ?",
                (index, self._run_id),
            )
            conn.execute(
                "DELETE FROM output_files WHERE idx = ? AND run_id = ?",
                (index, self._run_id),
            )
            conn.execute(
                "DELETE FROM stage_metrics WHERE idx = ? AND run_id = ?",
                (index, self._run_id),
            )
            conn.execute(
                "DELETE FROM index_results WHERE idx = ? AND run_id = ?",
                (index, self._run_id),
            )
            conn.commit()
        finally:
            conn.close()

    # -- Output file lookup ------------------------------------------------------

    def index_for_path(self, path: str) -> int | None:
        """Find which source index produced a given output file.

        Parameters
        ----------
        path : str
            Output file path to look up.

        Returns
        -------
        int | None
            Source index that produced the file, or ``None`` if not found.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT idx FROM output_files WHERE path = ? AND run_id = ?",
                (path, self._run_id),
            ).fetchone()
            return row[0] if row is not None else None
        finally:
            conn.close()

    def output_paths_for_index(self, index: int) -> list[str]:
        """Return the output file paths produced by a given source index.

        Parameters
        ----------
        index : int
            Source index to query.

        Returns
        -------
        list[str]
            Output file paths ordered by sequence, or empty list if none.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT path FROM output_files WHERE idx = ? AND run_id = ? ORDER BY seq",
                (index, self._run_id),
            ).fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()

    # -- Filter artifact tracking ------------------------------------------------

    def record_filter_artifacts(
        self,
        index: int,
        filter_name: str,
        filter_order: int,
        paths: list[str],
    ) -> None:
        """Record file artifacts produced by a filter for a given index.

        Parameters
        ----------
        index : int
            Source index that was processed.
        filter_name : str
            Human-readable name of the filter.
        filter_order : int
            Position of the filter in the pipeline (0-indexed).
        paths : list[str]
            File paths produced by the filter for this index.
        """
        if not paths:
            return
        conn = self._connect()
        try:
            for path in paths:
                conn.execute(
                    "INSERT OR REPLACE INTO filter_artifacts "
                    "(path, idx, run_id, filter_name, filter_order) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (path, index, self._run_id, filter_name, filter_order),
                )
            conn.commit()
        finally:
            conn.close()

    def replace_filter_artifacts(
        self,
        filter_name: str,
        filter_order: int,
        old_paths: list[str],
        merged_path: str,
    ) -> None:
        """Replace shard artifact paths with a single merged path.

        Removes all rows matching *old_paths* for the given filter and
        inserts one row pointing to *merged_path*.  This keeps the
        dashboard pointing at the final merged file after
        :func:`~physicsnemo_curator.run.gather_pipeline` completes.

        Parameters
        ----------
        filter_name : str
            Human-readable name of the filter.
        filter_order : int
            Position of the filter in the pipeline (0-indexed).
        old_paths : list[str]
            Shard file paths to remove from the artifact table.
        merged_path : str
            Path to the merged output file.
        """
        conn = self._connect()
        try:
            # Remove old shard records
            placeholders = ",".join("?" for _ in old_paths)
            conn.execute(
                f"DELETE FROM filter_artifacts WHERE run_id = ? AND path IN ({placeholders})",  # noqa: S608
                (self._run_id, *old_paths),
            )
            # Insert merged path (use idx=0 as representative)
            conn.execute(
                "INSERT OR REPLACE INTO filter_artifacts "
                "(path, idx, run_id, filter_name, filter_order) "
                "VALUES (?, ?, ?, ?, ?)",
                (merged_path, 0, self._run_id, filter_name, filter_order),
            )
            conn.commit()
        finally:
            conn.close()

    def filter_artifacts_for_index(self, index: int) -> dict[str, list[str]]:
        """Return filter artifact paths for a given source index.

        Parameters
        ----------
        index : int
            Source index to query.

        Returns
        -------
        dict[str, list[str]]
            Mapping of filter name to list of artifact paths.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT filter_name, path FROM filter_artifacts "
                "WHERE idx = ? AND run_id = ? ORDER BY filter_order, path",
                (index, self._run_id),
            ).fetchall()
            result: dict[str, list[str]] = {}
            for name, path in rows:
                result.setdefault(name, []).append(path)
            return result
        finally:
            conn.close()

    def all_filter_artifacts(self) -> dict[str, list[str]]:
        """Return all filter artifact paths grouped by filter name.

        Returns
        -------
        dict[str, list[str]]
            Mapping of filter name to list of all artifact paths.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT filter_name, path FROM filter_artifacts WHERE run_id = ? ORDER BY filter_order, idx, path",
                (self._run_id,),
            ).fetchall()
            result: dict[str, list[str]] = {}
            for name, path in rows:
                result.setdefault(name, []).append(path)
            return result
        finally:
            conn.close()

    # -- Worker progress tracking ------------------------------------------------

    def register_worker(self, worker_id: str, pid: int, hostname: str, invocation_id: str | None = None) -> None:
        """Register a worker or update its heartbeat if already known.

        Parameters
        ----------
        worker_id : str
            Unique identifier for this worker (UUID hex).
        pid : int
            OS process ID of the worker.
        hostname : str
            Hostname of the machine running the worker.
        invocation_id : str | None, optional
            Unique identifier for this ``run_pipeline`` invocation.
            Used to partition workers when the same pipeline is run
            concurrently with different index sets.
        """
        now = datetime.now(tz=UTC).isoformat()
        conn = self._connect()
        try:
            # Use INSERT OR REPLACE to reset worker state for new invocations.
            # This ensures completed_count starts at 0 for each run.
            conn.execute(
                "INSERT OR REPLACE INTO workers "
                "(worker_id, run_id, pid, hostname, started_at, last_heartbeat, "
                "current_index, completed_count, invocation_id) "
                "VALUES (?, ?, ?, ?, ?, ?, NULL, 0, ?)",
                (worker_id, self._run_id, pid, hostname, now, now, invocation_id),
            )
            conn.commit()
        finally:
            conn.close()

    def worker_start_index(self, worker_id: str, index: int) -> None:
        """Record that a worker is starting to process an index.

        Parameters
        ----------
        worker_id : str
            Unique identifier for this worker.
        index : int
            Source index being processed.
        """
        now = datetime.now(tz=UTC).isoformat()
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE workers SET current_index = ?, last_heartbeat = ? WHERE worker_id = ?",
                (index, now, worker_id),
            )
            conn.commit()
        finally:
            conn.close()

    def worker_finish_index(self, worker_id: str) -> None:
        """Record that a worker has finished processing its current index.

        Parameters
        ----------
        worker_id : str
            Unique identifier for this worker.
        """
        now = datetime.now(tz=UTC).isoformat()
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE workers SET current_index = NULL, last_heartbeat = ?,"
                " completed_count = completed_count + 1 WHERE worker_id = ?",
                (now, worker_id),
            )
            conn.commit()
        finally:
            conn.close()

    def active_workers(self, invocation_id: str | None = None) -> list[dict[str, Any]]:
        """Return all workers registered for this pipeline run.

        Parameters
        ----------
        invocation_id : str | None, optional
            If provided, only return workers from this invocation.

        Returns
        -------
        list[dict[str, Any]]
            List of worker dictionaries with keys: ``worker_id``, ``pid``,
            ``hostname``, ``started_at``, ``last_heartbeat``, ``current_index``,
            ``completed_count``, ``invocation_id``.
        """
        conn = self._connect()
        try:
            query = (
                "SELECT worker_id, pid, hostname, started_at, last_heartbeat, current_index,"
                " completed_count, invocation_id "
                "FROM workers WHERE run_id = ?"
            )
            params: list[Any] = [self._run_id]
            if invocation_id is not None:
                query += " AND invocation_id = ?"
                params.append(invocation_id)
            query += " ORDER BY started_at"
            rows = conn.execute(query, params).fetchall()
            return [
                {
                    "worker_id": r[0],
                    "pid": r[1],
                    "hostname": r[2],
                    "started_at": r[3],
                    "last_heartbeat": r[4],
                    "current_index": r[5],
                    "completed_count": r[6],
                    "invocation_id": r[7],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def metrics(self) -> PipelineMetrics:
        """Build aggregated metrics from the database.

        Returns
        -------
        PipelineMetrics
            Aggregated metrics across all completed indices in this run.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT idx, wall_time_ns, peak_memory_bytes, gpu_memory_bytes "
                "FROM index_results WHERE run_id = ? AND status = 'completed' "
                "ORDER BY idx",
                (self._run_id,),
            ).fetchall()

            index_metrics_list: list[IndexMetrics] = []
            for row in rows:
                idx, wall_ns, peak_mem, gpu_mem = row

                # Fetch stage metrics for this index
                stage_rows = conn.execute(
                    "SELECT stage_name, wall_time_ns FROM stage_metrics "
                    "WHERE idx = ? AND run_id = ? ORDER BY stage_order",
                    (idx, self._run_id),
                ).fetchall()
                stages = [StageMetrics(name=sr[0], wall_time_ns=sr[1]) for sr in stage_rows]

                index_metrics_list.append(
                    IndexMetrics(
                        index=idx,
                        stages=stages,
                        wall_time_ns=wall_ns,
                        peak_memory_bytes=peak_mem,
                        gpu_memory_bytes=gpu_mem,
                    )
                )

            return PipelineMetrics(indices=index_metrics_list)
        finally:
            conn.close()

    def index_metrics(self, index: int) -> IndexMetrics | None:
        """Retrieve metrics for a single index.

        Parameters
        ----------
        index : int
            Source index to query.

        Returns
        -------
        IndexMetrics | None
            Metrics for the index, or ``None`` if not found.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT wall_time_ns, peak_memory_bytes, gpu_memory_bytes "
                "FROM index_results WHERE idx = ? AND run_id = ? AND status = 'completed'",
                (index, self._run_id),
            ).fetchone()
            if row is None:
                return None

            wall_ns, peak_mem, gpu_mem = row

            stage_rows = conn.execute(
                "SELECT stage_name, wall_time_ns FROM stage_metrics WHERE idx = ? AND run_id = ? ORDER BY stage_order",
                (index, self._run_id),
            ).fetchall()
            stages = [StageMetrics(name=sr[0], wall_time_ns=sr[1]) for sr in stage_rows]

            return IndexMetrics(
                index=index,
                stages=stages,
                wall_time_ns=wall_ns,
                peak_memory_bytes=peak_mem,
                gpu_memory_bytes=gpu_mem,
            )
        finally:
            conn.close()
