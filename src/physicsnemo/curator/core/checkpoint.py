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

"""Pipeline checkpointing with SQLite-backed provenance tracking.

Provides :class:`CheckpointedPipeline`, a transparent wrapper around
:class:`~physicsnemo.curator.core.base.Pipeline` that records completed
indices in a SQLite database.  On restart the wrapper skips indices that
already finished, returning their cached output paths without re-running
source, filters, or sink.

The checkpoint also stores full pipeline provenance (source, filter, and
sink configurations) so that configuration drift between runs is detected
and logged as a warning.

Examples
--------
>>> from physicsnemo.curator import Pipeline, run_pipeline
>>> from physicsnemo.curator.core.checkpoint import CheckpointedPipeline
>>> pipeline = source.filter(filt).write(sink)
>>> cp = CheckpointedPipeline(pipeline, db_path="run.checkpoint.db")
>>> run_pipeline(cp, n_jobs=4)          # first run  # doctest: +SKIP
>>> run_pipeline(cp, n_jobs=4)          # restart — skips completed  # doctest: +SKIP
>>> cp.summary()                        # doctest: +SKIP
{'total': 80, 'completed': 80, 'remaining': 0, ...}
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import sqlite3
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from physicsnemo.curator.core.base import REQUIRED, Filter, Sink, Source

if TYPE_CHECKING:
    import pathlib

    from physicsnemo.curator.core.base import Pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    config_hash  TEXT    NOT NULL,
    config_json  TEXT    NOT NULL,
    started_at   TEXT    NOT NULL,
    UNIQUE(config_hash)
);

CREATE TABLE IF NOT EXISTS completed_indices (
    idx          INTEGER PRIMARY KEY,
    run_id       INTEGER NOT NULL REFERENCES pipeline_runs(run_id),
    output_paths TEXT    NOT NULL,
    completed_at TEXT    NOT NULL,
    elapsed_ns   INTEGER,
    error        TEXT
);
"""


# ---------------------------------------------------------------------------
# Config serialization helpers
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
    # pathlib.Path, other types → string
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
# CheckpointedPipeline
# ---------------------------------------------------------------------------


class CheckpointedPipeline[T]:
    """Transparent checkpointing wrapper around :class:`~physicsnemo.curator.core.base.Pipeline`.

    Duck-type compatible with ``Pipeline`` — exposes ``source``, ``filters``,
    ``sink``, ``__len__``, and ``__getitem__``.  Can be passed directly to
    :func:`~physicsnemo.curator.run.run_pipeline` without backend changes.

    On each ``__getitem__`` call the wrapper checks the SQLite database for
    a prior completion record.  If found, the cached output paths are
    returned immediately without re-executing the pipeline.  Otherwise the
    inner pipeline is executed, and the result is recorded.

    Pipeline provenance (source class, filter params, sink config, etc.) is
    hashed and stored.  On restart, a config mismatch triggers a warning but
    does **not** block resumption.

    Parameters
    ----------
    pipeline : Pipeline[T]
        The pipeline to wrap.
    db_path : str | pathlib.Path
        Path to the SQLite checkpoint database.  Created automatically
        if it does not exist.

    Examples
    --------
    >>> from physicsnemo.curator import Pipeline, run_pipeline
    >>> from physicsnemo.curator.core.checkpoint import CheckpointedPipeline
    >>> cp = CheckpointedPipeline(pipeline, db_path="run.db")
    >>> run_pipeline(cp, n_jobs=4)          # doctest: +SKIP
    >>> print(cp.summary())                 # doctest: +SKIP
    """

    def __init__(self, pipeline: Pipeline[T] | Any, db_path: str | pathlib.Path) -> None:
        """Initialize the checkpointing wrapper.

        Parameters
        ----------
        pipeline : Pipeline[T] | Any
            Inner pipeline to wrap.  Accepts ``Pipeline``,
            ``ProfiledPipeline``, or any duck-type-compatible object
            with ``source``, ``filters``, ``sink``, ``__len__``, and
            ``__getitem__``.
        db_path : str | pathlib.Path
            Filesystem path for the SQLite checkpoint file.
        """
        import pathlib as _pathlib

        self._pipeline = pipeline
        self._db_path = _pathlib.Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize pipeline config and compute hash
        self._config = _pipeline_config(pipeline)
        self._hash = _config_hash(self._config)

        # Initialize database
        self._init_db()

    # -- Database helpers -----------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open a WAL-mode connection to the checkpoint database.

        Returns
        -------
        sqlite3.Connection
            Database connection with WAL journal mode.
        """
        conn = sqlite3.connect(str(self._db_path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_db(self) -> None:
        """Create schema and register the current pipeline run."""
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA_SQL)

            # Check for existing run with same config hash
            row = conn.execute(
                "SELECT run_id, config_json FROM pipeline_runs WHERE config_hash = ?",
                (self._hash,),
            ).fetchone()

            if row is not None:
                self._run_id: int = row[0]
                logger.info("Resuming checkpoint run_id=%d (config hash %s…)", self._run_id, self._hash[:12])
            else:
                # Check for runs with a *different* hash (config drift)
                other = conn.execute("SELECT config_hash FROM pipeline_runs LIMIT 1").fetchone()
                if other is not None:
                    logger.warning(
                        "Pipeline config has changed since the original checkpoint "
                        "(stored hash %s…, current hash %s…). "
                        "Resuming anyway — completed indices from prior config will be kept.",
                        other[0][:12],
                        self._hash[:12],
                    )

                now = datetime.now(tz=UTC).isoformat()
                cur = conn.execute(
                    "INSERT INTO pipeline_runs (config_hash, config_json, started_at) VALUES (?, ?, ?)",
                    (self._hash, json.dumps(self._config, sort_keys=True, default=str), now),
                )
                self._run_id = cur.lastrowid  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
                logger.info("New checkpoint run_id=%d (config hash %s…)", self._run_id, self._hash[:12])

            conn.commit()
        finally:
            conn.close()

    def _is_completed(self, index: int) -> list[str] | None:
        """Check if an index has already been completed.

        Parameters
        ----------
        index : int
            Source index.

        Returns
        -------
        list[str] | None
            Cached output paths if completed, ``None`` otherwise.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT output_paths FROM completed_indices WHERE idx = ? AND error IS NULL",
                (index,),
            ).fetchone()
            if row is not None:
                return json.loads(row[0])  # type: ignore[no-any-return]
            return None
        finally:
            conn.close()

    def _record_completion(
        self,
        index: int,
        output_paths: list[str],
        elapsed_ns: int,
    ) -> None:
        """Record a successful index completion.

        Parameters
        ----------
        index : int
            Source index that completed.
        output_paths : list[str]
            Paths written by the sink.
        elapsed_ns : int
            Wall-clock time in nanoseconds.
        """
        now = datetime.now(tz=UTC).isoformat()
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO completed_indices "
                "(idx, run_id, output_paths, completed_at, elapsed_ns) "
                "VALUES (?, ?, ?, ?, ?)",
                (index, self._run_id, json.dumps(output_paths), now, elapsed_ns),
            )
            conn.commit()
        finally:
            conn.close()

    def _record_error(self, index: int, error: str, elapsed_ns: int) -> None:
        """Record a failed index execution.

        Parameters
        ----------
        index : int
            Source index that failed.
        error : str
            Error message.
        elapsed_ns : int
            Wall-clock time before the error.
        """
        now = datetime.now(tz=UTC).isoformat()
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO completed_indices "
                "(idx, run_id, output_paths, completed_at, elapsed_ns, error) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (index, self._run_id, "[]", now, elapsed_ns, error),
            )
            conn.commit()
        finally:
            conn.close()

    # -- Duck-type Pipeline protocol ------------------------------------------

    @property
    def source(self) -> Source[T]:
        """Return the wrapped pipeline's source.

        Returns
        -------
        Source[T]
            The underlying source.
        """
        return self._pipeline.source

    @property
    def filters(self) -> list[Filter[T]]:
        """Return the wrapped pipeline's filter list.

        Returns
        -------
        list[Filter[T]]
            The underlying filters.
        """
        return self._pipeline.filters

    @property
    def sink(self) -> Sink[T] | None:
        """Return the wrapped pipeline's sink.

        Returns
        -------
        Sink[T] | None
            The underlying sink, or ``None``.
        """
        return self._pipeline.sink

    def __len__(self) -> int:
        """Return the number of items in the source.

        Returns
        -------
        int
            Total number of source items.
        """
        return len(self._pipeline)

    def __getitem__(self, index: int) -> list[str]:
        """Process the given index, skipping if already checkpointed.

        If *index* was previously completed successfully, the cached output
        paths are returned immediately without executing the inner pipeline.
        Otherwise the full pipeline chain (source → filters → sink) runs
        and the result is recorded.

        Parameters
        ----------
        index : int
            Zero-based index into the source.

        Returns
        -------
        list[str]
            File paths produced by the sink.
        """
        # Check for cached completion
        cached = self._is_completed(index)
        if cached is not None:
            logger.debug("Checkpoint hit for index %d — returning cached paths", index)
            return cached

        # Execute the inner pipeline
        t0 = time.perf_counter_ns()
        try:
            result = self._pipeline[index]
        except Exception as exc:
            elapsed = time.perf_counter_ns() - t0
            self._record_error(index, str(exc), elapsed)
            raise
        elapsed = time.perf_counter_ns() - t0

        # Record success
        self._record_completion(index, result, elapsed)
        logger.debug("Checkpoint recorded for index %d (%d paths, %.2fs)", index, len(result), elapsed / 1e9)

        return result

    # -- Checkpoint query API -------------------------------------------------

    @property
    def db_path(self) -> pathlib.Path:
        """Return the path to the checkpoint database.

        Returns
        -------
        pathlib.Path
            Database file path.
        """
        return self._db_path

    @property
    def config_hash(self) -> str:
        """Return the SHA-256 hash of the current pipeline config.

        Returns
        -------
        str
            Hex-encoded hash string.
        """
        return self._hash

    @property
    def completed_indices(self) -> set[int]:
        """Return the set of successfully completed indices.

        Returns
        -------
        set[int]
            Indices that have been processed without error.
        """
        conn = self._connect()
        try:
            rows = conn.execute("SELECT idx FROM completed_indices WHERE error IS NULL").fetchall()
            return {r[0] for r in rows}
        finally:
            conn.close()

    @property
    def failed_indices(self) -> dict[int, str]:
        """Return indices that failed with their error messages.

        Returns
        -------
        dict[int, str]
            Mapping from index to error message string.
        """
        conn = self._connect()
        try:
            rows = conn.execute("SELECT idx, error FROM completed_indices WHERE error IS NOT NULL").fetchall()
            return {r[0]: r[1] for r in rows}
        finally:
            conn.close()

    @property
    def remaining_indices(self) -> list[int]:
        """Return indices that have not yet been completed successfully.

        Returns
        -------
        list[int]
            Sorted list of indices still needing processing.
        """
        done = self.completed_indices
        return sorted(i for i in range(len(self)) if i not in done)

    def reset(self) -> None:
        """Clear all checkpoint records and start fresh.

        This deletes all completion records and pipeline run metadata from
        the database.  The database file itself is kept.
        """
        conn = self._connect()
        try:
            conn.execute("DELETE FROM completed_indices")
            conn.execute("DELETE FROM pipeline_runs")
            conn.commit()
        finally:
            conn.close()

        # Re-initialize to register the current config
        self._init_db()
        logger.info("Checkpoint reset — all records cleared")

    def summary(self) -> dict[str, Any]:
        """Return a summary of the checkpoint state.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys: ``total``, ``completed``, ``failed``,
            ``remaining``, ``config_hash``, ``db_path``,
            ``total_elapsed_s``.
        """
        conn = self._connect()
        try:
            total = len(self)
            completed = conn.execute("SELECT COUNT(*) FROM completed_indices WHERE error IS NULL").fetchone()[0]
            failed = conn.execute("SELECT COUNT(*) FROM completed_indices WHERE error IS NOT NULL").fetchone()[0]
            elapsed_row = conn.execute(
                "SELECT COALESCE(SUM(elapsed_ns), 0) FROM completed_indices WHERE error IS NULL"
            ).fetchone()
            total_elapsed_ns: int = elapsed_row[0]
        finally:
            conn.close()

        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "remaining": total - completed,
            "config_hash": self._hash,
            "db_path": str(self._db_path),
            "total_elapsed_s": total_elapsed_ns / 1e9,
        }
