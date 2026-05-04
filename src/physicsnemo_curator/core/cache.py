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

"""Cache directory management and introspection for pipeline SQLite databases.

Provides utilities to locate, list, inspect, and clean up ``.db`` files
produced by pipeline runs.  The default cache location follows the
`XDG Base Directory Specification`_ and can be overridden with the
``PSNC_CACHE_DIR`` environment variable.

.. _XDG Base Directory Specification:
   https://specifications.freedesktop.org/basedir-spec/latest/

Usage
-----
>>> from physicsnemo_curator.core.cache import default_cache_dir, list_databases
>>> cache = default_cache_dir()
>>> for info in list_databases(cache):
...     print(info.hash_prefix, info.source_name, info.completed)
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default cache directory
# ---------------------------------------------------------------------------


def default_cache_dir() -> pathlib.Path:
    """Return the default cache directory for pipeline databases.

    Resolution order (highest priority first):

    1. ``PSNC_CACHE_DIR`` environment variable
    2. ``$XDG_CACHE_HOME/psnc/``
    3. ``~/.cache/psnc/``

    Returns
    -------
    pathlib.Path
        Absolute path to the cache directory (may not exist yet).

    Examples
    --------
    >>> import os
    >>> os.environ["PSNC_CACHE_DIR"] = "/tmp/my_cache"
    >>> default_cache_dir()
    PosixPath('/tmp/my_cache')
    """
    psnc = os.environ.get("PSNC_CACHE_DIR")
    if psnc:
        return pathlib.Path(psnc)

    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return pathlib.Path(xdg) / "psnc"

    return pathlib.Path.home() / ".cache" / "psnc"


def default_data_cache_dir(source_name: str) -> pathlib.Path:
    """Return the persistent cache directory for downloaded source data.

    Provides a standard location for remote sources to store downloaded
    files so they persist across pipeline runs.  The directory is created
    if it does not yet exist.

    Resolution order follows :func:`default_cache_dir`, with
    ``data/<source_name>`` appended:

    1. ``$PSNC_CACHE_DIR/data/<source_name>/``
    2. ``$XDG_CACHE_HOME/psnc/data/<source_name>/``
    3. ``~/.cache/psnc/data/<source_name>/``

    Parameters
    ----------
    source_name : str
        Short identifier for the source (e.g. ``"drivaerml"``,
        ``"ahmedml"``).  Used as the subdirectory name.

    Returns
    -------
    pathlib.Path
        Absolute path to the data cache directory (created if needed).

    Examples
    --------
    >>> default_data_cache_dir("drivaerml")  # doctest: +SKIP
    PosixPath('/home/user/.cache/psnc/data/drivaerml')
    """
    data_dir = default_cache_dir() / "data" / source_name
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


# ---------------------------------------------------------------------------
# DBInfo dataclass
# ---------------------------------------------------------------------------


@dataclass
class DBInfo:
    """Metadata about a single pipeline database file.

    Parameters
    ----------
    hash_prefix : str
        Filename stem (the config hash prefix used as the DB name).
    path : pathlib.Path
        Absolute path to the ``.db`` file.
    size_bytes : int
        File size in bytes.
    created : datetime
        Pipeline run start timestamp (from ``pipeline_runs.started_at``).
    source_name : str
        Registered source name extracted from the stored config JSON.
    sink_name : str
        Registered sink name extracted from the stored config JSON.
    filter_names : list[str]
        Registered filter names extracted from the stored config JSON.
    total : int
        Total number of ``index_results`` rows (completed + failed).
    completed : int
        Number of completed index results.
    failed : int
        Number of failed index results.
    """

    hash_prefix: str
    path: pathlib.Path
    size_bytes: int
    created: datetime
    source_name: str
    sink_name: str
    filter_names: list[str] = field(default_factory=list)
    total: int = 0
    completed: int = 0
    failed: int = 0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _read_db_info(db_path: pathlib.Path) -> DBInfo | None:
    """Read metadata from a single pipeline database file.

    Parameters
    ----------
    db_path : pathlib.Path
        Path to a ``.db`` file.

    Returns
    -------
    DBInfo | None
        Metadata if the DB is valid, or ``None`` if it is corrupt /
        unreadable.
    """
    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        try:
            row = conn.execute(
                "SELECT config_hash, config_json, started_at FROM pipeline_runs ORDER BY run_id DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return None

            config_hash, config_json, started_at = row

            # Parse config for source / sink / filter names
            config = json.loads(config_json)
            source_name = config.get("source", {}).get("name", "")
            sink_name = config.get("sink", {}).get("name", "")
            filter_names = [f.get("name", "") for f in config.get("filters", [])]

            # Parse started_at timestamp
            created = datetime.fromisoformat(started_at)
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)

            # Count index results
            completed_row = conn.execute("SELECT COUNT(*) FROM index_results WHERE status = 'completed'").fetchone()
            completed = completed_row[0] if completed_row else 0

            failed_row = conn.execute("SELECT COUNT(*) FROM index_results WHERE status = 'error'").fetchone()
            failed = failed_row[0] if failed_row else 0

            total = completed + failed

            return DBInfo(
                hash_prefix=db_path.stem,
                path=db_path.resolve(),
                size_bytes=db_path.stat().st_size,
                created=created,
                source_name=source_name,
                sink_name=sink_name,
                filter_names=filter_names,
                total=total,
                completed=completed,
                failed=failed,
            )
        finally:
            conn.close()
    except (sqlite3.Error, json.JSONDecodeError, OSError) as exc:
        logger.debug("Skipping corrupt or unreadable DB %s: %s", db_path, exc)
        return None


def _resolve_cache_dir(cache_dir: pathlib.Path | None) -> pathlib.Path:
    """Resolve the cache directory, falling back to the default.

    Parameters
    ----------
    cache_dir : pathlib.Path | None
        Explicit cache directory, or ``None`` to use :func:`default_cache_dir`.

    Returns
    -------
    pathlib.Path
        Resolved cache directory path.
    """
    if cache_dir is not None:
        return pathlib.Path(cache_dir)
    return default_cache_dir()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_databases(cache_dir: pathlib.Path | None = None) -> list[DBInfo]:
    """List all pipeline databases in the cache directory.

    Opens each ``.db`` file, reads the ``pipeline_runs`` and
    ``index_results`` tables, and returns metadata sorted newest first
    (by ``started_at`` timestamp).  Corrupt or unreadable databases are
    silently skipped.

    Parameters
    ----------
    cache_dir : pathlib.Path | None, optional
        Directory to scan.  Defaults to :func:`default_cache_dir`.

    Returns
    -------
    list[DBInfo]
        Metadata for each valid database, sorted newest first.
    """
    d = _resolve_cache_dir(cache_dir)
    if not d.is_dir():
        return []

    infos: list[DBInfo] = []
    for p in d.glob("*.db"):
        info = _read_db_info(p)
        if info is not None:
            infos.append(info)

    # Sort newest first by created timestamp
    infos.sort(key=lambda x: x.created, reverse=True)
    return infos


def remove_databases(
    identifiers: list[str],
    *,
    cache_dir: pathlib.Path | None = None,
) -> int:
    """Remove pipeline databases matching the given identifiers.

    Each identifier is first tested as an exact stem match.  If no exact
    match is found it is treated as a prefix and matched against ``.db``
    file stems.  A prefix that matches more than one file raises
    :class:`ValueError` to prevent accidental deletion.

    Parameters
    ----------
    identifiers : list[str]
        Full stems or prefix strings to match against DB file stems.
    cache_dir : pathlib.Path | None, optional
        Directory to scan.  Defaults to :func:`default_cache_dir`.

    Returns
    -------
    int
        Number of database files removed.

    Raises
    ------
    ValueError
        If a prefix is ambiguous (matches more than one ``.db`` file).
    """
    d = _resolve_cache_dir(cache_dir)
    if not d.is_dir():
        return 0

    db_files = list(d.glob("*.db"))
    stems_to_files = {f.stem: f for f in db_files}
    removed = 0

    for ident in identifiers:
        # Try exact stem match first
        if ident in stems_to_files:
            stems_to_files[ident].unlink()
            removed += 1
            continue

        # Fall back to prefix matching
        matches = [f for f in db_files if f.stem.startswith(ident)]
        if len(matches) > 1:
            stems = [f.stem for f in matches]
            msg = f"Identifier {ident!r} is ambiguous, matches {len(matches)} databases: {stems}"
            raise ValueError(msg)
        if len(matches) == 1:
            matches[0].unlink()
            removed += 1

    return removed


def remove_older_than(
    max_age: timedelta,
    *,
    cache_dir: pathlib.Path | None = None,
) -> int:
    """Remove pipeline databases older than *max_age* (by file mtime).

    Parameters
    ----------
    max_age : timedelta
        Maximum age.  Files with an mtime older than
        ``now - max_age`` are removed.
    cache_dir : pathlib.Path | None, optional
        Directory to scan.  Defaults to :func:`default_cache_dir`.

    Returns
    -------
    int
        Number of database files removed.
    """
    d = _resolve_cache_dir(cache_dir)
    if not d.is_dir():
        return 0

    import time

    cutoff = time.time() - max_age.total_seconds()
    removed = 0

    for p in d.glob("*.db"):
        if p.stat().st_mtime < cutoff:
            p.unlink()
            removed += 1

    return removed


def clear_cache(*, cache_dir: pathlib.Path | None = None) -> int:
    """Remove all ``.db`` files from the cache directory.

    Parameters
    ----------
    cache_dir : pathlib.Path | None, optional
        Directory to clear.  Defaults to :func:`default_cache_dir`.

    Returns
    -------
    int
        Number of database files removed.
    """
    d = _resolve_cache_dir(cache_dir)
    if not d.is_dir():
        return 0

    removed = 0
    for p in d.glob("*.db"):
        p.unlink()
        removed += 1

    return removed


def cache_size(*, cache_dir: pathlib.Path | None = None) -> int:
    """Return the total size in bytes of all ``.db`` files in the cache.

    Parameters
    ----------
    cache_dir : pathlib.Path | None, optional
        Directory to measure.  Defaults to :func:`default_cache_dir`.

    Returns
    -------
    int
        Total bytes occupied by ``.db`` files, or ``0`` if the
        directory is empty or does not exist.
    """
    d = _resolve_cache_dir(cache_dir)
    if not d.is_dir():
        return 0

    return sum(p.stat().st_size for p in d.glob("*.db"))
