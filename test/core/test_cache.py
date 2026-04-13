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

"""Tests for the cache directory management module."""

from __future__ import annotations

import json
import os
import pathlib
import sqlite3
import time
from datetime import UTC, datetime, timedelta

import pytest

from physicsnemo_curator.core.cache import (
    DBInfo,
    cache_size,
    clear_cache,
    default_cache_dir,
    list_databases,
    remove_databases,
    remove_older_than,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_fake_db(
    db_path: pathlib.Path,
    *,
    source_name: str = "TestSource",
    sink_name: str = "TestSink",
    filter_names: list[str] | None = None,
    num_completed: int = 3,
    num_failed: int = 1,
    config_hash: str = "abc123def456",
) -> pathlib.Path:
    """Create a minimal SQLite DB mimicking the pipeline schema.

    Parameters
    ----------
    db_path : pathlib.Path
        Where to write the DB.
    source_name : str
        Source name stored in config JSON.
    sink_name : str
        Sink name stored in config JSON.
    filter_names : list[str] | None
        Filter names stored in config JSON.
    num_completed : int
        Number of completed index_results rows.
    num_failed : int
        Number of failed index_results rows.
    config_hash : str
        The config hash to store.

    Returns
    -------
    pathlib.Path
        Path to the created DB file.
    """
    if filter_names is None:
        filter_names = ["FilterA"]

    db_path.parent.mkdir(parents=True, exist_ok=True)

    config_json = json.dumps(
        {
            "source": {"name": source_name},
            "filters": [{"name": n} for n in filter_names],
            "sink": {"name": sink_name},
        }
    )
    started_at = datetime.now(tz=UTC).isoformat()

    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """\
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                config_hash  TEXT    UNIQUE NOT NULL,
                config_json  TEXT    NOT NULL,
                started_at   TEXT    NOT NULL
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
                PRIMARY KEY (idx, run_id),
                FOREIGN KEY (run_id) REFERENCES pipeline_runs (run_id)
            );
        """
        )
        conn.execute(
            "INSERT INTO pipeline_runs (config_hash, config_json, started_at) VALUES (?, ?, ?)",
            (config_hash, config_json, started_at),
        )
        run_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        now = datetime.now(tz=UTC).isoformat()
        for i in range(num_completed):
            conn.execute(
                "INSERT INTO index_results (idx, run_id, status, output_paths, completed_at) "
                "VALUES (?, ?, 'completed', '[]', ?)",
                (i, run_id, now),
            )
        for i in range(num_failed):
            conn.execute(
                "INSERT INTO index_results (idx, run_id, status, completed_at, error) "
                "VALUES (?, ?, 'error', ?, 'some error')",
                (num_completed + i, run_id, now),
            )
        conn.commit()
    finally:
        conn.close()

    return db_path


# ---------------------------------------------------------------------------
# TestDefaultCacheDir
# ---------------------------------------------------------------------------


class TestDefaultCacheDir:
    """Tests for default_cache_dir()."""

    def test_xdg_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without env vars, falls back to ~/.cache/psnc/."""
        monkeypatch.delenv("PSNC_CACHE_DIR", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        result = default_cache_dir()
        expected = pathlib.Path.home() / ".cache" / "psnc"
        assert result == expected

    def test_xdg_cache_home_respected(self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
        """XDG_CACHE_HOME is respected when set."""
        monkeypatch.delenv("PSNC_CACHE_DIR", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "custom_cache"))
        result = default_cache_dir()
        assert result == tmp_path / "custom_cache" / "psnc"

    def test_psnc_cache_dir_overrides(self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
        """PSNC_CACHE_DIR takes highest priority."""
        monkeypatch.setenv("PSNC_CACHE_DIR", str(tmp_path / "my_cache"))
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        result = default_cache_dir()
        assert result == tmp_path / "my_cache"


# ---------------------------------------------------------------------------
# TestListDatabases
# ---------------------------------------------------------------------------


class TestListDatabases:
    """Tests for list_databases()."""

    def test_empty_dir(self, tmp_path: pathlib.Path) -> None:
        """Empty directory returns empty list."""
        result = list_databases(cache_dir=tmp_path)
        assert result == []

    def test_lists_with_metadata(self, tmp_path: pathlib.Path) -> None:
        """Lists DB files and extracts correct metadata."""
        _create_fake_db(
            tmp_path / "abc123def456.db",
            source_name="MySource",
            sink_name="MySink",
            filter_names=["F1", "F2"],
            num_completed=5,
            num_failed=2,
            config_hash="abc123def456",
        )

        result = list_databases(cache_dir=tmp_path)
        assert len(result) == 1

        info = result[0]
        assert isinstance(info, DBInfo)
        assert info.hash_prefix == "abc123def456"
        assert info.path == tmp_path / "abc123def456.db"
        assert info.size_bytes > 0
        assert info.source_name == "MySource"
        assert info.sink_name == "MySink"
        assert info.filter_names == ["F1", "F2"]
        assert info.total == 7  # 5 completed + 2 failed
        assert info.completed == 5
        assert info.failed == 2
        assert isinstance(info.created, datetime)

    def test_skips_corrupt_db(self, tmp_path: pathlib.Path) -> None:
        """Corrupt DB files are silently skipped."""
        # Create a valid DB
        _create_fake_db(tmp_path / "valid123.db", config_hash="valid123")

        # Create a corrupt "DB" file
        corrupt_path = tmp_path / "corrupt456.db"
        corrupt_path.write_text("this is not a sqlite database")

        result = list_databases(cache_dir=tmp_path)
        assert len(result) == 1
        assert result[0].hash_prefix == "valid123"

    def test_nonexistent_dir(self, tmp_path: pathlib.Path) -> None:
        """Nonexistent directory returns empty list."""
        result = list_databases(cache_dir=tmp_path / "nonexistent")
        assert result == []

    def test_sorted_newest_first(self, tmp_path: pathlib.Path) -> None:
        """Results are sorted with newest created timestamp first."""
        _create_fake_db(tmp_path / "older111.db", config_hash="older111")
        # Ensure different timestamps
        time.sleep(0.05)
        _create_fake_db(tmp_path / "newer222.db", config_hash="newer222")

        result = list_databases(cache_dir=tmp_path)
        assert len(result) == 2
        assert result[0].hash_prefix == "newer222"
        assert result[1].hash_prefix == "older111"


# ---------------------------------------------------------------------------
# TestRemoveDatabases
# ---------------------------------------------------------------------------


class TestRemoveDatabases:
    """Tests for remove_databases()."""

    def test_remove_by_prefix(self, tmp_path: pathlib.Path) -> None:
        """Remove a specific DB by its hash prefix."""
        _create_fake_db(tmp_path / "abc123.db", config_hash="abc123")
        _create_fake_db(tmp_path / "def456.db", config_hash="def456")

        count = remove_databases(["abc123"], cache_dir=tmp_path)
        assert count == 1
        assert not (tmp_path / "abc123.db").exists()
        assert (tmp_path / "def456.db").exists()

    def test_no_match_returns_zero(self, tmp_path: pathlib.Path) -> None:
        """No matching prefix returns 0."""
        _create_fake_db(tmp_path / "abc123.db", config_hash="abc123")

        count = remove_databases(["zzz999"], cache_dir=tmp_path)
        assert count == 0
        assert (tmp_path / "abc123.db").exists()

    def test_ambiguous_prefix_raises(self, tmp_path: pathlib.Path) -> None:
        """Ambiguous prefix (matches multiple DBs) raises ValueError."""
        _create_fake_db(tmp_path / "abc123.db", config_hash="abc123")
        _create_fake_db(tmp_path / "abc456.db", config_hash="abc456")

        with pytest.raises(ValueError, match="ambiguous"):
            remove_databases(["abc"], cache_dir=tmp_path)


# ---------------------------------------------------------------------------
# TestRemoveOlderThan
# ---------------------------------------------------------------------------


class TestRemoveOlderThan:
    """Tests for remove_older_than()."""

    def test_removes_old_keeps_new(self, tmp_path: pathlib.Path) -> None:
        """Remove DBs older than max_age by mtime, keep newer ones."""
        old_db = _create_fake_db(tmp_path / "old111.db", config_hash="old111")
        new_db = _create_fake_db(tmp_path / "new222.db", config_hash="new222")

        # Set the old DB mtime to 10 days ago
        ten_days_ago = time.time() - (10 * 86400)
        os.utime(old_db, (ten_days_ago, ten_days_ago))

        count = remove_older_than(timedelta(days=5), cache_dir=tmp_path)
        assert count == 1
        assert not old_db.exists()
        assert new_db.exists()

    def test_keeps_all_when_none_old(self, tmp_path: pathlib.Path) -> None:
        """No removals when all DBs are recent."""
        _create_fake_db(tmp_path / "recent.db", config_hash="recent")
        count = remove_older_than(timedelta(days=30), cache_dir=tmp_path)
        assert count == 0


# ---------------------------------------------------------------------------
# TestClearCache
# ---------------------------------------------------------------------------


class TestClearCache:
    """Tests for clear_cache()."""

    def test_removes_all_db_files(self, tmp_path: pathlib.Path) -> None:
        """Removes all .db files in the cache directory."""
        _create_fake_db(tmp_path / "a.db", config_hash="a")
        _create_fake_db(tmp_path / "b.db", config_hash="b")

        # Also create a non-db file that should NOT be removed
        (tmp_path / "keep_me.txt").write_text("safe")

        count = clear_cache(cache_dir=tmp_path)
        assert count == 2
        assert not (tmp_path / "a.db").exists()
        assert not (tmp_path / "b.db").exists()
        assert (tmp_path / "keep_me.txt").exists()

    def test_empty_dir_returns_zero(self, tmp_path: pathlib.Path) -> None:
        """Empty directory returns 0."""
        count = clear_cache(cache_dir=tmp_path)
        assert count == 0


# ---------------------------------------------------------------------------
# TestCacheSize
# ---------------------------------------------------------------------------


class TestCacheSize:
    """Tests for cache_size()."""

    def test_returns_total_size(self, tmp_path: pathlib.Path) -> None:
        """Returns total bytes of all .db files."""
        db1 = _create_fake_db(tmp_path / "x.db", config_hash="x")
        db2 = _create_fake_db(tmp_path / "y.db", config_hash="y")

        expected = db1.stat().st_size + db2.stat().st_size
        result = cache_size(cache_dir=tmp_path)
        assert result == expected

    def test_empty_dir_returns_zero(self, tmp_path: pathlib.Path) -> None:
        """Empty directory returns 0."""
        result = cache_size(cache_dir=tmp_path)
        assert result == 0
