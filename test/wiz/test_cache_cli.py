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

"""Tests for the psnc cache CLI commands."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

import pytest

click = pytest.importorskip("click", reason="click not installed (install wiz extra)")

from click.testing import CliRunner  # noqa: E402

from physicsnemo_curator.wiz import main  # noqa: E402


def _create_fake_db(db_path, source_name="TestSource", sink_name="TestSink"):
    """Create a minimal pipeline DB for CLI testing."""
    config = {
        "source": {"class": source_name, "module": "test", "name": source_name, "description": "test", "params": {}},
        "filters": [],
        "sink": {"class": sink_name, "module": "test", "name": sink_name, "description": "test", "params": {}},
    }
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_hash TEXT UNIQUE NOT NULL,
            config_json TEXT NOT NULL,
            started_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS index_results (
            idx INTEGER NOT NULL, run_id INTEGER NOT NULL,
            status TEXT NOT NULL, output_paths TEXT,
            completed_at TEXT NOT NULL, wall_time_ns INTEGER,
            peak_memory_bytes INTEGER, gpu_memory_bytes INTEGER,
            error TEXT, PRIMARY KEY (idx, run_id)
        );
    """)
    conn.execute(
        "INSERT INTO pipeline_runs (config_hash, config_json, started_at) VALUES (?, ?, ?)",
        (db_path.stem, json.dumps(config), datetime.now(UTC).isoformat()),
    )
    conn.commit()
    conn.close()


class TestCachePath:
    """Tests for psnc cache path."""

    def test_prints_cache_dir(self, monkeypatch, tmp_path):
        """Prints the resolved cache directory."""
        monkeypatch.setenv("PSNC_CACHE_DIR", str(tmp_path))
        runner = CliRunner()
        result = runner.invoke(main, ["cache", "path"])
        assert result.exit_code == 0
        assert str(tmp_path) in result.output


class TestCacheList:
    """Tests for psnc cache list."""

    def test_empty_cache(self, monkeypatch, tmp_path):
        """Shows message when no databases exist."""
        monkeypatch.setenv("PSNC_CACHE_DIR", str(tmp_path))
        runner = CliRunner()
        result = runner.invoke(main, ["cache", "list"])
        assert result.exit_code == 0
        assert "No cached databases" in result.output

    def test_lists_databases(self, monkeypatch, tmp_path):
        """Lists databases with metadata."""
        monkeypatch.setenv("PSNC_CACHE_DIR", str(tmp_path))
        _create_fake_db(tmp_path / "abc12345.db", source_name="VTKSource")
        runner = CliRunner()
        result = runner.invoke(main, ["cache", "list"])
        assert result.exit_code == 0
        assert "VTKSource" in result.output


class TestCacheRm:
    """Tests for psnc cache rm."""

    def test_rm_by_hash(self, monkeypatch, tmp_path):
        """Removes database by hash prefix."""
        monkeypatch.setenv("PSNC_CACHE_DIR", str(tmp_path))
        _create_fake_db(tmp_path / "abc12345.db")
        runner = CliRunner()
        result = runner.invoke(main, ["cache", "rm", "abc1"])
        assert result.exit_code == 0
        assert "Removed 1" in result.output
        assert not (tmp_path / "abc12345.db").exists()

    def test_rm_all(self, monkeypatch, tmp_path):
        """--all removes all databases."""
        monkeypatch.setenv("PSNC_CACHE_DIR", str(tmp_path))
        _create_fake_db(tmp_path / "abc.db")
        _create_fake_db(tmp_path / "def.db")
        runner = CliRunner()
        result = runner.invoke(main, ["cache", "rm", "--all", "-y"])
        assert result.exit_code == 0
        assert "Removed 2" in result.output

    def test_rm_no_args_fails(self, monkeypatch, tmp_path):
        """Fails when no hash prefixes or flags given."""
        monkeypatch.setenv("PSNC_CACHE_DIR", str(tmp_path))
        runner = CliRunner()
        result = runner.invoke(main, ["cache", "rm"])
        assert result.exit_code != 0
