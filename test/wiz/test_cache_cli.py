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

"""Tests for the CacheScreen in the Textual wizard app."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

import pytest

from physicsnemo_curator.wiz.app import CuratorApp
from physicsnemo_curator.wiz.screens.cache import CacheScreen


def _create_fake_db(db_path, source_name="TestSource", sink_name="TestSink"):
    """Create a minimal pipeline DB for testing.

    Parameters
    ----------
    db_path : pathlib.Path
        Path to the .db file.
    source_name : str
        Source name in the config.
    sink_name : str
        Sink name in the config.
    """
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


class TestCacheScreen:
    """Tests for CacheScreen using Textual's pilot API."""

    @pytest.mark.asyncio
    async def test_empty_cache(self, monkeypatch, tmp_path):
        """DataTable shows no rows when cache is empty."""
        monkeypatch.setenv("PSNC_CACHE_DIR", str(tmp_path))
        app = CuratorApp()
        async with app.run_test() as pilot:
            app.push_screen(CacheScreen())
            await pilot.pause()
            table = app.screen.query_one("#cache-table")
            assert table.row_count == 0

    @pytest.mark.asyncio
    async def test_lists_databases(self, monkeypatch, tmp_path):
        """DataTable shows rows for cached databases."""
        monkeypatch.setenv("PSNC_CACHE_DIR", str(tmp_path))
        _create_fake_db(tmp_path / "abc12345.db", source_name="VTKSource")
        app = CuratorApp()
        async with app.run_test() as pilot:
            app.push_screen(CacheScreen())
            await pilot.pause()
            table = app.screen.query_one("#cache-table")
            assert table.row_count == 1

    @pytest.mark.asyncio
    async def test_remove_all(self, monkeypatch, tmp_path):
        """Remove All button clears all databases."""
        monkeypatch.setenv("PSNC_CACHE_DIR", str(tmp_path))
        _create_fake_db(tmp_path / "abc.db")
        _create_fake_db(tmp_path / "def.db")
        app = CuratorApp()
        async with app.run_test() as pilot:
            app.push_screen(CacheScreen())
            await pilot.pause()
            table = app.screen.query_one("#cache-table")
            assert table.row_count == 2
            # Click Remove All
            await pilot.click("#rm-all-btn")
            await pilot.pause()
            table = app.screen.query_one("#cache-table")
            assert table.row_count == 0
