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

"""Tests for DashboardStore data layer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

pd = pytest.importorskip("pandas")
panel = pytest.importorskip("panel")

from physicsnemo_curator.core.pipeline_store import PipelineStore  # noqa: E402
from physicsnemo_curator.dashboard.data import DashboardStore  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture
def pipeline_store(tmp_path: Path) -> PipelineStore:
    """Create a fresh PipelineStore with some log entries."""
    db = tmp_path / "test.db"
    config = {"source": {"name": "TestSource"}, "filters": [], "sink": {"name": "TestSink"}}
    store = PipelineStore(db, config, "testhash")

    # Add some log entries
    logs: list[tuple[str, int, str, str, str, str | None, int | None]] = [
        ("2025-01-01T10:00:00Z", 20, "INFO", "test.source", "Starting index 0", "Worker-1", 0),
        ("2025-01-01T10:00:01Z", 10, "DEBUG", "test.source", "Debug message", "Worker-1", 0),
        ("2025-01-01T10:00:02Z", 20, "INFO", "test.filter", "Filtered index 0", "Worker-1", 0),
        ("2025-01-01T10:00:03Z", 20, "INFO", "test.source", "Starting index 1", "Worker-2", 1),
        ("2025-01-01T10:00:04Z", 30, "WARNING", "test.sink", "Warning message", None, 1),
        ("2025-01-01T10:00:05Z", 20, "INFO", "test.source", "Done", "Worker-2", 1),
    ]
    store.record_logs(logs)
    return store


@pytest.fixture
def dashboard_store(pipeline_store: PipelineStore, tmp_path: Path) -> DashboardStore:
    """Create a DashboardStore from the pipeline store."""
    db_path = tmp_path / "test.db"
    return DashboardStore(str(db_path))


class TestDashboardStoreLogs:
    """Tests for DashboardStore log retrieval methods."""

    def test_logs_df_returns_dataframe(self, dashboard_store: DashboardStore) -> None:
        """logs_df returns a DataFrame with expected columns."""
        df = dashboard_store.logs_df()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["time", "level", "worker_id", "index", "message"]

    def test_logs_df_contains_entries(self, dashboard_store: DashboardStore) -> None:
        """logs_df contains the log entries from the store."""
        df = dashboard_store.logs_df()
        assert len(df) == 6

    def test_logs_df_fills_none_worker_with_main(self, dashboard_store: DashboardStore) -> None:
        """logs_df replaces None worker_id with 'Main'."""
        df = dashboard_store.logs_df()
        # The warning message had None worker_id
        warning_row = df[df["level"] == "WARNING"]
        assert len(warning_row) == 1
        assert warning_row.iloc[0]["worker_id"] == "Main"

    def test_logs_df_min_level_filter(self, dashboard_store: DashboardStore) -> None:
        """logs_df respects min_level parameter."""
        # Only INFO and above (level >= 20)
        df = dashboard_store.logs_df(min_level=20)
        assert len(df) == 5  # Excludes DEBUG
        assert "DEBUG" not in df["level"].values

    def test_logs_df_limit(self, dashboard_store: DashboardStore) -> None:
        """logs_df respects limit parameter."""
        df = dashboard_store.logs_df(limit=3)
        assert len(df) == 3

    def test_logs_df_caches_results(self, dashboard_store: DashboardStore) -> None:
        """logs_df caches results for same parameters."""
        df1 = dashboard_store.logs_df(limit=100, min_level=20)
        df2 = dashboard_store.logs_df(limit=100, min_level=20)
        # Same object (cached)
        assert df1 is df2

    def test_logs_df_different_params_not_cached(self, dashboard_store: DashboardStore) -> None:
        """logs_df with different params returns different results."""
        df1 = dashboard_store.logs_df(limit=100, min_level=20)
        df2 = dashboard_store.logs_df(limit=100, min_level=10)
        # Different objects
        assert df1 is not df2

    def test_log_worker_ids(self, dashboard_store: DashboardStore) -> None:
        """log_worker_ids returns sorted unique worker IDs."""
        worker_ids = dashboard_store.log_worker_ids()
        assert "Main" in worker_ids  # None was converted to Main
        assert "Worker-1" in worker_ids
        assert "Worker-2" in worker_ids
        # Should be sorted
        assert worker_ids == sorted(worker_ids)

    def test_log_worker_ids_empty(self, tmp_path: Path) -> None:
        """log_worker_ids returns empty list when no logs."""
        db = tmp_path / "empty.db"
        config = {"source": {"name": "Test"}, "filters": [], "sink": {"name": "Test"}}
        PipelineStore(db, config, "hash")  # Create empty store

        store = DashboardStore(str(db))
        assert store.log_worker_ids() == []

    def test_logs_df_empty(self, tmp_path: Path) -> None:
        """logs_df returns empty DataFrame when no logs."""
        db = tmp_path / "empty.db"
        config = {"source": {"name": "Test"}, "filters": [], "sink": {"name": "Test"}}
        PipelineStore(db, config, "hash")

        store = DashboardStore(str(db))
        df = store.logs_df()
        assert len(df) == 0
        assert list(df.columns) == ["time", "level", "worker_id", "index", "message"]
