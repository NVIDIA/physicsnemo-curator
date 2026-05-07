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

"""Data layer wrapping PipelineStore for the dashboard.

:class:`DashboardStore` is a ``param.Parameterized`` adapter that
queries the SQLite database and exposes results as pandas DataFrames
suitable for Panel reactive updates.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import param

from physicsnemo_curator.core.pipeline_store import PipelineStore


class DashboardStore(param.Parameterized):
    """Reactive wrapper around :class:`PipelineStore`.

    Provides pandas DataFrame views of pipeline metrics.  Triggers a
    cache invalidation when the ``refresh`` event fires, causing the
    next property access to re-query the database.
    """

    refresh = param.Event(doc="Trigger a data refresh from the database.")
    selected_index = param.Integer(
        default=-1,
        allow_None=True,
        doc="Currently selected pipeline index (-1 = none).",
    )

    def __init__(self, db_path: str, **kwargs: Any) -> None:
        """Initialize the dashboard store.

        Parameters
        ----------
        db_path : str
            Path to an existing PipelineStore SQLite database.
        **kwargs : Any
            Additional param keyword arguments.
        """
        super().__init__(**kwargs)
        self._store = PipelineStore.from_db(db_path)
        self._cache: dict[str, Any] = {}

    def _invalidate(self) -> None:
        """Clear cached DataFrames so the next access re-queries."""
        self._cache.clear()

    @param.depends("refresh", watch=True)  # ty: ignore[invalid-argument-type]
    def _on_refresh(self) -> None:
        """Invalidate cache when refresh is triggered."""
        self._invalidate()

    @property
    def pipeline_config(self) -> dict:
        """Return the pipeline configuration dictionary.

        Returns
        -------
        dict
            Pipeline configuration as stored in the database.
        """
        return self._store._pipeline_config  # noqa: SLF001

    @property
    def index_df(self) -> pd.DataFrame:
        """DataFrame of per-index results.

        Columns: ``index``, ``status``, ``wall_time_s``,
        ``peak_memory_mb``, ``gpu_memory_mb``, ``error``.

        Returns
        -------
        pd.DataFrame
            One row per processed index.
        """
        if "index_df" not in self._cache:
            metrics = self._store.metrics()
            completed = self._store.completed_indices()
            failed = self._store.failed_indices()

            rows = []
            for im in metrics.indices:
                rows.append(
                    {
                        "index": im.index,
                        "status": "completed" if im.index in completed else "error",
                        "wall_time_s": im.wall_time_ns / 1e9,
                        "peak_memory_mb": im.peak_memory_bytes / (1024 * 1024),
                        "gpu_memory_mb": (im.gpu_memory_bytes or 0) / (1024 * 1024),
                        "error": failed.get(im.index, ""),
                    }
                )

            # Add failed indices not already in metrics
            for idx, err in failed.items():
                if not any(r["index"] == idx for r in rows):
                    rows.append(
                        {
                            "index": idx,
                            "status": "error",
                            "wall_time_s": 0.0,
                            "peak_memory_mb": 0.0,
                            "gpu_memory_mb": 0.0,
                            "error": err,
                        }
                    )

            df = (
                pd.DataFrame(rows)
                if rows
                else pd.DataFrame(
                    columns=["index", "status", "wall_time_s", "peak_memory_mb", "gpu_memory_mb", "error"]  # ty: ignore[invalid-argument-type]
                )
            )
            self._cache["index_df"] = df.sort_values("index").reset_index(drop=True)
        return self._cache["index_df"]

    @property
    def stage_df(self) -> pd.DataFrame:
        """DataFrame of per-stage timing for all indices.

        Columns: ``index``, ``stage_name``, ``stage_order``, ``wall_time_s``.

        Returns
        -------
        pd.DataFrame
            One row per (index, stage) combination.
        """
        if "stage_df" not in self._cache:
            metrics = self._store.metrics()
            rows = []
            for im in metrics.indices:
                for order, sm in enumerate(im.stages):
                    rows.append(
                        {
                            "index": im.index,
                            "stage_name": sm.name,
                            "stage_order": order,
                            "wall_time_s": sm.wall_time_ns / 1e9,
                        }
                    )
            df = (
                pd.DataFrame(rows)
                if rows
                else pd.DataFrame(columns=["index", "stage_name", "stage_order", "wall_time_s"])  # ty: ignore[invalid-argument-type]
            )
            self._cache["stage_df"] = df
        return self._cache["stage_df"]

    @property
    def summary(self) -> dict[str, Any]:
        """Summary of the pipeline run state.

        Returns
        -------
        dict[str, Any]
            Keys: ``total``, ``completed``, ``failed``, ``remaining``,
            ``elapsed_s``, ``config_hash``, ``db_path``, ``workers``.
        """
        if "summary" not in self._cache:
            # Use stored total_indices if available, fall back to computed value
            stored_total = self._store.get_total_indices()
            if stored_total is not None:
                total = stored_total
            else:
                # Fall back: compute from completed + failed + remaining
                total = len(self._store.completed_indices()) + len(self._store.failed_indices())
                remaining = self._store.remaining_indices(total)
                total += len(remaining)
            self._cache["summary"] = self._store.summary(total)
        return self._cache["summary"]

    @property
    def workers_df(self) -> pd.DataFrame:
        """DataFrame of registered workers.

        Columns: ``worker_id``, ``pid``, ``hostname``, ``started_at``,
        ``last_heartbeat``, ``current_index``.

        Returns
        -------
        pd.DataFrame
            One row per worker.
        """
        if "workers_df" not in self._cache:
            workers = self._store.active_workers()
            df = (
                pd.DataFrame(workers)
                if workers
                else pd.DataFrame(
                    columns=["worker_id", "pid", "hostname", "started_at", "last_heartbeat", "current_index"]  # ty: ignore[invalid-argument-type]
                )
            )
            self._cache["workers_df"] = df
        return self._cache["workers_df"]

    def output_paths(self, index: int) -> list[str]:
        """Return output file paths for a given index.

        Parameters
        ----------
        index : int
            Pipeline source index.

        Returns
        -------
        list[str]
            Ordered list of output file paths.
        """
        return self._store.output_paths_for_index(index)

    def artifacts(self, index: int) -> dict[str, list[str]]:
        """Return filter artifacts for a given index, resolved to absolute paths.

        Parameters
        ----------
        index : int
            Pipeline source index.

        Returns
        -------
        dict[str, list[str]]
            Mapping of filter name to list of resolved artifact paths.
        """
        raw = self._store.filter_artifacts_for_index(index)
        return {name: [str(self._store.resolve_artifact(p)) for p in paths] for name, paths in raw.items()}

    def all_artifacts(self) -> dict[str, list[str]]:
        """Return all filter artifacts across all indices, resolved to absolute paths.

        Returns
        -------
        dict[str, list[str]]
            Mapping of filter name to list of all resolved artifact paths.
        """
        raw = self._store.all_filter_artifacts()
        return {name: [str(self._store.resolve_artifact(p)) for p in paths] for name, paths in raw.items()}

    def logs_df(self, limit: int = 500, min_level: int = 0) -> pd.DataFrame:
        """DataFrame of log entries from the pipeline run.

        Parameters
        ----------
        limit : int
            Maximum number of log entries to retrieve (default: 500).
        min_level : int
            Minimum log level (0=DEBUG, 10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR).

        Returns
        -------
        pd.DataFrame
            Log entries with columns: timestamp, level_name, worker_id, idx, message.
        """
        cache_key = f"logs_df_{limit}_{min_level}"
        if cache_key not in self._cache:
            logs = self._store.get_logs(since_id=0, limit=limit, min_level=min_level)
            if logs:
                df = pd.DataFrame(logs)
                # Parse timestamp for display
                df["time"] = df["timestamp"].str[11:19]  # Extract HH:MM:SS
                # Fill None worker_id with "Main"
                df["worker_id"] = df["worker_id"].fillna("Main")
                # Select and order columns for display
                df = df[["time", "level_name", "worker_id", "idx", "message"]]
                df = df.rename(columns={"level_name": "level", "idx": "index"})
            else:
                df = pd.DataFrame(columns=["time", "level", "worker_id", "index", "message"])  # ty: ignore[invalid-argument-type]
            self._cache[cache_key] = df
        return self._cache[cache_key]

    def log_worker_ids(self) -> list[str]:
        """Return unique worker IDs from logs.

        Returns
        -------
        list[str]
            Sorted list of unique worker IDs (including "Main").
        """
        df = self.logs_df()
        if df.empty:
            return []
        return sorted(df["worker_id"].unique().tolist())
