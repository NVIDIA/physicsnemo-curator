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

"""Pipeline profiling utility.

Provides :class:`ProfiledPipeline`, a transparent proxy wrapper around
:class:`~physicsnemo_curator.core.base.Pipeline` that collects wall-clock,
memory, and GPU metrics at whole-pipeline, per-index, and per-stage
granularity.

Usage
-----
>>> from physicsnemo_curator import Pipeline, ProfiledPipeline, run_pipeline
>>> profiled = ProfiledPipeline(pipeline, track_gpu=True)
>>> results = run_pipeline(profiled, n_jobs=4, backend="process_pool")
>>> metrics = profiled.metrics
>>> metrics.to_console()
"""

from __future__ import annotations

import csv
import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")


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
