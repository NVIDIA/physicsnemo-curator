---
orphan: true
---

# Pipeline Profiling Utility — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `ProfiledPipeline` transparent proxy wrapper that collects wall-clock, memory, and GPU metrics at whole-pipeline, per-index, and per-stage granularity — with zero changes to existing backends.

**Architecture:** `ProfiledPipeline` wraps `Pipeline[T]` with duck-type compatibility (`source`, `filters`, `sink`, `__len__`, `__getitem__`). Per-stage timing uses `_TimedGenerator` chain-subtraction. Metrics are serialized to temp-directory JSON files (survives pickle for parallel backends). After `run_pipeline()` completes, `collect_metrics()` reads and aggregates them.

**Tech Stack:** Python stdlib (`time`, `tracemalloc`, `json`, `csv`, `dataclasses`, `pathlib`, `tempfile`, `uuid`, `shutil`), optional `torch.cuda` for GPU.

---

## File Structure

| Action | File | Purpose |
|--------|------|---------|
| Create | `src/physicsnemo_curator/core/profiling.py` | All profiling classes: `StageMetrics`, `IndexMetrics`, `PipelineMetrics`, `_TimedGenerator`, `ProfiledPipeline` |
| Modify | `src/physicsnemo_curator/__init__.py` | Export `ProfiledPipeline`, `PipelineMetrics` |
| Create | `test/core/test_profiling.py` | Unit tests (12 test cases) |
| Create | `test/run/test_profiling_backends.py` | Integration tests (3 backends) |
| Create | `benchmarks/bench_profiling.py` | ASV benchmarks for profiling overhead |
| Create | `docs/user-guide/profiling.md` | User-facing profiling guide |
| Modify | `docs/user-guide/parallel.md` | Add cross-reference to profiling |
| Modify | `docs/developer-guide/benchmarking.md` | Mention bench_profiling.py |

---

### Task 1: Core Data Classes — `StageMetrics`, `IndexMetrics`, `PipelineMetrics`

**Files:**
- Create: `src/physicsnemo_curator/core/profiling.py`
- Test: `test/core/test_profiling.py`

- [ ] **Step 1: Write failing tests for data classes**

Create `test/core/test_profiling.py`:

```python
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

"""Tests for the pipeline profiling utility."""

from __future__ import annotations

import json
import pathlib
import pickle
import sys
import tempfile
import time
from typing import TYPE_CHECKING, ClassVar
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

import pytest

from physicsnemo_curator.core.base import Filter, Param, Pipeline, Sink, Source

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Test implementations (module-level for pickle compatibility)
# ---------------------------------------------------------------------------


class _TimedSource(Source[int]):
    """Source with a small delay per item."""

    name: ClassVar[str] = "TimedSource"
    description: ClassVar[str] = "Yields ints with delay"

    @classmethod
    def params(cls) -> list[Param]:
        """Return empty params."""
        return []

    def __init__(self, n: int, delay: float = 0.01) -> None:
        """Initialize source."""
        self._n = n
        self._delay = delay

    def __len__(self) -> int:
        """Return count."""
        return self._n

    def __getitem__(self, index: int) -> Generator[int]:
        """Yield index with delay."""
        time.sleep(self._delay)
        yield index


class _SlowFilter(Filter[int]):
    """Filter that adds a small delay per item."""

    name: ClassVar[str] = "SlowFilter"
    description: ClassVar[str] = "Adds delay"

    @classmethod
    def params(cls) -> list[Param]:
        """Return empty params."""
        return []

    def __init__(self, delay: float = 0.01) -> None:
        """Initialize filter."""
        self._delay = delay

    def __call__(self, items: Generator[int]) -> Generator[int]:
        """Delay each item."""
        for item in items:
            time.sleep(self._delay)
            yield item * 2


class _DoubleFilter(Filter[int]):
    """Fast filter that doubles."""

    name: ClassVar[str] = "DoubleFilter"
    description: ClassVar[str] = "Doubles items"

    @classmethod
    def params(cls) -> list[Param]:
        """Return empty params."""
        return []

    def __call__(self, items: Generator[int]) -> Generator[int]:
        """Double each item."""
        for item in items:
            yield item * 2


class _CollectSink(Sink[int]):
    """Sink that writes items as strings."""

    name: ClassVar[str] = "CollectSink"
    description: ClassVar[str] = "Collects items"

    @classmethod
    def params(cls) -> list[Param]:
        """Return empty params."""
        return []

    def __call__(self, items: Iterator[int], index: int) -> list[str]:
        """Return string representations."""
        return [str(v) for v in items]


class _ErrorFilter(Filter[int]):
    """Filter that raises on first item."""

    name: ClassVar[str] = "ErrorFilter"
    description: ClassVar[str] = "Raises RuntimeError"

    @classmethod
    def params(cls) -> list[Param]:
        """Return empty params."""
        return []

    def __call__(self, items: Generator[int]) -> Generator[int]:
        """Raise on first item."""
        for item in items:
            msg = "intentional error"
            raise RuntimeError(msg)
            yield item  # unreachable


class _AllocSource(Source[int]):
    """Source that allocates a known amount of memory."""

    name: ClassVar[str] = "AllocSource"
    description: ClassVar[str] = "Allocates memory"

    @classmethod
    def params(cls) -> list[Param]:
        """Return empty params."""
        return []

    def __init__(self, n: int, alloc_bytes: int = 1_000_000) -> None:
        """Initialize source."""
        self._n = n
        self._alloc_bytes = alloc_bytes

    def __len__(self) -> int:
        """Return count."""
        return self._n

    def __getitem__(self, index: int) -> Generator[int]:
        """Yield index after allocating memory."""
        _big = bytearray(self._alloc_bytes)  # ~1 MB
        yield index


# ---------------------------------------------------------------------------
# Tests for StageMetrics, IndexMetrics, PipelineMetrics
# ---------------------------------------------------------------------------


class TestStageMetrics:
    """Tests for StageMetrics dataclass."""

    def test_creation(self):
        """StageMetrics holds name and wall_time_ns."""
        from physicsnemo_curator.core.profiling import StageMetrics

        m = StageMetrics(name="source", wall_time_ns=1_000_000)
        assert m.name == "source"
        assert m.wall_time_ns == 1_000_000

    def test_to_dict(self):
        """StageMetrics.to_dict() returns expected keys."""
        from physicsnemo_curator.core.profiling import StageMetrics

        m = StageMetrics(name="DoubleFilter", wall_time_ns=500_000)
        d = m.to_dict()
        assert d == {"name": "DoubleFilter", "wall_time_ns": 500_000}


class TestIndexMetrics:
    """Tests for IndexMetrics dataclass."""

    def test_creation(self):
        """IndexMetrics has index, stages, wall_time_ns, peak_memory_bytes, gpu_memory_bytes."""
        from physicsnemo_curator.core.profiling import IndexMetrics, StageMetrics

        stages = [StageMetrics(name="source", wall_time_ns=1000)]
        m = IndexMetrics(
            index=0,
            stages=stages,
            wall_time_ns=2000,
            peak_memory_bytes=1024,
            gpu_memory_bytes=None,
        )
        assert m.index == 0
        assert len(m.stages) == 1
        assert m.gpu_memory_bytes is None

    def test_to_dict(self):
        """IndexMetrics.to_dict() returns nested structure."""
        from physicsnemo_curator.core.profiling import IndexMetrics, StageMetrics

        m = IndexMetrics(
            index=3,
            stages=[StageMetrics(name="source", wall_time_ns=100)],
            wall_time_ns=200,
            peak_memory_bytes=512,
            gpu_memory_bytes=2048,
        )
        d = m.to_dict()
        assert d["index"] == 3
        assert d["gpu_memory_bytes"] == 2048
        assert len(d["stages"]) == 1


class TestPipelineMetrics:
    """Tests for PipelineMetrics dataclass."""

    def test_computed_properties(self):
        """Computed properties aggregate correctly."""
        from physicsnemo_curator.core.profiling import IndexMetrics, PipelineMetrics, StageMetrics

        idx0 = IndexMetrics(
            index=0,
            stages=[StageMetrics(name="source", wall_time_ns=100)],
            wall_time_ns=1000,
            peak_memory_bytes=500,
            gpu_memory_bytes=None,
        )
        idx1 = IndexMetrics(
            index=1,
            stages=[StageMetrics(name="source", wall_time_ns=200)],
            wall_time_ns=3000,
            peak_memory_bytes=700,
            gpu_memory_bytes=None,
        )
        pm = PipelineMetrics(indices=[idx0, idx1])
        assert pm.total_wall_time_ns == 4000
        assert pm.mean_index_time_ns == 2000.0
        assert pm.total_peak_memory_bytes == 700  # max, not sum

    def test_empty_metrics(self):
        """Empty PipelineMetrics has zero totals."""
        from physicsnemo_curator.core.profiling import PipelineMetrics

        pm = PipelineMetrics(indices=[])
        assert pm.total_wall_time_ns == 0
        assert pm.mean_index_time_ns == 0.0
        assert pm.total_peak_memory_bytes == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/core/test_profiling.py -v --no-header -x`
Expected: FAIL with `ModuleNotFoundError: No module named 'physicsnemo_curator.core.profiling'`

- [ ] **Step 3: Implement data classes**

Create `src/physicsnemo_curator/core/profiling.py` with this initial content:

```python
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
import io
import json
import pathlib
import shutil
import tempfile
import time
import tracemalloc
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

    from physicsnemo_curator.core.base import Filter, Pipeline, Sink, Source

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
```

- [ ] **Step 4: Run data class tests to verify they pass**

Run: `uv run pytest test/core/test_profiling.py::TestStageMetrics test/core/test_profiling.py::TestIndexMetrics test/core/test_profiling.py::TestPipelineMetrics -v --no-header`
Expected: 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/physicsnemo_curator/core/profiling.py test/core/test_profiling.py
git commit -m "feat(profiling): add StageMetrics, IndexMetrics, PipelineMetrics data classes"
```

---

### Task 2: `_TimedGenerator` and `ProfiledPipeline.__getitem__`

**Files:**
- Modify: `src/physicsnemo_curator/core/profiling.py`
- Modify: `test/core/test_profiling.py`

- [ ] **Step 1: Write failing tests for _TimedGenerator and ProfiledPipeline**

Append to `test/core/test_profiling.py`:

```python
class TestTimedGenerator:
    """Tests for the internal _TimedGenerator wrapper."""

    def test_iterates_correctly(self):
        """_TimedGenerator yields same values as wrapped generator."""
        from physicsnemo_curator.core.profiling import _TimedGenerator

        def gen():
            yield 1
            yield 2
            yield 3

        tg = _TimedGenerator(gen())
        assert list(tg) == [1, 2, 3]

    def test_accumulates_time(self):
        """_TimedGenerator.elapsed_ns is positive after iteration."""
        from physicsnemo_curator.core.profiling import _TimedGenerator

        def slow_gen():
            time.sleep(0.01)
            yield 1

        tg = _TimedGenerator(slow_gen())
        list(tg)  # consume
        assert tg.elapsed_ns > 0

    def test_empty_generator(self):
        """_TimedGenerator handles empty generators."""
        from physicsnemo_curator.core.profiling import _TimedGenerator

        def empty():
            return
            yield  # make it a generator

        tg = _TimedGenerator(empty())
        assert list(tg) == []
        assert tg.elapsed_ns >= 0


class TestProfiledPipeline:
    """Tests for ProfiledPipeline wrapper."""

    def test_duck_type_compatibility(self):
        """ProfiledPipeline exposes source, filters, sink, __len__."""
        from physicsnemo_curator.core.profiling import ProfiledPipeline

        pipeline = _TimedSource(3).filter(_DoubleFilter()).write(_CollectSink())
        profiled = ProfiledPipeline(pipeline)
        assert profiled.source is pipeline.source
        assert profiled.filters is pipeline.filters
        assert profiled.sink is pipeline.sink
        assert len(profiled) == 3

    def test_getitem_returns_correct_results(self):
        """ProfiledPipeline.__getitem__ returns same results as Pipeline."""
        from physicsnemo_curator.core.profiling import ProfiledPipeline

        pipeline = _TimedSource(3, delay=0.0).filter(_DoubleFilter()).write(_CollectSink())
        profiled = ProfiledPipeline(pipeline)

        for i in range(3):
            assert profiled[i] == pipeline[i]

    def test_basic_metrics_collection(self):
        """Single index produces IndexMetrics with correct stage count."""
        from physicsnemo_curator.core.profiling import ProfiledPipeline

        pipeline = _TimedSource(2, delay=0.001).filter(_SlowFilter(delay=0.001)).write(_CollectSink())
        profiled = ProfiledPipeline(pipeline)
        profiled[0]
        metrics = profiled.collect_metrics()

        assert len(metrics.indices) == 1
        idx_m = metrics.indices[0]
        assert idx_m.index == 0
        # 3 stages: source, SlowFilter, sink
        assert len(idx_m.stages) == 3
        assert idx_m.stages[0].name == "source"
        assert idx_m.stages[1].name == "SlowFilter"
        assert idx_m.stages[2].name == "sink"

    def test_per_stage_timing_positive(self):
        """Each stage has positive wall time."""
        from physicsnemo_curator.core.profiling import ProfiledPipeline

        pipeline = _TimedSource(1, delay=0.005).filter(_SlowFilter(delay=0.005)).write(_CollectSink())
        profiled = ProfiledPipeline(pipeline)
        profiled[0]
        metrics = profiled.collect_metrics()

        for stage in metrics.indices[0].stages:
            assert stage.wall_time_ns > 0, f"Stage {stage.name} has non-positive time"

    def test_stage_times_approximate_total(self):
        """Sum of stage times should approximate total index time."""
        from physicsnemo_curator.core.profiling import ProfiledPipeline

        pipeline = _TimedSource(1, delay=0.01).filter(_SlowFilter(delay=0.01)).write(_CollectSink())
        profiled = ProfiledPipeline(pipeline)
        profiled[0]
        metrics = profiled.collect_metrics()

        idx_m = metrics.indices[0]
        stage_sum = sum(s.wall_time_ns for s in idx_m.stages)
        # Stage sum should be within 50% of total (generous tolerance for CI)
        assert stage_sum <= idx_m.wall_time_ns * 1.5
        assert stage_sum >= idx_m.wall_time_ns * 0.3

    def test_memory_tracking(self):
        """Peak memory is non-trivial for source that allocates."""
        from physicsnemo_curator.core.profiling import ProfiledPipeline

        pipeline = _AllocSource(1, alloc_bytes=500_000).write(_CollectSink())
        profiled = ProfiledPipeline(pipeline)
        profiled[0]
        metrics = profiled.collect_metrics()
        # Should detect at least some memory (tracemalloc not perfectly precise)
        assert metrics.indices[0].peak_memory_bytes > 0

    def test_gpu_tracking_disabled_by_default(self):
        """GPU memory is None when track_gpu=False."""
        from physicsnemo_curator.core.profiling import ProfiledPipeline

        pipeline = _TimedSource(1, delay=0.0).write(_CollectSink())
        profiled = ProfiledPipeline(pipeline)
        profiled[0]
        metrics = profiled.collect_metrics()
        assert metrics.indices[0].gpu_memory_bytes is None

    def test_gpu_tracking_mocked(self):
        """GPU tracking works with mocked torch.cuda."""
        from physicsnemo_curator.core.profiling import ProfiledPipeline

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1000
        mock_torch.cuda.max_memory_allocated.return_value = 5000
        mock_torch.cuda.reset_peak_memory_stats.return_value = None

        with patch.dict(sys.modules, {"torch": mock_torch}):
            pipeline = _TimedSource(1, delay=0.0).write(_CollectSink())
            profiled = ProfiledPipeline(pipeline, track_gpu=True)
            profiled[0]
            metrics = profiled.collect_metrics()
            assert metrics.indices[0].gpu_memory_bytes == 4000  # 5000 - 1000

    def test_multiple_indices(self):
        """Metrics are collected for all indices."""
        from physicsnemo_curator.core.profiling import ProfiledPipeline

        pipeline = _TimedSource(5, delay=0.0).filter(_DoubleFilter()).write(_CollectSink())
        profiled = ProfiledPipeline(pipeline)
        for i in range(5):
            profiled[i]
        metrics = profiled.collect_metrics()
        assert len(metrics.indices) == 5
        collected_indices = {m.index for m in metrics.indices}
        assert collected_indices == {0, 1, 2, 3, 4}

    def test_no_filter_pipeline(self):
        """Pipeline with no filters still profiles source and sink."""
        from physicsnemo_curator.core.profiling import ProfiledPipeline

        pipeline = _TimedSource(1, delay=0.001).write(_CollectSink())
        profiled = ProfiledPipeline(pipeline)
        profiled[0]
        metrics = profiled.collect_metrics()

        idx_m = metrics.indices[0]
        assert len(idx_m.stages) == 2  # source, sink
        assert idx_m.stages[0].name == "source"
        assert idx_m.stages[1].name == "sink"

    def test_error_propagation(self):
        """Errors in stages propagate without being swallowed."""
        from physicsnemo_curator.core.profiling import ProfiledPipeline

        pipeline = _TimedSource(1, delay=0.0).filter(_ErrorFilter()).write(_CollectSink())
        profiled = ProfiledPipeline(pipeline)
        with pytest.raises(RuntimeError, match="intentional error"):
            profiled[0]

    def test_pickle_roundtrip(self):
        """ProfiledPipeline survives pickle round-trip."""
        from physicsnemo_curator.core.profiling import ProfiledPipeline

        pipeline = _TimedSource(3, delay=0.0).filter(_DoubleFilter()).write(_CollectSink())
        profiled = ProfiledPipeline(pipeline)

        data = pickle.dumps(profiled)
        restored = pickle.loads(data)  # noqa: S301
        assert len(restored) == 3
        assert restored[0] == profiled[0]

    def test_metrics_property(self):
        """The .metrics property is a shortcut for collect_metrics()."""
        from physicsnemo_curator.core.profiling import ProfiledPipeline

        pipeline = _TimedSource(1, delay=0.0).write(_CollectSink())
        profiled = ProfiledPipeline(pipeline)
        profiled[0]
        assert profiled.metrics.total_wall_time_ns == profiled.collect_metrics().total_wall_time_ns

    def test_cleanup(self):
        """cleanup() removes the temp metrics directory."""
        from physicsnemo_curator.core.profiling import ProfiledPipeline

        pipeline = _TimedSource(1, delay=0.0).write(_CollectSink())
        profiled = ProfiledPipeline(pipeline)
        profiled[0]
        metrics_dir = profiled._metrics_dir
        assert metrics_dir.exists()
        profiled.cleanup()
        assert not metrics_dir.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/core/test_profiling.py::TestTimedGenerator test/core/test_profiling.py::TestProfiledPipeline -v --no-header -x`
Expected: FAIL with `ImportError` for `_TimedGenerator` or `ProfiledPipeline`

- [ ] **Step 3: Implement _TimedGenerator and ProfiledPipeline**

Append to `src/physicsnemo_curator/core/profiling.py` (after `PipelineMetrics`):

```python
class _TimedGenerator(Generic[T]):
    """Generator wrapper that accumulates wall-clock time across ``__next__`` calls.

    This is used internally by :class:`ProfiledPipeline` to attribute time
    to each pipeline stage. The wrapper preserves the full iterator protocol.

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


class ProfiledPipeline(Generic[T]):
    """Transparent profiling wrapper around :class:`~physicsnemo_curator.core.base.Pipeline`.

    Duck-type compatible with ``Pipeline`` — exposes ``source``, ``filters``,
    ``sink``, ``__len__``, and ``__getitem__``. Can be passed directly to
    :func:`~physicsnemo_curator.run.run_pipeline` without any backend changes.

    Metrics are collected per-index and serialized to a temp directory as
    JSON files. After ``run_pipeline()`` completes, call :attr:`metrics` or
    :meth:`collect_metrics` to aggregate results.

    Parameters
    ----------
    pipeline : Pipeline[T]
        The pipeline to wrap.
    track_gpu : bool
        If ``True``, record GPU memory usage via ``torch.cuda``.
        Requires PyTorch with CUDA support.

    Examples
    --------
    >>> from physicsnemo_curator import Pipeline, ProfiledPipeline, run_pipeline
    >>> profiled = ProfiledPipeline(pipeline, track_gpu=True)
    >>> results = run_pipeline(profiled, n_jobs=4, backend="process_pool")
    >>> profiled.metrics.to_console()
    """

    def __init__(self, pipeline: Pipeline[T], *, track_gpu: bool = False) -> None:
        """Initialize the profiling wrapper."""
        self._pipeline = pipeline
        self._track_gpu = track_gpu
        self._session_id = uuid.uuid4().hex[:12]
        self._metrics_dir = pathlib.Path(tempfile.gettempdir()) / f"pnc_profile_{self._session_id}"
        self._metrics_dir.mkdir(exist_ok=True)

    # -- Duck-type compatibility with Pipeline --------------------------------

    @property
    def source(self) -> Source[T]:
        """The wrapped pipeline's source.

        Returns
        -------
        Source[T]
            The underlying source.
        """
        return self._pipeline.source

    @property
    def filters(self) -> list[Filter[T]]:
        """The wrapped pipeline's filter list.

        Returns
        -------
        list[Filter[T]]
            The underlying filters.
        """
        return self._pipeline.filters

    @property
    def sink(self) -> Sink[T] | None:
        """The wrapped pipeline's sink.

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
            Number of source items.
        """
        return len(self._pipeline)

    def __getitem__(self, index: int) -> list[str]:
        """Process the given index with full profiling instrumentation.

        Parameters
        ----------
        index : int
            Zero-based index into the source.

        Returns
        -------
        list[str]
            File paths produced by the sink (same contract as ``Pipeline``).
        """
        # --- GPU baseline ---
        gpu_baseline: int | None = None
        if self._track_gpu:
            gpu_baseline = self._gpu_setup()

        # --- Memory tracking ---
        was_tracing = tracemalloc.is_tracing()
        if not was_tracing:
            tracemalloc.start()
        tracemalloc.reset_peak()

        overall_start = time.perf_counter_ns()
        stage_metrics: list[StageMetrics] = []

        try:
            # 1. Wrap source generator with timing
            source_gen = self._pipeline.source[index]
            timed_source = _TimedGenerator(source_gen)

            # 2. Chain through filters, wrapping each output
            prev_wrapper = timed_source
            filter_wrappers: list[_TimedGenerator[T]] = []
            current_stream: _TimedGenerator[T] = timed_source

            for f in self._pipeline.filters:
                raw_output = f(current_stream)
                wrapped = _TimedGenerator(raw_output)
                filter_wrappers.append(wrapped)
                current_stream = wrapped

            # 3. Time the sink (forces full chain evaluation)
            sink_start = time.perf_counter_ns()
            result = self._pipeline.sink(current_stream, index)  # type: ignore[misc]
            sink_elapsed = time.perf_counter_ns() - sink_start

            overall_elapsed = time.perf_counter_ns() - overall_start

            # 4. Compute per-stage times using chain subtraction
            # Source time = timed_source.elapsed_ns
            source_time = timed_source.elapsed_ns
            stage_metrics.append(StageMetrics(name="source", wall_time_ns=source_time))

            # Filter times: filter N own time = wrapper N elapsed - wrapper N-1 elapsed
            prev_elapsed = source_time
            for i_f, fw in enumerate(filter_wrappers):
                filter_own_time = fw.elapsed_ns - prev_elapsed
                # Clamp to zero (rounding errors)
                filter_own_time = max(0, filter_own_time)
                fname = type(self._pipeline.filters[i_f]).name
                stage_metrics.append(StageMetrics(name=fname, wall_time_ns=filter_own_time))
                prev_elapsed = fw.elapsed_ns

            # Sink time: the sink_elapsed includes pulling from the last wrapper,
            # so sink's own time = sink_elapsed - (last_wrapper.elapsed - already accounted)
            # Simpler: sink_own = overall_elapsed - last_wrapper.elapsed (if filters exist)
            #          or sink_own = overall_elapsed - source_time (if no filters)
            last_elapsed = filter_wrappers[-1].elapsed_ns if filter_wrappers else source_time
            sink_own_time = max(0, overall_elapsed - last_elapsed)
            stage_metrics.append(StageMetrics(name="sink", wall_time_ns=sink_own_time))

        except BaseException:
            # Let exceptions propagate unchanged
            raise
        finally:
            # Memory measurement (even on error path, for partial metrics)
            _, peak = tracemalloc.get_traced_memory()
            if not was_tracing:
                tracemalloc.stop()

        # GPU measurement
        gpu_delta: int | None = None
        if self._track_gpu and gpu_baseline is not None:
            gpu_delta = self._gpu_measure(gpu_baseline)

        # Build and serialize IndexMetrics
        idx_metrics = IndexMetrics(
            index=index,
            stages=stage_metrics,
            wall_time_ns=overall_elapsed,
            peak_memory_bytes=peak,
            gpu_memory_bytes=gpu_delta,
        )
        self._write_index_metrics(idx_metrics)

        return result

    # -- Metrics collection ---------------------------------------------------

    def collect_metrics(self) -> PipelineMetrics:
        """Read serialized metrics from the temp directory and aggregate.

        Returns
        -------
        PipelineMetrics
            Aggregated metrics across all processed indices.
        """
        index_metrics: list[IndexMetrics] = []
        if self._metrics_dir.exists():
            for json_file in sorted(self._metrics_dir.glob("*.json")):
                data = json.loads(json_file.read_text())
                index_metrics.append(IndexMetrics.from_dict(data))
        return PipelineMetrics(indices=index_metrics)

    @property
    def metrics(self) -> PipelineMetrics:
        """Convenience property: calls :meth:`collect_metrics`.

        Returns
        -------
        PipelineMetrics
            Aggregated metrics.
        """
        return self.collect_metrics()

    def cleanup(self) -> None:
        """Remove the temporary metrics directory and all files."""
        if self._metrics_dir.exists():
            shutil.rmtree(self._metrics_dir)

    # -- Private helpers ------------------------------------------------------

    def _write_index_metrics(self, idx_metrics: IndexMetrics) -> None:
        """Serialize IndexMetrics to a JSON file in the temp directory.

        Parameters
        ----------
        idx_metrics : IndexMetrics
            Metrics to serialize.
        """
        # Use a unique filename to avoid collisions in parallel execution
        fname = f"{idx_metrics.index:010d}_{uuid.uuid4().hex[:8]}.json"
        filepath = self._metrics_dir / fname
        # Atomic write: write to temp, then rename
        tmp_path = filepath.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(idx_metrics.to_dict()))
        tmp_path.rename(filepath)

    def _gpu_setup(self) -> int | None:
        """Reset GPU peak stats and return baseline memory.

        Returns
        -------
        int | None
            Baseline GPU memory in bytes, or None if unavailable.
        """
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                return torch.cuda.memory_allocated()
        except ImportError:
            pass
        return None

    def _gpu_measure(self, baseline: int) -> int:
        """Measure peak GPU memory delta from baseline.

        Parameters
        ----------
        baseline : int
            GPU memory at start of ``__getitem__``.

        Returns
        -------
        int
            Peak GPU memory minus baseline (bytes).
        """
        import torch

        return torch.cuda.max_memory_allocated() - baseline
```

- [ ] **Step 4: Run all profiling unit tests**

Run: `uv run pytest test/core/test_profiling.py -v --no-header`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/physicsnemo_curator/core/profiling.py test/core/test_profiling.py
git commit -m "feat(profiling): add _TimedGenerator and ProfiledPipeline with instrumented __getitem__"
```

---

### Task 3: Output Format Tests — JSON, CSV, Console

**Files:**
- Modify: `test/core/test_profiling.py`

- [ ] **Step 1: Write failing tests for output formats**

Append to `test/core/test_profiling.py`:

```python
class TestOutputFormats:
    """Tests for PipelineMetrics output methods."""

    def _make_metrics(self):
        """Build a small PipelineMetrics for output testing."""
        from physicsnemo_curator.core.profiling import IndexMetrics, PipelineMetrics, StageMetrics

        idx0 = IndexMetrics(
            index=0,
            stages=[
                StageMetrics(name="source", wall_time_ns=1_000_000),
                StageMetrics(name="DoubleFilter", wall_time_ns=500_000),
                StageMetrics(name="sink", wall_time_ns=200_000),
            ],
            wall_time_ns=1_700_000,
            peak_memory_bytes=1_048_576,
            gpu_memory_bytes=None,
        )
        idx1 = IndexMetrics(
            index=1,
            stages=[
                StageMetrics(name="source", wall_time_ns=900_000),
                StageMetrics(name="DoubleFilter", wall_time_ns=400_000),
                StageMetrics(name="sink", wall_time_ns=300_000),
            ],
            wall_time_ns=1_600_000,
            peak_memory_bytes=2_097_152,
            gpu_memory_bytes=4096,
        )
        return PipelineMetrics(indices=[idx0, idx1])

    def test_to_json(self, tmp_path):
        """to_json writes valid JSON with expected keys."""
        pm = self._make_metrics()
        out = tmp_path / "metrics.json"
        pm.to_json(out)

        data = json.loads(out.read_text())
        assert data["num_indices"] == 2
        assert data["total_wall_time_ns"] == 3_300_000
        assert len(data["indices"]) == 2
        assert data["indices"][0]["index"] == 0

    def test_to_csv(self, tmp_path):
        """to_csv writes valid CSV with per-index rows and stage columns."""
        pm = self._make_metrics()
        out = tmp_path / "metrics.csv"
        pm.to_csv(out)

        lines = out.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 rows
        header = lines[0]
        assert "index" in header
        assert "stage_source_ns" in header
        assert "stage_DoubleFilter_ns" in header

    def test_to_csv_empty(self, tmp_path):
        """to_csv with no indices writes empty file."""
        from physicsnemo_curator.core.profiling import PipelineMetrics

        pm = PipelineMetrics(indices=[])
        out = tmp_path / "empty.csv"
        pm.to_csv(out)
        assert out.read_text() == ""

    def test_to_console(self, capsys):
        """to_console writes to stdout without errors."""
        pm = self._make_metrics()
        pm.to_console()
        captured = capsys.readouterr()
        assert "Pipeline Profiling Results" in captured.out
        assert "1.70" in captured.out or "1,70" in captured.out  # total_ms for index 0

    def test_to_console_empty(self, capsys):
        """to_console with empty metrics prints a message."""
        from physicsnemo_curator.core.profiling import PipelineMetrics

        pm = PipelineMetrics(indices=[])
        pm.to_console()
        captured = capsys.readouterr()
        assert "No profiling metrics" in captured.out

    def test_summary_dict(self):
        """summary() returns dict with expected structure."""
        pm = self._make_metrics()
        s = pm.summary()
        assert s["num_indices"] == 2
        assert "total_wall_time_ns" in s
        assert "mean_index_time_ns" in s
        assert "indices" in s
```

- [ ] **Step 2: Run output format tests**

Run: `uv run pytest test/core/test_profiling.py::TestOutputFormats -v --no-header`
Expected: All 6 tests PASS (implementation already done in Task 1)

- [ ] **Step 3: Commit**

```bash
git add test/core/test_profiling.py
git commit -m "test(profiling): add output format tests (JSON, CSV, console, summary)"
```

---

### Task 4: Thread-Safety Test (Simulated Parallel)

**Files:**
- Modify: `test/core/test_profiling.py`

- [ ] **Step 1: Write thread-safety test**

Append to `test/core/test_profiling.py`:

```python
import concurrent.futures


class TestConcurrentMetrics:
    """Test metrics collection under concurrent access."""

    def test_concurrent_getitem(self):
        """Multiple threads calling __getitem__ concurrently."""
        from physicsnemo_curator.core.profiling import ProfiledPipeline

        pipeline = _TimedSource(20, delay=0.0).filter(_DoubleFilter()).write(_CollectSink())
        profiled = ProfiledPipeline(pipeline)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(profiled.__getitem__, i) for i in range(20)]
            results = [f.result() for f in futures]

        assert len(results) == 20
        metrics = profiled.collect_metrics()
        assert len(metrics.indices) == 20
```

- [ ] **Step 2: Run concurrent test**

Run: `uv run pytest test/core/test_profiling.py::TestConcurrentMetrics -v --no-header`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add test/core/test_profiling.py
git commit -m "test(profiling): add concurrent metrics collection test"
```

---

### Task 5: Public API Export

**Files:**
- Modify: `src/physicsnemo_curator/__init__.py`

- [ ] **Step 1: Write failing test for public API**

Append to `test/core/test_profiling.py`:

```python
class TestPublicAPI:
    """Test that profiling classes are exported from the package."""

    def test_import_profiled_pipeline(self):
        """ProfiledPipeline is importable from top-level package."""
        from physicsnemo_curator import ProfiledPipeline

        assert ProfiledPipeline is not None

    def test_import_pipeline_metrics(self):
        """PipelineMetrics is importable from top-level package."""
        from physicsnemo_curator import PipelineMetrics

        assert PipelineMetrics is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/core/test_profiling.py::TestPublicAPI -v --no-header -x`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Add exports to `__init__.py`**

Edit `src/physicsnemo_curator/__init__.py`:

Add after the existing imports (around line 24):
```python
from physicsnemo_curator.core.profiling import PipelineMetrics, ProfiledPipeline
```

Add to the `__all__` list (alphabetical order):
```python
    "PipelineMetrics",
    "ProfiledPipeline",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/core/test_profiling.py::TestPublicAPI -v --no-header`
Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/physicsnemo_curator/__init__.py test/core/test_profiling.py
git commit -m "feat(profiling): export ProfiledPipeline and PipelineMetrics from public API"
```

---

### Task 6: Integration Tests with Backends

**Files:**
- Create: `test/run/test_profiling_backends.py`

- [ ] **Step 1: Write integration tests**

Create `test/run/test_profiling_backends.py`:

```python
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

"""Integration tests for ProfiledPipeline with execution backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

import pytest

from physicsnemo_curator.core.base import Filter, Param, Pipeline, Sink, Source
from physicsnemo_curator.core.profiling import ProfiledPipeline
from physicsnemo_curator.run import run_pipeline

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Module-level test components (pickle-safe for multiprocessing)
# ---------------------------------------------------------------------------


class _ProfNumberSource(Source[int]):
    """Source that yields sequential integers."""

    name: ClassVar[str] = "ProfNumbers"
    description: ClassVar[str] = "Yields ints for profiling tests"

    @classmethod
    def params(cls) -> list[Param]:
        """Return params."""
        return []

    def __init__(self, count: int) -> None:
        """Initialize source."""
        self._count = count

    def __len__(self) -> int:
        """Return count."""
        return self._count

    def __getitem__(self, index: int) -> Generator[int]:
        """Yield index."""
        yield index


class _ProfDoubleFilter(Filter[int]):
    """Filter that doubles each value."""

    name: ClassVar[str] = "ProfDouble"
    description: ClassVar[str] = "Doubles items"

    @classmethod
    def params(cls) -> list[Param]:
        """Return empty params."""
        return []

    def __call__(self, items: Generator[int]) -> Generator[int]:
        """Double each item."""
        for item in items:
            yield item * 2


class _ProfListSink(Sink[int]):
    """Sink that returns items as formatted strings."""

    name: ClassVar[str] = "ProfListSink"
    description: ClassVar[str] = "Returns items as strings"

    @classmethod
    def params(cls) -> list[Param]:
        """Return empty params."""
        return []

    def __call__(self, items: Iterator[int], index: int) -> list[str]:
        """Return formatted strings."""
        return [f"prof_{index}_{v}" for v in items]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def profiled_pipeline() -> ProfiledPipeline[int]:
    """A 5-item profiled pipeline: source -> double -> list sink."""
    pipeline = _ProfNumberSource(5).filter(_ProfDoubleFilter()).write(_ProfListSink())
    return ProfiledPipeline(pipeline)


# ---------------------------------------------------------------------------
# Sequential backend
# ---------------------------------------------------------------------------


class TestSequentialProfiling:
    """ProfiledPipeline with sequential backend."""

    def test_results_match(self, profiled_pipeline):
        """Profiled results should match non-profiled results."""
        raw_pipeline = _ProfNumberSource(5).filter(_ProfDoubleFilter()).write(_ProfListSink())
        raw_results = run_pipeline(raw_pipeline, backend="sequential", progress=False)
        profiled_results = run_pipeline(profiled_pipeline, backend="sequential", progress=False)
        assert profiled_results == raw_results

    def test_metrics_collected(self, profiled_pipeline):
        """Metrics should be collected for all indices."""
        run_pipeline(profiled_pipeline, backend="sequential", progress=False)
        metrics = profiled_pipeline.collect_metrics()
        assert len(metrics.indices) == 5

    def test_per_stage_present(self, profiled_pipeline):
        """Each index should have source, filter, sink stages."""
        run_pipeline(profiled_pipeline, backend="sequential", progress=False)
        metrics = profiled_pipeline.collect_metrics()
        for idx_m in metrics.indices:
            names = [s.name for s in idx_m.stages]
            assert "source" in names
            assert "ProfDouble" in names
            assert "sink" in names

    def test_cleanup(self, profiled_pipeline):
        """cleanup() removes temp files after run."""
        run_pipeline(profiled_pipeline, backend="sequential", progress=False)
        profiled_pipeline.cleanup()
        assert not profiled_pipeline._metrics_dir.exists()


# ---------------------------------------------------------------------------
# Thread pool backend
# ---------------------------------------------------------------------------


class TestThreadPoolProfiling:
    """ProfiledPipeline with thread_pool backend."""

    def test_results_match(self):
        """Thread pool profiled results match raw results."""
        raw_pipeline = _ProfNumberSource(5).filter(_ProfDoubleFilter()).write(_ProfListSink())
        profiled = ProfiledPipeline(raw_pipeline)

        raw_results = run_pipeline(raw_pipeline, n_jobs=2, backend="thread_pool", progress=False)
        profiled_results = run_pipeline(profiled, n_jobs=2, backend="thread_pool", progress=False)
        assert profiled_results == raw_results

    def test_metrics_collected(self):
        """Thread pool collects metrics for all indices."""
        pipeline = _ProfNumberSource(8).filter(_ProfDoubleFilter()).write(_ProfListSink())
        profiled = ProfiledPipeline(pipeline)
        run_pipeline(profiled, n_jobs=3, backend="thread_pool", progress=False)
        metrics = profiled.collect_metrics()
        assert len(metrics.indices) == 8
        profiled.cleanup()


# ---------------------------------------------------------------------------
# Process pool backend
# ---------------------------------------------------------------------------


class TestProcessPoolProfiling:
    """ProfiledPipeline with process_pool backend."""

    def test_results_match(self):
        """Process pool profiled results match raw results."""
        raw_pipeline = _ProfNumberSource(5).filter(_ProfDoubleFilter()).write(_ProfListSink())
        profiled = ProfiledPipeline(raw_pipeline)

        raw_results = run_pipeline(raw_pipeline, n_jobs=2, backend="process_pool", progress=False)
        profiled_results = run_pipeline(profiled, n_jobs=2, backend="process_pool", progress=False)
        assert profiled_results == raw_results

    def test_metrics_collected_across_processes(self):
        """Temp-file metrics survive process boundaries."""
        pipeline = _ProfNumberSource(6).filter(_ProfDoubleFilter()).write(_ProfListSink())
        profiled = ProfiledPipeline(pipeline)
        run_pipeline(profiled, n_jobs=2, backend="process_pool", progress=False)
        metrics = profiled.collect_metrics()
        # All 6 indices should have metrics via temp files
        assert len(metrics.indices) == 6
        collected_indices = {m.index for m in metrics.indices}
        assert collected_indices == {0, 1, 2, 3, 4, 5}
        profiled.cleanup()

    def test_subset_indices(self):
        """Process pool with subset indices collects correct metrics."""
        pipeline = _ProfNumberSource(10).filter(_ProfDoubleFilter()).write(_ProfListSink())
        profiled = ProfiledPipeline(pipeline)
        run_pipeline(profiled, n_jobs=2, backend="process_pool", indices=[1, 3, 7], progress=False)
        metrics = profiled.collect_metrics()
        assert len(metrics.indices) == 3
        collected_indices = {m.index for m in metrics.indices}
        assert collected_indices == {1, 3, 7}
        profiled.cleanup()
```

- [ ] **Step 2: Run integration tests**

Run: `uv run pytest test/run/test_profiling_backends.py -v --no-header`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add test/run/test_profiling_backends.py
git commit -m "test(profiling): add integration tests for sequential, thread_pool, and process_pool backends"
```

---

### Task 7: ASV Benchmarks

**Files:**
- Create: `benchmarks/bench_profiling.py`
- Modify: `docs/developer-guide/benchmarking.md`

- [ ] **Step 1: Create ASV benchmark file**

Create `benchmarks/bench_profiling.py`:

```python
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

"""Benchmarks for profiling overhead (ProfiledPipeline vs raw Pipeline)."""

from collections.abc import Generator

from curator.core.base import Filter, Param, Pipeline, Sink, Source
from curator.core.profiling import ProfiledPipeline

# ── helpers ──────────────────────────────────────────────────────────────────


class _NumberSource(Source[int]):
    """Emit sequential integers."""

    name = "number-source"
    description = "Benchmark helper: sequential integer source"

    @classmethod
    def params(cls) -> list[Param]:  # noqa: D102
        return [Param(name="n", description="Number of items", type=int)]

    def __init__(self, n: int) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> Generator[int]:  # type: ignore[override]  # noqa: D105
        yield index


class _DoubleFilter(Filter[int]):
    """Multiply each item by two."""

    name = "double-filter"
    description = "Benchmark helper: doubles items"

    @classmethod
    def params(cls) -> list[Param]:  # noqa: D102
        return []

    def __call__(self, items: Generator[int]) -> Generator[int]:  # noqa: D102
        for item in items:
            yield item * 2


class _NullSink(Sink[int]):
    """Discard all items."""

    name = "null-sink"
    description = "Benchmark helper: discards all items"

    @classmethod
    def params(cls) -> list[Param]:  # noqa: D102
        return []

    def __call__(self, items: Generator[int], index: int) -> list[str]:  # noqa: D102
        for _ in items:
            pass
        return []


# ── benchmarks ───────────────────────────────────────────────────────────────


class TimeProfilingOverhead:
    """Benchmark wall-clock overhead of ProfiledPipeline vs raw Pipeline."""

    params = [[10, 100, 1000]]
    param_names = ["n_indices"]

    def setup(self, n_indices):
        """Build raw and profiled pipelines."""
        self.pipeline = _NumberSource(n_indices).filter(_DoubleFilter()).write(_NullSink())
        self.profiled = ProfiledPipeline(self.pipeline)

    def time_raw_pipeline(self, n_indices):
        """Time iterating raw pipeline over all indices."""
        for i in range(n_indices):
            self.pipeline[i]

    def time_profiled_pipeline(self, n_indices):
        """Time iterating profiled pipeline over all indices."""
        for i in range(n_indices):
            self.profiled[i]
        self.profiled.cleanup()

    def track_overhead_percent(self, n_indices):
        """Compute profiling overhead as a percentage."""
        import time

        # Raw
        start = time.perf_counter_ns()
        for i in range(n_indices):
            self.pipeline[i]
        raw_ns = time.perf_counter_ns() - start

        # Profiled
        profiled = ProfiledPipeline(self.pipeline)
        start = time.perf_counter_ns()
        for i in range(n_indices):
            profiled[i]
        profiled_ns = time.perf_counter_ns() - start
        profiled.cleanup()

        if raw_ns == 0:
            return 0.0
        return ((profiled_ns - raw_ns) / raw_ns) * 100

    track_overhead_percent.unit = "percent"  # type: ignore[attr-defined]


class MemProfilingOverhead:
    """Benchmark memory overhead of ProfiledPipeline."""

    params = [[10, 100]]
    param_names = ["n_indices"]

    def setup(self, n_indices):
        """Build pipelines."""
        self.pipeline = _NumberSource(n_indices).filter(_DoubleFilter()).write(_NullSink())

    def peakmem_raw_pipeline(self, n_indices):
        """Peak memory for raw pipeline iteration."""
        for i in range(n_indices):
            self.pipeline[i]

    def peakmem_profiled_pipeline(self, n_indices):
        """Peak memory for profiled pipeline iteration."""
        profiled = ProfiledPipeline(self.pipeline)
        for i in range(n_indices):
            profiled[i]
        profiled.cleanup()


class TimeMetricsCollection:
    """Benchmark metrics collection and serialization."""

    params = [[10, 100, 1000]]
    param_names = ["n_indices"]

    def setup(self, n_indices):
        """Run profiled pipeline to generate metrics files."""
        pipeline = _NumberSource(n_indices).filter(_DoubleFilter()).write(_NullSink())
        self.profiled = ProfiledPipeline(pipeline)
        for i in range(n_indices):
            self.profiled[i]

    def time_collect_metrics(self, n_indices):
        """Time reading and aggregating metrics from temp files."""
        self.profiled.collect_metrics()

    def time_to_json(self, n_indices):
        """Time serializing metrics to JSON."""
        import tempfile

        metrics = self.profiled.collect_metrics()
        out = tempfile.mktemp(suffix=".json")
        metrics.to_json(out)

    def teardown(self, n_indices):
        """Clean up temp files."""
        self.profiled.cleanup()
```

- [ ] **Step 2: Update benchmarking docs**

Edit `docs/developer-guide/benchmarking.md` line 79-83 to add `bench_profiling.py`:

Change:
```text
benchmarks/
├── __init__.py
├── bench_pipeline.py    # Pipeline construction & iteration
├── bench_store.py       # FileStore creation & indexing
└── bench_import.py      # Package import time
```

To:
```text
benchmarks/
├── __init__.py
├── bench_pipeline.py    # Pipeline construction & iteration
├── bench_profiling.py   # Profiling overhead measurement
├── bench_store.py       # FileStore creation & indexing
└── bench_import.py      # Package import time
```

- [ ] **Step 3: Verify ASV can discover the benchmark**

Run: `uv run asv check --config asv.conf.json 2>&1 | head -20`
Expected: No errors for `bench_profiling.py`

- [ ] **Step 4: Commit**

```bash
git add benchmarks/bench_profiling.py docs/developer-guide/benchmarking.md
git commit -m "bench(profiling): add ASV benchmarks for ProfiledPipeline overhead"
```

---

### Task 8: User Documentation

**Files:**
- Create: `docs/user-guide/profiling.md`
- Modify: `docs/user-guide/parallel.md`

- [ ] **Step 1: Create profiling user guide**

Create `docs/user-guide/profiling.md`:

```markdown
# Profiling Pipelines

`ProfiledPipeline` is a transparent wrapper around `Pipeline` that collects
wall-clock time, memory, and (optionally) GPU metrics at whole-pipeline,
per-index, and per-stage granularity — without requiring any changes to
your pipeline or backend configuration.

## Quick Start

```python
from physicsnemo_curator import Pipeline, ProfiledPipeline, run_pipeline

# Wrap any existing pipeline
profiled = ProfiledPipeline(pipeline)

# Run exactly as before — works with all backends
results = run_pipeline(profiled, n_jobs=4, backend="process_pool")

# Inspect metrics
metrics = profiled.metrics
metrics.to_console()

# Clean up temp files when done
profiled.cleanup()
```

## Metrics Granularity

Profiling collects data at three levels:

| Level | What's measured |
|-------|----------------|
| **Whole-pipeline** | Total wall time, peak memory across all indices |
| **Per-index** | Wall time, peak memory, GPU memory for each source index |
| **Per-stage** | Wall time for source, each filter, and sink |

### Per-Stage Timing

The pipeline chain `source → filter₁ → filter₂ → … → sink` uses lazy
generators.  `ProfiledPipeline` wraps each stage's generator with an
internal timer to attribute time accurately using chain subtraction:

- **Source time** = time spent yielding items from the source
- **Filter N time** = time spent in filter N's own logic (excluding upstream)
- **Sink time** = time spent in the sink (excluding all upstream generators)

### Memory Tracking

Peak Python memory per index is tracked via `tracemalloc`. This captures
Python-level allocations accurately but does not cover C-extension or
Rust-extension memory. Per-stage memory is not tracked because chained
lazy generators make per-stage attribution unreliable.

### GPU Memory Tracking

To track GPU memory, pass `track_gpu=True`:

```python
profiled = ProfiledPipeline(pipeline, track_gpu=True)
```

This uses `torch.cuda.max_memory_allocated()` to capture peak GPU memory
per index. Requires PyTorch with CUDA support.  If `torch` is not installed
or CUDA is unavailable, GPU fields will be `None`.

## Output Formats

### Console Table

```python
metrics.to_console()
```

Prints a human-readable summary with per-index breakdown and stage averages.

### JSON File

```python
metrics.to_json("profile.json")
```

Writes full metrics (including per-stage breakdowns) as structured JSON.

### CSV File

```python
metrics.to_csv("profile.csv")
```

Writes one row per index with columns for wall time, memory, GPU memory,
and per-stage timing.

### Programmatic Access

```python
info = metrics.summary()      # dict
info["total_wall_time_ns"]    # int
info["mean_index_time_ns"]    # float
info["indices"][0]["stages"]  # list of stage dicts
```

## Using with Parallel Backends

`ProfiledPipeline` works with **all** backends — sequential, thread_pool,
process_pool, loky, dask, and prefect — without any backend modifications.

For multiprocess backends (`process_pool`, `loky`, `dask`, `prefect`),
metrics are serialized to a temporary directory as JSON files. Each worker
process writes its own index metrics to a uniquely-named file. After
`run_pipeline()` returns, call `profiled.metrics` (or
`profiled.collect_metrics()`) to read and aggregate all results.

```python
profiled = ProfiledPipeline(pipeline)
results = run_pipeline(profiled, n_jobs=8, backend="process_pool")

# Reads all temp files and aggregates
metrics = profiled.metrics
metrics.to_console()

# Clean up temp directory
profiled.cleanup()
```

## Full Example

```python
from physicsnemo_curator import Pipeline, ProfiledPipeline, run_pipeline

# Build a pipeline
pipeline = (
    MySource(path="/data/cfd/")
    .filter(NormalizeFilter())
    .filter(ResampleFilter(target_resolution=0.01))
    .write(MeshSink(output_dir="/output/"))
)

# Profile it
profiled = ProfiledPipeline(pipeline, track_gpu=True)
results = run_pipeline(profiled, n_jobs=4, backend="process_pool")

# Analyze
metrics = profiled.metrics
metrics.to_console()             # Quick visual summary
metrics.to_json("profile.json")  # Detailed JSON for analysis
metrics.to_csv("profile.csv")    # CSV for spreadsheet

# Programmatic access
summary = metrics.summary()
print(f"Processed {summary['num_indices']} indices")
print(f"Total time: {summary['total_wall_time_ns'] / 1e9:.2f}s")
print(f"Peak memory: {summary['total_peak_memory_bytes'] / 1e6:.1f} MB")

# Clean up
profiled.cleanup()
```

## API Reference

### `ProfiledPipeline`

```python
class ProfiledPipeline(Generic[T]):
    def __init__(self, pipeline: Pipeline[T], *, track_gpu: bool = False) -> None: ...

    # Duck-type compatibility
    source: Source[T]          # property
    filters: list[Filter[T]]  # property
    sink: Sink[T] | None      # property
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> list[str]: ...

    # Metrics
    def collect_metrics(self) -> PipelineMetrics: ...
    metrics: PipelineMetrics   # property (shortcut for collect_metrics)
    def cleanup(self) -> None: ...
```

### `PipelineMetrics`

```python
class PipelineMetrics:
    indices: list[IndexMetrics]

    # Properties
    total_wall_time_ns: int
    mean_index_time_ns: float
    total_peak_memory_bytes: int

    # Output
    def to_console(self) -> None: ...
    def to_json(self, path: str | Path) -> None: ...
    def to_csv(self, path: str | Path) -> None: ...
    def summary(self) -> dict: ...
```

### `IndexMetrics`

```python
class IndexMetrics:
    index: int
    stages: list[StageMetrics]
    wall_time_ns: int
    peak_memory_bytes: int
    gpu_memory_bytes: int | None
```

### `StageMetrics`

```python
class StageMetrics:
    name: str
    wall_time_ns: int
```
```

- [ ] **Step 2: Add cross-reference in parallel.md**

Edit `docs/user-guide/parallel.md`. After the "Process Isolation" section (after line 113), add:

```markdown

## Profiling

To measure wall-clock time, memory, and GPU usage across parallel backends,
wrap your pipeline with `ProfiledPipeline`. See [Profiling](profiling.md)
for details.
```

- [ ] **Step 3: Commit**

```bash
git add docs/user-guide/profiling.md docs/user-guide/parallel.md
git commit -m "docs(profiling): add user guide and cross-reference in parallel docs"
```

---

### Task 9: Linting, Type-Checking, and Interrogate

**Files:**
- All new/modified files

- [ ] **Step 1: Run ruff formatter**

Run: `uv run ruff format src/physicsnemo_curator/core/profiling.py test/core/test_profiling.py test/run/test_profiling_backends.py benchmarks/bench_profiling.py`

- [ ] **Step 2: Run ruff linter**

Run: `uv run ruff check --fix src/physicsnemo_curator/core/profiling.py test/core/test_profiling.py test/run/test_profiling_backends.py benchmarks/bench_profiling.py`

- [ ] **Step 3: Run type checker**

Run: `uv run ty check`
Expected: No new errors in profiling files

- [ ] **Step 4: Run interrogate**

Run: `uv run interrogate src/physicsnemo_curator/core/profiling.py`
Expected: 100% docstring coverage

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest test/core/test_profiling.py test/run/test_profiling_backends.py -v --tb=short`
Expected: All tests PASS

- [ ] **Step 6: Fix any issues and commit**

```bash
git add -A
git commit -m "chore(profiling): fix linting, formatting, and type-checking issues"
```

---

### Task 10: Final Validation

- [ ] **Step 1: Run all tests with coverage**

Run: `uv run pytest test/ -v --cov=physicsnemo_curator.core.profiling --cov-report=term-missing --tb=short -x`
Expected: High coverage (>90%) on `profiling.py`

- [ ] **Step 2: Run full project checks**

Run: `make check` (runs format + lint + typecheck + interrogate + deny)
Expected: All pass

- [ ] **Step 3: Verify all tests pass**

Run: `uv run pytest test/ -v --tb=short`
Expected: All existing + new tests PASS, no regressions

- [ ] **Step 4: Commit any remaining fixes**

```bash
git add -A
git commit -m "chore(profiling): final validation and coverage improvements"
```
