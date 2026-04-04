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

import concurrent.futures
import json
import pickle
import sys
import time
from typing import TYPE_CHECKING, ClassVar
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

import pytest

from physicsnemo_curator.core.base import Filter, Param, Sink, Source
from physicsnemo_curator.core.profiling import (
    IndexMetrics,
    PipelineMetrics,
    ProfiledPipeline,
    StageMetrics,
    _TimedGenerator,
)

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
        m = StageMetrics(name="source", wall_time_ns=1_000_000)
        assert m.name == "source"
        assert m.wall_time_ns == 1_000_000

    def test_to_dict(self):
        """StageMetrics.to_dict() returns expected keys."""
        m = StageMetrics(name="DoubleFilter", wall_time_ns=500_000)
        d = m.to_dict()
        assert d == {"name": "DoubleFilter", "wall_time_ns": 500_000}


class TestIndexMetrics:
    """Tests for IndexMetrics dataclass."""

    def test_creation(self):
        """IndexMetrics has index, stages, wall_time_ns, peak_memory_bytes, gpu_memory_bytes."""
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
        pm = PipelineMetrics(indices=[])
        assert pm.total_wall_time_ns == 0
        assert pm.mean_index_time_ns == 0.0
        assert pm.total_peak_memory_bytes == 0


class TestTimedGenerator:
    """Tests for the internal _TimedGenerator wrapper."""

    def test_iterates_correctly(self):
        """_TimedGenerator yields same values as wrapped generator."""

        def gen():
            yield 1
            yield 2
            yield 3

        tg = _TimedGenerator(gen())
        assert list(tg) == [1, 2, 3]

    def test_accumulates_time(self):
        """_TimedGenerator.elapsed_ns is positive after iteration."""

        def slow_gen():
            time.sleep(0.01)
            yield 1

        tg = _TimedGenerator(slow_gen())
        list(tg)  # consume
        assert tg.elapsed_ns > 0

    def test_empty_generator(self):
        """_TimedGenerator handles empty generators."""

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

        pipeline = _TimedSource(3).filter(_DoubleFilter()).write(_CollectSink())
        profiled = ProfiledPipeline(pipeline)
        assert profiled.source is pipeline.source
        assert profiled.filters is pipeline.filters
        assert profiled.sink is pipeline.sink
        assert len(profiled) == 3

    def test_getitem_returns_correct_results(self):
        """ProfiledPipeline.__getitem__ returns same results as Pipeline."""

        pipeline = _TimedSource(3, delay=0.0).filter(_DoubleFilter()).write(_CollectSink())
        profiled = ProfiledPipeline(pipeline)

        for i in range(3):
            assert profiled[i] == pipeline[i]

    def test_basic_metrics_collection(self):
        """Single index produces IndexMetrics with correct stage count."""

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

        pipeline = _TimedSource(1, delay=0.005).filter(_SlowFilter(delay=0.005)).write(_CollectSink())
        profiled = ProfiledPipeline(pipeline)
        profiled[0]
        metrics = profiled.collect_metrics()

        for stage in metrics.indices[0].stages:
            assert stage.wall_time_ns > 0, f"Stage {stage.name} has non-positive time"

    def test_stage_times_approximate_total(self):
        """Sum of stage times should approximate total index time."""

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

        pipeline = _AllocSource(1, alloc_bytes=500_000).write(_CollectSink())
        profiled = ProfiledPipeline(pipeline)
        profiled[0]
        metrics = profiled.collect_metrics()
        # Should detect at least some memory (tracemalloc not perfectly precise)
        assert metrics.indices[0].peak_memory_bytes > 0

    def test_gpu_tracking_disabled_by_default(self):
        """GPU memory is None when track_gpu=False."""

        pipeline = _TimedSource(1, delay=0.0).write(_CollectSink())
        profiled = ProfiledPipeline(pipeline)
        profiled[0]
        metrics = profiled.collect_metrics()
        assert metrics.indices[0].gpu_memory_bytes is None

    def test_gpu_tracking_mocked(self):
        """GPU tracking works with mocked torch.cuda."""

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

        pipeline = _TimedSource(1, delay=0.0).filter(_ErrorFilter()).write(_CollectSink())
        profiled = ProfiledPipeline(pipeline)
        with pytest.raises(RuntimeError, match="intentional error"):
            profiled[0]

    def test_pickle_roundtrip(self):
        """ProfiledPipeline survives pickle round-trip."""

        pipeline = _TimedSource(3, delay=0.0).filter(_DoubleFilter()).write(_CollectSink())
        profiled = ProfiledPipeline(pipeline)

        data = pickle.dumps(profiled)
        restored = pickle.loads(data)  # noqa: S301
        assert len(restored) == 3
        assert restored[0] == profiled[0]

    def test_metrics_property(self):
        """The .metrics property is a shortcut for collect_metrics()."""

        pipeline = _TimedSource(1, delay=0.0).write(_CollectSink())
        profiled = ProfiledPipeline(pipeline)
        profiled[0]
        assert profiled.metrics.total_wall_time_ns == profiled.collect_metrics().total_wall_time_ns

    def test_cleanup(self):
        """cleanup() removes the temp metrics directory."""

        pipeline = _TimedSource(1, delay=0.0).write(_CollectSink())
        profiled = ProfiledPipeline(pipeline)
        profiled[0]
        metrics_dir = profiled._metrics_dir
        assert metrics_dir.exists()
        profiled.cleanup()
        assert not metrics_dir.exists()


class TestOutputFormats:
    """Tests for PipelineMetrics output methods."""

    def _make_metrics(self):
        """Build a small PipelineMetrics for output testing."""
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


class TestConcurrentMetrics:
    """Test metrics collection under concurrent access."""

    def test_concurrent_getitem(self):
        """Multiple threads calling __getitem__ concurrently."""
        pipeline = _TimedSource(20, delay=0.0).filter(_DoubleFilter()).write(_CollectSink())
        profiled = ProfiledPipeline(pipeline)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(profiled.__getitem__, i) for i in range(20)]
            results = [f.result() for f in futures]

        assert len(results) == 20
        metrics = profiled.collect_metrics()
        assert len(metrics.indices) == 20


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
