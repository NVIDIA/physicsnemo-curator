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

"""Integration tests for :mod:`physicsnemo_curator.run` backends.

These tests execute actual pipelines using each backend to verify
correct behavior in realistic scenarios.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar
from unittest.mock import patch

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

import pytest

from physicsnemo_curator.core.base import Filter, Param, Pipeline, Sink, Source
from physicsnemo_curator.run import _pick_auto_backend, run_pipeline

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Concrete test components (picklable at module level for multiprocessing)
# ---------------------------------------------------------------------------


class NumberSource(Source[int]):
    """Source that yields integers 0..n-1."""

    name: ClassVar[str] = "Numbers"
    description: ClassVar[str] = "Yields sequential integers"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter definitions."""
        return [Param(name="count", description="How many items", type=int)]

    def __init__(self, count: int) -> None:
        """Initialize with count."""
        self._count = count

    def __len__(self) -> int:
        """Return number of items."""
        return self._count

    def __getitem__(self, index: int) -> Generator[int]:
        """Yield the index value."""
        yield index


class TripleFilter(Filter[int]):
    """Filter that triples each value."""

    name: ClassVar[str] = "Triple"
    description: ClassVar[str] = "Multiplies by 3"

    @classmethod
    def params(cls) -> list[Param]:
        """Return empty params list."""
        return []

    def __call__(self, items: Generator[int]) -> Generator[int]:
        """Triple each item."""
        for item in items:
            yield item * 3


class ListSink(Sink[int]):
    """Sink that returns items as string paths."""

    name: ClassVar[str] = "ListSink"
    description: ClassVar[str] = "Returns items as strings"

    @classmethod
    def params(cls) -> list[Param]:
        """Return empty params list."""
        return []

    def __call__(self, items: Iterator[int], index: int) -> list[str]:
        """Return items as formatted strings."""
        return [f"item_{index}_{v}" for v in items]


class StatefulSink(Sink[int]):
    """Sink that tracks how many times it was called (in-process only)."""

    name: ClassVar[str] = "StatefulSink"
    description: ClassVar[str] = "Counts calls"

    @classmethod
    def params(cls) -> list[Param]:
        """Return empty params list."""
        return []

    def __init__(self) -> None:
        """Initialize call counter."""
        self.call_count = 0

    def __call__(self, items: Iterator[int], index: int) -> list[str]:
        """Increment counter and return items as strings."""
        self.call_count += 1
        return [str(v) for v in items]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_pipeline() -> Pipeline[int]:
    """A 5-item pipeline: source → triple → list sink."""
    p = NumberSource(5).filter(TripleFilter()).write(ListSink())
    p.track_metrics = False
    return p


@pytest.fixture
def no_filter_pipeline() -> Pipeline[int]:
    """A 3-item pipeline with no filters."""
    p = NumberSource(3).write(ListSink())
    p.track_metrics = False
    return p


# ---------------------------------------------------------------------------
# Sequential backend tests
# ---------------------------------------------------------------------------


class TestSequentialBackend:
    """Integration tests for the sequential backend."""

    def test_processes_all_indices(self, simple_pipeline):
        """Should process all indices in order."""
        results = run_pipeline(simple_pipeline, n_jobs=1, progress=False)
        assert len(results) == 5
        # Index 0 → 0*3=0, Index 4 → 4*3=12
        assert results[0] == ["item_0_0"]
        assert results[4] == ["item_4_12"]

    def test_explicit_sequential_backend(self, simple_pipeline):
        """Explicit backend="sequential" should work."""
        results = run_pipeline(simple_pipeline, backend="sequential", progress=False)
        assert len(results) == 5

    def test_subset_indices(self, simple_pipeline):
        """Should process only specified indices."""
        results = run_pipeline(simple_pipeline, indices=[1, 3], progress=False)
        assert len(results) == 2
        assert results[0] == ["item_1_3"]  # index 1 → 1*3=3
        assert results[1] == ["item_3_9"]  # index 3 → 3*3=9

    def test_empty_indices(self, simple_pipeline):
        """Empty indices should return empty results."""
        results = run_pipeline(simple_pipeline, indices=[], progress=False)
        assert results == []

    def test_no_filter_pipeline(self, no_filter_pipeline):
        """Pipeline without filters should work."""
        results = run_pipeline(no_filter_pipeline, progress=False)
        assert len(results) == 3
        assert results[0] == ["item_0_0"]
        assert results[2] == ["item_2_2"]

    def test_stateful_sink_sequential(self):
        """Sequential execution should preserve state in sink."""
        sink = StatefulSink()
        pipeline = NumberSource(3).write(sink)
        pipeline.track_metrics = False
        run_pipeline(pipeline, n_jobs=1, progress=False)
        # Sequential: sink is shared, call_count should increment
        assert sink.call_count == 3

    def test_with_progress(self, simple_pipeline):
        """Should not raise even if tqdm is not available."""
        results = run_pipeline(simple_pipeline, n_jobs=1, progress=True)
        assert len(results) == 5


# ---------------------------------------------------------------------------
# Process pool backend tests
# ---------------------------------------------------------------------------


class TestProcessPoolBackend:
    """Integration tests for the process_pool backend."""

    def test_basic_parallel(self, simple_pipeline):
        """Basic parallel execution with process pool."""
        results = run_pipeline(simple_pipeline, n_jobs=2, backend="process_pool", progress=False)
        assert len(results) == 5
        # Order should be preserved
        assert results[0] == ["item_0_0"]
        assert results[4] == ["item_4_12"]

    def test_subset_parallel(self, simple_pipeline):
        """Process pool with subset of indices."""
        results = run_pipeline(simple_pipeline, n_jobs=2, backend="process_pool", indices=[0, 2, 4], progress=False)
        assert len(results) == 3
        assert results[0] == ["item_0_0"]
        assert results[1] == ["item_2_6"]
        assert results[2] == ["item_4_12"]

    def test_stateful_sink_parallel_isolation(self):
        """Stateful sink in parent is not mutated by child processes."""
        sink = StatefulSink()
        pipeline = NumberSource(4).write(sink)
        pipeline.track_metrics = False
        results = run_pipeline(pipeline, n_jobs=2, backend="process_pool", progress=False)
        # Results should still be correct
        assert len(results) == 4
        # But parent sink should NOT have been called (children have copies)
        assert sink.call_count == 0


# ---------------------------------------------------------------------------
# Loky backend tests (optional dependency)
# ---------------------------------------------------------------------------


class TestLokyBackend:
    """Integration tests for the loky backend (requires joblib)."""

    def test_loky_runs(self, simple_pipeline):
        """Loky backend should produce correct results."""
        pytest.importorskip("joblib")
        results = run_pipeline(simple_pipeline, n_jobs=2, backend="loky", progress=False)
        assert len(results) == 5
        assert results[0] == ["item_0_0"]
        assert results[4] == ["item_4_12"]

    def test_loky_missing_raises(self, simple_pipeline):
        """Missing joblib should raise ImportError with helpful message."""
        with patch.dict("sys.modules", {"joblib": None}), pytest.raises(ImportError, match="joblib"):
            run_pipeline(simple_pipeline, n_jobs=2, backend="loky", progress=False)


# ---------------------------------------------------------------------------
# Dask backend tests (optional dependency)
# ---------------------------------------------------------------------------


class TestDaskBackend:
    """Integration tests for the dask backend (requires dask)."""

    def test_dask_runs(self, simple_pipeline):
        """Dask backend should produce correct results."""
        pytest.importorskip("dask")
        results = run_pipeline(simple_pipeline, n_jobs=2, backend="dask", progress=False)
        assert len(results) == 5
        assert results[0] == ["item_0_0"]
        assert results[4] == ["item_4_12"]

    def test_dask_missing_raises(self, simple_pipeline):
        """Missing dask should raise ImportError with helpful message."""
        with patch.dict("sys.modules", {"dask": None, "dask.bag": None}), pytest.raises(ImportError, match="dask"):
            run_pipeline(simple_pipeline, n_jobs=2, backend="dask", progress=False)


# ---------------------------------------------------------------------------
# Auto backend tests
# ---------------------------------------------------------------------------


class TestAutoBackend:
    """Integration tests for auto backend selection."""

    def test_auto_selects_something(self, simple_pipeline):
        """Auto backend should select a working backend."""
        results = run_pipeline(simple_pipeline, n_jobs=2, backend="auto", progress=False)
        assert len(results) == 5

    def test_pick_auto_backend_returns_valid(self):
        """_pick_auto_backend should return a valid backend name."""
        result = _pick_auto_backend()
        assert result in ("dask", "loky", "process_pool")


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error conditions."""

    def test_unknown_backend_raises(self, simple_pipeline):
        """Unknown backend should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            run_pipeline(simple_pipeline, backend="magic")

    def test_no_sink_raises(self):
        """Pipeline without sink should raise RuntimeError."""
        pipeline = NumberSource(3).filter(TripleFilter())
        pipeline.track_metrics = False
        with pytest.raises(RuntimeError, match="no sink"):
            run_pipeline(pipeline)

    def test_n_jobs_one_forces_sequential(self, simple_pipeline):
        """n_jobs=1 should force sequential even with different backend."""
        sink = StatefulSink()
        pipeline = NumberSource(3).write(sink)
        pipeline.track_metrics = False
        # Even with backend="process_pool", n_jobs=1 should use sequential
        run_pipeline(pipeline, n_jobs=1, backend="process_pool", progress=False)
        # Sequential preserves state
        assert sink.call_count == 3
