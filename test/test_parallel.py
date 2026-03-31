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

"""Tests for :mod:`curator.core.parallel` — ``run_pipeline()``."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar
from unittest.mock import patch

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

import pytest

from curator.core.base import Filter, Param, Pipeline, Sink, Source
from curator.core.parallel import (
    _pick_auto_backend,
    _resolve_n_jobs,
    run_pipeline,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Concrete test components (picklable at module level)
# ---------------------------------------------------------------------------


class NumberSource(Source[int]):
    """Source that yields integers 0..n-1."""

    name: ClassVar[str] = "Numbers"
    description: ClassVar[str] = "Yields sequential integers"

    @classmethod
    def params(cls) -> list[Param]:
        return [Param(name="count", description="How many items", type=int)]

    def __init__(self, count: int) -> None:
        self._count = count

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int) -> Generator[int]:
        yield index


class TripleFilter(Filter[int]):
    """Filter that triples each value."""

    name: ClassVar[str] = "Triple"
    description: ClassVar[str] = "Multiplies by 3"

    @classmethod
    def params(cls) -> list[Param]:
        return []

    def __call__(self, items: Generator[int]) -> Generator[int]:
        for item in items:
            yield item * 3


class ListSink(Sink[int]):
    """Sink that returns items as string paths."""

    name: ClassVar[str] = "ListSink"
    description: ClassVar[str] = "Returns items as strings"

    @classmethod
    def params(cls) -> list[Param]:
        return []

    def __call__(self, items: Iterator[int], index: int) -> list[str]:
        return [f"item_{index}_{v}" for v in items]


class StatefulSink(Sink[int]):
    """Sink that tracks how many times it was called (in-process only)."""

    name: ClassVar[str] = "StatefulSink"
    description: ClassVar[str] = "Counts calls"

    @classmethod
    def params(cls) -> list[Param]:
        return []

    def __init__(self) -> None:
        self.call_count = 0

    def __call__(self, items: Iterator[int], index: int) -> list[str]:
        self.call_count += 1
        return [str(v) for v in items]


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_pipeline() -> Pipeline[int]:
    """A 5-item pipeline: source → triple → list sink."""
    return NumberSource(5).filter(TripleFilter()).write(ListSink())


@pytest.fixture
def no_filter_pipeline() -> Pipeline[int]:
    """A 3-item pipeline with no filters."""
    return NumberSource(3).write(ListSink())


# ---------------------------------------------------------------------------
# _resolve_n_jobs tests
# ---------------------------------------------------------------------------


class TestResolveNJobs:
    def test_positive_passthrough(self):
        assert _resolve_n_jobs(4) == 4

    def test_one_is_one(self):
        assert _resolve_n_jobs(1) == 1

    def test_negative_one_uses_all_cpus(self):
        import os

        expected = os.cpu_count() or 1
        assert _resolve_n_jobs(-1) == expected

    def test_negative_two_is_cpus_minus_one(self):
        import os

        cpu = os.cpu_count() or 1
        assert _resolve_n_jobs(-2) == max(1, cpu - 1)

    def test_very_negative_floors_at_one(self):
        assert _resolve_n_jobs(-999) >= 1


# ---------------------------------------------------------------------------
# run_pipeline — sequential backend
# ---------------------------------------------------------------------------


class TestSequentialBackend:
    def test_processes_all_indices(self, simple_pipeline):
        results = run_pipeline(simple_pipeline, n_jobs=1, progress=False)
        assert len(results) == 5
        # Index 0 → 0*3=0, Index 4 → 4*3=12
        assert results[0] == ["item_0_0"]
        assert results[4] == ["item_4_12"]

    def test_explicit_sequential_backend(self, simple_pipeline):
        results = run_pipeline(simple_pipeline, backend="sequential", progress=False)
        assert len(results) == 5

    def test_subset_indices(self, simple_pipeline):
        results = run_pipeline(simple_pipeline, indices=[1, 3], progress=False)
        assert len(results) == 2
        assert results[0] == ["item_1_3"]  # index 1 → 1*3=3
        assert results[1] == ["item_3_9"]  # index 3 → 3*3=9

    def test_empty_indices(self, simple_pipeline):
        results = run_pipeline(simple_pipeline, indices=[], progress=False)
        assert results == []

    def test_no_filter_pipeline(self, no_filter_pipeline):
        results = run_pipeline(no_filter_pipeline, progress=False)
        assert len(results) == 3
        assert results[0] == ["item_0_0"]
        assert results[2] == ["item_2_2"]

    def test_stateful_sink_sequential(self):
        sink = StatefulSink()
        pipeline = NumberSource(3).write(sink)
        run_pipeline(pipeline, n_jobs=1, progress=False)
        # Sequential: sink is shared, call_count should increment
        assert sink.call_count == 3

    def test_with_progress(self, simple_pipeline):
        # Should not raise even if tqdm is not available
        results = run_pipeline(simple_pipeline, n_jobs=1, progress=True)
        assert len(results) == 5


# ---------------------------------------------------------------------------
# run_pipeline — processes backend
# ---------------------------------------------------------------------------


class TestProcessesBackend:
    def test_basic_parallel(self, simple_pipeline):
        results = run_pipeline(simple_pipeline, n_jobs=2, backend="processes", progress=False)
        assert len(results) == 5
        # Order should be preserved
        assert results[0] == ["item_0_0"]
        assert results[4] == ["item_4_12"]

    def test_subset_parallel(self, simple_pipeline):
        results = run_pipeline(simple_pipeline, n_jobs=2, backend="processes", indices=[0, 2, 4], progress=False)
        assert len(results) == 3
        assert results[0] == ["item_0_0"]
        assert results[1] == ["item_2_6"]
        assert results[2] == ["item_4_12"]

    def test_stateful_sink_parallel_isolation(self):
        """Stateful sink in parent is not mutated by child processes."""
        sink = StatefulSink()
        pipeline = NumberSource(4).write(sink)
        results = run_pipeline(pipeline, n_jobs=2, backend="processes", progress=False)
        # Results should still be correct
        assert len(results) == 4
        # But parent sink should NOT have been called (children have copies)
        assert sink.call_count == 0


# ---------------------------------------------------------------------------
# run_pipeline — loky backend (optional)
# ---------------------------------------------------------------------------


class TestLokyBackend:
    def test_loky_runs(self, simple_pipeline):
        pytest.importorskip("joblib")
        results = run_pipeline(simple_pipeline, n_jobs=2, backend="loky", progress=False)
        assert len(results) == 5
        assert results[0] == ["item_0_0"]
        assert results[4] == ["item_4_12"]

    def test_loky_missing_raises(self, simple_pipeline):
        with patch.dict("sys.modules", {"joblib": None}), pytest.raises(ImportError, match="joblib"):
            run_pipeline(simple_pipeline, n_jobs=2, backend="loky", progress=False)


# ---------------------------------------------------------------------------
# run_pipeline — dask backend (optional)
# ---------------------------------------------------------------------------


class TestDaskBackend:
    def test_dask_runs(self, simple_pipeline):
        pytest.importorskip("dask")
        results = run_pipeline(simple_pipeline, n_jobs=2, backend="dask", progress=False)
        assert len(results) == 5
        assert results[0] == ["item_0_0"]
        assert results[4] == ["item_4_12"]

    def test_dask_missing_raises(self, simple_pipeline):
        with patch.dict("sys.modules", {"dask": None, "dask.bag": None}), pytest.raises(ImportError, match="dask"):
            run_pipeline(simple_pipeline, n_jobs=2, backend="dask", progress=False)


# ---------------------------------------------------------------------------
# run_pipeline — auto backend
# ---------------------------------------------------------------------------


class TestAutoBackend:
    def test_auto_selects_something(self, simple_pipeline):
        results = run_pipeline(simple_pipeline, n_jobs=2, backend="auto", progress=False)
        assert len(results) == 5

    def test_pick_auto_backend_returns_valid(self):
        result = _pick_auto_backend()
        assert result in ("dask", "loky", "processes")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_unknown_backend_raises(self, simple_pipeline):
        with pytest.raises(ValueError, match="Unknown backend"):
            run_pipeline(simple_pipeline, backend="magic")

    def test_no_sink_raises(self):
        pipeline = NumberSource(3).filter(TripleFilter())
        with pytest.raises(RuntimeError, match="no sink"):
            run_pipeline(pipeline)


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestImports:
    def test_import_from_core(self):
        from curator.core import run_pipeline as rp

        assert callable(rp)

    def test_import_from_top_level(self):
        from curator import run_pipeline as rp

        assert callable(rp)
