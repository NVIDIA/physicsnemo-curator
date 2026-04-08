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

from physicsnemo_curator.core.base import Filter, Param, Sink, Source
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
