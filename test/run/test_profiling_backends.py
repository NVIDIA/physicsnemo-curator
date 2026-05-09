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

"""Integration tests for Pipeline metrics with execution backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

import pytest

from physicsnemo_curator.core.base import Filter, Param, Pipeline, Sink, Source
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


def _make_profiled_pipeline(tmp_path, count: int = 5) -> Pipeline[int]:
    """Create a profiled pipeline with metrics enabled."""
    return Pipeline(
        source=_ProfNumberSource(count),
        filters=[_ProfDoubleFilter()],
        sink=_ProfListSink(),
        track_metrics=True,
        db_dir=tmp_path / ".pnc",
    )


# ---------------------------------------------------------------------------
# Sequential backend
# ---------------------------------------------------------------------------


class TestSequentialProfiling:
    """Pipeline with metrics and sequential backend."""

    def test_results_match(self, tmp_path):
        """Profiled results should match non-profiled results."""
        raw_pipeline = Pipeline(
            source=_ProfNumberSource(5),
            filters=[_ProfDoubleFilter()],
            sink=_ProfListSink(),
            track_metrics=False,
        )
        profiled = _make_profiled_pipeline(tmp_path)
        raw_results = run_pipeline(raw_pipeline, backend="sequential", use_tui=False)
        profiled_results = run_pipeline(profiled, backend="sequential", use_tui=False)
        assert profiled_results == raw_results

    def test_metrics_collected(self, tmp_path):
        """Metrics should be collected for all indices."""
        profiled = _make_profiled_pipeline(tmp_path)
        run_pipeline(profiled, backend="sequential", use_tui=False)
        metrics = profiled.metrics
        assert len(metrics.indices) == 5

    def test_per_stage_present(self, tmp_path):
        """Each index should have source, filter, sink stages."""
        profiled = _make_profiled_pipeline(tmp_path)
        run_pipeline(profiled, backend="sequential", use_tui=False)
        metrics = profiled.metrics
        for idx_m in metrics.indices:
            names = [s.name for s in idx_m.stages]
            assert "source" in names
            assert "ProfDouble" in names
            assert "sink" in names


# ---------------------------------------------------------------------------
# Thread pool backend
# ---------------------------------------------------------------------------


class TestThreadPoolProfiling:
    """Pipeline with metrics and process_pool backend."""

    def test_results_match(self, tmp_path):
        """Process pool profiled results match raw results."""
        raw_pipeline = Pipeline(
            source=_ProfNumberSource(5),
            filters=[_ProfDoubleFilter()],
            sink=_ProfListSink(),
            track_metrics=False,
        )
        profiled = _make_profiled_pipeline(tmp_path)

        raw_results = run_pipeline(raw_pipeline, n_jobs=2, backend="process_pool", use_tui=False)
        profiled_results = run_pipeline(profiled, n_jobs=2, backend="process_pool", use_tui=False)
        assert profiled_results == raw_results

    def test_metrics_collected(self, tmp_path):
        """Process pool collects metrics for all indices."""
        profiled = _make_profiled_pipeline(tmp_path, count=8)
        run_pipeline(profiled, n_jobs=3, backend="process_pool", use_tui=False)
        metrics = profiled.metrics
        assert len(metrics.indices) == 8


# ---------------------------------------------------------------------------
# Process pool backend
# ---------------------------------------------------------------------------


class TestProcessPoolProfiling:
    """Pipeline with metrics and process_pool backend."""

    def test_results_match(self, tmp_path):
        """Process pool profiled results match raw results."""
        raw_pipeline = Pipeline(
            source=_ProfNumberSource(5),
            filters=[_ProfDoubleFilter()],
            sink=_ProfListSink(),
            track_metrics=False,
        )
        profiled = _make_profiled_pipeline(tmp_path)

        raw_results = run_pipeline(raw_pipeline, n_jobs=2, backend="process_pool", use_tui=False)
        profiled_results = run_pipeline(profiled, n_jobs=2, backend="process_pool", use_tui=False)
        assert profiled_results == raw_results

    def test_metrics_collected_across_processes(self, tmp_path):
        """Metrics survive process boundaries via shared SQLite DB."""
        profiled = _make_profiled_pipeline(tmp_path, count=6)
        run_pipeline(profiled, n_jobs=2, backend="process_pool", use_tui=False)
        metrics = profiled.metrics
        # All 6 indices should have metrics via SQLite WAL
        assert len(metrics.indices) == 6
        collected_indices = {m.index for m in metrics.indices}
        assert collected_indices == {0, 1, 2, 3, 4, 5}

    def test_subset_indices(self, tmp_path):
        """Process pool with subset indices collects correct metrics."""
        profiled = _make_profiled_pipeline(tmp_path, count=10)
        run_pipeline(profiled, n_jobs=2, backend="process_pool", indices=[1, 3, 7], use_tui=False)
        metrics = profiled.metrics
        assert len(metrics.indices) == 3
        collected_indices = {m.index for m in metrics.indices}
        assert collected_indices == {1, 3, 7}
