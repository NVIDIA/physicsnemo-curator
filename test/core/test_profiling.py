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

import time
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

import pytest

from physicsnemo_curator.core.base import Filter, Param, Sink, Source
from physicsnemo_curator.core.profiling import IndexMetrics, PipelineMetrics, StageMetrics

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
