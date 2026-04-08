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

import os
from collections.abc import Generator, Iterator

from physicsnemo_curator.core.base import Filter, Param, Sink, Source
from physicsnemo_curator.core.profiling import ProfiledPipeline

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

    def __call__(self, items: Iterator[int], index: int) -> list[str]:  # noqa: D102
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

    unit = "percent"


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
        fd, out = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        metrics.to_json(out)

    def teardown(self, n_indices):
        """Clean up temp files."""
        self.profiled.cleanup()
