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

"""Benchmarks for the core pipeline (Source → Filter → Sink)."""

from collections.abc import Generator, Iterator

from physicsnemo_curator.core.base import Filter, Param, Sink, Source

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


class TimePipelineConstruction:
    """Benchmark pipeline construction overhead."""

    params = [1, 10, 50]
    param_names = ["num_filters"]

    def time_build_pipeline(self, num_filters):
        """Time pipeline construction with N chained filters."""
        src = _NumberSource(100)
        pipe = src
        for _ in range(num_filters):
            pipe = pipe.filter(_DoubleFilter())
        pipe.write(_NullSink())


class TimePipelineIteration:
    """Benchmark per-item pipeline throughput."""

    params = [10, 100, 1000]
    param_names = ["num_items"]

    def setup(self, num_items):
        """Build the pipeline once."""
        src = _NumberSource(num_items)
        self.pipeline = src.filter(_DoubleFilter()).write(_NullSink())

    def time_iterate_all(self, num_items):
        """Time iterating through every item in the pipeline."""
        for i in range(len(self.pipeline)):
            self.pipeline[i]


class TimeSourceIndexing:
    """Benchmark raw source __getitem__ speed."""

    params = [100, 10_000]
    param_names = ["num_items"]

    def setup(self, num_items):
        """Create source."""
        self.source = _NumberSource(num_items)

    def time_getitem(self, num_items):
        """Time indexing every item from the source."""
        for i in range(num_items):
            self.source[i]


class MemPipeline:
    """Memory benchmarks for pipeline objects."""

    params = [1, 10, 50]
    param_names = ["num_filters"]

    def mem_pipeline_object(self, num_filters):
        """Track memory of a pipeline with N filters."""
        src = _NumberSource(100)
        pipe = src
        for _ in range(num_filters):
            pipe = pipe.filter(_DoubleFilter())
        return pipe.write(_NullSink())
