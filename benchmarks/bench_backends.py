# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""ASV benchmarks for run_pipeline backend comparison.

Compares Sequential, ThreadPool, ProcessPool, and Loky backends using both a
lightweight synthetic pipeline (to isolate overhead) and a real VTK I/O pipeline
(to show practical scaling).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

from physicsnemo_curator.core.base import Filter, Param, Sink, Source

from ._helpers import cleanup_temp_dir, create_temp_dir, write_synthetic_vtu


# ---------------------------------------------------------------------------
# Lightweight synthetic pipeline components (picklable, minimal overhead)
# ---------------------------------------------------------------------------
class _NumberSource(Source[int]):
    """Source that yields sequential integers.

    Parameters
    ----------
    n_items : int
        Number of pipeline indices.
    """

    name = "number_source"
    description = "Synthetic integer source."

    def __init__(self, n_items: int) -> None:
        self._n_items = n_items

    @classmethod
    def params(cls) -> list[Param]:
        """Return empty params list."""
        return []

    def __len__(self) -> int:
        return self._n_items

    def __getitem__(self, index: int) -> Generator:
        """Yield the index value."""
        yield index


class _DoubleFilter(Filter[int]):
    """Filter that doubles each item (pass-through with trivial computation).

    This is intentionally lightweight so backend overhead dominates.
    """

    name = "double_filter"
    description = "Doubles values."

    @classmethod
    def params(cls) -> list[Param]:
        """Return empty params list."""
        return []

    def __call__(self, items: Generator) -> Generator:
        """Yield doubled items."""
        for item in items:
            yield item * 2

    def artifacts(self) -> list[str]:
        """Return empty artifacts list."""
        return []


class _NullSink(Sink[int]):
    """Sink that drains the iterator and returns empty list."""

    name = "null_sink"
    description = "Discards all items."

    @classmethod
    def params(cls) -> list[Param]:
        """Return empty params list."""
        return []

    def __call__(self, items: Iterator, index: int) -> list[str]:
        """Drain the iterator."""
        for _ in items:
            pass
        return []


# ---------------------------------------------------------------------------
# Backend scaling with lightweight pipeline
# ---------------------------------------------------------------------------
class TimeBackendScaling:
    """Compare backend overhead using lightweight synthetic pipeline.

    Isolates serialization cost, process spawn time, and GIL effects.
    """

    params = [
        ["sequential", "thread_pool", "process_pool", "loky"],
        [1, 2, 4],
        [100, 1000],
    ]
    param_names = ["backend", "n_workers", "n_items"]

    def setup(self, backend, n_workers, n_items):
        """Build the lightweight pipeline."""
        from physicsnemo_curator.core.base import Pipeline

        source = _NumberSource(n_items)
        filt = _DoubleFilter()
        sink = _NullSink()

        self.pipeline = Pipeline(
            source=source,
            filters=[filt],  # ty: ignore[invalid-argument-type]
            sink=sink,
            track_metrics=False,
            track_memory=False,
        )

    def time_run_pipeline(self, backend, n_workers, n_items):
        """Run the pipeline with the specified backend and worker count."""
        from physicsnemo_curator.run import run_pipeline

        run_pipeline(
            self.pipeline,
            backend=backend,
            n_jobs=n_workers,
            progress=False,
        )


# ---------------------------------------------------------------------------
# Backend scaling with real I/O (VTK pipeline)
# ---------------------------------------------------------------------------
class TimeBackendWithIO:
    """Compare backend performance with real VTK file I/O.

    Pipeline: VTKSource -> PrecisionFilter -> MeshSink
    """

    params = [
        ["sequential", "thread_pool", "process_pool", "loky"],
        [1, 2, 4],
        [10, 50],
    ]
    param_names = ["backend", "n_workers", "n_files"]

    def setup(self, backend, n_workers, n_files):
        """Generate VTU files and build the pipeline."""
        from physicsnemo_curator.core.base import Pipeline
        from physicsnemo_curator.core.store import LocalFileStore
        from physicsnemo_curator.mesh.filters.precision import PrecisionFilter
        from physicsnemo_curator.mesh.sinks.mesh_writer import MeshSink
        from physicsnemo_curator.mesh.sources.vtk import VTKSource

        self._input_dir = create_temp_dir()
        self._output_dir = create_temp_dir()
        n_points, n_cells = 500, 250

        for i in range(n_files):
            write_synthetic_vtu(
                Path(self._input_dir) / f"mesh_{i:04d}.vtu",
                n_points,
                n_cells,
                seed=42 + i,
            )

        store = LocalFileStore(self._input_dir, extensions=frozenset({".vtu"}))
        source = VTKSource(store, backend="pyvista")
        precision = PrecisionFilter(target_dtype="float32")
        sink = MeshSink(output_dir=self._output_dir)

        self.pipeline = Pipeline(
            source=source,
            filters=[precision],  # ty: ignore[invalid-argument-type]
            sink=sink,
            track_metrics=False,
            track_memory=False,
        )

    def time_run_pipeline(self, backend, n_workers, n_files):
        """Run the pipeline with the specified backend and worker count."""
        from physicsnemo_curator.run import run_pipeline

        run_pipeline(
            self.pipeline,
            backend=backend,
            n_jobs=n_workers,
            progress=False,
        )

    def teardown(self, backend, n_workers, n_files):
        """Remove temporary directories."""
        cleanup_temp_dir(self._input_dir)
        cleanup_temp_dir(self._output_dir)


# ---------------------------------------------------------------------------
# Memory overhead per backend
# ---------------------------------------------------------------------------
class MemBackendOverhead:
    """Peak memory comparison across backends."""

    params = [
        ["sequential", "thread_pool", "process_pool", "loky"],
        [1, 2, 4],
    ]
    param_names = ["backend", "n_workers"]

    def setup(self, backend, n_workers):
        """Build a lightweight pipeline with 100 items."""
        from physicsnemo_curator.core.base import Pipeline

        source = _NumberSource(100)
        filt = _DoubleFilter()
        sink = _NullSink()

        self.pipeline = Pipeline(
            source=source,
            filters=[filt],  # ty: ignore[invalid-argument-type]
            sink=sink,
            track_metrics=False,
            track_memory=False,
        )

    def peakmem_run(self, backend, n_workers):
        """Run pipeline tracking peak RSS."""
        from physicsnemo_curator.run import run_pipeline

        run_pipeline(
            self.pipeline,
            backend=backend,
            n_jobs=n_workers,
            progress=False,
        )


# ---------------------------------------------------------------------------
# Speedup tracking
# ---------------------------------------------------------------------------
class TrackBackendSpeedup:
    """Track actual speedup factor relative to sequential baseline.

    Returns ``time_sequential / time_parallel`` so >1.0 indicates speedup.
    """

    params = [
        ["thread_pool", "process_pool", "loky"],
        [2, 4],
    ]
    param_names = ["backend", "n_workers"]

    def setup(self, backend, n_workers):
        """Build a lightweight pipeline with 1000 items."""
        from physicsnemo_curator.core.base import Pipeline

        source = _NumberSource(1000)
        filt = _DoubleFilter()
        sink = _NullSink()

        self.pipeline = Pipeline(
            source=source,
            filters=[filt],  # ty: ignore[invalid-argument-type]
            sink=sink,
            track_metrics=False,
            track_memory=False,
        )

    def track_speedup(self, backend, n_workers):
        """Measure speedup: sequential_time / parallel_time."""
        from physicsnemo_curator.run import run_pipeline

        # Sequential baseline
        t0 = time.perf_counter()
        run_pipeline(
            self.pipeline,
            backend="sequential",
            n_jobs=1,
            progress=False,
        )
        t_seq = time.perf_counter() - t0

        # Parallel run
        t0 = time.perf_counter()
        run_pipeline(
            self.pipeline,
            backend=backend,
            n_jobs=n_workers,
            progress=False,
        )
        t_par = time.perf_counter() - t0

        if t_par == 0:
            return float("inf")
        return t_seq / t_par

    track_speedup.unit = "speedup"  # ty: ignore[unresolved-attribute]
