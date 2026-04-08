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
:class:`~physicsnemo.curator.core.base.Pipeline` that collects wall-clock,
memory, and GPU metrics at whole-pipeline, per-index, and per-stage
granularity.

Usage
-----
>>> from physicsnemo.curator import Pipeline, ProfiledPipeline, run_pipeline
>>> profiled = ProfiledPipeline(pipeline, track_gpu=True)
>>> results = run_pipeline(profiled, n_jobs=4, backend="process_pool")
>>> metrics = profiled.metrics
>>> metrics.to_console()
"""

from __future__ import annotations

import csv
import json
import pathlib
import shutil
import tempfile
import time
import tracemalloc
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    from physicsnemo.curator.core.base import Filter, Pipeline, Sink, Source


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


class _TimedGenerator[T]:
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


class ProfiledPipeline[T]:
    """Transparent profiling wrapper around :class:`~physicsnemo.curator.core.base.Pipeline`.

    Duck-type compatible with ``Pipeline`` — exposes ``source``, ``filters``,
    ``sink``, ``__len__``, and ``__getitem__``. Can be passed directly to
    :func:`~physicsnemo.curator.run.run_pipeline` without any backend changes.

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
    >>> from physicsnemo.curator import Pipeline, ProfiledPipeline, run_pipeline
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
            filter_wrappers: list[_TimedGenerator[T]] = []
            current_stream: _TimedGenerator[T] = timed_source

            for f in self._pipeline.filters:
                raw_output = f(current_stream)  # type: ignore
                wrapped = _TimedGenerator(raw_output)
                filter_wrappers.append(wrapped)
                current_stream = wrapped

            # 3. Run the sink (forces full chain evaluation)
            result = self._pipeline.sink(current_stream, index)  # type: ignore

            overall_elapsed = time.perf_counter_ns() - overall_start

            # 4. Compute per-stage times using chain subtraction
            source_time = timed_source.elapsed_ns
            stage_metrics.append(StageMetrics(name="source", wall_time_ns=source_time))

            # Filter times: filter N own time = wrapper N elapsed - wrapper N-1 elapsed
            prev_elapsed = source_time
            for i_f, fw in enumerate(filter_wrappers):
                filter_own_time = fw.elapsed_ns - prev_elapsed
                filter_own_time = max(0, filter_own_time)
                fname = type(self._pipeline.filters[i_f]).name
                stage_metrics.append(StageMetrics(name=fname, wall_time_ns=filter_own_time))
                prev_elapsed = fw.elapsed_ns

            # Sink time
            last_elapsed = filter_wrappers[-1].elapsed_ns if filter_wrappers else source_time
            sink_own_time = max(0, overall_elapsed - last_elapsed)
            stage_metrics.append(StageMetrics(name="sink", wall_time_ns=sink_own_time))

        finally:
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
        fname = f"{idx_metrics.index:010d}_{uuid.uuid4().hex[:8]}.json"
        filepath = self._metrics_dir / fname
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
