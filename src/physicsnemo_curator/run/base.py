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

"""Base classes for pipeline execution backends.

This module defines the abstract interface that all execution backends
must implement, along with common utilities.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

if TYPE_CHECKING:
    from physicsnemo_curator.core.base import Pipeline


@dataclass
class RunConfig:
    """Configuration for pipeline execution.

    Parameters
    ----------
    n_jobs : int
        Number of parallel workers. ``1`` forces sequential execution.
        ``-1`` uses all available CPUs. Values ``<= 0`` follow the
        convention ``cpu_count + 1 + n_jobs``.
    progress : bool | Literal["log"]
        Whether to show a progress indicator. ``True`` shows the
        full-screen Textual TUI (requires an interactive terminal).
        ``"log"`` prints simple timestamped percentage lines suitable
        for notebooks and non-interactive scripts. ``False`` disables
        all progress output.
    indices : list[int] | None
        Specific source indices to process. ``None`` processes all indices.
    backend_options : dict[str, Any]
        Additional backend-specific options.
    """

    n_jobs: int = 1
    progress: bool | Literal["log"] = True
    indices: list[int] | None = None
    backend_options: dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_n_jobs(self) -> int:
        """Return the concrete positive worker count.

        Returns
        -------
        int
            Positive integer number of workers.
        """
        if self.n_jobs > 0:
            return self.n_jobs
        cpu = os.cpu_count() or 1
        resolved = cpu + 1 + self.n_jobs  # -1 → cpu, -2 → cpu-1, …
        return max(1, resolved)


class RunBackend(ABC):
    """Abstract base class for pipeline execution backends.

    Subclasses implement different parallelization strategies (threading,
    multiprocessing, distributed computing, workflow orchestrators, etc.).

    Class Attributes
    ----------------
    name : str
        Unique identifier for this backend (e.g., "sequential", "process_pool").
    description : str
        Human-readable description of the backend.
    requires : tuple[str, ...]
        Optional package dependencies required by this backend.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    requires: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def is_available(cls) -> bool:
        """Check if this backend's dependencies are installed.

        Returns
        -------
        bool
            True if all required packages are available.
        """
        for package in cls.requires:
            try:
                __import__(package)
            except ImportError:
                return False
        return True

    @abstractmethod
    def run(
        self,
        pipeline: Pipeline[Any],
        config: RunConfig,
    ) -> list[list[str]]:
        """Execute the pipeline over the configured indices.

        Parameters
        ----------
        pipeline : Pipeline
            A fully-configured pipeline (source + filters + sink).
        config : RunConfig
            Execution configuration.

        Returns
        -------
        list[list[str]]
            Outer list is ordered by the input indices; each inner list
            contains the file paths returned by the sink for that index.
        """
        ...


def _get_worker_id() -> str:
    """Return a unique worker identifier using PID.

    For process-based backends each forked process has a distinct PID,
    which is sufficient to produce unique per-worker shard files.

    Returns
    -------
    str
        A string like ``"12345"`` (pid).
    """
    import os

    return str(os.getpid())


def _flush_filters(pipeline: Pipeline[Any], index: int) -> None:
    """Flush stateful filters after processing an index.

    For each filter that has a ``flush`` method and an ``_output_path``
    attribute, this function resolves a worker-specific output path and
    flushes the filter's accumulated state.

    If the original path contains ``{worker_id}`` it is treated as a
    template and the placeholder is substituted with the unique worker
    identifier.  Otherwise the path is rewritten as
    ``{stem}_worker_{worker_id}{suffix}``.

    The worker ID is derived from the PID so that process-based backends
    produce unique per-worker output files.

    After flushing, any filter artifacts (reported via
    :meth:`~Filter.artifacts`) are recorded in the pipeline store when
    metrics tracking is enabled.

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline whose filters should be flushed.
    index : int
        The source index that was just processed (used for artifact
        tracking).
    """
    import pathlib

    worker_id = _get_worker_id()

    for i_f, f in enumerate(pipeline.filters):
        if not (hasattr(f, "flush") and hasattr(f, "_output_path")):
            continue

        # Store the original template path once (first call in this process).
        if not hasattr(f, "_output_path_template"):
            f._output_path_template = f._output_path  # noqa: SLF001  # ty: ignore[invalid-assignment]

        template_str = str(f._output_path_template)  # noqa: SLF001  # ty: ignore[unresolved-attribute]

        if "{worker_id}" in template_str:
            worker_path = pathlib.Path(template_str.format(worker_id=worker_id))
        else:
            p = pathlib.Path(template_str)
            worker_path = p.parent / f"{p.stem}_worker_{worker_id}{p.suffix}"

        f._output_path = worker_path  # noqa: SLF001  # ty: ignore[invalid-assignment]
        f.flush()  # ty: ignore[call-non-callable]

        # Record filter artifacts if metrics tracking is enabled
        if pipeline.track_metrics:
            artifact_paths = f.artifacts()
            if artifact_paths:
                store = pipeline._get_store()  # noqa: SLF001
                store.record_filter_artifacts(index, type(f).name, i_f, artifact_paths)


def process_single_index(pipeline: Pipeline[Any], index: int) -> list[str]:
    """Process a single pipeline index.

    This is a module-level function to support pickling for multiprocess
    backends.  After processing, any stateful filters with ``flush``
    methods are automatically flushed to shard files.

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline to execute.
    index : int
        The index to process.

    Returns
    -------
    list[str]
        File paths written by the sink.
    """
    result = pipeline[index]
    _flush_filters(pipeline, index)
    return result


def process_single_index_packed(args: tuple[Pipeline[Any], int]) -> list[str]:
    """Process a single pipeline index (packed arguments for map functions).

    Parameters
    ----------
    args : tuple[Pipeline, int]
        A ``(pipeline, index)`` pair.

    Returns
    -------
    list[str]
        File paths written by the sink.
    """
    pipeline, index = args
    result = pipeline[index]
    _flush_filters(pipeline, index)
    return result


def intersect_partitions(
    source_groups: list[list[int]] | None,
    sink_groups: list[list[int]] | None,
) -> list[list[int]] | None:
    """Intersect source and sink partition constraints.

    Both the source and sink may independently declare that certain
    indices MUST be processed by the same worker.  This function
    computes the finest partition that satisfies both constraints,
    or raises :class:`ValueError` if the constraints are incompatible.

    Parameters
    ----------
    source_groups : list[list[int]] | None
        Groups from :meth:`Source.partition_indices`, or ``None``.
    sink_groups : list[list[int]] | None
        Groups from :meth:`Sink.partition_indices`, or ``None``.

    Returns
    -------
    list[list[int]] | None
        Merged groups satisfying both constraints, or ``None`` if
        neither source nor sink requires partitioning.

    Raises
    ------
    ValueError
        If the source and sink constraints are incompatible (one
        requires indices together that the other requires apart).
    """
    if source_groups is None and sink_groups is None:
        return None
    if source_groups is None:
        return sink_groups
    if sink_groups is None:
        return source_groups

    # Build index → group_id mappings.
    source_map: dict[int, int] = {}
    for gid, group in enumerate(source_groups):
        for idx in group:
            source_map[idx] = gid

    sink_map: dict[int, int] = {}
    for gid, group in enumerate(sink_groups):
        for idx in group:
            sink_map[idx] = gid

    # Group by (source_group_id, sink_group_id) pair.
    from collections import defaultdict

    pair_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    all_indices = set(source_map.keys()) | set(sink_map.keys())
    for idx in all_indices:
        s_gid = source_map.get(idx, -1)
        k_gid = sink_map.get(idx, -1)
        pair_groups[(s_gid, k_gid)].append(idx)

    # Validate: no original group was split.
    # For each source group, all its indices must map to the same
    # intersection group.  If they don't, the constraints conflict.
    intersection_groups = list(pair_groups.values())

    # Build reverse: idx → intersection group id
    idx_to_intersection: dict[int, int] = {}
    for ig_id, ig in enumerate(intersection_groups):
        for idx in ig:
            idx_to_intersection[idx] = ig_id

    # Check source groups are not split.
    for s_gid, s_group in enumerate(source_groups):
        ig_ids = {idx_to_intersection[idx] for idx in s_group}
        if len(ig_ids) > 1:
            # Find conflicting sink groups
            conflicting_sinks = {sink_map.get(idx, -1) for idx in s_group}
            msg = (
                f"Incompatible partition constraints: source requires indices "
                f"{s_group} to be processed together (source group {s_gid}), "
                f"but they span {len(conflicting_sinks)} different sink groups. "
                f"Adjust sink chunk_size so chunk boundaries align with source "
                f"file boundaries."
            )
            raise ValueError(msg)

    # Check sink groups are not split.
    for k_gid, k_group in enumerate(sink_groups):
        ig_ids = {idx_to_intersection[idx] for idx in k_group}
        if len(ig_ids) > 1:
            # Find conflicting source groups
            conflicting_sources = {source_map.get(idx, -1) for idx in k_group}
            msg = (
                f"Incompatible partition constraints: sink requires indices "
                f"{k_group} to be processed together (sink group {k_gid}), "
                f"but they span {len(conflicting_sources)} different source groups. "
                f"Adjust sink chunk_size so chunk boundaries align with source "
                f"file boundaries."
            )
            raise ValueError(msg)

    # Sort groups by their minimum index for deterministic ordering.
    intersection_groups.sort(key=lambda g: min(g))
    # Sort indices within each group.
    for g in intersection_groups:
        g.sort()

    return intersection_groups


def batch_groups(groups: list[list[int]], n_workers: int) -> list[list[int]]:
    """Merge partition groups into at most *n_workers* batches.

    When there are more groups than workers, groups are distributed
    across workers using a greedy bin-packing strategy (assign each
    group to the lightest batch) to balance load.

    Each batch is a flat list of indices preserving the constraint
    that indices from the same original group are always together.

    Parameters
    ----------
    groups : list[list[int]]
        Partition groups (from :func:`intersect_partitions`).
    n_workers : int
        Maximum number of worker batches.

    Returns
    -------
    list[list[int]]
        At most *n_workers* batches, each a list of indices.
    """
    if len(groups) <= n_workers:
        return groups

    import heapq

    # Greedy: assign largest groups first to lightest batch.
    # Sort groups descending by size for best packing.
    sorted_groups = sorted(groups, key=len, reverse=True)

    # Min-heap of (batch_size, batch_index)
    batches: list[list[int]] = [[] for _ in range(n_workers)]
    heap: list[tuple[int, int]] = [(0, i) for i in range(n_workers)]
    heapq.heapify(heap)

    for group in sorted_groups:
        size, batch_idx = heapq.heappop(heap)
        batches[batch_idx].extend(group)  # ty: ignore[invalid-argument-type]
        heapq.heappush(heap, (size + len(group), batch_idx))

    # Remove empty batches (if n_workers > n_groups, already handled above).
    return [b for b in batches if b]


def process_index_group(pipeline: Pipeline[Any], indices: list[int]) -> dict[int, list[str]]:
    """Process a group of pipeline indices sequentially.

    Used when the sink provides :meth:`partition_indices` to batch
    related indices onto the same worker (e.g. for chunk-aligned
    parallel writes).

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline to execute.
    indices : list[int]
        The indices to process (in order).

    Returns
    -------
    dict[int, list[str]]
        Mapping of index to sink output paths.
    """
    results: dict[int, list[str]] = {}
    for idx in indices:
        results[idx] = pipeline[idx]
        _flush_filters(pipeline, idx)
    return results


def make_progress_bar(total: int, *, enabled: bool, desc: str = "run_pipeline") -> Any:
    """Return a tqdm progress bar or None.

    Parameters
    ----------
    total : int
        Number of items.
    enabled : bool
        Whether to attempt tqdm import.
    desc : str
        Description for the progress bar.

    Returns
    -------
    Any
        A tqdm progress bar, or None if disabled or unavailable.
    """
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm

        return tqdm(total=total, desc=desc, unit="item")
    except ImportError:
        import warnings

        warnings.warn(
            "progress=True was requested but tqdm is not installed. Install tqdm for progress bars: pip install tqdm",
            stacklevel=2,
        )
        return None


_MAX_WORKER_BARS = 8
"""Maximum number of per-worker progress bars to display."""


class WorkerProgressDisplay:
    """Multi-line progress display showing per-worker activity.

    Renders an overall progress bar plus optionally one bar per active worker
    (up to :data:`_MAX_WORKER_BARS`).  Falls back gracefully when
    *tqdm* is not installed or progress is disabled.

    Parameters
    ----------
    total : int
        Total number of items to process.
    n_workers : int
        Number of parallel workers.
    enabled : bool
        Whether to show progress at all.
    desc : str
        Description label for the overall bar.
    show_worker_bars : bool
        Whether to show per-worker progress bars. Defaults to False
        to avoid console conflicts with multiple processes.
    """

    def __init__(
        self,
        total: int,
        n_workers: int,
        *,
        enabled: bool = True,
        desc: str = "run_pipeline",
        show_worker_bars: bool = False,
    ) -> None:
        self._enabled = enabled
        self._show_worker_bars = show_worker_bars
        self._n_display = min(n_workers, _MAX_WORKER_BARS) if show_worker_bars else 0
        self._main_bar: Any = None
        self._worker_bars: list[Any] = []
        self._tqdm_cls: Any = None

        if not enabled:
            return

        try:
            from tqdm.auto import tqdm

            self._tqdm_cls = tqdm
        except ImportError:
            import warnings

            warnings.warn(
                "progress=True was requested but tqdm is not installed. "
                "Install tqdm for progress bars: pip install tqdm",
                stacklevel=3,
            )
            return

        # Position 0: overall bar
        self._main_bar = tqdm(
            total=total,
            desc=desc,
            unit="item",
            position=0,
            leave=True,
        )

        # Positions 1..n_display: per-worker bars (only if requested)
        if show_worker_bars:
            for w in range(self._n_display):
                bar = tqdm(
                    total=0,
                    desc=f"  Worker {w}",
                    bar_format="  {desc}",
                    position=w + 1,
                    leave=False,
                )
                self._worker_bars.append(bar)

    @property
    def active(self) -> bool:
        """Return whether the display is active."""
        return self._main_bar is not None

    def worker_start(self, worker_id: int, index: int) -> None:
        """Mark a worker as starting to process an index.

        Parameters
        ----------
        worker_id : int
            Zero-based worker identifier.
        index : int
            The source index being processed.
        """
        if worker_id < self._n_display and self._worker_bars:
            bar = self._worker_bars[worker_id]
            bar.set_description_str(f"  Worker {worker_id}: index {index}")
            bar.refresh()

    def worker_done(self, worker_id: int) -> None:
        """Mark a worker as idle (does NOT update main bar - use complete_item).

        Parameters
        ----------
        worker_id : int
            Zero-based worker identifier.
        """
        if worker_id < self._n_display and self._worker_bars:
            bar = self._worker_bars[worker_id]
            bar.set_description_str(f"  Worker {worker_id}: idle")
            bar.refresh()

    def complete_item(self) -> None:
        """Increment the overall bar without worker tracking.

        Use this for backends where individual worker identity is not
        available.
        """
        if self._main_bar is not None:
            self._main_bar.update(1)

    def close(self) -> None:
        """Close all bars and clean up terminal lines."""
        for bar in reversed(self._worker_bars):
            bar.close()
        self._worker_bars.clear()
        if self._main_bar is not None:
            self._main_bar.close()
            self._main_bar = None
