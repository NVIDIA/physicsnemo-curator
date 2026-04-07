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
from typing import TYPE_CHECKING, Any, ClassVar

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
    progress : bool
        Whether to show a progress indicator (if supported by backend).
    indices : list[int] | None
        Specific source indices to process. ``None`` processes all indices.
    backend_options : dict[str, Any]
        Additional backend-specific options.
    """

    n_jobs: int = 1
    progress: bool = True
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
        Unique identifier for this backend (e.g., "sequential", "thread_pool").
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


def _flush_filters(pipeline: Pipeline[Any], index: int) -> None:
    """Flush stateful filters after processing an index.

    For each filter that has a ``flush`` method and an ``_output_path``
    attribute, this function temporarily swaps the output path to a
    shard-specific path (``{stem}_shard_{index:06d}{suffix}``) before
    flushing, then restores the original path.

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline whose filters should be flushed.
    index : int
        The source index that was just processed (used to generate
        unique shard filenames).
    """
    import pathlib

    for f in pipeline.filters:
        if not (hasattr(f, "flush") and hasattr(f, "_output_path")):
            continue

        original = f._output_path  # noqa: SLF001
        p = pathlib.Path(str(original))
        shard_path = p.parent / f"{p.stem}_shard_{index:06d}{p.suffix}"
        f._output_path = shard_path  # noqa: SLF001  # ty: ignore[invalid-assignment]
        try:
            f.flush()  # ty: ignore[call-non-callable]
        finally:
            f._output_path = original  # noqa: SLF001  # ty: ignore[invalid-assignment]


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
        return None


_MAX_WORKER_BARS = 8
"""Maximum number of per-worker progress bars to display."""


class WorkerProgressDisplay:
    """Multi-line progress display showing per-worker activity.

    Renders an overall progress bar plus one bar per active worker
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
    """

    def __init__(
        self,
        total: int,
        n_workers: int,
        *,
        enabled: bool = True,
        desc: str = "run_pipeline",
    ) -> None:
        self._enabled = enabled
        self._n_display = min(n_workers, _MAX_WORKER_BARS)
        self._main_bar: Any = None
        self._worker_bars: list[Any] = []
        self._tqdm_cls: Any = None

        if not enabled:
            return

        try:
            from tqdm.auto import tqdm

            self._tqdm_cls = tqdm
        except ImportError:
            return

        # Position 0: overall bar
        self._main_bar = tqdm(
            total=total,
            desc=desc,
            unit="item",
            position=0,
            leave=True,
        )

        # Positions 1..n_display: per-worker bars
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
        """Mark a worker as idle and update the overall bar.

        Parameters
        ----------
        worker_id : int
            Zero-based worker identifier.
        """
        if self._main_bar is not None:
            self._main_bar.update(1)
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
