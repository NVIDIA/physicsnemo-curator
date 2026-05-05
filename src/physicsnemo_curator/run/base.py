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


def _get_worker_id() -> str:
    """Return a unique worker identifier using PID and thread ID.

    For process-based backends each process has a distinct PID.  For
    thread-based backends all threads share the same PID, so we append
    the thread identifier to disambiguate.

    Returns
    -------
    str
        A string like ``"12345_1234567890"`` (pid_threadid).
    """
    import os
    import threading

    pid = os.getpid()
    tid = threading.current_thread().ident or 0
    return f"{pid}_{tid}"


def _flush_filters(pipeline: Pipeline[Any], index: int) -> None:
    """Flush stateful filters after processing an index.

    For each filter that has a ``flush`` method and an ``_output_path``
    attribute, this function resolves a worker-specific output path and
    flushes the filter's accumulated state.

    If the original path contains ``{worker_id}`` it is treated as a
    template and the placeholder is substituted with the unique worker
    identifier.  Otherwise the path is rewritten as
    ``{stem}_worker_{worker_id}{suffix}``.

    The worker ID is derived from a combination of PID and thread ID so
    that both process-based and thread-based backends produce unique
    per-worker output files.

    The worker-specific path is stored in a thread-local attribute on the
    filter (``_worker_output_path``) so that concurrent threads do not
    race on a shared ``_output_path`` attribute.

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
    import threading

    worker_id = _get_worker_id()

    for i_f, f in enumerate(pipeline.filters):
        if not (hasattr(f, "flush") and hasattr(f, "_output_path")):
            continue

        # Store the original template path once (first thread to arrive).
        if not hasattr(f, "_output_path_template"):
            f._output_path_template = f._output_path  # noqa: SLF001  # ty: ignore[invalid-assignment]

        template_str = str(f._output_path_template)  # noqa: SLF001  # ty: ignore[unresolved-attribute]

        if "{worker_id}" in template_str:
            worker_path = pathlib.Path(template_str.format(worker_id=worker_id))
        else:
            p = pathlib.Path(template_str)
            worker_path = p.parent / f"{p.stem}_worker_{worker_id}{p.suffix}"

        # Store the resolved path in a thread-local so that flush() can
        # pick it up without racing with other threads.
        if not hasattr(f, "_local"):
            f._local = threading.local()  # noqa: SLF001  # ty: ignore[invalid-assignment]
        f._local.output_path = worker_path  # noqa: SLF001  # ty: ignore[unresolved-attribute]

        # Also set _output_path for backward compat with filters that
        # read self._output_path in flush() (single-threaded/process case).
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
