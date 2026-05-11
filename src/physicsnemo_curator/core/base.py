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

"""Abstract base classes for pipeline components and the Pipeline builder."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Generator, Iterator

    from physicsnemo_curator.core.pipeline_store import PipelineMetrics, PipelineStore

logger = logging.getLogger(__name__)

# Sentinel for required parameters (no default).
_REQUIRED = object()
REQUIRED: Any = _REQUIRED
"""Sentinel value indicating a :class:`Param` has no default and must be provided."""


@dataclass(frozen=True)
class Param:
    """Descriptor for a configurable parameter on a pipeline component.

    Parameters
    ----------
    name : str
        Parameter name (should match the ``__init__`` keyword argument).
    description : str
        Human-readable help text shown in the interactive CLI.
    type : type
        Expected Python type (``str``, ``int``, ``float``, ``pathlib.Path``, …).
    default : Any
        Default value.  Use :data:`REQUIRED` (the default) to indicate the
        parameter must be supplied by the user.
    choices : list[str] | None
        If not *None*, the CLI will present a selection prompt instead of
        free-text input.
    """

    name: str
    description: str
    type: type = str  # ty: ignore[invalid-type-form]
    default: Any = REQUIRED
    choices: list[str] | None = None

    @property
    def required(self) -> bool:
        """Return ``True`` if this parameter has no default value."""
        return self.default is _REQUIRED


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


class Source[T](ABC):
    """Abstract data source that yields items of type *T*.

    A source represents a collection of data items (e.g. files on disk).
    Each item is accessed by integer index and may yield one or more *T*
    objects (generator semantics allow a single source item to expand into
    multiple outputs).

    Subclasses must set the class-level :attr:`name` and :attr:`description`
    attributes and implement :meth:`params`, :meth:`__len__`, and
    :meth:`__getitem__`.

    Examples
    --------
    >>> pipeline = MySource(path="/data").filter(MyFilter()).write(MySink())
    >>> pipeline[0]  # process first source item lazily
    """

    name: ClassVar[str]
    """Human-readable display name for the interactive CLI."""
    description: ClassVar[str]
    """Short description shown in the interactive CLI."""

    @classmethod
    @abstractmethod
    def params(cls) -> list[Param]:
        """Declare the configurable parameters for this source.

        Returns
        -------
        list[Param]
            Ordered list of parameter descriptors.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of items available in this source."""
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> Generator[T]:
        """Yield one or more *T* items for the given *index*.

        Parameters
        ----------
        index : int
            Zero-based index into the source's item collection.

        Yields
        ------
        T
            Data item(s) produced from the source at *index*.
        """
        ...

    def partition_indices(self, indices: list[int]) -> list[list[int]] | None:
        """Group indices into partitions that MUST be processed by the same worker.

        Each returned group is a list of indices that must be handled
        sequentially by a single worker.  The runner will never split a
        group across workers.

        Override this method when the source has constraints on concurrent
        access (e.g., LMDB allows only one environment open per file per
        process).

        Parameters
        ----------
        indices : list[int]
            The indices to partition.

        Returns
        -------
        list[list[int]] | None
            Partitioned groups, or ``None`` if no partitioning is required
            (the default).
        """
        return None

    # -- Convenience builder methods -----------------------------------------

    def filter(self, f: Filter[T]) -> Pipeline[T]:
        """Create a :class:`Pipeline` with this source and a single filter.

        Parameters
        ----------
        f : Filter[T]
            The filter to append.

        Returns
        -------
        Pipeline[T]
            A new pipeline containing this source and the given filter.
        """
        return Pipeline(source=self, filters=[f])

    def write(self, s: Sink[T]) -> Pipeline[T]:
        """Create a :class:`Pipeline` with this source and a sink (no filters).

        If the sink exposes a ``set_source`` method, the source is
        automatically injected so the sink can resolve naming
        placeholders (e.g. ``{relpath}``, ``{stem}``) from the source.

        Parameters
        ----------
        s : Sink[T]
            The sink to attach.

        Returns
        -------
        Pipeline[T]
            A new pipeline containing this source and the given sink.
        """
        if hasattr(s, "set_source"):
            s.set_source(self)  # ty: ignore[call-non-callable]
        return Pipeline(source=self, sink=s)


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


class Filter[T](ABC):
    """Abstract filter/transform that processes a stream of *T* items.

    Filters receive a generator of items and yield zero or more items per
    input (full generator semantics — can expand, contract, or pass through).

    Subclasses must set :attr:`name` and :attr:`description` and implement
    :meth:`params` and :meth:`__call__`.
    """

    name: ClassVar[str]
    """Human-readable display name for the interactive CLI."""
    description: ClassVar[str]
    """Short description shown in the interactive CLI."""

    @classmethod
    @abstractmethod
    def params(cls) -> list[Param]:
        """Declare the configurable parameters for this filter.

        Returns
        -------
        list[Param]
            Ordered list of parameter descriptors.
        """
        ...

    @abstractmethod
    def __call__(self, items: Generator[T]) -> Generator[T]:
        """Process a stream of items, yielding transformed results.

        Parameters
        ----------
        items : Generator[T]
            Incoming stream of data items.

        Yields
        ------
        T
            Transformed data item(s).
        """
        ...

    def artifacts(self) -> list[str]:
        """Return paths of files produced by this filter since the last call.

        Stateful filters that write side-effect files (statistics, logs,
        etc.) should override this to report the paths written during the
        most recent :meth:`flush` or :meth:`__call__` cycle.  The
        framework calls this after each index to record filter artifacts
        in the pipeline store.

        The default implementation returns an empty list, which is
        correct for stateless (pass-through) filters.

        Returns
        -------
        list[str]
            Paths of files written, or ``[]`` if none.
        """
        return []

    @classmethod
    def dashboard_panel(
        cls,
        artifact_paths: list[str],
        selected_index: int | None = None,
    ) -> Any:
        """Return a Panel component visualizing this filter's artifacts.

        Override in subclasses to provide a custom dashboard widget.
        The default returns ``None`` (no widget).  Panel is only needed
        at runtime when a subclass actually imports it inside the method
        body — the base class has no Panel dependency.

        Parameters
        ----------
        artifact_paths : list[str]
            Paths to artifact files produced by the filter.
        selected_index : int or None
            Currently selected pipeline index, if any.

        Returns
        -------
        pn.viewable.Viewable or None
            A Panel component, or ``None`` if this filter has no widget.
        """
        return None

    @classmethod
    def dashboard_layout_hints(cls) -> dict[str, int]:
        """Declare grid space preferences for dashboard GridStack placement.

        Override in subclasses to customize.  The default spans half the
        grid width and 2 rows.

        Returns
        -------
        dict[str, int]
            ``cols``: number of GridStack columns to span (1–12).
            ``rows``: number of GridStack rows to span (1+).
        """
        return {"cols": 6, "rows": 2}


# ---------------------------------------------------------------------------
# Sink
# ---------------------------------------------------------------------------


class Sink[T](ABC):
    """Abstract sink that persists items and returns output file paths.

    The sink consumes a generator of items and writes each one to storage,
    returning the file paths of the written outputs.

    Subclasses must set :attr:`name` and :attr:`description` and implement
    :meth:`params` and :meth:`__call__`.
    """

    name: ClassVar[str]
    """Human-readable display name for the interactive CLI."""
    description: ClassVar[str]
    """Short description shown in the interactive CLI."""

    @classmethod
    @abstractmethod
    def params(cls) -> list[Param]:
        """Declare the configurable parameters for this sink.

        Returns
        -------
        list[Param]
            Ordered list of parameter descriptors.
        """
        ...

    @abstractmethod
    def __call__(self, items: Iterator[T], index: int) -> list[str]:
        """Consume items and persist them to storage.

        Parameters
        ----------
        items : Iterator[T]
            Stream of data items to write.
        index : int
            Source index being processed (useful for naming output files).

        Returns
        -------
        list[str]
            Paths of the files written.
        """
        ...

    def partition_indices(self, indices: list[int]) -> list[list[int]] | None:
        """Group indices into partitions that MUST be processed by the same worker.

        Each returned group is a list of indices that must be handled
        sequentially by a single worker.  The runner will never split a
        group across workers.

        Override this method when the sink has constraints on concurrent
        writes (e.g., multiple indices writing to the same Zarr chunk must
        go through the same worker).

        Parameters
        ----------
        indices : list[int]
            The indices to partition.

        Returns
        -------
        list[list[int]] | None
            Partitioned groups, or ``None`` if no partitioning is required
            (the default).
        """
        return None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class Pipeline[T]:
    """Lazy pipeline that chains a source through filters into a sink.

    The pipeline is built incrementally using the :meth:`filter` and
    :meth:`write` builder methods.  Execution is deferred until the
    pipeline is indexed with ``pipeline[i]``, which processes only
    the *i*-th source item.

    Parameters
    ----------
    source : Source[T]
        The data source.
    filters : list[Filter[T]]
        Ordered list of filters to apply.
    sink : Sink[T] | None
        Optional sink for writing output.
    db_dir : pathlib.Path or None
        Directory for the checkpoint database. When ``None``, defaults to
        the platform cache directory (``~/.cache/psnc/``).
    db_file : pathlib.Path or None
        Exact path to the checkpoint database file. When set, overrides
        both ``db_dir`` and the auto-generated filename. Useful for
        specifying a custom database name or location.
    resume : bool
        If ``True``, reuse the checkpoint database across runs so that
        completed indices are skipped on restart.

    Examples
    --------
    >>> pipeline = (
    ...     MySource(path="/data")
    ...     .filter(FilterA())
    ...     .filter(FilterB())
    ...     .write(MySink(output="/out"))
    ... )
    >>> pipeline[0]   # lazily process source item 0
    ['/out/item_0']
    """

    source: Source[T]
    filters: list[Filter[T]] = field(default_factory=list)
    sink: Sink[T] | None = None
    track_metrics: bool = True
    track_memory: bool = True
    track_gpu: bool = False
    db_dir: pathlib.Path | None = None
    db_file: pathlib.Path | None = None
    resume: bool = False
    _store: PipelineStore | None = field(default=None, init=False, repr=False, compare=False)
    _db_path: pathlib.Path | None = field(default=None, init=False, repr=False, compare=False)
    invocation_id: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Eagerly create the pipeline store when metrics are enabled.

        The store is only created for complete pipelines (those with a
        sink attached).  Intermediate pipelines built via :meth:`filter`
        skip store creation — the final pipeline produced by
        :meth:`write` will create it.
        """
        if self.track_metrics and self.sink is not None:
            self._init_store()

    def filter(self, f: Filter[T]) -> Pipeline[T]:
        """Return a new pipeline with an additional filter appended.

        Parameters
        ----------
        f : Filter[T]
            The filter to append.

        Returns
        -------
        Pipeline[T]
            A new pipeline instance (the original is unchanged).
        """
        return Pipeline(
            source=self.source,
            filters=[*self.filters, f],
            sink=self.sink,
            track_metrics=self.track_metrics,
            track_memory=self.track_memory,
            track_gpu=self.track_gpu,
            db_dir=self.db_dir,
            resume=self.resume,
        )

    def write(self, s: Sink[T]) -> Pipeline[T]:
        """Return a new pipeline with the given sink attached.

        If the sink exposes a ``set_source`` method, the pipeline's
        source is automatically injected so the sink can resolve
        naming placeholders (e.g. ``{relpath}``, ``{stem}``).

        Parameters
        ----------
        s : Sink[T]
            The sink to attach.

        Returns
        -------
        Pipeline[T]
            A new pipeline instance (the original is unchanged).
        """
        if hasattr(s, "set_source"):
            s.set_source(self.source)  # ty: ignore[call-non-callable]
        return Pipeline(
            source=self.source,
            filters=list(self.filters),
            sink=s,
            track_metrics=self.track_metrics,
            track_memory=self.track_memory,
            track_gpu=self.track_gpu,
            db_dir=self.db_dir,
            resume=self.resume,
        )

    def __len__(self) -> int:
        """Return the number of items in the source."""
        return len(self.source)

    def __getitem__(self, index: int) -> list[str]:
        """Lazily process the *index*-th source item through the full chain.

        When :attr:`track_metrics` is ``True``, each stage is wrapped with
        :class:`~physicsnemo_curator.core.pipeline_store._TimedGenerator` for
        per-stage timing, memory tracking via ``tracemalloc``, and optional
        GPU memory tracking.  Results and errors are recorded in the
        :class:`~physicsnemo_curator.core.pipeline_store.PipelineStore`.

        Parameters
        ----------
        index : int
            Zero-based index into the source.

        Returns
        -------
        list[str]
            File paths produced by the sink.

        Raises
        ------
        RuntimeError
            If no sink has been attached to the pipeline.
        IndexError
            If *index* is out of range.
        """
        if self.sink is None:
            msg = "Pipeline has no sink. Call .write(sink) before indexing."
            raise RuntimeError(msg)

        n = len(self.source)
        if index < 0:
            index += n
        if index < 0 or index >= n:
            msg = f"Index {index} out of range for source with {n} items."
            raise IndexError(msg)

        # Fast path: no instrumentation
        if not self.track_metrics:
            stream: Generator[T] = self.source[index]
            for f in self.filters:
                stream = f(stream)
            return self.sink(stream, index)

        # Instrumented path
        return self._getitem_instrumented(index)

    def _getitem_instrumented(self, index: int) -> list[str]:
        """Execute index with full metrics instrumentation.

        Parameters
        ----------
        index : int
            Validated, non-negative index into the source.

        Returns
        -------
        list[str]
            File paths produced by the sink.
        """
        import contextlib
        import os
        import socket
        import sqlite3
        import time
        import tracemalloc

        from physicsnemo_curator.core.pipeline_store import (
            StageMetrics,
            _get_worker_id,
            _TimedGenerator,
        )

        store = self._get_store()

        # --- Worker registration ---
        worker_id = _get_worker_id()
        store._resilient_write(
            "register_worker",
            store.register_worker,
            worker_id,
            os.getpid(),
            socket.gethostname(),
            invocation_id=self.invocation_id,
        )
        store._resilient_write("worker_start_index", store.worker_start_index, worker_id, index)

        # Checkpoint hit — return cached paths
        cached: list[str] | None = None
        with contextlib.suppress(sqlite3.DatabaseError, sqlite3.OperationalError):
            cached = store.is_completed(index)
        if cached is not None:
            logger.debug("Checkpoint hit for index %d — returning cached paths", index)
            store._resilient_write("worker_finish_index", store.worker_finish_index, worker_id)
            return cached

        # --- GPU baseline ---
        gpu_baseline: int | None = None
        if self.track_gpu:
            gpu_baseline = Pipeline._gpu_setup()

        # --- Memory tracking ---
        was_tracing = tracemalloc.is_tracing()
        if self.track_memory:
            if not was_tracing:
                tracemalloc.start()
            tracemalloc.reset_peak()

        overall_start = time.perf_counter_ns()
        started_tracemalloc = self.track_memory and not was_tracing

        try:
            # 1. Wrap source generator with timing
            source_gen = self.source[index]
            timed_source: _TimedGenerator[T] = _TimedGenerator(source_gen)

            # 2. Chain through filters, wrapping each output
            filter_wrappers: list[_TimedGenerator[T]] = []
            current_stream: _TimedGenerator[T] = timed_source

            for f in self.filters:
                raw_output = f(current_stream)  # type: ignore[arg-type]
                wrapped: _TimedGenerator[T] = _TimedGenerator(raw_output)
                filter_wrappers.append(wrapped)
                current_stream = wrapped

            # 3. Run the sink (forces full chain evaluation)
            assert self.sink is not None  # guaranteed by caller
            result = self.sink(current_stream, index)

            overall_elapsed = time.perf_counter_ns() - overall_start

            # 4. Compute per-stage times using chain subtraction
            stage_metrics: list[StageMetrics] = []
            source_time = timed_source.elapsed_ns
            stage_metrics.append(StageMetrics(name="source", wall_time_ns=source_time))

            prev_elapsed = source_time
            for i_f, fw in enumerate(filter_wrappers):
                filter_own_time = max(0, fw.elapsed_ns - prev_elapsed)
                fname = type(self.filters[i_f]).name
                stage_metrics.append(StageMetrics(name=fname, wall_time_ns=filter_own_time))
                prev_elapsed = fw.elapsed_ns

            last_elapsed = filter_wrappers[-1].elapsed_ns if filter_wrappers else source_time
            sink_own_time = max(0, overall_elapsed - last_elapsed)
            stage_metrics.append(StageMetrics(name="sink", wall_time_ns=sink_own_time))

            # 5. Memory measurement
            peak_memory: int = 0
            if self.track_memory:
                _, peak_memory = tracemalloc.get_traced_memory()

            # 6. GPU measurement
            gpu_delta: int | None = None
            if self.track_gpu and gpu_baseline is not None:
                gpu_delta = Pipeline._gpu_measure(gpu_baseline)

            # 7. Record success — DB failure must not discard the sink result
            store._resilient_write(
                "record_success",
                store.record_success,
                index,
                result,
                overall_elapsed,
                peak_memory,
                gpu_delta,
                stage_metrics,
            )

            return result

        except Exception as exc:
            elapsed = time.perf_counter_ns() - overall_start
            store._resilient_write("record_error", store.record_error, index, str(exc), elapsed)
            raise

        finally:
            store._resilient_write("worker_finish_index", store.worker_finish_index, worker_id)
            if started_tracemalloc:
                tracemalloc.stop()

    def _get_store(self) -> PipelineStore:
        """Return the pipeline store, creating it if necessary.

        The store is normally created eagerly during ``__post_init__``
        for complete pipelines.  This method handles the edge case where
        a user accesses store-backed properties before attaching a sink,
        or after deserialization.

        The database path is resolved in priority order:

        1. ``db_dir`` field (explicit per-pipeline override)
        2. :func:`~physicsnemo_curator.core.cache.default_cache_dir` which
           honours the ``PSNC_CACHE_DIR`` environment variable, then
           ``$XDG_CACHE_HOME/psnc/``, then ``~/.cache/psnc/``

        Returns
        -------
        PipelineStore
            The SQLite-backed pipeline store for this pipeline.
        """
        if self._store is not None:
            return self._store

        self._init_store()
        assert self._store is not None  # noqa: S101
        return self._store

    def _init_store(self) -> None:
        """Create the SQLite-backed pipeline store.

        Idempotent — if the store already exists, this is a no-op.

        When :attr:`db_file` is set, it is used as the exact database path,
        overriding both :attr:`db_dir` and the auto-generated filename.

        When :attr:`resume` is ``False`` (default), a new database file is
        created with a unique timestamp suffix so each pipeline run starts
        fresh.  When ``True``, the stable config-hash name is reused,
        enabling checkpoint resumption from a previous run.

        If ``_db_path`` is already set (e.g. after unpickling), the store
        reconnects to that exact database rather than generating a new path.
        """
        if self._store is not None:
            return

        import pathlib

        from physicsnemo_curator.core.pipeline_store import (
            PipelineStore,
            _config_hash,
            _pipeline_config,
        )

        config = _pipeline_config(self)
        hash_ = _config_hash(config)

        if self._db_path is not None:
            # Reconnect to an existing DB (e.g. child process after pickle).
            # Use _worker=True to skip schema creation and avoid lock contention.
            db_path = self._db_path
            self._store = PipelineStore(db_path=db_path, pipeline_config=config, config_hash=hash_, _worker=True)
            return
        elif self.db_file is not None:
            # User-specified exact database file path
            db_path = pathlib.Path(self.db_file)
        elif self.resume:
            # Stable filename — reuses existing DB for checkpoint resumption
            from physicsnemo_curator.core.cache import default_cache_dir

            filename = f"{hash_[:16]}.db"
            base_dir = pathlib.Path(self.db_dir) if self.db_dir is not None else default_cache_dir()
            db_path = base_dir / filename
        else:
            # Unique filename — each pipeline gets a fresh database
            import time

            from physicsnemo_curator.core.cache import default_cache_dir

            ts = int(time.time() * 1_000_000)
            filename = f"{hash_[:16]}_{ts}.db"
            base_dir = pathlib.Path(self.db_dir) if self.db_dir is not None else default_cache_dir()
            db_path = base_dir / filename

        self._store = PipelineStore(db_path=db_path, pipeline_config=config, config_hash=hash_)
        self._db_path = db_path

    @staticmethod
    def _gpu_setup() -> int | None:
        """Reset GPU peak stats and return baseline memory.

        Returns
        -------
        int | None
            Baseline GPU memory in bytes, or ``None`` if unavailable.
        """
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                return torch.cuda.memory_allocated()
        except ImportError:
            pass
        return None

    @staticmethod
    def _gpu_measure(baseline: int) -> int:
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

    def __getstate__(self) -> dict[str, Any]:
        """Return picklable state, dropping the non-serializable store.

        Returns
        -------
        dict[str, Any]
            Instance state with ``_store`` excluded.
        """
        state = self.__dict__.copy()
        state["_store"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state from pickle, ensuring ``_store`` is ``None``.

        Parameters
        ----------
        state : dict[str, Any]
            Pickled state dictionary.
        """
        state["_store"] = None
        self.__dict__.update(state)

    # -- Query API (delegates to store) ----------------------------------------

    def _require_metrics(self) -> PipelineStore:
        """Return the store or raise if metrics are disabled.

        Returns
        -------
        PipelineStore
            The pipeline store.

        Raises
        ------
        RuntimeError
            If ``track_metrics`` is ``False``.
        """
        if not self.track_metrics:
            msg = "Pipeline metrics are disabled (track_metrics=False)"
            raise RuntimeError(msg)
        return self._get_store()

    @property
    def completed_indices(self) -> set[int]:
        """Return the set of successfully completed indices.

        Returns
        -------
        set[int]
            Indices with recorded successful completions.

        Raises
        ------
        RuntimeError
            If ``track_metrics`` is ``False``.
        """
        return self._require_metrics().completed_indices()

    @property
    def db_path(self) -> pathlib.Path | None:
        """Return the resolved database path, or ``None`` if metrics are disabled.

        Returns
        -------
        pathlib.Path or None
            Absolute path to the SQLite database file, or ``None`` when
            ``track_metrics`` is ``False``.
        """
        if not self.track_metrics:
            return None
        return self._get_store()._db_path

    @property
    def failed_indices(self) -> dict[int, str]:
        """Return indices that failed with their error messages.

        Returns
        -------
        dict[int, str]
            Mapping from index to error message string.

        Raises
        ------
        RuntimeError
            If ``track_metrics`` is ``False``.
        """
        return self._require_metrics().failed_indices()

    @property
    def metrics(self) -> PipelineMetrics:
        """Return aggregated metrics from the store.

        Returns
        -------
        PipelineMetrics
            Aggregated metrics across all completed indices.

        Raises
        ------
        RuntimeError
            If ``track_metrics`` is ``False``.
        """
        return self._require_metrics().metrics()

    @property
    def active_workers(self) -> list[dict[str, Any]]:
        """Return all workers registered for this pipeline run.

        Returns
        -------
        list[dict[str, Any]]
            List of worker dictionaries with keys: ``worker_id``, ``pid``,
            ``hostname``, ``started_at``, ``last_heartbeat``, ``current_index``.

        Raises
        ------
        RuntimeError
            If ``track_metrics`` is ``False``.
        """
        return self._require_metrics().active_workers()

    def remaining_indices(self) -> list[int]:
        """Return indices not yet completed or failed.

        Returns
        -------
        list[int]
            Sorted list of indices still needing processing.

        Raises
        ------
        RuntimeError
            If ``track_metrics`` is ``False``.
        """
        store = self._require_metrics()
        return store.remaining_indices(len(self.source))

    def summary(self) -> dict[str, Any]:
        """Return a summary of the store state.

        Returns
        -------
        dict[str, Any]
            Dictionary with ``total``, ``completed``, ``failed``,
            ``remaining``, ``config_hash``, ``db_path``, ``total_elapsed_s``.

        Raises
        ------
        RuntimeError
            If ``track_metrics`` is ``False``.
        """
        store = self._require_metrics()
        return store.summary(len(self.source))

    def reset(self) -> None:
        """Clear all records for this pipeline run.

        Raises
        ------
        RuntimeError
            If ``track_metrics`` is ``False``.
        """
        self._require_metrics().reset()

    def reset_index(self, index: int) -> None:
        """Remove records for a single index.

        Parameters
        ----------
        index : int
            Source index to remove.

        Raises
        ------
        RuntimeError
            If ``track_metrics`` is ``False``.
        """
        self._require_metrics().reset_index(index)

    def index_for_path(self, path: str) -> int | None:
        """Find which source index produced a given output file.

        Parameters
        ----------
        path : str
            Output file path to look up.

        Returns
        -------
        int | None
            Source index that produced the file, or ``None`` if not found.

        Raises
        ------
        RuntimeError
            If ``track_metrics`` is ``False``.
        """
        return self._require_metrics().index_for_path(path)

    def output_paths_for_index(self, index: int) -> list[str]:
        """Return the output file paths produced by a given source index.

        Parameters
        ----------
        index : int
            Source index to query.

        Returns
        -------
        list[str]
            Output file paths ordered by sequence, or empty list if none.

        Raises
        ------
        RuntimeError
            If ``track_metrics`` is ``False``.
        """
        return self._require_metrics().output_paths_for_index(index)

    def filter_artifacts_for_index(self, index: int) -> dict[str, list[str]]:
        """Return filter artifact paths for a given source index.

        Parameters
        ----------
        index : int
            Source index to query.

        Returns
        -------
        dict[str, list[str]]
            Mapping of filter name to list of artifact paths.

        Raises
        ------
        RuntimeError
            If ``track_metrics`` is ``False``.
        """
        return self._require_metrics().filter_artifacts_for_index(index)

    def all_filter_artifacts(self) -> dict[str, list[str]]:
        """Return all filter artifact paths grouped by filter name.

        Returns
        -------
        dict[str, list[str]]
            Mapping of filter name to list of all artifact paths.

        Raises
        ------
        RuntimeError
            If ``track_metrics`` is ``False``.
        """
        return self._require_metrics().all_filter_artifacts()

    def save(self, path: str | pathlib.Path) -> None:
        """Save this pipeline's configuration to a YAML or JSON file.

        The file format is determined by the extension:
        ``.yaml`` / ``.yml`` → YAML, ``.json`` → JSON.

        Parameters
        ----------
        path : str | pathlib.Path
            Destination file path.

        Raises
        ------
        ValueError
            If the file extension is not supported.

        See Also
        --------
        Pipeline.load : Restore a pipeline from a saved file.
        """
        from physicsnemo_curator.core.serialization import save_pipeline

        save_pipeline(self, path)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> Pipeline[Any]:
        """Load a pipeline from a YAML or JSON configuration file.

        Parameters
        ----------
        path : str | pathlib.Path
            Path to the pipeline configuration file.

        Returns
        -------
        Pipeline
            Fully constructed pipeline ready for execution.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file extension is not supported.

        See Also
        --------
        Pipeline.save : Save a pipeline configuration.
        """
        from physicsnemo_curator.core.serialization import load_pipeline

        return load_pipeline(path)
