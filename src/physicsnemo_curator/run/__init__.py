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

"""Pipeline execution with pluggable backends.

This module provides :func:`run_pipeline`, which processes every index
of a :class:`~physicsnemo_curator.core.base.Pipeline` — optionally in
parallel — and returns the collected sink outputs.

Available Backends
------------------
* ``"sequential"`` — simple ``for``-loop (default when *n_jobs=1*).
* ``"thread_pool"`` — :class:`concurrent.futures.ThreadPoolExecutor`
  (good for I/O-bound tasks).
* ``"process_pool"`` — :class:`concurrent.futures.ProcessPoolExecutor`
  (true parallelism for CPU-bound tasks).
* ``"loky"`` — ``joblib.Parallel`` with the ``loky`` backend
  (requires ``joblib``).
* ``"dask"`` — ``dask.bag`` for parallel/distributed execution
  (requires ``dask``).
* ``"prefect"`` — Prefect workflow orchestration with observability
  (requires ``prefect``).
* ``"auto"`` — picks the best available backend automatically.

Custom Backends
---------------
You can register custom backends using :func:`register_backend`::

    from physicsnemo_curator.run import register_backend, RunBackend, RunConfig

    class MyBackend(RunBackend):
        name = "my_backend"
        description = "My custom execution backend"

        def run(self, pipeline, config):
            # Custom execution logic
            ...

    register_backend(MyBackend)

.. warning::

   Stateful filters (e.g. those with ``flush()`` methods) automatically
   write per-index shard files when using parallel backends.  Call
   :func:`gather_pipeline` after :func:`run_pipeline` to merge all
   shards into a single output file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from physicsnemo_curator.run.base import RunBackend, RunConfig
from physicsnemo_curator.run.dask import DaskBackend
from physicsnemo_curator.run.loky import LokyBackend
from physicsnemo_curator.run.prefect import PrefectBackend
from physicsnemo_curator.run.process_pool import ProcessPoolBackend
from physicsnemo_curator.run.sequential import SequentialBackend
from physicsnemo_curator.run.thread_pool import ThreadPoolBackend

if TYPE_CHECKING:
    from collections.abc import Iterable

    from physicsnemo_curator.core.base import Pipeline
    from physicsnemo_curator.core.profiling import ProfiledPipeline

    #: Type alias for objects accepted by :func:`run_pipeline`.
    #: Both ``Pipeline`` and ``ProfiledPipeline`` are supported.
    PipelineLike = Pipeline[Any] | ProfiledPipeline[Any]

# ---------------------------------------------------------------------------
# Backend Registry
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, type[RunBackend]] = {}


def register_backend(backend_cls: type[RunBackend]) -> None:
    """Register a custom execution backend.

    Parameters
    ----------
    backend_cls : type[RunBackend]
        The backend class to register. Must have a ``name`` class attribute.

    Raises
    ------
    ValueError
        If a backend with the same name is already registered.

    Examples
    --------
    >>> from physicsnemo_curator.run import register_backend, RunBackend
    >>> class MyBackend(RunBackend):
    ...     name = "my_backend"
    ...     description = "Custom backend"
    ...     def run(self, pipeline, config):
    ...         return []
    >>> register_backend(MyBackend)
    """
    name = backend_cls.name
    if name in _BACKENDS:
        msg = f"Backend {name!r} is already registered."
        raise ValueError(msg)
    _BACKENDS[name] = backend_cls


def get_backend(name: str) -> RunBackend:
    """Get an instance of a registered backend by name.

    Parameters
    ----------
    name : str
        The backend name.

    Returns
    -------
    RunBackend
        An instance of the requested backend.

    Raises
    ------
    ValueError
        If the backend is not registered.
    """
    if name not in _BACKENDS:
        available = ", ".join(sorted(_BACKENDS.keys()))
        msg = f"Unknown backend {name!r}. Available: {available}"
        raise ValueError(msg)
    return _BACKENDS[name]()


def list_backends() -> dict[str, dict[str, Any]]:
    """List all registered backends and their availability.

    Returns
    -------
    dict[str, dict[str, Any]]
        Dictionary mapping backend names to info dicts containing:
        - description: Human-readable description
        - available: Whether dependencies are installed
        - requires: Tuple of required packages
    """
    return {
        name: {
            "description": cls.description,
            "available": cls.is_available(),
            "requires": cls.requires,
        }
        for name, cls in _BACKENDS.items()
    }


def _pick_auto_backend() -> str:
    """Select the best available parallel backend.

    Returns
    -------
    str
        Name of the best available backend.
    """
    # Priority order for parallel execution
    priority = ["dask", "loky", "process_pool"]
    for name in priority:
        if name in _BACKENDS and _BACKENDS[name].is_available():
            return name
    return "process_pool"


# Register built-in backends
register_backend(SequentialBackend)
register_backend(ThreadPoolBackend)
register_backend(ProcessPoolBackend)
register_backend(LokyBackend)
register_backend(DaskBackend)
register_backend(PrefectBackend)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_pipeline(
    pipeline: PipelineLike,
    *,
    n_jobs: int = 1,
    backend: str = "auto",
    indices: Iterable[int] | None = None,
    progress: bool = True,
    **backend_kwargs: Any,
) -> list[list[str]]:
    """Execute a pipeline over all (or selected) source indices.

    This is the primary entry-point for batch or parallel pipeline
    execution. It dispatches to the chosen backend and collects results.

    Parameters
    ----------
    pipeline : Pipeline | ProfiledPipeline
        A fully-configured pipeline (source + filters + sink).
    n_jobs : int
        Number of parallel workers. ``1`` forces sequential execution.
        ``-1`` uses all available CPUs.
    backend : str
        Execution backend. One of ``"auto"``, ``"sequential"``,
        ``"thread_pool"``, ``"process_pool"``, ``"loky"``, ``"dask"``,
        ``"prefect"``, or any custom registered backend.
    indices : Iterable[int] | None
        Specific source indices to process. ``None`` (default) processes
        all indices ``range(len(pipeline))``.
    progress : bool
        Show a progress indicator if the chosen backend supports it.
    **backend_kwargs : Any
        Extra keyword arguments forwarded to the backend.

    Returns
    -------
    list[list[str]]
        Outer list is ordered by the input indices; each inner list
        contains the file paths returned by the sink for that index.

    Raises
    ------
    ValueError
        If *backend* is not a recognised name.
    RuntimeError
        If the pipeline has no sink attached.
    ImportError
        If the selected backend's optional dependency is missing.

    Notes
    -----
    When using parallel backends, each worker operates on an independent
    copy of the pipeline. Stateful filters accumulate per-worker state
    that is **not** merged back. Use sequential execution when filter
    side-effects must be aggregated, or implement a custom reduce step.

    Examples
    --------
    Sequential execution (default):

    >>> results = run_pipeline(pipeline)

    Parallel with 4 processes:

    >>> results = run_pipeline(pipeline, n_jobs=4, backend="process_pool")

    Using Prefect with retries:

    >>> results = run_pipeline(
    ...     pipeline,
    ...     n_jobs=4,
    ...     backend="prefect",
    ...     retries=3,
    ...     retry_delay_seconds=10,
    ... )

    Process only a subset of indices:

    >>> results = run_pipeline(pipeline, indices=[0, 5, 10])
    """
    # Validate backend
    if backend != "auto" and backend not in _BACKENDS:
        available = ", ".join(sorted(_BACKENDS.keys()))
        msg = f"Unknown backend {backend!r}. Available: {available}, auto"
        raise ValueError(msg)

    # Validate pipeline
    if pipeline.sink is None:
        msg = "Pipeline has no sink. Call .write(sink) before run_pipeline()."
        raise RuntimeError(msg)

    # Build configuration
    idx_list: list[int] | None = list(indices) if indices is not None else None
    config = RunConfig(
        n_jobs=n_jobs,
        progress=progress,
        indices=idx_list,
        backend_options=backend_kwargs,
    )

    # Handle empty indices
    if idx_list is not None and not idx_list:
        return []

    # Resolve backend
    resolved_n_jobs = config.resolved_n_jobs
    if resolved_n_jobs == 1 or backend == "sequential":
        effective_backend = "sequential"
    elif backend == "auto":
        effective_backend = _pick_auto_backend()
    else:
        effective_backend = backend

    # Check backend availability
    backend_cls = _BACKENDS[effective_backend]
    if not backend_cls.is_available():
        requires = ", ".join(backend_cls.requires)
        msg = f"Backend {effective_backend!r} requires: {requires}"
        raise ImportError(msg)

    # Execute
    runner = backend_cls()
    return runner.run(pipeline, config)  # ty: ignore[invalid-argument-type]  # ProfiledPipeline duck-types Pipeline


def gather_pipeline(pipeline: PipelineLike) -> list[str]:
    """Merge per-index shard files produced by stateful filters.

    When :func:`run_pipeline` runs with a parallel backend, each worker
    flushes stateful filters to shard files named
    ``{stem}_shard_{index:06d}{suffix}``.  This function discovers those
    shards, calls the filter's :meth:`merge` method to combine them into
    a single output file, and removes the shard files.

    Call this **after** :func:`run_pipeline` completes.

    Parameters
    ----------
    pipeline : Pipeline | ProfiledPipeline
        The same pipeline that was passed to :func:`run_pipeline`.

    Returns
    -------
    list[str]
        Paths to the merged output files (one per stateful filter).

    Examples
    --------
    >>> results = run_pipeline(pipeline, n_jobs=4, backend="process_pool")
    >>> merged = gather_pipeline(pipeline)
    >>> print(merged)
    ['outputs/mean_stats.parquet']
    """
    import pathlib

    merged_paths: list[str] = []

    for f in pipeline.filters:
        if not (hasattr(f, "flush") and hasattr(f, "_output_path") and hasattr(f, "merge")):
            continue

        output_path = pathlib.Path(str(f._output_path))  # noqa: SLF001
        shard_pattern = f"{output_path.stem}_shard_*{output_path.suffix}"
        shard_files = sorted(str(p) for p in output_path.parent.glob(shard_pattern))

        if not shard_files:
            continue

        merged = f.merge(shard_files, str(output_path))  # ty: ignore[call-non-callable]
        merged_paths.append(merged)

        # Clean up shard files
        for shard in shard_files:
            pathlib.Path(shard).unlink(missing_ok=True)

    return merged_paths


__all__ = [
    "DaskBackend",
    "LokyBackend",
    "PrefectBackend",
    "ProcessPoolBackend",
    "RunBackend",
    "RunConfig",
    "SequentialBackend",
    "ThreadPoolBackend",
    "gather_pipeline",
    "get_backend",
    "list_backends",
    "register_backend",
    "run_pipeline",
]
