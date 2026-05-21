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
* ``"sequential"`` — simple ``for``-loop (default).
* ``"process_pool"`` — :class:`concurrent.futures.ProcessPoolExecutor`
  (true parallelism for CPU-bound tasks).
* ``"loky"`` — ``joblib.Parallel`` with the ``loky`` backend
  (requires ``joblib``).
* ``"dask"`` — ``dask.bag`` for parallel/distributed execution
  (requires ``dask``).

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

from physicsnemo_curator.core.logging import _ensure_logging_configured
from physicsnemo_curator.run.base import RunBackend, RunConfig
from physicsnemo_curator.run.dask import DaskBackend
from physicsnemo_curator.run.loky import LokyBackend
from physicsnemo_curator.run.process_pool import ProcessPoolBackend
from physicsnemo_curator.run.sequential import SequentialBackend

if TYPE_CHECKING:
    from collections.abc import Iterable

    from physicsnemo_curator.core.base import Pipeline

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


# Register built-in backends
register_backend(SequentialBackend)
register_backend(ProcessPoolBackend)
register_backend(LokyBackend)
register_backend(DaskBackend)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_pipeline(
    pipeline: Pipeline[Any],
    *,
    n_jobs: int = 1,
    backend: str = "sequential",
    indices: Iterable[int] | None = None,
    use_tui: bool = True,
    **backend_kwargs: Any,
) -> list[list[str]]:
    """Execute a pipeline over all (or selected) source indices.

    This is the primary entry-point for batch or parallel pipeline
    execution. It dispatches to the chosen backend and collects results.

    Parameters
    ----------
    pipeline : Pipeline
        A fully-configured pipeline (source + filters + sink).
    n_jobs : int
        Number of parallel workers. ``1`` forces sequential execution.
        ``-1`` uses all available CPUs.
    backend : str
        Execution backend. One of ``"sequential"``,
        ``"process_pool"``, ``"loky"``, ``"dask"``,
        or any custom registered backend.
    indices : Iterable[int] | None
        Specific source indices to process. ``None`` (default) processes
        all indices ``range(len(pipeline))``.
    use_tui : bool
        Whether to show the full-screen Textual TUI for progress
        (requires an interactive terminal). When ``False``, prints
        simple timestamped log lines to the console instead.
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

    Process only a subset of indices:

    >>> results = run_pipeline(pipeline, indices=[0, 5, 10])
    """
    # Ensure console logging is configured so users see output even if they
    # never called configure_logging() explicitly.
    _ensure_logging_configured()

    # Validate backend
    if backend not in _BACKENDS:
        available = ", ".join(sorted(_BACKENDS.keys()))
        msg = f"Unknown backend {backend!r}. Available: {available}"
        raise ValueError(msg)

    # Validate pipeline
    if pipeline.sink is None:
        msg = "Pipeline has no sink. Call .write(sink) before run_pipeline()."
        raise RuntimeError(msg)

    # Build configuration
    idx_list: list[int] | None = list(indices) if indices is not None else None
    config = RunConfig(
        n_jobs=n_jobs,
        use_tui=use_tui,
        indices=idx_list,
        backend_options=backend_kwargs,
    )

    # Handle empty indices
    if idx_list is not None and not idx_list:
        return []

    # Resolve backend
    resolved_n_jobs = config.resolved_n_jobs
    effective_backend = "sequential" if resolved_n_jobs == 1 or backend == "sequential" else backend

    # Check backend availability
    backend_cls = _BACKENDS[effective_backend]
    if not backend_cls.is_available():
        requires = ", ".join(backend_cls.requires)
        msg = f"Backend {effective_backend!r} requires: {requires}"
        raise ImportError(msg)

    # Execute
    runner = backend_cls()
    results = runner.run(pipeline, config)

    # Checkpoint the WAL to ensure all data is flushed to the main database
    # file. This guarantees the dashboard and other out-of-process readers
    # can see the complete results immediately after run_pipeline returns.
    if pipeline.track_metrics:
        import contextlib

        with contextlib.suppress(Exception):
            store = pipeline._get_store()  # noqa: SLF001
            store.checkpoint()

    return results


def gather_pipeline(pipeline: Pipeline[Any]) -> list[str]:
    """Merge per-worker shard files produced by stateful filters.

    When :func:`run_pipeline` runs with a parallel backend, each worker
    flushes stateful filters to shard files named
    ``{stem}_worker_{worker_id}{suffix}``.  This function discovers those
    shards, calls the filter's :meth:`merge` method to combine them into
    a single output file, and removes the shard files.

    Call this **after** :func:`run_pipeline` completes.

    Parameters
    ----------
    pipeline : Pipeline
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

    # Access the pipeline store (if available) to update artifact records
    store = None
    if pipeline.track_metrics:
        import contextlib

        with contextlib.suppress(Exception):
            store = pipeline._get_store()  # noqa: SLF001

    for i_f, f in enumerate(pipeline.filters):
        if not (hasattr(f, "flush") and hasattr(f, "_output_path") and hasattr(f, "merge")):
            continue

        # Use the original template path (before worker rewriting) as the
        # canonical output and glob base.  _flush_filters overwrites
        # _output_path with the worker-specific path, so reading it here
        # would give a single worker's path rather than the original.
        template_attr = getattr(f, "_output_path_template", None)
        output_path = pathlib.Path(str(template_attr if template_attr is not None else f._output_path))  # noqa: SLF001
        worker_pattern = f"{output_path.stem}_worker_*{output_path.suffix}"
        shard_files = sorted(str(p) for p in output_path.parent.glob(worker_pattern))

        if not shard_files:
            continue

        merged = f.merge(shard_files, str(output_path))  # ty: ignore[call-non-callable]
        merged_paths.append(merged)

        # Update artifact records in the DB: replace shard paths with merged
        if store is not None:
            filter_name = type(f).name
            store.replace_filter_artifacts(filter_name, i_f, shard_files, merged)

        # Clean up shard files/directories unless keep_shards is True
        keep_shards = getattr(f, "keep_shards", False)
        if not keep_shards:
            for shard in shard_files:
                shard_path = pathlib.Path(shard)
                if shard_path.is_dir():
                    import shutil

                    shutil.rmtree(shard_path, ignore_errors=True)
                else:
                    shard_path.unlink(missing_ok=True)

    return merged_paths


__all__ = [
    "DaskBackend",
    "LokyBackend",
    "ProcessPoolBackend",
    "RunBackend",
    "RunConfig",
    "SequentialBackend",
    "gather_pipeline",
    "get_backend",
    "list_backends",
    "register_backend",
    "run_pipeline",
]
