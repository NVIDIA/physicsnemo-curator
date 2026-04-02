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

"""Parallel execution of pipeline items across multiple backends.

This module provides :func:`run_pipeline`, which processes every index
of a :class:`~curator.core.base.Pipeline` — optionally in parallel —
and returns the collected sink outputs.

Backends are chosen via the *backend* parameter:

* ``"sequential"`` — simple ``for``-loop (default when *n_jobs=1*).
* ``"processes"`` — :class:`concurrent.futures.ProcessPoolExecutor`
  (zero extra dependencies).
* ``"loky"`` — ``joblib.Parallel`` with the ``loky`` backend
  (requires ``joblib``).
* ``"dask"`` — ``dask.bag`` for distributed execution
  (requires ``dask``).
* ``"auto"`` — picks the best available backend automatically.

.. warning::

   Stateful filters (e.g. ``MeanFilter._rows``) accumulate per-process
   state when using multiprocess backends.  Their side-effects are
   **not** merged back into the parent process.  Call ``filter.flush()``
   only when running sequentially, or design a post-hoc merge strategy.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

    from physicsnemo_curator.core.base import Pipeline

# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_BACKENDS = ("sequential", "processes", "loky", "dask", "auto")


def _resolve_n_jobs(n_jobs: int) -> int:
    """Translate *n_jobs* into a concrete positive worker count.

    Parameters
    ----------
    n_jobs : int
        Requested worker count.  ``-1`` means "all CPUs".
        Values ``<= 0`` follow the convention ``cpu_count + 1 + n_jobs``.

    Returns
    -------
    int
        Positive integer number of workers.
    """
    if n_jobs > 0:
        return n_jobs
    cpu = os.cpu_count() or 1
    resolved = cpu + 1 + n_jobs  # -1 → cpu, -2 → cpu-1, …
    return max(1, resolved)


def _make_progress(total: int, *, enabled: bool) -> Any:
    """Return a tqdm progress bar or a no-op context manager.

    Parameters
    ----------
    total : int
        Number of items.
    enabled : bool
        Whether to attempt tqdm import.

    Returns
    -------
    Any
        A tqdm progress bar, or a simple list passthrough.
    """
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm

        return tqdm(total=total, desc="run_pipeline", unit="item")
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Worker function — must be top-level for pickling
# ---------------------------------------------------------------------------


def _process_one(args: tuple[Pipeline[Any], int]) -> list[str]:
    """Process a single pipeline index.

    Parameters
    ----------
    args : tuple[Pipeline, int]
        A ``(pipeline, index)`` pair.  Packed as a single argument so
        that :func:`concurrent.futures.ProcessPoolExecutor.map` can use
        a simple iterable.

    Returns
    -------
    list[str]
        File paths written by the sink.
    """
    pipeline, index = args
    return pipeline[index]


# ---------------------------------------------------------------------------
# Sequential backend
# ---------------------------------------------------------------------------


def _run_sequential(
    pipeline: Pipeline[Any],
    indices: list[int],
    *,
    progress: bool,
) -> list[list[str]]:
    """Run pipeline indices sequentially with optional progress bar.

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline to execute.
    indices : list[int]
        Indices to process.
    progress : bool
        Show tqdm progress bar if available.

    Returns
    -------
    list[list[str]]
        Sink outputs, one list per index.
    """
    pbar = _make_progress(len(indices), enabled=progress)
    results: list[list[str]] = []
    try:
        for idx in indices:
            results.append(pipeline[idx])
            if pbar is not None:
                pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()
    return results


# ---------------------------------------------------------------------------
# ProcessPoolExecutor backend
# ---------------------------------------------------------------------------


def _run_processes(
    pipeline: Pipeline[Any],
    indices: list[int],
    *,
    n_jobs: int,
    progress: bool,
    **kwargs: Any,
) -> list[list[str]]:
    """Run pipeline indices via :class:`ProcessPoolExecutor`.

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline to execute.
    indices : list[int]
        Indices to process.
    n_jobs : int
        Number of worker processes.
    progress : bool
        Show tqdm progress bar if available.
    **kwargs : Any
        Forwarded to :class:`ProcessPoolExecutor`.

    Returns
    -------
    list[list[str]]
        Sink outputs, one list per index.
    """
    from concurrent.futures import ProcessPoolExecutor

    pbar = _make_progress(len(indices), enabled=progress)
    results: list[list[str]] = []
    try:
        with ProcessPoolExecutor(max_workers=n_jobs, **kwargs) as executor:
            for result in executor.map(_process_one, ((pipeline, i) for i in indices)):
                results.append(result)
                if pbar is not None:
                    pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()
    return results


# ---------------------------------------------------------------------------
# Joblib (loky) backend
# ---------------------------------------------------------------------------


def _run_loky(
    pipeline: Pipeline[Any],
    indices: list[int],
    *,
    n_jobs: int,
    progress: bool,
    **kwargs: Any,
) -> list[list[str]]:
    """Run pipeline indices via ``joblib.Parallel`` with the loky backend.

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline to execute.
    indices : list[int]
        Indices to process.
    n_jobs : int
        Number of worker processes.
    progress : bool
        Show tqdm progress bar if available.
    **kwargs : Any
        Forwarded to :class:`joblib.Parallel`.

    Returns
    -------
    list[list[str]]
        Sink outputs, one list per index.

    Raises
    ------
    ImportError
        If ``joblib`` is not installed.
    """
    try:
        from joblib import Parallel, delayed
    except ImportError:
        msg = "The 'loky' backend requires joblib. Install it with: pip install 'physicsnemo-curator[parallel]'"
        raise ImportError(msg) from None

    verbose = 10 if progress else 0
    results: list[list[str]] = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        verbose=verbose,
        **kwargs,
    )(delayed(pipeline.__getitem__)(i) for i in indices)
    return results


# ---------------------------------------------------------------------------
# Dask backend
# ---------------------------------------------------------------------------


def _run_dask(
    pipeline: Pipeline[Any],
    indices: list[int],
    *,
    n_jobs: int,
    progress: bool,
    **kwargs: Any,
) -> list[list[str]]:
    """Run pipeline indices via ``dask.bag``.

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline to execute.
    indices : list[int]
        Indices to process.
    n_jobs : int
        Number of workers / partitions.
    progress : bool
        Show dask progress bar if available.
    **kwargs : Any
        Forwarded to ``dask.bag.map``.

    Returns
    -------
    list[list[str]]
        Sink outputs, one list per index.

    Raises
    ------
    ImportError
        If ``dask`` is not installed.
    """
    try:
        import dask.bag as db
    except ImportError:
        msg = "The 'dask' backend requires dask. Install it with: pip install 'physicsnemo-curator[parallel]'"
        raise ImportError(msg) from None

    if progress:
        try:
            from dask.diagnostics import ProgressBar

            pbar: Any = ProgressBar()
            pbar.register()
        except ImportError:
            pass

    bag = db.from_sequence(
        [(pipeline, i) for i in indices],
        npartitions=min(n_jobs, len(indices)),
    )
    results: list[list[str]] = bag.map(_process_one, **kwargs).compute()
    return results


# ---------------------------------------------------------------------------
# Auto backend selection
# ---------------------------------------------------------------------------


def _pick_auto_backend() -> str:
    """Select the best available backend.

    Returns
    -------
    str
        One of ``"dask"``, ``"loky"``, or ``"processes"``.
    """
    try:
        import dask.bag  # noqa: F401

        return "dask"
    except ImportError:
        pass
    try:
        import joblib  # noqa: F401

        return "loky"
    except ImportError:
        pass
    return "processes"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_pipeline(
    pipeline: Pipeline[Any],
    *,
    n_jobs: int = 1,
    backend: str = "auto",
    indices: Iterable[int] | None = None,
    progress: bool = True,
    **backend_kwargs: Any,
) -> list[list[str]]:
    """Execute a pipeline over all (or selected) source indices.

    This is the primary entry-point for batch or parallel pipeline
    execution.  It replaces the manual ``for i in range(len(pipeline))``
    loop and transparently dispatches to the chosen backend.

    Parameters
    ----------
    pipeline : Pipeline
        A fully-configured pipeline (source + filters + sink).
    n_jobs : int
        Number of parallel workers.  ``1`` forces sequential execution.
        ``-1`` uses all available CPUs.
    backend : str
        Execution backend.  One of ``"auto"``, ``"sequential"``,
        ``"processes"``, ``"loky"``, ``"dask"``.
    indices : Iterable[int] | None
        Specific source indices to process.  ``None`` (default) processes
        all indices ``range(len(pipeline))``.
    progress : bool
        Show a progress indicator if the chosen backend supports it.
    **backend_kwargs : Any
        Extra keyword arguments forwarded to the backend executor.

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
    When using multiprocess backends (``"processes"``, ``"loky"``,
    ``"dask"``), each worker operates on an independent copy of the
    pipeline.  Stateful filters (e.g. ``MeanFilter``) accumulate
    per-process state that is **not** merged back.  Use sequential
    execution when filter side-effects must be aggregated, or implement
    a custom reduce step after ``run_pipeline`` returns.

    Examples
    --------
    Sequential execution (default):

    >>> results = run_pipeline(pipeline)

    Parallel with 4 processes:

    >>> results = run_pipeline(pipeline, n_jobs=4, backend="processes")

    Process only a subset of indices:

    >>> results = run_pipeline(pipeline, indices=[0, 5, 10])
    """
    if backend not in _BACKENDS:
        msg = f"Unknown backend {backend!r}. Choose from {_BACKENDS}."
        raise ValueError(msg)

    if pipeline.sink is None:
        msg = "Pipeline has no sink. Call .write(sink) before run_pipeline()."
        raise RuntimeError(msg)

    # Materialise index list.
    idx_list: list[int] = list(indices) if indices is not None else list(range(len(pipeline)))
    if not idx_list:
        return []

    n = _resolve_n_jobs(n_jobs)

    # Force sequential for single-worker requests.
    if n == 1 or backend == "sequential":
        return _run_sequential(pipeline, idx_list, progress=progress)

    # Resolve "auto".
    effective = backend if backend != "auto" else _pick_auto_backend()

    if effective == "processes":
        return _run_processes(pipeline, idx_list, n_jobs=n, progress=progress, **backend_kwargs)
    if effective == "loky":
        return _run_loky(pipeline, idx_list, n_jobs=n, progress=progress, **backend_kwargs)
    if effective == "dask":
        return _run_dask(pipeline, idx_list, n_jobs=n, progress=progress, **backend_kwargs)

    # Fallback (should never be reached given the validation above).
    return _run_sequential(pipeline, idx_list, progress=progress)  # pragma: no cover
