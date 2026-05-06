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

"""Process pool execution backend.

Uses :class:`concurrent.futures.ProcessPoolExecutor` for parallel execution.
Suitable for CPU-bound workloads that benefit from true parallelism.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from typing import TYPE_CHECKING, Any, ClassVar

from physicsnemo_curator.run.base import (
    RunBackend,
    RunConfig,
    batch_groups,
    intersect_partitions,
    process_index_group,
    process_single_index_packed,
)
from physicsnemo_curator.run.progress_monitor import start_progress_monitor

if TYPE_CHECKING:
    from physicsnemo_curator.core.base import Pipeline


class ProcessPoolBackend(RunBackend):
    """Execute pipeline items using a process pool.

    This backend uses Python's :class:`concurrent.futures.ProcessPoolExecutor`.
    It provides true parallelism for CPU-bound workloads by spawning separate
    processes that bypass the GIL.

    .. warning::

       Stateful filters accumulate per-process state that is **not** merged
       back into the parent process. Use sequential execution when filter
       side-effects must be aggregated.

    Backend Options
    ---------------
    max_workers : int | None
        Maximum number of processes. Defaults to ``config.resolved_n_jobs``.
    mp_context : str | None
        Multiprocessing context ("spawn", "fork", "forkserver").
    initializer : Callable | None
        Callable to run at the start of each worker process.
    initargs : tuple
        Arguments for the initializer.
    """

    name: ClassVar[str] = "process_pool"
    description: ClassVar[str] = "Process pool executor (true parallelism for CPU-bound tasks)"
    requires: ClassVar[tuple[str, ...]] = ()

    def run(
        self,
        pipeline: Pipeline[Any],
        config: RunConfig,
    ) -> list[list[str]]:
        """Execute pipeline indices using a process pool.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline to execute.
        config : RunConfig
            Execution configuration.

        Returns
        -------
        list[list[str]]
            Sink outputs, one list per index.
        """
        indices = config.indices if config.indices is not None else list(range(len(pipeline)))
        n_jobs = config.resolved_n_jobs

        # Extract ProcessPoolExecutor-specific options
        executor_kwargs = {
            k: v
            for k, v in config.backend_options.items()
            if k in ("max_workers", "mp_context", "initializer", "initargs", "max_tasks_per_child")
        }
        if "max_workers" not in executor_kwargs:
            executor_kwargs["max_workers"] = n_jobs

        result_map: dict[int, list[str]] = {}
        with start_progress_monitor(pipeline, config), ProcessPoolExecutor(**executor_kwargs) as executor:
            # Compute partition groups from source and sink constraints.
            source_groups = pipeline.source.partition_indices(indices)
            sink_groups = pipeline.sink.partition_indices(indices) if pipeline.sink else None
            groups = intersect_partitions(source_groups, sink_groups)

            if groups is not None:
                # Batch groups to at most n_workers batches for efficiency.
                batches = batch_groups(groups, n_jobs)
                # Dispatch batches with work-stealing.
                pending: set[Future[dict[int, list[str]]]] = set()
                future_to_group: dict[Future[dict[int, list[str]]], list[int]] = {}

                next_submit = 0
                for _ in range(min(n_jobs, len(batches))):
                    batch = batches[next_submit]
                    fut: Future[dict[int, list[str]]] = executor.submit(_process_group_packed, (pipeline, batch))
                    future_to_group[fut] = batch
                    pending.add(fut)
                    next_submit += 1

                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        group_results = future.result()
                        result_map.update(group_results)

                        if next_submit < len(batches):
                            batch = batches[next_submit]
                            fut_next: Future[dict[int, list[str]]] = executor.submit(
                                _process_group_packed, (pipeline, batch)
                            )
                            future_to_group[fut_next] = batch
                            pending.add(fut_next)
                            next_submit += 1
            else:
                # Default: one index per future with work-stealing.
                pending_idx: set[Future[list[str]]] = set()
                future_to_idx: dict[Future[list[str]], int] = {}

                # Submit initial batch (one per worker)
                next_submit = 0
                for _ in range(min(n_jobs, len(indices))):
                    idx = indices[next_submit]
                    fut_idx: Future[list[str]] = executor.submit(process_single_index_packed, (pipeline, idx))
                    future_to_idx[fut_idx] = idx
                    pending_idx.add(fut_idx)
                    next_submit += 1

                # Process completions and submit new tasks
                while pending_idx:
                    done_idx, pending_idx = wait(pending_idx, return_when=FIRST_COMPLETED)
                    for future in done_idx:
                        idx = future_to_idx[future]
                        result_map[idx] = future.result()

                        # Submit next task if available
                        if next_submit < len(indices):
                            next_idx = indices[next_submit]
                            fut_next_idx: Future[list[str]] = executor.submit(
                                process_single_index_packed, (pipeline, next_idx)
                            )
                            future_to_idx[fut_next_idx] = next_idx
                            pending_idx.add(fut_next_idx)
                            next_submit += 1

        return [result_map[idx] for idx in indices]


def _process_group_packed(args: tuple[Pipeline[Any], list[int]]) -> dict[int, list[str]]:
    """Process an index group (packed arguments for ProcessPoolExecutor).

    Parameters
    ----------
    args : tuple[Pipeline, list[int]]
        A ``(pipeline, indices)`` pair.

    Returns
    -------
    dict[int, list[str]]
        Mapping of index to sink output paths.
    """
    pipeline, group_indices = args
    return process_index_group(pipeline, group_indices)
