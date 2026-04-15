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
    WorkerProgressDisplay,
    process_single_index_packed,
)

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

        display = WorkerProgressDisplay(
            total=len(indices),
            n_workers=n_jobs,
            enabled=config.progress,
        )

        # Extract ProcessPoolExecutor-specific options
        executor_kwargs = {
            k: v
            for k, v in config.backend_options.items()
            if k in ("max_workers", "mp_context", "initializer", "initargs", "max_tasks_per_child")
        }
        if "max_workers" not in executor_kwargs:
            executor_kwargs["max_workers"] = n_jobs

        result_map: dict[int, list[str]] = {}
        try:
            with ProcessPoolExecutor(**executor_kwargs) as executor:
                future_to_idx: dict[Future[list[str]], int] = {}
                future_to_slot: dict[Future[list[str]], int] = {}
                pending: set[Future[list[str]]] = set()

                # Submit initial batch (one per worker)
                next_submit = 0
                for slot in range(min(n_jobs, len(indices))):
                    idx = indices[next_submit]
                    fut: Future[list[str]] = executor.submit(process_single_index_packed, (pipeline, idx))
                    future_to_idx[fut] = idx
                    future_to_slot[fut] = slot
                    pending.add(fut)
                    display.worker_start(slot, idx)
                    next_submit += 1

                # Process completions and submit new tasks
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        idx = future_to_idx[future]
                        slot = future_to_slot[future]
                        result_map[idx] = future.result()

                        # Submit next task to this slot if available
                        if next_submit < len(indices):
                            next_idx = indices[next_submit]
                            fut_next: Future[list[str]] = executor.submit(
                                process_single_index_packed, (pipeline, next_idx)
                            )
                            future_to_idx[fut_next] = next_idx
                            future_to_slot[fut_next] = slot
                            pending.add(fut_next)
                            display.worker_start(slot, next_idx)
                            next_submit += 1
                        else:
                            display.worker_done(slot)
        finally:
            display.close()

        return [result_map[idx] for idx in indices]
