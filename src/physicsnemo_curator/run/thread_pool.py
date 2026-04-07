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

"""Thread pool execution backend.

Uses :class:`concurrent.futures.ThreadPoolExecutor` for parallel execution.
Suitable for I/O-bound workloads where the GIL is not a bottleneck.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, ClassVar

from physicsnemo_curator.run.base import (
    RunBackend,
    RunConfig,
    WorkerProgressDisplay,
    process_single_index,
)

if TYPE_CHECKING:
    from physicsnemo_curator.core.base import Pipeline


class ThreadPoolBackend(RunBackend):
    """Execute pipeline items using a thread pool.

    This backend uses Python's :class:`concurrent.futures.ThreadPoolExecutor`.
    It's suitable for I/O-bound workloads but may not provide speedup for
    CPU-bound tasks due to the GIL.

    Backend Options
    ---------------
    max_workers : int | None
        Maximum number of threads. Defaults to ``config.resolved_n_jobs``.
    thread_name_prefix : str
        Prefix for thread names.
    """

    name: ClassVar[str] = "thread_pool"
    description: ClassVar[str] = "Thread pool executor (good for I/O-bound tasks)"
    requires: ClassVar[tuple[str, ...]] = ()

    def run(
        self,
        pipeline: Pipeline[Any],
        config: RunConfig,
    ) -> list[list[str]]:
        """Execute pipeline indices using a thread pool.

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

        # Extract ThreadPoolExecutor-specific options
        executor_kwargs = {
            k: v
            for k, v in config.backend_options.items()
            if k in ("max_workers", "thread_name_prefix", "initializer", "initargs")
        }
        if "max_workers" not in executor_kwargs:
            executor_kwargs["max_workers"] = n_jobs

        display = WorkerProgressDisplay(
            total=len(indices),
            n_workers=n_jobs,
            enabled=config.progress,
        )

        # Map worker threads to display slots
        _slot_lock = threading.Lock()
        _thread_slots: dict[int, int] = {}
        _next_slot = [0]

        def _get_slot() -> int:
            tid = threading.get_ident()
            with _slot_lock:
                if tid not in _thread_slots:
                    _thread_slots[tid] = _next_slot[0]
                    _next_slot[0] += 1
                return _thread_slots[tid]

        def _process_tracked(idx: int) -> list[str]:
            slot = _get_slot()
            display.worker_start(slot, idx)
            result = process_single_index(pipeline, idx)
            display.worker_done(slot)
            return result

        # Use as_completed so the main bar updates as soon as each
        # future finishes rather than waiting for ordered completion.
        result_map: dict[int, list[str]] = {}
        try:
            with ThreadPoolExecutor(**executor_kwargs) as executor:
                future_to_idx = {executor.submit(_process_tracked, idx): idx for idx in indices}

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    result_map[idx] = future.result()
        finally:
            display.close()

        # Return results in original index order
        return [result_map[idx] for idx in indices]
