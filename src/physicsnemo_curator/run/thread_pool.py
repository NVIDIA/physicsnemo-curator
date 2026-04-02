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

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, ClassVar

from physicsnemo_curator.run.base import RunBackend, RunConfig, make_progress_bar, process_single_index

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
        pbar = make_progress_bar(len(indices), enabled=config.progress)

        # Extract ThreadPoolExecutor-specific options
        executor_kwargs = {
            k: v
            for k, v in config.backend_options.items()
            if k in ("max_workers", "thread_name_prefix", "initializer", "initargs")
        }
        if "max_workers" not in executor_kwargs:
            executor_kwargs["max_workers"] = n_jobs

        results: list[list[str]] = []
        try:
            with ThreadPoolExecutor(**executor_kwargs) as executor:
                futures = [executor.submit(process_single_index, pipeline, idx) for idx in indices]
                for future in futures:
                    result = future.result()
                    results.append(result)
                    if pbar is not None:
                        pbar.update(1)
        finally:
            if pbar is not None:
                pbar.close()

        return results
