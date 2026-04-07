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

"""Prefect execution backend.

Uses Prefect for workflow orchestration with observability, retries,
and scheduling capabilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from physicsnemo_curator.run.base import RunBackend, RunConfig

if TYPE_CHECKING:
    from physicsnemo_curator.core.base import Pipeline


class PrefectBackend(RunBackend):
    """Execute pipeline items using Prefect tasks.

    Prefect provides workflow orchestration with features like:
    - Automatic retries on failure
    - Observability and logging
    - Caching of task results
    - Scheduling and triggers
    - Distributed execution with Dask or Ray

    .. warning::

       Stateful filters accumulate per-task state that is **not** merged
       back. Design a post-hoc merge strategy if needed.

    Backend Options
    ---------------
    task_runner : prefect.task_runners.BaseTaskRunner | None
        Custom task runner (e.g., DaskTaskRunner, RayTaskRunner).
    retries : int
        Number of retries for failed tasks. Default: 0.
    retry_delay_seconds : float
        Delay between retries. Default: 0.
    timeout_seconds : float | None
        Timeout for each task.
    cache_key_fn : Callable | None
        Function to generate cache keys.
    persist_result : bool
        Whether to persist task results. Default: False.
    flow_name : str
        Name for the Prefect flow. Default: "run_pipeline".
    """

    name: ClassVar[str] = "prefect"
    description: ClassVar[str] = "Prefect workflow orchestration with observability"
    requires: ClassVar[tuple[str, ...]] = ("prefect",)

    def run(
        self,
        pipeline: Pipeline[Any],
        config: RunConfig,
    ) -> list[list[str]]:
        """Execute pipeline indices using Prefect.

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

        Raises
        ------
        ImportError
            If prefect is not installed.
        """
        try:
            from prefect import flow, task
            from prefect.futures import wait
        except ImportError:
            msg = "The 'prefect' backend requires prefect. Install with: pip install 'physicsnemo-curator[prefect]'"
            raise ImportError(msg) from None

        indices = config.indices if config.indices is not None else list(range(len(pipeline)))
        n_jobs = config.resolved_n_jobs
        opts = config.backend_options

        # Extract task options
        task_kwargs: dict[str, Any] = {}
        if "retries" in opts:
            task_kwargs["retries"] = opts["retries"]
        if "retry_delay_seconds" in opts:
            task_kwargs["retry_delay_seconds"] = opts["retry_delay_seconds"]
        if "timeout_seconds" in opts:
            task_kwargs["timeout_seconds"] = opts["timeout_seconds"]
        if "cache_key_fn" in opts:
            task_kwargs["cache_key_fn"] = opts["cache_key_fn"]
        if "persist_result" in opts:
            task_kwargs["persist_result"] = opts["persist_result"]

        # Create the task dynamically
        @task(name="process_pipeline_index", **task_kwargs)
        def process_index(idx: int) -> list[str]:
            """Process a single pipeline index and flush stateful filters."""
            from physicsnemo_curator.run.base import _flush_filters

            result = pipeline[idx]
            _flush_filters(pipeline, idx)
            return result

        # Extract flow options
        flow_name = opts.get("flow_name", "run_pipeline")
        task_runner = opts.get("task_runner", None)

        # Determine task runner based on n_jobs
        if task_runner is None and n_jobs > 1:
            try:
                from prefect.task_runners import ConcurrentTaskRunner

                task_runner = ConcurrentTaskRunner(max_workers=n_jobs)
            except ImportError:
                pass  # Fall back to default

        flow_kwargs: dict[str, Any] = {"name": flow_name}
        if task_runner is not None:
            flow_kwargs["task_runner"] = task_runner

        @flow(**flow_kwargs)
        def pipeline_flow() -> list[list[str]]:
            """Execute all pipeline indices."""
            futures = [process_index.submit(idx) for idx in indices]
            wait(futures)
            return [f.result() for f in futures]

        return pipeline_flow()
