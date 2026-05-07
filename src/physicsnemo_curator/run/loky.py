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

"""Loky (joblib) execution backend.

Uses ``joblib.Parallel`` with the loky backend for robust parallel execution.
Loky provides better process management than the standard multiprocessing module,
including automatic worker restart on crashes and better memory cleanup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from physicsnemo_curator.run.base import (
    RunBackend,
    RunConfig,
    batch_groups,
    intersect_partitions,
    process_index_group,
    process_single_index,
)
from physicsnemo_curator.run.progress_monitor import start_progress_monitor

if TYPE_CHECKING:
    from physicsnemo_curator.core.base import Pipeline


class LokyBackend(RunBackend):
    """Execute pipeline items using joblib with the loky backend.

    Loky is a robust process executor that handles worker crashes gracefully
    and provides better memory management than standard multiprocessing.
    Key advantages over ProcessPoolExecutor:

    - **Automatic worker restart**: If a worker crashes or is killed, loky
      automatically restarts it without failing the entire job.
    - **Memory cleanup**: Workers are recycled after processing a configurable
      number of tasks to prevent memory leaks.
    - **Robust serialization**: Uses cloudpickle for better serialization of
      complex objects including lambdas and closures.
    - **Timeout handling**: Built-in timeout support per task.

    .. warning::

       Stateful filters accumulate per-process state that is **not** merged
       back into the parent process.

    Backend Options
    ---------------
    prefer : str
        Soft hint for parallelization ("processes" or "threads").
    require : str
        Hard constraint for parallelization.
    verbose : int
        Verbosity level (0-50). If not set, uses 0 (quiet).
    batch_size : int | str
        Number of tasks per batch ("auto" or int).
    pre_dispatch : str | int
        Number of batches to pre-dispatch.
    temp_folder : str | None
        Folder for memmapping large arrays.
    timeout : float | None
        Timeout in seconds for retrieving results.
    max_nbytes : str | int | None
        Threshold for automatic memmapping (e.g., "1M").
    """

    name: ClassVar[str] = "loky"
    description: ClassVar[str] = "Joblib with loky backend (robust process management)"
    requires: ClassVar[tuple[str, ...]] = ("joblib",)

    def run(
        self,
        pipeline: Pipeline[Any],
        config: RunConfig,
    ) -> list[list[str]]:
        """Execute pipeline indices using joblib/loky.

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
            If joblib is not installed.
        """
        try:
            from joblib import Parallel, delayed
        except ImportError:
            msg = "The 'loky' backend requires joblib. Install with: pip install 'physicsnemo-curator[loky]'"
            raise ImportError(msg) from None

        indices = config.indices if config.indices is not None else list(range(len(pipeline)))
        n_jobs = config.resolved_n_jobs

        # Extract joblib-specific options
        parallel_kwargs = dict(config.backend_options)
        # Default to quiet since we have our own progress monitor
        if "verbose" not in parallel_kwargs:
            parallel_kwargs["verbose"] = 0

        # Compute partition groups from source and sink constraints
        source_groups = pipeline.source.partition_indices(indices)
        sink_groups = pipeline.sink.partition_indices(indices) if pipeline.sink else None
        groups = intersect_partitions(source_groups, sink_groups)

        result_map: dict[int, list[str]] = {}

        with start_progress_monitor(pipeline, config):
            if groups is not None:
                # Batch groups for efficiency
                batches = batch_groups(groups, n_jobs)

                # Process batched groups
                batch_results: list[dict[int, list[str]]] = Parallel(
                    n_jobs=n_jobs,
                    backend="loky",
                    **parallel_kwargs,
                )(delayed(process_index_group)(pipeline, batch) for batch in batches)

                for batch_result in batch_results:
                    result_map.update(batch_result)
            else:
                # Default: one index per task
                results: list[list[str]] = Parallel(
                    n_jobs=n_jobs,
                    backend="loky",
                    **parallel_kwargs,
                )(delayed(process_single_index)(pipeline, i) for i in indices)

                for i, result in zip(indices, results, strict=True):
                    result_map[i] = result

        return [result_map[i] for i in indices]
