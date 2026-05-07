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

"""Dask execution backend.

Uses ``dask.bag`` for parallel and distributed execution.
Supports local execution and can scale to clusters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from physicsnemo_curator.run.base import (
    RunBackend,
    RunConfig,
    process_single_index_packed,
)

if TYPE_CHECKING:
    from physicsnemo_curator.core.base import Pipeline


class DaskBackend(RunBackend):
    """Execute pipeline items using Dask bags.

    Dask provides parallel execution that can scale from a single machine
    to a distributed cluster. This backend uses ``dask.bag`` for task-parallel
    execution.

    .. warning::

       Stateful filters accumulate per-worker state that is **not** merged
       back. Design a post-hoc merge strategy if needed.

    Backend Options
    ---------------
    scheduler : str
        Dask scheduler ("synchronous", "threads", "processes", "distributed").
    num_workers : int | None
        Number of workers (for local schedulers).
    client : distributed.Client | None
        Pre-configured Dask distributed client.
    """

    name: ClassVar[str] = "dask"
    description: ClassVar[str] = "Dask bags for parallel/distributed execution"
    requires: ClassVar[tuple[str, ...]] = ("dask",)

    def run(
        self,
        pipeline: Pipeline[Any],
        config: RunConfig,
    ) -> list[list[str]]:
        """Execute pipeline indices using Dask.

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
            If dask is not installed.
        """
        try:
            import dask.bag as db
        except ImportError:
            msg = "The 'dask' backend requires dask. Install with: pip install 'physicsnemo-curator[dask]'"
            raise ImportError(msg) from None

        indices = config.indices if config.indices is not None else list(range(len(pipeline)))
        n_jobs = config.resolved_n_jobs

        # Set up progress bar if TUI requested (dask uses its own progress bar)
        if config.use_tui:
            try:
                from dask.diagnostics import ProgressBar

                pbar: Any = ProgressBar()
                pbar.register()
            except ImportError:
                pass

        # Create dask bag from index-pipeline pairs
        npartitions = min(n_jobs, len(indices))
        bag = db.from_sequence(
            [(pipeline, i) for i in indices],
            npartitions=npartitions,
        )

        # Extract compute options
        compute_kwargs = {k: v for k, v in config.backend_options.items() if k in ("scheduler", "num_workers")}

        results: list[list[str]] = bag.map(process_single_index_packed).compute(**compute_kwargs)
        return results
