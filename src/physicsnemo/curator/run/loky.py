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
Loky provides better process management than the standard multiprocessing module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from physicsnemo.curator.run.base import (
    RunBackend,
    RunConfig,
    process_single_index,
)

if TYPE_CHECKING:
    from physicsnemo.curator.core.base import Pipeline


class LokyBackend(RunBackend):
    """Execute pipeline items using joblib with the loky backend.

    Loky is a robust process executor that handles worker crashes gracefully
    and provides better memory management than standard multiprocessing.

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
        Verbosity level (0-50). If not set, uses 10 when progress=True.
    batch_size : int | str
        Number of tasks per batch ("auto" or int).
    pre_dispatch : str | int
        Number of batches to pre-dispatch.
    temp_folder : str | None
        Folder for memmapping large arrays.
    timeout : float | None
        Timeout in seconds for retrieving results.
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
        if "verbose" not in parallel_kwargs:
            parallel_kwargs["verbose"] = 10 if config.progress else 0

        results: list[list[str]] = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            **parallel_kwargs,
        )(delayed(process_single_index)(pipeline, i) for i in indices)

        return results
