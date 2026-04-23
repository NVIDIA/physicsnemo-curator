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

"""Sequential execution backend.

Processes pipeline items one at a time in a simple for-loop.
This is the default when ``n_jobs=1``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from physicsnemo_curator.run.base import (
    RunBackend,
    RunConfig,
    _flush_filters,
)
from physicsnemo_curator.run.progress_monitor import start_progress_monitor

if TYPE_CHECKING:
    from physicsnemo_curator.core.base import Pipeline


class SequentialBackend(RunBackend):
    """Execute pipeline items sequentially in a for-loop.

    This backend has no dependencies and is always available. It's the
    safest choice for pipelines with stateful filters that need to
    accumulate results across all items.
    """

    name: ClassVar[str] = "sequential"
    description: ClassVar[str] = "Sequential for-loop execution (no parallelism)"
    requires: ClassVar[tuple[str, ...]] = ()

    def run(
        self,
        pipeline: Pipeline[Any],
        config: RunConfig,
    ) -> list[list[str]]:
        """Execute pipeline indices sequentially.

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

        results: list[list[str]] = []
        with start_progress_monitor(pipeline, config):
            for idx in indices:
                results.append(pipeline[idx])
                _flush_filters(pipeline, idx)

        return results
