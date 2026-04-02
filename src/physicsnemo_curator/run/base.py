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

"""Base classes for pipeline execution backends.

This module defines the abstract interface that all execution backends
must implement, along with common utilities.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from physicsnemo_curator.core.base import Pipeline


@dataclass
class RunConfig:
    """Configuration for pipeline execution.

    Parameters
    ----------
    n_jobs : int
        Number of parallel workers. ``1`` forces sequential execution.
        ``-1`` uses all available CPUs. Values ``<= 0`` follow the
        convention ``cpu_count + 1 + n_jobs``.
    progress : bool
        Whether to show a progress indicator (if supported by backend).
    indices : list[int] | None
        Specific source indices to process. ``None`` processes all indices.
    backend_options : dict[str, Any]
        Additional backend-specific options.
    """

    n_jobs: int = 1
    progress: bool = True
    indices: list[int] | None = None
    backend_options: dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_n_jobs(self) -> int:
        """Return the concrete positive worker count.

        Returns
        -------
        int
            Positive integer number of workers.
        """
        if self.n_jobs > 0:
            return self.n_jobs
        cpu = os.cpu_count() or 1
        resolved = cpu + 1 + self.n_jobs  # -1 → cpu, -2 → cpu-1, …
        return max(1, resolved)


class RunBackend(ABC):
    """Abstract base class for pipeline execution backends.

    Subclasses implement different parallelization strategies (threading,
    multiprocessing, distributed computing, workflow orchestrators, etc.).

    Class Attributes
    ----------------
    name : str
        Unique identifier for this backend (e.g., "sequential", "thread_pool").
    description : str
        Human-readable description of the backend.
    requires : tuple[str, ...]
        Optional package dependencies required by this backend.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    requires: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def is_available(cls) -> bool:
        """Check if this backend's dependencies are installed.

        Returns
        -------
        bool
            True if all required packages are available.
        """
        for package in cls.requires:
            try:
                __import__(package)
            except ImportError:
                return False
        return True

    @abstractmethod
    def run(
        self,
        pipeline: Pipeline[Any],
        config: RunConfig,
    ) -> list[list[str]]:
        """Execute the pipeline over the configured indices.

        Parameters
        ----------
        pipeline : Pipeline
            A fully-configured pipeline (source + filters + sink).
        config : RunConfig
            Execution configuration.

        Returns
        -------
        list[list[str]]
            Outer list is ordered by the input indices; each inner list
            contains the file paths returned by the sink for that index.
        """
        ...


def process_single_index(pipeline: Pipeline[Any], index: int) -> list[str]:
    """Process a single pipeline index.

    This is a module-level function to support pickling for multiprocess
    backends.

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline to execute.
    index : int
        The index to process.

    Returns
    -------
    list[str]
        File paths written by the sink.
    """
    return pipeline[index]


def process_single_index_packed(args: tuple[Pipeline[Any], int]) -> list[str]:
    """Process a single pipeline index (packed arguments for map functions).

    Parameters
    ----------
    args : tuple[Pipeline, int]
        A ``(pipeline, index)`` pair.

    Returns
    -------
    list[str]
        File paths written by the sink.
    """
    pipeline, index = args
    return pipeline[index]


def make_progress_bar(total: int, *, enabled: bool, desc: str = "run_pipeline") -> Any:
    """Return a tqdm progress bar or None.

    Parameters
    ----------
    total : int
        Number of items.
    enabled : bool
        Whether to attempt tqdm import.
    desc : str
        Description for the progress bar.

    Returns
    -------
    Any
        A tqdm progress bar, or None if disabled or unavailable.
    """
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm

        return tqdm(total=total, desc=desc, unit="item")
    except ImportError:
        return None
