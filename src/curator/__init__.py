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

"""PhysicsNeMo Curator: ETL toolkit for deep learning data curation."""

from __future__ import annotations

from curator.core.base import Filter, Param, Pipeline, Sink, Source
from curator.core.parallel import run_pipeline
from curator.core.registry import registry
from curator.core.store import FileStore, FsspecFileStore, LocalFileStore

__version__ = "0.1.0"

try:
    from curator._lib import rust_version
except ImportError:
    # Native extension not built yet (e.g. during docs build or initial setup).
    def rust_version() -> str:  # type: ignore[misc]
        """Return a placeholder when the native library is unavailable."""
        return "not built"


__all__ = [
    "FileStore",
    "Filter",
    "FsspecFileStore",
    "LocalFileStore",
    "Param",
    "Pipeline",
    "Sink",
    "Source",
    "__version__",
    "registry",
    "run_pipeline",
    "rust_version",
]
