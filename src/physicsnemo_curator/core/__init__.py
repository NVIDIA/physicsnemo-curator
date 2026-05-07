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

"""Core pipeline framework for PhysicsNeMo Curator."""

from __future__ import annotations

from physicsnemo_curator.core.base import Filter, Param, Pipeline, Sink, Source
from physicsnemo_curator.core.logging import (
    DatabaseLogHandler,
    configure_logging,
    get_logger,
    setup_worker_logging,
)
from physicsnemo_curator.core.pipeline_store import PipelineMetrics, PipelineStore
from physicsnemo_curator.core.registry import registry
from physicsnemo_curator.core.serialization import (
    deserialize_pipeline,
    load_pipeline,
    save_pipeline,
    serialize_pipeline,
)
from physicsnemo_curator.run import run_pipeline

__all__ = [
    "DatabaseLogHandler",
    "Filter",
    "Param",
    "Pipeline",
    "PipelineMetrics",
    "PipelineStore",
    "Sink",
    "Source",
    "configure_logging",
    "deserialize_pipeline",
    "get_logger",
    "load_pipeline",
    "registry",
    "run_pipeline",
    "save_pipeline",
    "serialize_pipeline",
    "setup_worker_logging",
]
