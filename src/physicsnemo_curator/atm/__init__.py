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

"""Atomic (atm) submodule for PhysicsNeMo Curator.

Provides pipeline components for reading, transforming, and writing
:class:`~nvalchemi.data.AtomicData` objects.  Requires the ``atm``
dependency group (nvalchemi, ase, ase-db-backends, torch).

This module registers its components with the global
:data:`~physicsnemo_curator.core.registry.registry` at import time.
"""

from __future__ import annotations

from physicsnemo_curator.atm.filters.stats import AtomicStatsFilter, merge_welford_stats
from physicsnemo_curator.atm.sinks.zarr_writer import AtomicDataZarrSink
from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource
from physicsnemo_curator.core.registry import registry

# Register submodule and components with the global registry.
registry.register_submodule("atm", "Atomic data curation (nvalchemi.data.AtomicData)", "nvalchemi.data")
registry.register_source("atm", ASELMDBSource)
registry.register_filter("atm", AtomicStatsFilter)
registry.register_sink("atm", AtomicDataZarrSink)

__all__ = [
    "ASELMDBSource",
    "AtomicDataZarrSink",
    "AtomicStatsFilter",
    "merge_welford_stats",
]
