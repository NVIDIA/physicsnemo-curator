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

"""DataArray submodule for PhysicsNeMo Curator.

Provides pipeline components for reading, transforming, and writing
:class:`xarray.DataArray` objects.  Requires the ``da`` dependency group
(xarray, earth2studio, zarr, gcsfs).

This module registers its components with the global
:data:`~curator.core.registry.registry` at import time.
"""

from __future__ import annotations

from curator.core.registry import registry
from curator.da.filters.moments import MomentsFilter
from curator.da.sinks.zarr_writer import ZarrSink
from curator.da.sources.era5 import ERA5Source

# Register submodule and components with the global registry.
registry.register_submodule("da", "DataArray processing (xarray.DataArray)", "xarray")
registry.register_source("da", ERA5Source)
registry.register_filter("da", MomentsFilter)
registry.register_sink("da", ZarrSink)

__all__ = [
    "ERA5Source",
    "MomentsFilter",
    "ZarrSink",
]
