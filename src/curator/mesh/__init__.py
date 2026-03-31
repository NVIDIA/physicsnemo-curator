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

"""Mesh submodule for PhysicsNeMo Curator.

Provides pipeline components for reading, transforming, and writing
:class:`physicsnemo.mesh.Mesh` objects.  Requires the ``curator[mesh]``
extra (physicsnemo, pyvista, pyarrow, torch).

This module registers its components with the global
:data:`~curator.core.registry.registry` at import time.
"""

from __future__ import annotations

from curator.core.registry import registry
from curator.core.store import FsspecFileStore, LocalFileStore, RunIndexedFileStore
from curator.mesh.filters.mean import MeanFilter
from curator.mesh.sinks.mesh_writer import MeshSink
from curator.mesh.sources.ahmedml import AhmedMLSource
from curator.mesh.sources.drivaerml import DrivAerMLSource
from curator.mesh.sources.vtk import VTKSource
from curator.mesh.sources.windsorml import WindsorMLSource
from curator.mesh.sources.windtunnel import WindTunnelSource

# Register submodule and components with the global registry.
registry.register_submodule("mesh", "Mesh processing (physicsnemo.mesh.Mesh)", "physicsnemo.mesh")
registry.register_store("mesh", "Local directory", LocalFileStore)
registry.register_store("mesh", "Remote (fsspec)", FsspecFileStore)
registry.register_store("mesh", "Run-indexed (remote)", RunIndexedFileStore)
registry.register_source("mesh", VTKSource)
registry.register_source("mesh", DrivAerMLSource)
registry.register_source("mesh", AhmedMLSource)
registry.register_source("mesh", WindsorMLSource)
registry.register_source("mesh", WindTunnelSource)
registry.register_filter("mesh", MeanFilter)
registry.register_sink("mesh", MeshSink)

__all__ = [
    "AhmedMLSource",
    "DrivAerMLSource",
    "MeanFilter",
    "MeshSink",
    "VTKSource",
    "WindsorMLSource",
    "WindTunnelSource",
]
