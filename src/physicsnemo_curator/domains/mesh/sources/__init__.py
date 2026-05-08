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

"""Mesh data sources.

Provides pipeline sources for reading mesh data from local files,
remote URLs, and curated HuggingFace Hub datasets.
"""

from physicsnemo_curator.domains.mesh.sources.ahmedml import AhmedMLSource
from physicsnemo_curator.domains.mesh.sources.ansys_rst import AnsysRSTSource
from physicsnemo_curator.domains.mesh.sources.d3plot import D3PlotSource
from physicsnemo_curator.domains.mesh.sources.drivaerml import DrivAerMLSource
from physicsnemo_curator.domains.mesh.sources.ns_cylinder import NavierStokesCylinderSource
from physicsnemo_curator.domains.mesh.sources.openradioss import OpenRadiossSource
from physicsnemo_curator.domains.mesh.sources.vtk import VTKSource

__all__ = [
    "AhmedMLSource",
    "AnsysRSTSource",
    "D3PlotSource",
    "DrivAerMLSource",
    "NavierStokesCylinderSource",
    "OpenRadiossSource",
    "VTKSource",
]
