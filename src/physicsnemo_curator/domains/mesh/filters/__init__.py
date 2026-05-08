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

"""Mesh data filters/transforms."""

from physicsnemo_curator.domains.mesh.filters.edge_compute import EdgeComputeFilter
from physicsnemo_curator.domains.mesh.filters.field_select import FieldSelectFilter
from physicsnemo_curator.domains.mesh.filters.mean import MeanFilter
from physicsnemo_curator.domains.mesh.filters.mesh_info import MeshInfoFilter
from physicsnemo_curator.domains.mesh.filters.precision import PrecisionFilter
from physicsnemo_curator.domains.mesh.filters.quality import MeshQualityFilter
from physicsnemo_curator.domains.mesh.filters.random_permutation import RandomPermutationFilter
from physicsnemo_curator.domains.mesh.filters.wall_node import WallNodeFilter

__all__ = [
    "EdgeComputeFilter",
    "FieldSelectFilter",
    "MeanFilter",
    "MeshInfoFilter",
    "MeshQualityFilter",
    "PrecisionFilter",
    "RandomPermutationFilter",
    "WallNodeFilter",
]
