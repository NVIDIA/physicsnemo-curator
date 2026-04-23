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

from dataclasses import dataclass

import numpy as np


@dataclass
class DropTestMetadata:
    """Metadata for drop test simulation data.

    Version history:
    - 1.0: Initial version with expected metadata fields.
    """

    # Simulation identifiers
    filename: str


@dataclass
class DropTestExtractedDataInMemory:
    """Container for drop test (solid element) simulation data and metadata.

    Version history:
    - 1.0: Initial version with nodal and element fields for solids.
    """

    # Metadata
    metadata: DropTestMetadata = None

    # Raw nodal data
    pos_raw: np.ndarray = None  # (T, N, 3) displacements
    node_velocity: np.ndarray = None  # (T, N, 3) or None
    node_acceleration: np.ndarray = None  # (T, N, 3) or None
    node_temperature: np.ndarray = None  # (T, N) or None
    node_residual_forces: np.ndarray = None  # (T, N, 3) or None
    node_stress_voigt: np.ndarray = None  # (T, N, 6) or None
    node_stress_vm: np.ndarray = None  # (T, N) or None

    # Raw element data (solid)
    mesh_connectivity: np.ndarray = None  # list of cells (hex=8, tet=4 nodes)
    part_ids: np.ndarray = None
    element_solid_stress: np.ndarray = None  # (T, E, n_layers, 6) or None
    element_solid_stress_vm: np.ndarray = None  # (T, E) von Mises when stress tensor not available
    element_solid_effective_plastic_strain: np.ndarray = None  # (T, E, n_layers) or None
    element_solid_strain: np.ndarray = None  # (T, E, n_layers, 6) or None
    element_solid_plastic_strain_tensor: np.ndarray = None  # (T, E, n_layers, 6) or None

    # Processed data
    filtered_pos_raw: np.ndarray = None
    filtered_node_velocity: np.ndarray = None
    filtered_node_acceleration: np.ndarray = None
    filtered_node_temperature: np.ndarray = None
    filtered_node_residual_forces: np.ndarray = None
    filtered_node_stress_voigt: np.ndarray = None  # (T, N, 6) or None
    filtered_node_stress_vm: np.ndarray = None  # (T, N) or None
    filtered_mesh_connectivity: np.ndarray = None
    filtered_node_thickness: np.ndarray = None  # zeros for solids (no thickness)
    edges: np.ndarray = None

    # Processed element fields
    filtered_element_stress_voigt: np.ndarray = None  # (T, E, 6)
    filtered_element_stress_vm: np.ndarray = None  # (T, E)
    filtered_element_effective_plastic_strain: np.ndarray = None  # (T, E)
    filtered_element_strain_voigt: np.ndarray = None  # (T, E, 6) or None
    filtered_element_plastic_strain_voigt: np.ndarray = None  # (T, E, 6) or None
