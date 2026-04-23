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

from typing import Optional

import numpy as np


def compute_node_type(pos_raw: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    """
    Identify structural vs wall nodes based on displacement variation.

    Args:
        pos_raw: (timesteps, num_nodes, 3) raw displacement trajectories
        threshold: max displacement below which a node is considered "wall"

    Returns:
        node_type: (num_nodes,) uint8 array where 1=wall, 0=structure
    """
    variation = np.max(np.abs(pos_raw - pos_raw[0:1, :, :]), axis=0)
    variation = np.max(variation, axis=1)
    is_wall = variation < threshold
    return np.where(is_wall, 1, 0).astype(np.uint8)


def build_edges_from_mesh_connectivity(mesh_connectivity) -> set:
    """
    Build unique edges from mesh connectivity.

    Args:
        mesh_connectivity: list of elements (list[int])

    Returns:
        Set of unique edges (i,j)
    """
    edges = set()
    for cell in mesh_connectivity:
        n = len(cell)
        for idx in range(n):
            edges.add(tuple(sorted((cell[idx], cell[(idx + 1) % n]))))
    return edges


def reduce_solid_layers_scalar(x: np.ndarray) -> np.ndarray:
    """
    Average over integration-point layers (axis=2) for scalar solid fields.
    Input shape: (T, E, n_layers) -> Output shape: (T, E)
    """
    return np.nanmean(x, axis=2)


def reduce_solid_layers_voigt(x: np.ndarray) -> np.ndarray:
    """
    Average over integration-point layers (axis=2) for Voigt-form fields.
    Input shape: (T, E, n_layers, 6) -> Output shape: (T, E, 6)
    """
    return np.nanmean(x, axis=2)


def von_mises_from_voigt(sig: np.ndarray) -> np.ndarray:
    """
    Compute von Mises stress from Voigt components [sx, sy, sz, txy, tyz, tzx].
    Input shape: (T, E, 6) -> Output shape: (T, E)
    """
    sx = sig[..., 0]
    sy = sig[..., 1]
    sz = sig[..., 2]
    txy = sig[..., 3]
    tyz = sig[..., 4]
    tzx = sig[..., 5]
    j2 = 0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3.0 * (
        txy**2 + tyz**2 + tzx**2
    )
    return np.sqrt(np.maximum(j2 * 2.0 / 3.0, 0.0))


def compute_element_fields(
    element_solid_stress: Optional[np.ndarray] = None,
    element_solid_effective_plastic_strain: Optional[np.ndarray] = None,
    element_solid_strain: Optional[np.ndarray] = None,
    element_solid_plastic_strain_tensor: Optional[np.ndarray] = None,
    compute_von_mises: bool = True,
) -> dict:
    """
    Compute reduced element fields for solid elements.
    Inputs can be None; outputs will carry None accordingly.
    Returns dict with key 'element' containing arrays or None.
    """
    out = {"element": {}}

    # Effective plastic strain (scalar)
    if element_solid_effective_plastic_strain is not None:
        eps_elem = reduce_solid_layers_scalar(element_solid_effective_plastic_strain)
        out["element"]["effective_plastic_strain"] = eps_elem
    else:
        out["element"]["effective_plastic_strain"] = None

    # Stress
    if element_solid_stress is not None:
        stress_elem_voigt = reduce_solid_layers_voigt(element_solid_stress)
        out["element"]["stress_voigt"] = stress_elem_voigt
        if compute_von_mises:
            out["element"]["stress_vm"] = von_mises_from_voigt(stress_elem_voigt)
        else:
            out["element"]["stress_vm"] = None
    else:
        out["element"]["stress_voigt"] = None
        out["element"]["stress_vm"] = None

    # Strain
    if element_solid_strain is not None:
        out["element"]["strain_voigt"] = reduce_solid_layers_voigt(element_solid_strain)
    else:
        out["element"]["strain_voigt"] = None

    # Plastic strain tensor
    if element_solid_plastic_strain_tensor is not None:
        out["element"]["plastic_strain_voigt"] = reduce_solid_layers_voigt(
            element_solid_plastic_strain_tensor
        )
    else:
        out["element"]["plastic_strain_voigt"] = None

    return out
