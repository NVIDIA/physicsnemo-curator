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

from pathlib import Path
from typing import Optional

import numpy as np
from lasso.dyna import ArrayType, D3plot


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


def load_d3plot_data(data_path: str):
    """Load node coordinates and displacements from a d3plot file."""
    dp = D3plot(data_path)
    coords = dp.arrays[ArrayType.node_coordinates]  # (num_nodes, 3)
    pos_raw = dp.arrays[ArrayType.node_displacement]  # (timesteps, num_nodes, 3)
    mesh_connectivity = dp.arrays[ArrayType.element_shell_node_indexes]
    part_ids = dp.arrays[ArrayType.element_shell_part_indexes]

    # Get actual part IDs if available
    actual_part_ids = None
    if ArrayType.part_ids in dp.arrays:
        actual_part_ids = dp.arrays[ArrayType.part_ids]

    return coords, pos_raw, mesh_connectivity, part_ids, actual_part_ids


def find_k_file(run_dir: Path) -> Optional[Path]:
    """Find .k file in run directory.

    Args:
        run_dir: Path to run directory.

    Returns:
        Path to .k file.
    """
    k_files = list(run_dir.glob("*.k"))
    return k_files[0] if k_files else None


def parse_k_file(k_file_path: Path) -> dict[int, float]:
    """Parse LS-DYNA .k file to extract part thickness information.

    Args:
        k_file_path: Path to .k file.

    Returns:
        Dictionary mapping part ID to thickness.
    """

    part_to_section = {}
    section_thickness = {}

    with open(k_file_path, "r") as f:
        lines = [
            line.strip() for line in f if line.strip() and not line.startswith("$")
        ]

    i = 0
    while i < len(lines):
        line = lines[i]
        if "*PART" in line.upper():
            # After *PART:
            # i+1 = part name (skip)
            # i+2 = part id, section id, material id
            if i + 2 < len(lines):
                tokens = lines[i + 2].split()
                if len(tokens) >= 2:
                    part_id = int(tokens[0])
                    section_id = int(tokens[1])
                    part_to_section[part_id] = section_id
            i += 3
        elif "*SECTION_SHELL" in line.upper():
            # Multiple sections can be defined under one *SECTION_SHELL keyword
            # Each section has two lines: header line and thickness line
            i += 1  # Skip the *SECTION_SHELL line
            while i < len(lines) and not lines[i].startswith("*"):
                # Check if this line looks like a section header (starts with a number)
                if i < len(lines) and lines[i].strip() and lines[i][0].isdigit():
                    header_line = lines[i]
                    thickness_line = lines[i + 1] if i + 1 < len(lines) else ""

                    # Extract section ID from header line (first number)
                    header_tokens = header_line.split()
                    if len(header_tokens) >= 1:
                        try:  # noqa: PERF203
                            section_id = int(header_tokens[0])
                        except ValueError:
                            section_id = None
                    else:
                        section_id = None

                    # Extract thickness values from thickness line
                    thickness_values = []
                    thickness_tokens = thickness_line.split()
                    for t in thickness_tokens:
                        try:
                            thickness_values.append(float(t))
                        except ValueError:  # noqa: PERF203
                            thickness_values.append(0.0)
                    # Calculate average thickness (ignore zeros)
                    non_zero_thicknesses = [t for t in thickness_values if t > 0.0]
                    if non_zero_thicknesses:
                        thickness = sum(non_zero_thicknesses) / len(
                            non_zero_thicknesses
                        )
                    elif thickness_values:
                        thickness = sum(thickness_values) / len(thickness_values)
                    else:
                        thickness = 0.0
                    if section_id is not None:
                        section_thickness[section_id] = thickness

                    i += 2  # Skip both header and thickness lines
                else:
                    i += 1
        else:
            i += 1

    part_thickness = {
        pid: section_thickness.get(sid, 0.0) for pid, sid in part_to_section.items()
    }
    return part_thickness


def compute_node_thickness(
    mesh_connectivity: np.ndarray,
    part_ids: np.ndarray,
    part_thickness_map: dict[int, float],
    actual_part_ids: np.ndarray = None,
) -> np.ndarray:
    """
    Compute thickness for each node based on elements connected to it.

    Args:
        mesh_connectivity: Element connectivity array (num_elements, num_nodes_per_element)
        part_ids: Part IDs for each element (num_elements)
        part_thickness_map: Mapping from part ID to thickness
        actual_part_ids: Actual part IDs if available (num_parts)

    Returns:
        node_thickness: Array of thickness values for each node (num_nodes)
    """
    # Create mapping from part index to actual part ID
    if actual_part_ids is not None:
        part_index_to_id = {
            i: actual_part_id
            for i, actual_part_id in enumerate(actual_part_ids)
            if i > 0  # Skip index 0
        }
    else:
        sorted_part_ids = sorted(part_thickness_map.keys())
        part_index_to_id = {i: part_id for i, part_id in enumerate(sorted_part_ids, 1)}

    # Get element thickness
    element_thickness = np.zeros(len(part_ids))
    for i, part_index in enumerate(part_ids):
        actual_part_id = part_index_to_id.get(part_index)
        if actual_part_id is not None:
            thickness = part_thickness_map.get(actual_part_id, 0.0)
            element_thickness[i] = thickness

    # Find maximum node index to initialize node thickness array
    max_node_idx = 0
    for element in mesh_connectivity:
        max_node_idx = max(max_node_idx, max(element))

    node_thickness = np.zeros(max_node_idx + 1)
    node_thickness_count = np.zeros(max_node_idx + 1)

    # Accumulate thickness from all elements connected to each node
    for i, element in enumerate(mesh_connectivity):
        thickness = element_thickness[i]
        for node_idx in element:
            node_thickness[node_idx] += thickness
            node_thickness_count[node_idx] += 1

    # Average thickness for nodes connected to multiple elements
    for i in range(len(node_thickness)):
        if node_thickness_count[i] > 0:
            node_thickness[i] /= node_thickness_count[i]

    return node_thickness
