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

import tempfile
from pathlib import Path

import numpy as np
import pytest
from crash_data_processors import (
    build_edges_from_mesh_connectivity,
    compute_node_thickness,
    compute_node_type,
    parse_k_file,
)


def test_build_edges_single_quad():
    """Test edge building from single quad."""
    mesh_connectivity = [[0, 1, 2, 3]]
    edges = build_edges_from_mesh_connectivity(mesh_connectivity)

    expected_edges = {(0, 1), (1, 2), (2, 3), (0, 3)}
    assert edges == expected_edges, f"Expected {expected_edges}, got {edges}"


def test_build_edges_triangle():
    """Test edge building from triangle."""
    mesh_connectivity = [[0, 1, 2]]
    edges = build_edges_from_mesh_connectivity(mesh_connectivity)

    expected_edges = {(0, 1), (1, 2), (0, 2)}
    assert edges == expected_edges, f"Expected {expected_edges}, got {edges}"


def test_build_edges_shared():
    """Test that shared edges between cells are not duplicated."""
    # Two quads sharing edge (1, 2)
    mesh_connectivity = [
        [0, 1, 2, 3],
        [1, 4, 5, 2],
    ]
    edges = build_edges_from_mesh_connectivity(mesh_connectivity)

    # Should have 7 unique edges (4 + 4 - 1 shared)
    assert len(edges) == 7, f"Expected 7 unique edges, got {len(edges)}"

    # Shared edge should appear once
    assert (1, 2) in edges, "Shared edge (1, 2) should be present"


def test_build_edges_empty():
    """Test edge building with empty connectivity."""
    mesh_connectivity = []
    edges = build_edges_from_mesh_connectivity(mesh_connectivity)

    assert len(edges) == 0, "Empty connectivity should produce no edges"


def test_compute_node_type_all_wall():
    """Test node type computation when all nodes are walls."""
    # 3 timesteps, 4 nodes, no displacement
    pos_raw = np.zeros((3, 4, 3), dtype=np.float32)

    node_type = compute_node_type(pos_raw, threshold=1.0)

    # All nodes should be classified as wall (type=1)
    assert np.all(node_type == 1), "All stationary nodes should be classified as wall"


def test_compute_node_type_all_structure():
    """Test node type computation when all nodes are structure."""
    # 3 timesteps, 4 nodes, large displacement
    pos_raw = np.array(
        [
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
            [[2, 0, 0], [3, 0, 0], [2, 1, 0], [3, 1, 0]],  # Large displacement
            [[4, 0, 0], [5, 0, 0], [4, 1, 0], [5, 1, 0]],
        ],
        dtype=np.float32,
    )

    node_type = compute_node_type(pos_raw, threshold=1.0)

    # All nodes should be classified as structure (type=0)
    assert np.all(node_type == 0), "All moving nodes should be classified as structure"


def test_compute_node_type_mixed():
    """Test node type computation with mixed wall and structure nodes."""
    # First 2 nodes stationary (wall), last 2 nodes moving (structure)
    pos_raw = np.array(
        [
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
            [[0, 0, 0], [1, 0, 0], [3, 0, 0], [4, 0, 0]],  # Last 2 move
            [[0, 0, 0], [1, 0, 0], [4, 0, 0], [5, 0, 0]],  # Last 2 move more
        ],
        dtype=np.float32,
    )

    node_type = compute_node_type(pos_raw, threshold=1.0)

    assert node_type[0] == 1, "Node 0 should be wall"
    assert node_type[1] == 1, "Node 1 should be wall"
    assert node_type[2] == 0, "Node 2 should be structure"
    assert node_type[3] == 0, "Node 3 should be structure"


def test_compute_node_type_threshold():
    """Test that threshold correctly separates wall from structure."""
    # Node moves by exactly 0.5 units
    pos_raw = np.array(
        [
            [[0, 0, 0]],
            [[0.5, 0, 0]],
        ],
        dtype=np.float32,
    )

    # With threshold=1.0, should be wall
    node_type = compute_node_type(pos_raw, threshold=1.0)
    assert node_type[0] == 1, "Node with displacement < threshold should be wall"

    # With threshold=0.3, should be structure
    node_type = compute_node_type(pos_raw, threshold=0.3)
    assert node_type[0] == 0, "Node with displacement > threshold should be structure"


@pytest.fixture
def temp_k_file():
    """Create a temporary .k file for testing."""
    k_content = """$# LS-DYNA Keyword file
*PART
Part 1
1   1   1
*SECTION_SHELL
1
2.5
*PART
Part 2
2   2   1
*SECTION_SHELL
2
1.5
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".k", delete=False) as f:
        f.write(k_content)
        temp_path = f.name

    yield Path(temp_path)
    Path(temp_path).unlink(missing_ok=True)


def test_parse_k_file(temp_k_file):
    """Test parsing of .k file for thickness information."""
    part_thickness_map = parse_k_file(temp_k_file)

    # Should have 2 parts
    assert (
        len(part_thickness_map) == 2
    ), f"Expected 2 parts, got {len(part_thickness_map)}"

    # Check thickness values
    assert (
        part_thickness_map[1] == 2.5
    ), f"Expected part 1 thickness 2.5, got {part_thickness_map[1]}"
    assert (
        part_thickness_map[2] == 1.5
    ), f"Expected part 2 thickness 1.5, got {part_thickness_map[2]}"


def test_compute_node_thickness_basic():
    """Test node thickness computation from element thickness."""
    # Simple mesh: 2 elements, 6 nodes (sharing 2 nodes)
    mesh_connectivity = [
        [0, 1, 2, 3],  # Element 0 (quad)
        [1, 4, 5, 2],  # Element 1 (quad, shares nodes 1 and 2)
    ]

    part_ids = np.array([1, 2])  # Element 0 is part 1, element 1 is part 2
    part_thickness_map = {1: 2.0, 2: 1.0}
    actual_part_ids = np.array([0, 1, 2])  # Index 0 unused, 1->part1, 2->part2

    node_thickness = compute_node_thickness(
        mesh_connectivity, part_ids, part_thickness_map, actual_part_ids
    )

    # Node 0 and 3 only in element 0: thickness = 2.0
    assert (
        node_thickness[0] == 2.0
    ), f"Expected thickness 2.0 for node 0, got {node_thickness[0]}"
    assert (
        node_thickness[3] == 2.0
    ), f"Expected thickness 2.0 for node 3, got {node_thickness[3]}"

    # Nodes 1 and 2 in both elements: thickness = (2.0 + 1.0) / 2 = 1.5
    assert (
        node_thickness[1] == 1.5
    ), f"Expected thickness 1.5 for node 1, got {node_thickness[1]}"
    assert (
        node_thickness[2] == 1.5
    ), f"Expected thickness 1.5 for node 2, got {node_thickness[2]}"

    # Nodes 4 and 5 only in element 1: thickness = 1.0
    assert (
        node_thickness[4] == 1.0
    ), f"Expected thickness 1.0 for node 4, got {node_thickness[4]}"
    assert (
        node_thickness[5] == 1.0
    ), f"Expected thickness 1.0 for node 5, got {node_thickness[5]}"


def test_compute_node_thickness_no_actual_ids():
    """Test node thickness computation without actual_part_ids."""
    mesh_connectivity = [[0, 1, 2]]
    part_ids = np.array([1])
    part_thickness_map = {10: 3.0}  # Part ID 10 (will be mapped to index 1)

    node_thickness = compute_node_thickness(
        mesh_connectivity, part_ids, part_thickness_map, actual_part_ids=None
    )

    # All nodes should have thickness 3.0
    assert node_thickness[0] == 3.0
    assert node_thickness[1] == 3.0
    assert node_thickness[2] == 3.0
