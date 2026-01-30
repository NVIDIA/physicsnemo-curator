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

import numpy as np
import pytest
from data_transformations import CrashDataTransformation
from schemas import CrashExtractedDataInMemory, CrashMetadata

from physicsnemo_curator.etl.processing_config import ProcessingConfig


@pytest.fixture
def simple_crash_data():
    """Create simple crash data with wall and structure nodes."""
    # 6 nodes: 2 wall nodes (0, 1) with no displacement, 4 structure nodes (2-5) with displacement
    pos_raw = np.array(
        [
            # t0: all at reference position
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [2, 1, 0], [3, 1, 0]],
            # t1: nodes 0,1 stationary (wall), nodes 2-5 move
            [[0, 0, 0], [1, 0, 0], [2.5, 0, 0], [3.5, 0, 0], [2.5, 1, 0], [3.5, 1, 0]],
            # t2: nodes 0,1 still stationary, nodes 2-5 move more
            [[0, 0, 0], [1, 0, 0], [3.0, 0, 0], [4.0, 0, 0], [3.0, 1, 0], [4.0, 1, 0]],
        ],
        dtype=np.float32,
    )

    # Mesh: 2 quads, first uses wall nodes, second uses only structure nodes
    mesh_connectivity = [
        [0, 1, 3, 2],  # Uses nodes 0, 1 (wall) and 2, 3 (structure)
        [2, 3, 5, 4],  # Uses only structure nodes 2, 3, 4, 5
    ]

    node_thickness = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32)

    return CrashExtractedDataInMemory(
        metadata=CrashMetadata(filename="test_run"),
        pos_raw=pos_raw,
        mesh_connectivity=mesh_connectivity,
        node_thickness=node_thickness,
    )


def test_transform_filters_wall_nodes(simple_crash_data):
    """Test that transformation filters out wall nodes."""
    cfg = ProcessingConfig(num_processes=1)
    transform = CrashDataTransformation(cfg, wall_threshold=1.0)

    result = transform.transform(simple_crash_data)

    # Should filter out 2 wall nodes (0, 1), keeping 4 structure nodes
    assert (
        result.filtered_pos_raw.shape[1] == 4
    ), f"Expected 4 nodes after filtering, got {result.filtered_pos_raw.shape[1]}"


def test_transform_remaps_connectivity(simple_crash_data):
    """Test that connectivity is correctly remapped after filtering."""
    cfg = ProcessingConfig(num_processes=1)
    transform = CrashDataTransformation(cfg, wall_threshold=1.0)

    result = transform.transform(simple_crash_data)

    # After filtering nodes 0, 1, original nodes [2,3,4,5] become [0,1,2,3]
    # Original connectivity: [[0,1,3,2], [2,3,5,4]]
    # After filtering, first cell loses nodes 0,1: only [3,2] = [1,0] in new indexing
    # This cell has < 3 nodes, so it gets dropped
    # Second cell: [2,3,5,4] = [0,1,3,2] in new indexing

    assert (
        len(result.filtered_mesh_connectivity) >= 1
    ), "Should have at least one cell after filtering"

    # All node indices should be in range [0, 3]
    for cell in result.filtered_mesh_connectivity:
        assert all(0 <= idx < 4 for idx in cell), f"Cell indices out of range: {cell}"


def test_transform_builds_edges(simple_crash_data):
    """Test that edges are built from filtered connectivity."""
    cfg = ProcessingConfig(num_processes=1)
    transform = CrashDataTransformation(cfg, wall_threshold=1.0)

    result = transform.transform(simple_crash_data)

    # Should have edges
    assert len(result.edges) > 0, "Should have edges after transformation"

    # All edge indices should be in valid range
    for edge in result.edges:
        assert (
            0 <= edge[0] < result.filtered_pos_raw.shape[1]
        ), f"Edge node {edge[0]} out of range"
        assert (
            0 <= edge[1] < result.filtered_pos_raw.shape[1]
        ), f"Edge node {edge[1]} out of range"


def test_transform_preserves_timesteps(simple_crash_data):
    """Test that number of timesteps is preserved."""
    cfg = ProcessingConfig(num_processes=1)
    transform = CrashDataTransformation(cfg, wall_threshold=1.0)

    original_timesteps = simple_crash_data.pos_raw.shape[0]
    result = transform.transform(simple_crash_data)

    assert (
        result.filtered_pos_raw.shape[0] == original_timesteps
    ), f"Expected {original_timesteps} timesteps, got {result.filtered_pos_raw.shape[0]}"


def test_transform_filters_thickness(simple_crash_data):
    """Test that thickness is correctly filtered for remaining nodes."""
    cfg = ProcessingConfig(num_processes=1)
    transform = CrashDataTransformation(cfg, wall_threshold=1.0)

    result = transform.transform(simple_crash_data)

    # Thickness should match number of filtered nodes
    assert (
        len(result.filtered_node_thickness) == result.filtered_pos_raw.shape[1]
    ), "Thickness array size should match filtered node count"


def test_transform_no_cells_error():
    """Test that transformation raises error when all cells are filtered."""
    # Create data where all nodes are wall nodes
    pos_raw = np.zeros((3, 4, 3), dtype=np.float32)
    mesh_connectivity = [[0, 1, 2, 3]]
    node_thickness = np.ones(4, dtype=np.float32)

    data = CrashExtractedDataInMemory(
        metadata=CrashMetadata(filename="test_run"),
        pos_raw=pos_raw,
        mesh_connectivity=mesh_connectivity,
        node_thickness=node_thickness,
    )

    cfg = ProcessingConfig(num_processes=1)
    transform = CrashDataTransformation(cfg, wall_threshold=1.0)

    with pytest.raises(ValueError, match="No cells left after filtering"):
        transform.transform(data)


def test_transform_aggressive_threshold():
    """Test transformation with very aggressive wall threshold."""
    # Create data with small displacement
    # Node 0 has zero displacement (wall), nodes 1-3 have small displacement
    pos_raw = np.array(
        [
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
            [[0, 0, 0], [1.1, 0, 0], [2.1, 0, 0], [3.1, 0, 0]],  # Small displacement
        ],
        dtype=np.float32,
    )

    mesh_connectivity = [[0, 1, 2, 3]]
    node_thickness = np.ones(4, dtype=np.float32)

    data = CrashExtractedDataInMemory(
        metadata=CrashMetadata(filename="test_run"),
        pos_raw=pos_raw,
        mesh_connectivity=mesh_connectivity,
        node_thickness=node_thickness,
    )

    cfg = ProcessingConfig(num_processes=1)

    # With low threshold (0.05), nodes with displacement > 0.05 are kept
    # Node 0: displacement = 0.0 (filtered as wall)
    # Nodes 1-3: displacement = 0.1 > 0.05 (kept as structure)
    transform = CrashDataTransformation(cfg, wall_threshold=0.05)
    result = transform.transform(data)

    # 3 nodes should be kept (nodes 1, 2, 3)
    assert (
        result.filtered_pos_raw.shape[1] == 3
    ), "Nodes with displacement > threshold should be kept"


def test_transform_clears_raw_data(simple_crash_data):
    """Test that transformation clears raw data to save memory."""
    cfg = ProcessingConfig(num_processes=1)
    transform = CrashDataTransformation(cfg, wall_threshold=1.0)

    # Copy data since fixture is reused
    import copy

    data = copy.deepcopy(simple_crash_data)

    result = transform.transform(data)

    # Original data arrays should be cleared
    assert data.pos_raw is None, "Raw pos_raw should be cleared"
    assert data.mesh_connectivity is None, "Raw mesh_connectivity should be cleared"
    assert data.node_thickness is None, "Raw node_thickness should be cleared"

    # Result should have filtered data
    assert result.filtered_pos_raw is not None
    assert result.filtered_mesh_connectivity is not None
    assert result.filtered_node_thickness is not None
