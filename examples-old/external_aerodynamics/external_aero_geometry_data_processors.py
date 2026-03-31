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

import logging

import numpy as np
from external_aero_utils import to_float32
from schemas import ExternalAerodynamicsExtractedDataInMemory

logging.basicConfig(
    format="%(asctime)s - Process %(process)d - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def default_geometry_processing_for_external_aerodynamics(
    data: ExternalAerodynamicsExtractedDataInMemory,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Default geometry processing for External Aerodynamics."""

    data.stl_coordinates = data.stl_polydata.points
    data.stl_faces = (
        np.array(data.stl_polydata.faces).reshape((-1, 4))[:, 1:].astype(np.int32)
    ).flatten()  # Assuming triangular elements
    data.stl_areas = data.stl_polydata.compute_cell_sizes(
        length=False, area=True, volume=False
    )
    data.stl_areas = np.array(data.stl_areas.cell_data["Area"])
    data.stl_centers = np.array(data.stl_polydata.cell_centers().points)

    # Update metadata
    bounds = data.stl_polydata.bounds
    data.metadata.x_bound = bounds[0:2]  # xmin, xmax
    data.metadata.y_bound = bounds[2:4]  # ymin, ymax
    data.metadata.z_bound = bounds[4:6]  # zmin, zmax
    data.metadata.num_points = len(data.stl_polydata.points)
    data.metadata.num_faces = len(data.stl_faces)

    return data


def filter_geometry_invalid_faces(
    data: ExternalAerodynamicsExtractedDataInMemory,
    tolerance: float = 1e-6,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """
    Filter out invalid geometry faces based on area criteria.

    Removes faces where:
    - Area is <= tolerance (zero or negative area)

    After filtering faces, removes unused vertices and reindexes the face array.
    Updates metadata including bounds, num_points, and num_faces.

    Args:
        data: External aerodynamics data with geometry information
        tolerance: Minimum valid value for face area (default: 1e-6)

    Returns:
        Data with invalid faces filtered out and vertices reindexed
    """

    if data.stl_areas is None or len(data.stl_areas) == 0:
        logger.warning("STL areas are empty, skipping geometry filter")
        return data

    if data.stl_faces is None or len(data.stl_faces) == 0:
        logger.warning("STL faces are empty, skipping geometry filter")
        return data

    if data.stl_coordinates is None or len(data.stl_coordinates) == 0:
        logger.warning("STL coordinates are empty, skipping geometry filter")
        return data

    # Calculate initial counts
    n_total_faces = len(data.stl_areas)
    n_total_vertices = len(data.stl_coordinates)

    # Create validity mask for faces
    valid_area_mask = data.stl_areas > tolerance

    # Count filtered faces
    n_valid_faces = valid_area_mask.sum()
    n_filtered_faces = n_total_faces - n_valid_faces

    # Log filtering statistics
    if n_filtered_faces == 0:
        logger.info(
            f"No invalid geometry faces found (all {n_total_faces} faces are valid)"
        )
        return data

    if n_valid_faces == 0:
        logger.error(
            f"All {n_total_faces} geometry faces filtered out! "
            f"Check tolerance ({tolerance}) and data quality."
        )
        raise ValueError("Filtering removed all geometry faces")

    logger.info(
        f"Filtered {n_filtered_faces} invalid geometry faces "
        f"({n_filtered_faces / n_total_faces * 100:.2f}% of {n_total_faces} total faces):"
    )
    logger.info(f"  - {n_filtered_faces} faces with area <= {tolerance}")

    # Reshape faces to 2D for easier processing (each row is a triangle)
    stl_faces_2d = data.stl_faces.reshape(-1, 3)

    # Filter face-level data
    filtered_faces = stl_faces_2d[valid_area_mask]
    data.stl_areas = data.stl_areas[valid_area_mask]
    data.stl_centers = data.stl_centers[valid_area_mask]

    # Find still-referenced vertices
    used_vertex_indices = np.unique(filtered_faces)
    n_used_vertices = len(used_vertex_indices)
    n_removed_vertices = n_total_vertices - n_used_vertices

    # Create old->new index mapping
    # Initialize with -1 (unused vertices)
    vertex_map = np.full(n_total_vertices, -1, dtype=np.int32)
    # Map used vertices to new consecutive indices
    vertex_map[used_vertex_indices] = np.arange(n_used_vertices, dtype=np.int32)

    # Remap face indices to new vertex indices
    data.stl_faces = vertex_map[filtered_faces].flatten()

    # Filter vertices to only keep used ones
    data.stl_coordinates = data.stl_coordinates[used_vertex_indices]

    # Update metadata with new counts
    data.metadata.num_faces = len(
        data.stl_faces
    )  # Number of indices in flattened array
    data.metadata.num_points = len(data.stl_coordinates)

    # Update bounds based on remaining vertices
    if len(data.stl_coordinates) > 0:
        x_coords = data.stl_coordinates[:, 0]
        y_coords = data.stl_coordinates[:, 1]
        z_coords = data.stl_coordinates[:, 2]

        data.metadata.x_bound = (np.min(x_coords), np.max(x_coords))
        data.metadata.y_bound = (np.min(y_coords), np.max(y_coords))
        data.metadata.z_bound = (np.min(z_coords), np.max(z_coords))

    logger.info(f"  - Removed {n_removed_vertices} unused vertices")
    logger.info(
        f"  - {n_valid_faces} valid faces and {n_used_vertices} vertices remaining"
    )

    return data


def update_geometry_data_to_float32(
    data: ExternalAerodynamicsExtractedDataInMemory,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Update geometry data to float32."""

    data.stl_coordinates = to_float32(data.stl_coordinates)
    data.stl_centers = to_float32(data.stl_centers)
    data.stl_areas = to_float32(data.stl_areas)
    # data.stl_faces will be left as is (np.int32)

    return data
