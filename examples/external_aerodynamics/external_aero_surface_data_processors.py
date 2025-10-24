# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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
import warnings
from typing import Optional

import numpy as np

from examples.external_aerodynamics.constants import PhysicsConstants
from examples.external_aerodynamics.external_aero_utils import to_float32
from examples.external_aerodynamics.external_aero_validation_utils import (
    check_field_statistics,
    check_surface_physics_bounds,
)
from examples.external_aerodynamics.schemas import (
    ExternalAerodynamicsExtractedDataInMemory,
)

logging.basicConfig(
    format="%(asctime)s - Process %(process)d - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def default_surface_processing_for_external_aerodynamics(
    data: ExternalAerodynamicsExtractedDataInMemory,
    surface_variables: list[str],
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Default surface processing for External Aerodynamics."""

    cell_data = (data.surface_polydata.cell_data[k] for k in surface_variables)
    data.surface_fields = np.concatenate(
        [d if d.ndim > 1 else d[:, np.newaxis] for d in cell_data], axis=-1
    )
    data.surface_mesh_centers = np.array(data.surface_polydata.cell_centers().points)
    data.surface_normals = np.array(data.surface_polydata.cell_normals)
    data.surface_areas = data.surface_polydata.compute_cell_sizes(
        length=False, area=True, volume=False
    )
    data.surface_areas = np.array(data.surface_areas.cell_data["Area"])

    return data


def filter_invalid_surface_cells(
    data: ExternalAerodynamicsExtractedDataInMemory,
    tolerance: float = 1e-6,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """
    Filter out invalid surface cells based on area and normal criteria.

    Removes cells where:
    - Area is <= tolerance (zero or negative area)
    - Normal vector has L2-norm <= tolerance (degenerate normal)

    Args:
        data: External aerodynamics data with surface information
        tolerance: Minimum valid value for area and normal magnitude (default: 1e-6)

    Returns:
        Data with invalid cells filtered out
    """

    if data.surface_areas is None or len(data.surface_areas) == 0:
        logger.warning("Surface areas are empty, skipping filter")
        return data

    if data.surface_normals is None or len(data.surface_normals) == 0:
        logger.warning("Surface normals are empty, skipping filter")
        return data

    # Calculate initial count
    n_total = len(data.surface_areas)

    # Create validity masks
    valid_area_mask = data.surface_areas > tolerance
    normal_norms = np.linalg.norm(data.surface_normals, axis=1)
    valid_normal_mask = normal_norms > tolerance

    # Combine masks (both conditions must be true)
    valid_mask = valid_area_mask & valid_normal_mask

    # Count filtered cells
    n_valid = valid_mask.sum()
    n_filtered = n_total - n_valid
    n_area_filtered = (~valid_area_mask).sum()
    n_normal_filtered = (~valid_normal_mask).sum()

    # Log filtering statistics
    if n_filtered == 0:
        logger.info(f"No invalid surface cells found (all {n_total} cells are valid)")
        return data

    if n_valid == 0:
        logger.error(
            f"All {n_total} surface cells filtered out! "
            f"({n_area_filtered} due to area, {n_normal_filtered} due to normals). "
            "Check tolerance and data quality."
        )
        raise ValueError("Filtering removed all surface cells")

    logger.info(
        f"Filtered {n_filtered} invalid surface cells "
        f"({n_filtered/n_total*100:.2f}% of {n_total} total cells):"
    )
    logger.info(f"  - {n_area_filtered} cells with area <= {tolerance}")
    logger.info(f"  - {n_normal_filtered} cells with normal L2-norm <= {tolerance}")
    logger.info(f"  - {n_valid} valid cells remaining")

    # Apply filter to all surface arrays
    data.surface_mesh_centers = data.surface_mesh_centers[valid_mask]
    data.surface_normals = data.surface_normals[valid_mask]
    data.surface_areas = data.surface_areas[valid_mask]
    data.surface_fields = data.surface_fields[valid_mask]

    return data


def normalize_surface_normals(
    data: ExternalAerodynamicsExtractedDataInMemory,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Normalize surface normals."""

    if data.surface_normals.shape[0] == 0:
        logger.error(f"Surface normals are empty: {data.surface_normals}")
        return data

    # Normalize cell normals
    data.surface_normals = (
        data.surface_normals
        / np.linalg.norm(data.surface_normals, axis=1)[:, np.newaxis]
    )

    return data


def non_dimensionalize_surface_fields(
    data: ExternalAerodynamicsExtractedDataInMemory,
    air_density: float = PhysicsConstants.AIR_DENSITY,
    stream_velocity: float = PhysicsConstants.STREAM_VELOCITY,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Non-dimensionalize surface fields."""

    if data.surface_fields.shape[0] == 0:
        logger.error(f"Surface fields are empty: {data.surface_fields}")
        return data

    if air_density <= 0:
        logger.error(f"Air density must be > 0: {air_density}")
    if stream_velocity <= 0:
        logger.error(f"Stream velocity must be > 0: {stream_velocity}")

    # Non-dimensionalize surface fields
    data.surface_fields = data.surface_fields / (air_density * stream_velocity**2.0)

    return data


def update_surface_data_to_float32(
    data: ExternalAerodynamicsExtractedDataInMemory,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Update surface data to float32."""

    # Update processed surface data
    data.surface_mesh_centers = to_float32(data.surface_mesh_centers)
    data.surface_normals = to_float32(data.surface_normals)
    data.surface_areas = to_float32(data.surface_areas)
    data.surface_fields = to_float32(data.surface_fields)

    return data


def validate_surface_sample_quality(
    data: ExternalAerodynamicsExtractedDataInMemory,
    statistical_tolerance: float = 7.0,
    pressure_max: float = 4.0,
) -> Optional[ExternalAerodynamicsExtractedDataInMemory]:
    """
    Validate surface sample quality and reject entire sample if it fails checks.

    This validator checks:
    1. Statistical outliers: If all data points are beyond mean ± tolerance*std
    2. Physics bounds: If max non-dimensionalized pressure exceeds threshold

    Note: This should be applied AFTER non-dimensionalization.

    Args:
        data: External aerodynamics data with surface information
        statistical_tolerance: Number of standard deviations for outlier detection (default: 7.0)
        pressure_max: Maximum allowed non-dimensionalized pressure (default: 4.0)

    Returns:
        Data unchanged if valid, None if sample should be rejected
    """

    if data.surface_fields is None or len(data.surface_fields) == 0:
        logger.warning("Surface fields are empty, skipping validation")
        return data

    # 1. Check field statistics and perform statistical outlier filtering
    is_invalid, vmax, vmin = check_field_statistics(
        data.surface_fields, field_type="surface", tolerance=statistical_tolerance
    )

    if is_invalid:
        logger.error(
            "Sample rejected: Statistical outlier detection filtered all cells"
        )
        return None

    # 2. Check physics-based bounds
    exceeds_bounds, error_msg = check_surface_physics_bounds(
        vmax, pressure_max=pressure_max
    )

    if exceeds_bounds:
        logger.error(f"Sample rejected: {error_msg}")
        return None

    logger.info("Surface sample passed quality checks")
    return data


def decimate_mesh(
    data: ExternalAerodynamicsExtractedDataInMemory,
    algo: str = None,
    reduction: float = 0.0,
    **kwargs,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Decimate mesh using pyvista."""

    if reduction < 0:
        logger.error(f"Reduction must be >= 0: {reduction}")
        return data

    if not algo or reduction == 0:
        return data

    mesh = data.surface_polydata

    # Need point_data to interpolate target mesh node values.
    mesh = mesh.cell_data_to_point_data()
    # Decimation algos require tri-mesh.
    mesh = mesh.triangulate()
    match algo:
        case "decimate_pro":
            mesh = mesh.decimate_pro(reduction, **kwargs)
        case "decimate":
            if mesh.n_points > 400_000:
                warnings.warn("decimate algo may hang on meshes of size more than 400K")
            mesh = mesh.decimate(
                reduction,
                attribute_error=True,
                scalars=True,
                vectors=True,
                **kwargs,
            )
        case _:
            logger.error(f"Unsupported decimation algo {algo}")
            return data

    # Compute cell data.
    data.surface_polydata = mesh.point_data_to_cell_data()

    # Update metadata
    data.metadata.decimation_algo = algo
    data.metadata.decimation_reduction = reduction

    return data
