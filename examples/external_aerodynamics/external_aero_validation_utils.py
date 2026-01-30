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

"""
Common validation utilities for External Aerodynamics data quality checks.
"""

import logging
from typing import Tuple

import numpy as np

logging.basicConfig(
    format="%(asctime)s - Process %(process)d - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def mean_std_sampling(
    fields: np.ndarray, tolerance: float = 7.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample fields within mean ± tolerance * std to identify statistical outliers.

    Returns indices of values within mean ± tolerance * std.
    This removes extreme outliers from consideration.

    Args:
        fields: Array of shape (n_cells, n_features) to sample
        tolerance: Number of standard deviations for outlier detection (default: 7.0)

    Returns:
        Tuple of (valid_mask, sampled_fields)
        - valid_mask: Boolean mask of shape (n_cells,) indicating valid cells
        - sampled_fields: Filtered array containing only statistically valid cells
    """
    if fields.size == 0:
        logger.warning("Empty field array provided to mean_std_sampling")
        return np.array([], dtype=bool), np.array([])

    # Compute statistics per field component
    field_mean = np.mean(fields, axis=0)
    field_std = np.std(fields, axis=0)

    # Check if all components are within range
    within_range = np.abs(fields - field_mean) <= tolerance * field_std
    valid_mask = np.all(within_range, axis=1)

    sampled_fields = fields[valid_mask]

    return valid_mask, sampled_fields


def check_field_statistics(
    fields: np.ndarray, field_type: str = "unknown", tolerance: float = 7.0
) -> Tuple[bool, np.ndarray, np.ndarray, int, int]:
    """
    Check field statistics and identify if data is acceptable.

    Performs statistical outlier filtering and computes min/max on sampled data.

    Args:
        fields: Array of shape (n_cells, n_features) to check
        field_type: Descriptive name for logging (e.g., "volume", "surface")
        tolerance: Number of standard deviations for outlier detection (default: 7.0)

    Returns:
        Tuple of (is_invalid, vmax, vmin, n_filtered, n_total):
        - is_invalid: True if sample should be rejected due to statistical issues
        - vmax: Maximum absolute values per component (after outlier filtering)
        - vmin: Minimum values per component (after outlier filtering)
        - n_filtered: Number of cells filtered out as outliers
        - n_total: Total number of cells before filtering
    """
    if fields.size == 0:
        logger.warning(f"Empty field for {field_type}")
        return True, np.array([]), np.array([]), 0, 0

    # Apply statistical outlier filtering
    valid_mask, sampled_fields = mean_std_sampling(fields, tolerance)

    n_total = len(fields)
    n_valid = valid_mask.sum()
    n_filtered = n_total - n_valid

    if n_valid == 0:
        return True, np.array([]), np.array([]), n_filtered, n_total

    # Compute max/min on statistically valid data
    vmax = np.amax(np.abs(sampled_fields), axis=0)
    vmin = np.amin(sampled_fields, axis=0)

    return False, vmax, vmin, n_filtered, n_total


def check_volume_physics_bounds(
    vmax: np.ndarray,
    velocity_max: float = 3.5,
    pressure_max: float = 4.0,
) -> Tuple[bool, str]:
    """
    Check if volume field values exceed physics-based thresholds.

    Assumes fields are [u, v, w, p, ...] after non-dimensionalization.

    Args:
        vmax: Maximum absolute values per field component
        velocity_max: Maximum allowed non-dimensionalized velocity (default: 3.5)
        pressure_max: Maximum allowed non-dimensionalized pressure (default: 4.0)

    Returns:
        Tuple of (exceeds_bounds, error_message):
        - exceeds_bounds: True if any threshold is exceeded
        - error_message: Descriptive error message if bounds exceeded, empty string otherwise
    """
    if vmax.shape[0] < 4:
        return False, ""

    # Check velocity components (first 3 components: u, v, w)
    if vmax[0] > velocity_max or vmax[1] > velocity_max or vmax[2] > velocity_max:
        return True, (
            f"Velocity exceeds threshold: "
            f"vmax=[{vmax[0]:.3f}, {vmax[1]:.3f}, {vmax[2]:.3f}] > {velocity_max}"
        )

    # Check pressure component (4th component)
    if vmax[3] > pressure_max:
        return (
            True,
            f"Pressure exceeds threshold: vmax[p]={vmax[3]:.3f} > {pressure_max}",
        )

    return False, ""


def check_surface_physics_bounds(
    vmax: np.ndarray,
    pressure_max: float = 4.0,
) -> Tuple[bool, str]:
    """
    Check if surface field values exceed physics-based thresholds.

    Assumes first component is pressure after non-dimensionalization.

    Args:
        vmax: Maximum absolute values per field component
        pressure_max: Maximum allowed non-dimensionalized pressure (default: 4.0)

    Returns:
        Tuple of (exceeds_bounds, error_message):
        - exceeds_bounds: True if threshold is exceeded
        - error_message: Descriptive error message if bounds exceeded, empty string otherwise
    """
    if vmax.shape[0] < 1:
        return False, ""

    # Check pressure (first component)
    if vmax[0] > pressure_max:
        return (
            True,
            f"Pressure exceeds threshold: vmax[p]={vmax[0]:.3f} > {pressure_max}",
        )

    return False, ""
