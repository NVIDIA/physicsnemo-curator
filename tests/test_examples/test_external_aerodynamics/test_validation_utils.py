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
from external_aero_validation_utils import (
    check_field_statistics,
    check_surface_physics_bounds,
    check_volume_physics_bounds,
    mean_std_sampling,
)


def test_mean_std_sampling_all_valid_data():
    """Test that all data points are valid."""
    fields = np.random.randn(100, 3) * 0.1  # Small std dev
    valid_mask, sampled_fields = mean_std_sampling(fields, tolerance=7.0)

    assert valid_mask.sum() == 100
    np.testing.assert_array_equal(sampled_fields, fields)


def test_mean_std_sampling_with_outliers():
    """Test that outliers are filtered."""
    fields = np.random.randn(100, 3)
    fields[0, :] = [100, 100, 100]  # Extreme outlier
    fields[1, :] = [-100, -100, -100]  # Extreme outlier
    valid_mask, sampled_fields = mean_std_sampling(fields, tolerance=7.0)
    assert valid_mask.sum() < 100
    assert not valid_mask[0]
    assert not valid_mask[1]
    assert sampled_fields.shape[0] == valid_mask.sum()


def test_mean_std_sampling_empty_array():
    """Test that empty array is handled correctly."""
    fields = np.array([]).reshape(0, 3)
    valid_mask, sampled_fields = mean_std_sampling(fields, tolerance=7.0)
    assert len(valid_mask) == 0
    assert len(sampled_fields) == 0


def test_mean_std_sampling_different_tolerances():
    """Test that different tolerances affect filtering."""
    fields = np.random.randn(100, 3)

    # Add a moderate outlier
    fields[0, :] = [5, 5, 5]

    # With high tolerance, should include more points
    valid_high, _ = mean_std_sampling(fields, tolerance=10.0)

    # With low tolerance, should filter more
    valid_low, _ = mean_std_sampling(fields, tolerance=2.0)

    assert valid_high.sum() >= valid_low.sum()


def test_check_field_statistics_valid_data():
    """Test with valid data."""
    fields = np.random.randn(100, 4) * 0.5

    is_invalid, vmax, vmin, n_filtered, n_total = check_field_statistics(
        fields, field_type="test", tolerance=7.0
    )

    assert not is_invalid
    assert vmax.shape[0] == 4
    assert vmin.shape[0] == 4
    assert np.all(vmax >= 0)  # vmax should be absolute values
    assert n_total == 100
    assert n_filtered >= 0
    assert n_filtered < n_total  # Should have some valid data


def test_check_field_statistics_all_outliers():
    """Test when all data points are outliers."""
    # Create data where everything is far from mean
    fields = np.array([[1000, 1000, 1000], [2000, 2000, 2000]])

    is_invalid, vmax, vmin, n_filtered, n_total = check_field_statistics(
        fields, field_type="test", tolerance=0.001  # Very tight tolerance
    )
    assert is_invalid
    assert n_total == 2
    assert n_filtered == 2  # All cells should be filtered


def test_check_field_statistics_empty_array():
    """Test with empty array."""
    fields = np.array([]).reshape(0, 3)

    is_invalid, vmax, vmin, n_filtered, n_total = check_field_statistics(
        fields, field_type="test", tolerance=7.0
    )

    assert is_invalid
    assert len(vmax) == 0
    assert len(vmin) == 0
    assert n_total == 0
    assert n_filtered == 0


def test_check_volume_physics_bounds_within_bounds():
    """Test with values within bounds."""
    vmax = np.array([2.0, 2.5, 2.8, 3.0])  # All within bounds

    exceeds_bounds, error_msg = check_volume_physics_bounds(
        vmax, velocity_max=3.5, pressure_max=4.0
    )

    assert not exceeds_bounds
    assert error_msg == ""


def test_check_volume_physics_bounds_velocity_exceeds_bounds():
    """Test when velocity exceeds bounds."""
    vmax = np.array([4.0, 2.0, 2.0, 3.0])  # First component exceeds

    exceeds_bounds, error_msg = check_volume_physics_bounds(
        vmax, velocity_max=3.5, pressure_max=4.0
    )

    assert exceeds_bounds
    assert "Velocity" in error_msg
    assert "4.000" in error_msg


def test_check_volume_physics_bounds_pressure_exceeds_bounds():
    """Test when pressure exceeds bounds."""
    vmax = np.array([2.0, 2.0, 2.0, 5.0])  # Pressure exceeds

    exceeds_bounds, error_msg = check_volume_physics_bounds(
        vmax, velocity_max=3.5, pressure_max=4.0
    )

    assert exceeds_bounds
    assert "Pressure" in error_msg
    assert "5.000" in error_msg


def test_check_volume_physics_bounds_multiple_components_exceed():
    """Test when multiple components exceed bounds."""
    vmax = np.array([4.0, 4.0, 2.0, 3.0])  # Two velocity components exceed

    exceeds_bounds, error_msg = check_volume_physics_bounds(
        vmax, velocity_max=3.5, pressure_max=4.0
    )

    assert exceeds_bounds
    assert "Velocity" in error_msg


def test_check_volume_physics_bounds_edge_case_exactly_at_threshold():
    """Test when values are exactly at threshold."""
    vmax = np.array([3.5, 3.5, 3.5, 4.0])  # Exactly at limits

    exceeds_bounds, error_msg = check_volume_physics_bounds(
        vmax, velocity_max=3.5, pressure_max=4.0
    )

    assert not exceeds_bounds
    assert error_msg == ""


def test_check_surface_physics_bounds_within_bounds():
    """Test with pressure within bounds."""
    vmax = np.array([3.0, 2.0, 1.5])  # Pressure is first component

    exceeds_bounds, error_msg = check_surface_physics_bounds(vmax, pressure_max=4.0)

    assert not exceeds_bounds
    assert error_msg == ""


def test_check_surface_physics_bounds_pressure_exceeds_bounds():
    """Test when pressure exceeds bounds."""
    vmax = np.array([5.0, 2.0, 1.5])  # Pressure exceeds

    exceeds_bounds, error_msg = check_surface_physics_bounds(vmax, pressure_max=4.0)

    assert exceeds_bounds
    assert "Pressure" in error_msg
    assert "5.000" in error_msg


def test_check_surface_physics_bounds_empty_array():
    """Test with empty array."""
    vmax = np.array([])

    exceeds_bounds, error_msg = check_surface_physics_bounds(vmax, pressure_max=4.0)

    assert not exceeds_bounds
    assert error_msg == ""


def test_check_surface_physics_bounds_edge_case_exactly_at_threshold():
    """Test when pressure is exactly at threshold."""
    vmax = np.array([4.0, 2.0, 1.5])  # Exactly at limit

    exceeds_bounds, error_msg = check_surface_physics_bounds(vmax, pressure_max=4.0)

    assert not exceeds_bounds
    assert error_msg == ""
