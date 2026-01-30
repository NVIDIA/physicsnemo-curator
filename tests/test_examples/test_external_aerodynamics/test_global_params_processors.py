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
from constants import ModelType
from external_aero_global_params_data_processors import (
    default_global_params_processing_for_external_aerodynamics,
    process_global_params,
    process_global_params_hlpw,
)
from schemas import (
    ExternalAerodynamicsExtractedDataInMemory,
    ExternalAerodynamicsMetadata,
)


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    return ExternalAerodynamicsMetadata(
        filename="test_sample",
        dataset_type=ModelType.COMBINED,
    )


@pytest.fixture
def sample_data(sample_metadata):
    """Create sample extracted data container for testing."""
    return ExternalAerodynamicsExtractedDataInMemory(
        metadata=sample_metadata,
    )


class TestDefaultGlobalParamsProcessing:
    """Test the default_global_params_processing_for_external_aerodynamics function."""

    def test_scalar_parameters(self, sample_data):
        """Test processing with scalar parameters only."""
        global_parameters = {
            "air_density": {"type": "scalar", "reference": 1.205},
            "pressure": {"type": "scalar", "reference": 101325.0},
        }

        result = default_global_params_processing_for_external_aerodynamics(
            sample_data, global_parameters
        )

        # Check that global_params_reference is set
        assert result.global_params_reference is not None

        # Check dtype is float32
        assert result.global_params_reference.dtype == np.float32

        # Check values are flattened correctly: [1.205, 101325.0]
        expected = np.array([1.205, 101325.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result.global_params_reference, expected)

    def test_vector_parameters(self, sample_data):
        """Test processing with vector parameters only.

        Vectors can have variable length:
        - [30.0] represents velocity in x-direction only
        - [30.0, 30.0] represents 2D velocity
        - [30.0, 30.0, 30.0] represents full 3D velocity
        """
        global_parameters = {
            "inlet_velocity_1d": {"type": "vector", "reference": [30.0]},
            "inlet_velocity_2d": {"type": "vector", "reference": [25.0, 10.0]},
        }

        result = default_global_params_processing_for_external_aerodynamics(
            sample_data, global_parameters
        )

        # Check dtype is float32
        assert result.global_params_reference.dtype == np.float32

        # Vectors are flattened in order: [30.0] + [25.0, 10.0] = [30.0, 25.0, 10.0]
        expected = np.array([30.0, 25.0, 10.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result.global_params_reference, expected)

    def test_mixed_parameters(self, sample_data):
        """Test processing with mixed scalar and vector parameters.

        Mirrors drivaerml.yaml structure:
        - inlet_velocity: vector
        - air_density: scalar
        - pressure: scalar
        """
        global_parameters = {
            "inlet_velocity": {"type": "vector", "reference": [30.0]},
            "air_density": {"type": "scalar", "reference": 1.205},
            "pressure": {"type": "scalar", "reference": 101325.0},
        }

        result = default_global_params_processing_for_external_aerodynamics(
            sample_data, global_parameters
        )

        # Check dtype is float32
        assert result.global_params_reference.dtype == np.float32

        # Flattened in order: [30.0] + 1.205 + 101325.0 = [30.0, 1.205, 101325.0]
        expected = np.array([30.0, 1.205, 101325.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result.global_params_reference, expected)

    def test_invalid_type_raises_error(self, sample_data):
        """Test that unsupported parameter type raises ValueError."""
        global_parameters = {
            "temperature": {"type": "tensor", "reference": [[1, 2], [3, 4]]},
        }

        with pytest.raises(ValueError, match="unsupported type"):
            default_global_params_processing_for_external_aerodynamics(
                sample_data, global_parameters
            )


class TestProcessGlobalParams:
    """Test the process_global_params function."""

    def test_copies_reference_to_values(self, sample_data):
        """Test that process_global_params copies reference to values.

        This is the default behavior when simulation conditions match reference.
        """
        global_parameters = {
            "inlet_velocity": {"type": "vector", "reference": [30.0]},
            "air_density": {"type": "scalar", "reference": 1.205},
        }

        # First, set up global_params_reference
        data = default_global_params_processing_for_external_aerodynamics(
            sample_data, global_parameters
        )

        # Then apply process_global_params
        result = process_global_params(data, global_parameters)

        # Check that global_params_values is set
        assert result.global_params_values is not None

        # Check that values equal reference
        np.testing.assert_array_equal(
            result.global_params_values, result.global_params_reference
        )

        # Verify it's a copy, not the same object
        assert result.global_params_values is not result.global_params_reference


class TestProcessGlobalParamsHLPW:
    """Test the process_global_params_hlpw function."""

    @pytest.mark.parametrize(
        "filename, expected_aoa",
        [
            ("geo_LHC001_AoA_16", 16.0),  # Two-digit AoA
            ("geo_LHC002_AoA_4", 4.0),  # Single-digit AoA
        ],
    )
    def test_extracts_aoa_from_filename(self, filename, expected_aoa):
        """Test that AoA is correctly extracted from HLPW filename patterns."""
        metadata = ExternalAerodynamicsMetadata(
            filename=filename,
            dataset_type=ModelType.COMBINED,
        )
        data = ExternalAerodynamicsExtractedDataInMemory(metadata=metadata)

        global_parameters = {
            "AoA": {"type": "scalar", "reference": 22.0},
        }

        result = process_global_params_hlpw(data, global_parameters)

        # Check extracted value matches expected (not the reference 22.0)
        assert result.global_params_values is not None
        expected = np.array([expected_aoa], dtype=np.float32)
        np.testing.assert_array_equal(result.global_params_values, expected)

    def test_missing_aoa_pattern_raises_error(self):
        """Test that missing AoA pattern in filename raises ValueError."""
        metadata = ExternalAerodynamicsMetadata(
            filename="geo_LHC001_no_angle_info",
            dataset_type=ModelType.COMBINED,
        )
        data = ExternalAerodynamicsExtractedDataInMemory(metadata=metadata)

        global_parameters = {
            "AoA": {"type": "scalar", "reference": 22.0},
        }

        with pytest.raises(ValueError, match="AoA pattern not found"):
            process_global_params_hlpw(data, global_parameters)
