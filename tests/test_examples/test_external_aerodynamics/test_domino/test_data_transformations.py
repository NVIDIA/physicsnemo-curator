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

import warnings

import numpy as np
import pytest

from physicsnemo_curator.etl.processing_config import ProcessingConfig
from physicsnemo_curator.examples.external_aerodynamics.domino.constants import (
    ModelType,
)
from physicsnemo_curator.examples.external_aerodynamics.domino.data_transformations import (
    DoMINONumpyTransformation,
    DoMINOZarrTransformation,
)
from physicsnemo_curator.examples.external_aerodynamics.domino.schemas import (
    DoMINOExtractedDataInMemory,
    DoMINOMetadata,
    DoMINONumpyDataInMemory,
    DoMINOZarrDataInMemory,
    PreparedZarrArrayInfo,
)


@pytest.fixture
def sample_data():
    """Create sample DoMINO data for testing."""
    return DoMINOExtractedDataInMemory(
        metadata=DoMINOMetadata(
            filename="run_1234",
            dataset_type=ModelType.COMBINED,
            stream_velocity=30.0,
            air_density=1.205,
            x_bound=(0.0, 1.0),
            y_bound=(0.0, 1.0),
            z_bound=(0.0, 1.0),
            num_points=3,
            num_faces=3,
            decimation_reduction=0.5,
            decimation_algo="decimate_pro",
        ),
        stl_coordinates=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64),
        stl_centers=np.array([[0.5, 0.5, 0.5]], dtype=np.float64),
        stl_faces=np.array([[0, 1, 2]], dtype=np.int32),
        stl_areas=np.array([1.0], dtype=np.float64),
        surface_mesh_centers=np.array([[0, 0, 0]], dtype=np.float64),
        surface_normals=np.array([[0, 0, 1]], dtype=np.float64),
        surface_areas=np.array([1.0], dtype=np.float64),
        surface_fields=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        volume_mesh_centers=np.array([[0, 0, 0]], dtype=np.float64),
        volume_fields=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
    )


class TestDoMINONumpyTransformation:
    """Test the DoMINONumpyTransformation class."""

    def test_initialization(self):
        """Test initialization of NumPy transformation."""
        config = ProcessingConfig(num_processes=1)
        transform = DoMINONumpyTransformation(config)
        assert transform.config == config

    def test_transform(self, sample_data):
        """Test NumPy transformation of DoMINO data."""
        config = ProcessingConfig(num_processes=1)
        transform = DoMINONumpyTransformation(config)

        result = transform.transform(sample_data)
        assert isinstance(result, DoMINONumpyDataInMemory)

        # Check a couple of STL fields
        np.testing.assert_array_equal(
            result.stl_coordinates, sample_data.stl_coordinates
        )
        np.testing.assert_array_equal(result.stl_areas, sample_data.stl_areas)

        # Check a couple of surface fields
        np.testing.assert_array_equal(
            result.surface_mesh_centers, sample_data.surface_mesh_centers
        )
        np.testing.assert_array_equal(
            result.surface_normals, sample_data.surface_normals
        )

        # Check a couple of volume fields
        np.testing.assert_array_equal(
            result.volume_mesh_centers, sample_data.volume_mesh_centers
        )
        np.testing.assert_array_equal(result.volume_fields, sample_data.volume_fields)


class TestDoMINOZarrTransformation:
    """Test the DoMINOZarrTransformation class."""

    def test_initialization(self):
        """Test initialization of Zarr transformation."""
        config = ProcessingConfig(num_processes=1)
        transform = DoMINOZarrTransformation(config)
        assert transform.config == config
        assert transform.compressor.cname == "zstd"
        assert transform.compressor.clevel == 5

    def test_transform(self, sample_data):
        """Test Zarr transformation of DoMINO data."""
        config = ProcessingConfig(
            num_processes=1,
        )
        transform = DoMINOZarrTransformation(config)

        result = transform.transform(sample_data)

        # Check overall structure
        assert isinstance(result, DoMINOZarrDataInMemory)

        # Check metadata
        assert result.metadata == sample_data.metadata

        # Check one STL field
        assert isinstance(result.stl_coordinates, PreparedZarrArrayInfo)
        np.testing.assert_array_equal(
            result.stl_coordinates.data, sample_data.stl_coordinates
        )
        assert result.stl_coordinates.chunks == (2, 3)
        assert result.stl_coordinates.compressor == transform.compressor

        # Check one surface field
        assert isinstance(result.surface_mesh_centers, PreparedZarrArrayInfo)
        np.testing.assert_array_equal(
            result.surface_mesh_centers.data, sample_data.surface_mesh_centers
        )
        assert result.surface_mesh_centers.chunks == (1, 3)
        assert result.surface_mesh_centers.compressor == transform.compressor

        # Check one volume field
        assert isinstance(result.volume_mesh_centers, PreparedZarrArrayInfo)
        np.testing.assert_array_equal(
            result.volume_mesh_centers.data, sample_data.volume_mesh_centers
        )
        assert result.volume_mesh_centers.chunks == (1, 3)
        assert result.volume_mesh_centers.compressor == transform.compressor

    def test_prepare_array(self):
        """Test array preparation for Zarr storage."""
        config = ProcessingConfig(num_processes=1)
        transform = DoMINOZarrTransformation(config)

        # Test 1D array
        array_1d = np.array([1, 2, 3], dtype=np.float64)
        result_1d = transform._prepare_array(array_1d)
        assert isinstance(result_1d, PreparedZarrArrayInfo)
        assert result_1d.data.dtype == np.float32
        assert len(result_1d.chunks) == 1

        # Test 2D array
        array_2d = np.array([[1, 2], [3, 4]], dtype=np.float64)
        result_2d = transform._prepare_array(array_2d)
        assert isinstance(result_2d, PreparedZarrArrayInfo)
        assert result_2d.data.dtype == np.float32
        assert len(result_2d.chunks) == 2

        # Test None input
        assert transform._prepare_array(None) is None

    @pytest.mark.parametrize(
        "chunk_size_mb, should_warn",
        [
            (0.5, True),  # Too small
            (1.0, False),  # Default - OK
            (25.0, False),  # OK
            (75.0, True),  # Too large
        ],
    )
    def test_chunk_size_warnings(self, chunk_size_mb, should_warn):
        """Test warnings for different chunk sizes."""
        config = ProcessingConfig(num_processes=1)

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered
            warnings.simplefilter("always")

            # Create transformation
            DoMINOZarrTransformation(config, chunk_size_mb=chunk_size_mb)

            if should_warn:
                assert len(w) == 1
                assert issubclass(w[-1].category, UserWarning)
                if chunk_size_mb < 1.0:
                    assert "too small" in str(w[-1].message)
                else:
                    assert "too large" in str(w[-1].message)
            else:
                assert len(w) == 0

    def test_chunk_size_effect(self, sample_data):
        """Test that different chunk sizes result in different chunking."""
        # Create larger test data for meaningful chunk size testing
        large_data = DoMINOExtractedDataInMemory(
            metadata=sample_data.metadata,
            stl_coordinates=np.random.rand(100000, 3),  # Roughly 2.4 MB
            stl_centers=sample_data.stl_centers,
            stl_faces=sample_data.stl_faces,
            stl_areas=sample_data.stl_areas,
            surface_mesh_centers=sample_data.surface_mesh_centers,
            surface_normals=sample_data.surface_normals,
            surface_areas=sample_data.surface_areas,
            surface_fields=np.random.rand(100000, 3),  # Roughly 2.4 MB
            volume_mesh_centers=sample_data.volume_mesh_centers,
            volume_fields=sample_data.volume_fields,
        )

        config = ProcessingConfig(num_processes=1)

        # Create transformations with different chunk sizes
        transform_small = DoMINOZarrTransformation(config, chunk_size_mb=1.0)
        transform_large = DoMINOZarrTransformation(config, chunk_size_mb=10.0)

        # Transform data with both
        result_small = transform_small.transform(large_data)
        result_large = transform_large.transform(large_data)

        # Check chunk sizes for arrays that should be large enough to show differences
        assert (
            result_small.stl_coordinates.chunks != result_large.stl_coordinates.chunks
        ), "Chunk sizes should differ for stl_coordinates"
        assert (
            result_small.surface_fields.chunks != result_large.surface_fields.chunks
        ), "Chunk sizes should differ for surface_fields"

        # Verify chunk sizes are proportional (approximately)

        # Small data
        small_data_elements_per_chunk = result_small.stl_coordinates.chunks[0]
        # 3 chunks; (2.4 MB array size / 1.0 MB chunk size)
        small_data_num_chunks = np.ceil(100000 / small_data_elements_per_chunk)
        assert small_data_num_chunks == 3

        # Large data
        large_data_elements_per_chunk = result_large.stl_coordinates.chunks[0]
        # 1 chunk; (2.4 MB array size / 10.0 MB chunk size)
        large_data_num_chunks = np.ceil(100000 / large_data_elements_per_chunk)
        assert large_data_num_chunks == 1
