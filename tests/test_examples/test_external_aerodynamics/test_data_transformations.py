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

import shutil
import tempfile
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv
import vtk
import zarr
from constants import ModelType
from data_transformations import (
    ExternalAerodynamicsNumpyTransformation,
    ExternalAerodynamicsSTLTransformation,
    ExternalAerodynamicsSurfaceTransformation,
    ExternalAerodynamicsVolumeTransformation,
    ExternalAerodynamicsZarrTransformation,
)
from external_aero_geometry_data_processors import (
    filter_geometry_invalid_faces,
    update_geometry_data_to_float32,
)
from external_aero_surface_data_processors import (
    filter_invalid_surface_cells,
    non_dimensionalize_surface_fields,
    normalize_surface_normals,
    update_surface_data_to_float32,
    validate_surface_sample_quality,
)
from external_aero_utils import get_volume_data
from external_aero_volume_data_processors import (
    filter_volume_invalid_cells,
    non_dimensionalize_volume_fields,
    shuffle_volume_data,
    update_volume_data_to_float32,
    validate_volume_sample_quality,
)
from schemas import (
    ExternalAerodynamicsExtractedDataInMemory,
    ExternalAerodynamicsMetadata,
    ExternalAerodynamicsNumpyDataInMemory,
    ExternalAerodynamicsZarrDataInMemory,
    PreparedZarrArrayInfo,
)

from physicsnemo_curator.etl.processing_config import ProcessingConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_data_raw(temp_dir):
    """Create sample Raw DrivAerML data for testing."""

    # Simple STL geometry - triangle
    stl_points = np.array(
        [
            [0.0, 0.0, 0.0],  # min corner
            [1.0, 0.0, 0.0],  # x max
            [0.5, 1.0, 1.0],  # y and z max
        ],
        dtype=np.float64,
    )
    stl_faces = np.array([3, 0, 1, 2], dtype=np.int32)  # Single triangle
    stl_polydata = pv.PolyData(stl_points, faces=stl_faces)

    # Simple surface mesh with basic fields
    surface_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    surface_faces = np.array([3, 0, 1, 2], dtype=np.int32)
    surface_polydata = pv.PolyData(surface_points, faces=surface_faces)

    # Add basic cell data fields (1 value per cell/face)
    surface_polydata["pMeanTrim"] = np.array([101325.0], dtype=np.float64)
    surface_polydata["wallShearStressMeanTrim"] = np.array(
        [[1.0, 0.5, 0.2]], dtype=np.float64
    )

    # Simple volume grid with basic fields
    volume_grid = vtk.vtkUnstructuredGrid()
    volume_points = vtk.vtkPoints()
    for point in [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]:
        volume_points.InsertNextPoint(point)
    volume_grid.SetPoints(volume_points)

    # Add single tetrahedron cell
    tetra = vtk.vtkTetra()
    for i in range(4):
        tetra.GetPointIds().SetId(i, i)
    volume_grid.InsertNextCell(tetra.GetCellType(), tetra.GetPointIds())

    # Add volume fields
    velocity_array = vtk.vtkFloatArray()
    velocity_array.SetName("UMeanTrim")
    velocity_array.SetNumberOfComponents(3)
    velocity_array.SetNumberOfTuples(4)
    for i, vel in enumerate([[30, 0, 0], [25, 0, 0], [28, 1, 0], [27, 0, 1]]):
        velocity_array.SetTuple3(i, vel[0], vel[1], vel[2])
    volume_grid.GetPointData().AddArray(velocity_array)

    pressure_array = vtk.vtkFloatArray()
    pressure_array.SetName("pMeanTrim")
    pressure_array.SetNumberOfComponents(1)
    pressure_array.SetNumberOfTuples(4)
    for i, p in enumerate([101325, 101300, 101320, 101310]):
        pressure_array.SetValue(i, p)
    volume_grid.GetPointData().AddArray(pressure_array)

    return ExternalAerodynamicsExtractedDataInMemory(
        metadata=ExternalAerodynamicsMetadata(
            filename="test_sample",
            dataset_type=ModelType.COMBINED,
            physics_constants={"stream_velocity": 30.0, "air_density": 1.205},
        ),
        stl_polydata=stl_polydata,
        surface_polydata=surface_polydata,
        volume_unstructured_grid=volume_grid,
        # Everything below is None, because this sample data is the raw data.
        stl_coordinates=None,
        stl_centers=None,
        stl_faces=None,
        stl_areas=None,
        surface_mesh_centers=None,
        surface_normals=None,
        surface_areas=None,
        surface_fields=None,
        volume_mesh_centers=None,
        volume_fields=None,
    )


@pytest.fixture
def sample_data_processed():
    """Create sample processed External Aerodynamics data for testing."""
    return ExternalAerodynamicsExtractedDataInMemory(
        metadata=ExternalAerodynamicsMetadata(
            filename="run_1234",
            dataset_type=ModelType.COMBINED,
            physics_constants={"stream_velocity": 30.0, "air_density": 1.205},
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
        global_params_values=np.array([30.0, 1.205], dtype=np.float32),
        global_params_reference=np.array([30.0, 1.205], dtype=np.float32),
    )


class TestExternalAerodynamicsNumpyTransformation:
    """Test the ExternalAerodynamicsNumpyTransformation class."""

    def test_initialization(self):
        """Test initialization of NumPy transformation."""
        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsNumpyTransformation(config)
        assert transform.config == config

    def test_transform(self, sample_data_processed):
        """Test NumPy transformation of External Aerodynamics data."""
        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsNumpyTransformation(config)

        result = transform.transform(sample_data_processed)
        assert isinstance(result, ExternalAerodynamicsNumpyDataInMemory)

        # Check a couple of STL fields
        np.testing.assert_array_equal(
            result.stl_coordinates, sample_data_processed.stl_coordinates
        )
        np.testing.assert_array_equal(result.stl_areas, sample_data_processed.stl_areas)

        # Check a couple of surface fields
        np.testing.assert_array_equal(
            result.surface_mesh_centers, sample_data_processed.surface_mesh_centers
        )
        np.testing.assert_array_equal(
            result.surface_normals, sample_data_processed.surface_normals
        )

        # Check a couple of volume fields
        np.testing.assert_array_equal(
            result.volume_mesh_centers, sample_data_processed.volume_mesh_centers
        )
        np.testing.assert_array_equal(
            result.volume_fields, sample_data_processed.volume_fields
        )

        # Check global params fields
        np.testing.assert_array_equal(
            result.global_params_values, sample_data_processed.global_params_values
        )
        np.testing.assert_array_equal(
            result.global_params_reference,
            sample_data_processed.global_params_reference,
        )


class TestExternalAerodynamicsZarrTransformation:
    """Test the ExternalAerodynamicsZarrTransformation class."""

    def test_initialization(self):
        """Test initialization of Zarr transformation."""
        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsZarrTransformation(config)
        assert transform.config == config
        assert transform.compressor.cname == zarr.codecs.BloscCname.zstd
        assert transform.compressor.clevel == 5

    def test_transform(self, sample_data_processed):
        """Test Zarr transformation of External Aerodynamics data."""
        config = ProcessingConfig(
            num_processes=1,
        )
        transform = ExternalAerodynamicsZarrTransformation(config)

        result = transform.transform(sample_data_processed)

        # Check overall structure
        assert isinstance(result, ExternalAerodynamicsZarrDataInMemory)

        # Check metadata
        assert result.metadata == sample_data_processed.metadata

        # Check one STL field
        assert isinstance(result.stl_coordinates, PreparedZarrArrayInfo)
        np.testing.assert_array_equal(
            result.stl_coordinates.data, sample_data_processed.stl_coordinates
        )
        assert result.stl_coordinates.chunks == (2, 3)
        assert result.stl_coordinates.compressor == transform.compressor

        # Check one surface field
        assert isinstance(result.surface_mesh_centers, PreparedZarrArrayInfo)
        np.testing.assert_array_equal(
            result.surface_mesh_centers.data, sample_data_processed.surface_mesh_centers
        )
        assert result.surface_mesh_centers.chunks == (1, 3)
        assert result.surface_mesh_centers.compressor == transform.compressor

        # Check one volume field
        assert isinstance(result.volume_mesh_centers, PreparedZarrArrayInfo)
        np.testing.assert_array_equal(
            result.volume_mesh_centers.data, sample_data_processed.volume_mesh_centers
        )
        assert result.volume_mesh_centers.chunks == (1, 3)
        assert result.volume_mesh_centers.compressor == transform.compressor

        # Check global params fields (no compression for small 1xN arrays)
        assert isinstance(result.global_params_values, PreparedZarrArrayInfo)
        np.testing.assert_array_equal(
            result.global_params_values.data, sample_data_processed.global_params_values
        )
        assert result.global_params_values.compressor is None

        assert isinstance(result.global_params_reference, PreparedZarrArrayInfo)
        np.testing.assert_array_equal(
            result.global_params_reference.data,
            sample_data_processed.global_params_reference,
        )
        assert result.global_params_reference.compressor is None

    def test_prepare_array(self):
        """Test array preparation for Zarr storage."""
        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsZarrTransformation(config)

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
            ExternalAerodynamicsZarrTransformation(config, chunk_size_mb=chunk_size_mb)

            if should_warn:
                assert len(w) == 1
                assert issubclass(w[-1].category, UserWarning)
                if chunk_size_mb < 1.0:
                    assert "too small" in str(w[-1].message)
                else:
                    assert "too large" in str(w[-1].message)
            else:
                assert len(w) == 0

    def test_chunk_size_effect(self, sample_data_processed):
        """Test that different chunk sizes result in different chunking."""
        # Create larger test data for meaningful chunk size testing
        large_data = ExternalAerodynamicsExtractedDataInMemory(
            metadata=sample_data_processed.metadata,
            stl_coordinates=np.random.rand(100000, 3),  # Roughly 2.4 MB
            stl_centers=sample_data_processed.stl_centers,
            stl_faces=sample_data_processed.stl_faces,
            stl_areas=sample_data_processed.stl_areas,
            surface_mesh_centers=sample_data_processed.surface_mesh_centers,
            surface_normals=sample_data_processed.surface_normals,
            surface_areas=sample_data_processed.surface_areas,
            surface_fields=np.random.rand(100000, 3),  # Roughly 2.4 MB
            volume_mesh_centers=sample_data_processed.volume_mesh_centers,
            volume_fields=sample_data_processed.volume_fields,
        )

        config = ProcessingConfig(num_processes=1)

        # Create transformations with different chunk sizes
        transform_small = ExternalAerodynamicsZarrTransformation(
            config, chunk_size_mb=1.0
        )
        transform_large = ExternalAerodynamicsZarrTransformation(
            config, chunk_size_mb=10.0
        )

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

    def test_sharding_always_enabled(self, sample_data_processed):
        """Test that sharding is always enabled for all arrays."""
        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsZarrTransformation(
            config,
            chunk_size_mb=1.0,
            chunks_per_shard=1000,
        )

        result = transform.transform(sample_data_processed)

        # All arrays should have sharding enabled
        assert result.stl_coordinates.shards is not None
        assert result.stl_centers.shards is not None
        assert result.stl_faces.shards is not None
        assert result.stl_areas.shards is not None
        assert result.surface_mesh_centers.shards is not None
        assert result.surface_normals.shards is not None
        assert result.surface_areas.shards is not None
        assert result.surface_fields.shards is not None
        assert result.volume_mesh_centers.shards is not None
        assert result.volume_fields.shards is not None

    def test_sharding_shape_for_large_arrays(self):
        """Test that sharding shape is calculated correctly for large arrays."""
        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsZarrTransformation(
            config,
            chunk_size_mb=1.0,
            chunks_per_shard=1000,
        )

        # Create a large array > 1GB
        # For float32, 1 GB = 1024^3 / 4 = 268,435,456 elements
        # Create a 2D array: 300M elements / 3 columns = 100M rows
        # This is 300M * 4 bytes = 1.2 GB
        large_array = np.random.rand(100_000_000, 3).astype(np.float64)

        metadata = ExternalAerodynamicsMetadata(
            filename="test_large",
            dataset_type=ModelType.COMBINED,
            physics_constants={"stream_velocity": 30.0, "air_density": 1.205},
        )

        large_data = ExternalAerodynamicsExtractedDataInMemory(
            metadata=metadata,
            stl_coordinates=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64),
            stl_centers=np.array([[0.5, 0.5, 0.5]], dtype=np.float64),
            stl_faces=np.array([[0, 1, 2]], dtype=np.int32),
            stl_areas=np.array([1.0], dtype=np.float64),
            volume_fields=large_array,  # Large array
        )

        result = transform.transform(large_data)

        # All arrays should have sharding enabled
        assert result.stl_coordinates.shards is not None
        assert result.stl_centers.shards is not None

        # Check large array shard shape
        assert result.volume_fields.shards is not None
        assert len(result.volume_fields.shards) == 2  # 2D array
        assert (
            result.volume_fields.shards[1] == 3
        )  # Second dimension should match array
        # First dimension should be chunk_rows * chunks_per_shard (or array size if smaller)
        expected_shard_rows = min(100_000_000, result.volume_fields.chunks[0] * 1000)
        assert result.volume_fields.shards[0] == expected_shard_rows

    def test_sharding_calculation_1d_array(self):
        """Test correct shard shape calculation for 1D arrays."""
        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsZarrTransformation(
            config,
            chunk_size_mb=1.0,
            chunks_per_shard=100,  # Use smaller value for easier testing
        )

        # Create 1D array (400MB as float64, 200MB as float32)
        # 50M elements * 8 bytes = 400 MB
        large_1d_array = np.random.rand(50_000_000).astype(np.float64)

        result = transform._prepare_array(large_1d_array)

        # Check that sharding is enabled
        assert result.shards is not None
        assert len(result.shards) == 1

        # For 1D array: shard_size = chunks[0] * chunks_per_shard
        # With 1MB chunks and float32 (4 bytes): chunk_size = 1MB / 4 = 262144 elements
        # So shards should be ~26,214,400 elements (100 chunks)
        expected_shard_size = result.chunks[0] * 100
        assert result.shards[0] == expected_shard_size

    def test_sharding_calculation_2d_array(self):
        """Test correct shard shape calculation for 2D arrays."""
        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsZarrTransformation(
            config,
            chunk_size_mb=1.0,
            chunks_per_shard=50,  # Use smaller value for easier testing
        )

        # Create 2D array (2.4GB as float64, 1.2GB as float32)
        # 100M rows * 3 cols * 8 bytes = 2.4 GB
        large_2d_array = np.random.rand(100_000_000, 3).astype(np.float64)

        result = transform._prepare_array(large_2d_array)

        # Check that sharding is enabled
        assert result.shards is not None
        assert len(result.shards) == 2

        # For 2D arrays: shard_rows = chunk_rows * chunks_per_shard
        # Second dimension should match array shape
        expected_shard_rows = result.chunks[0] * 50
        assert result.shards[0] == expected_shard_rows
        assert result.shards[1] == 3  # Matches second dimension

    def test_sharding_initialization_defaults(self):
        """Test that default sharding parameters are set correctly."""
        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsZarrTransformation(config)

        # Check defaults match Zarr recommendations
        assert transform.chunks_per_shard == 1000  # Zarr recommended (~1GB shards)
        assert transform.chunk_size_mb == 1.0

    def test_sharding_divisibility_requirement(self):
        """Test that shard shapes are always divisible by chunk shapes."""
        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsZarrTransformation(
            config,
            chunk_size_mb=1.0,
            chunks_per_shard=1000,
        )

        # Test with various array sizes that might not divide evenly
        test_cases = [
            (1234, 3),  # Odd number of rows
            (999, 5),  # Another odd number
            (1000000, 4),  # Large array
            (50, 3),  # Very small array
            (262144 + 100, 3),  # Just over one chunk
        ]

        for rows, cols in test_cases:
            array = np.random.rand(rows, cols).astype(np.float64)
            result = transform._prepare_array(array)

            # Check that shards are divisible by chunks in each dimension
            assert result.shards[0] % result.chunks[0] == 0, (
                f"Shard rows {result.shards[0]} not divisible by chunk rows {result.chunks[0]} "
                f"for array shape ({rows}, {cols})"
            )
            assert (
                result.shards[1] == result.chunks[1]
            ), f"Shard cols {result.shards[1]} should equal chunk cols {result.chunks[1]}"

    def test_sharding_small_array_one_shard(self):
        """Test that small arrays fit into a single shard with proper rounding."""
        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsZarrTransformation(
            config,
            chunk_size_mb=1.0,
            chunks_per_shard=1000,
        )

        # Create small array that would need multiple chunks but < chunks_per_shard
        # With 1MB chunks and float64: ~262144 elements per chunk for 1D
        # Create array with 500,000 elements (needs ~2 chunks)
        small_array = np.random.rand(500_000).astype(np.float64)
        result = transform._prepare_array(small_array)

        # Should have 1 shard that contains all chunks
        num_chunks_needed = (500_000 + result.chunks[0] - 1) // result.chunks[0]
        expected_shard_size = num_chunks_needed * result.chunks[0]
        assert result.shards[0] == expected_shard_size
        assert result.shards[0] >= 500_000  # Shard is big enough for array
        assert result.shards[0] % result.chunks[0] == 0  # Properly divisible


class TestExternalAerodynamicsSTLTransformation:
    """Test the ExternalAerodynamicsSTLTransformation class."""

    def test_initialization(self):
        """Test initialization of STL transformation class."""
        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsSTLTransformation(config)
        assert transform.config == config
        assert transform.geometry_processors is None

    def test_initialization_with_processors(self):
        """Test initialization with geometry processors."""
        config = ProcessingConfig(num_processes=1)

        def dummy_processor(data):
            return data

        transform = ExternalAerodynamicsSTLTransformation(
            config, geometry_processors=(dummy_processor,)
        )
        assert transform.geometry_processors == (dummy_processor,)

    def test_transform_basic(self, sample_data_raw):
        """Test basic STL transformation of External Aerodynamics data."""
        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsSTLTransformation(config)

        result = transform.transform(sample_data_raw)
        assert isinstance(result, ExternalAerodynamicsExtractedDataInMemory)

        # Check that raw STL data is cleaned up
        assert result.stl_polydata is None

        # Check that processed STL data is present
        assert result.stl_coordinates is not None
        assert result.stl_centers is not None
        assert result.stl_faces is not None
        assert result.stl_areas is not None

        # Check metadata
        assert result.metadata.num_points == 3
        assert result.metadata.num_faces == 3

    def test_transform_with_simple_processors(self, sample_data_raw):
        """Test STL transformation with custom, simple processors."""
        config = ProcessingConfig(num_processes=1)

        # Create a mock processor that modifies the data
        def mock_processor(data):
            # Just modify the data slightly to verify it was called
            if data.stl_coordinates is not None:
                data.stl_coordinates = data.stl_coordinates * 2.0
            return data

        transform = ExternalAerodynamicsSTLTransformation(
            config, geometry_processors=(mock_processor,)
        )

        result = transform.transform(sample_data_raw)

        # Check that the processor was applied
        original_coords = np.array(
            [
                [0.0, 0.0, 0.0],  # min corner
                [1.0, 0.0, 0.0],  # x max
                [0.5, 1.0, 1.0],  # y and z max
            ],
            dtype=np.float64,
        )  # Same as what's in sample_data_raw
        np.testing.assert_array_equal(result.stl_coordinates, original_coords * 2.0)

    def test_transform_with_default_processor_and_float32_conversion(
        self, sample_data_raw
    ):
        """Test STL transformation with float32 conversion on top of the default processor."""
        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsSTLTransformation(
            config, geometry_processors=(update_geometry_data_to_float32,)
        )
        result = transform.transform(sample_data_raw)

        # Check data types
        assert result.stl_coordinates.dtype == np.float32
        assert result.stl_centers.dtype == np.float32
        assert result.stl_faces.dtype == np.int32  # This should remain int32
        assert result.stl_areas.dtype == np.float32

    def test_transform_with_default_processor_and_filter_invalid_geometry_faces(
        self, sample_data_raw
    ):
        """Test STL transformation with filter invalid geometry faces on top of the default processor."""
        # Start with sample_data_raw which has 1 valid triangle with 3 vertices
        stl_polydata = sample_data_raw.stl_polydata

        # Get existing points and faces
        stl_points = stl_polydata.points  # Shape: (3, 3) - 3 vertices
        stl_faces = stl_polydata.faces  # [3, 0, 1, 2] - 1 triangle

        # Add 3 collinear points for a degenerate triangle (zero area)
        # These points lie on a line, so they won't form a valid triangle
        additional_points = np.array(
            [
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],  # All on x-axis -> collinear
            ],
            dtype=np.float64,
        )
        updated_points = np.vstack([stl_points, additional_points])

        # Add the degenerate triangle face
        # Format: [num_vertices, vertex_idx_0, vertex_idx_1, vertex_idx_2]
        updated_faces = np.concatenate(
            [
                stl_faces,  # Original valid triangle: [3, 0, 1, 2]
                np.array([3, 3, 4, 5], dtype=np.int32),  # Degenerate triangle
            ]
        )

        # Create new polydata with updated geometry
        sample_data_raw.stl_polydata = pv.PolyData(updated_points, faces=updated_faces)

        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsSTLTransformation(
            config, geometry_processors=(filter_geometry_invalid_faces,)
        )
        result = transform.transform(sample_data_raw)

        # Check that the invalid geometry face was filtered out
        # We started with 2 faces (1 valid + 1 invalid), should have 1 valid face after filtering
        assert (
            result.metadata.num_faces == 3
        ), "Should have 3 face indices (1 triangle) after filtering"
        assert len(result.stl_areas) == 1, "Should have 1 area value after filtering"
        assert len(result.stl_centers) == 1, "Should have 1 center after filtering"

        # The remaining face should have valid area (> 0)
        assert np.all(result.stl_areas > 0), "Remaining area should be positive"

        # Check that unused vertices were removed
        # Original: 6 vertices (3 valid + 3 for degenerate triangle)
        # After filtering: only 3 vertices (from the valid triangle) should remain
        assert result.metadata.num_points == 3, "Should have 3 vertices after filtering"
        assert len(result.stl_coordinates) == 3, "Should have 3 coordinate sets"

        # Check that face indices were reindexed correctly (should be [0, 1, 2])
        assert len(result.stl_faces) == 3, "Should have 3 face indices"
        np.testing.assert_array_equal(
            np.sort(result.stl_faces),
            np.array([0, 1, 2]),
            err_msg="Face indices should be reindexed to [0, 1, 2]",
        )

        # Check that bounds were updated correctly
        assert result.metadata.x_bound is not None
        assert result.metadata.y_bound is not None
        assert result.metadata.z_bound is not None


class TestExternalAerodynamicsSurfaceTransformation:
    """Test the ExternalAerodynamicsSurfaceTransformation class."""

    def test_initialization_with_variables(self):
        """Test initialization with surface variables."""
        config = ProcessingConfig(num_processes=1)
        surface_variables = {
            "pMeanTrim": "scalar",
            "wallShearStressMeanTrim": "vector",
        }
        transform = ExternalAerodynamicsSurfaceTransformation(
            config,
            surface_variables=surface_variables,
        )
        assert transform.surface_variables == surface_variables

    def test_initialization_with_no_variables_raises_error(self):
        """Test initialization with no surface variables."""
        config = ProcessingConfig(num_processes=1)
        with pytest.raises(ValueError):
            ExternalAerodynamicsSurfaceTransformation(config)

    def test_transform_with_surface_data(self, sample_data_raw):
        """Test surface transformation with surface data."""
        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsSurfaceTransformation(
            config,
            surface_variables={
                "pMeanTrim": "scalar",
                "wallShearStressMeanTrim": "vector",
            },
        )

        result = transform.transform(sample_data_raw)

        # Check that raw surface data is cleaned up
        assert result.surface_polydata is None

        # Check surface data processing
        assert result.surface_mesh_centers is not None
        assert result.surface_normals is not None
        assert result.surface_areas is not None
        assert result.surface_fields is not None

    def test_transform_no_surface_data(self, sample_data_raw):
        """Test surface transformation without surface data."""
        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsSurfaceTransformation(
            config,
            surface_variables={
                "pMeanTrim": "scalar",
                "wallShearStressMeanTrim": "vector",
            },
        )

        # Set surface data to None.
        sample_data_raw.surface_polydata = None

        result = transform.transform(sample_data_raw)
        assert result.surface_mesh_centers is None
        assert result.surface_normals is None
        assert result.surface_areas is None
        assert result.surface_fields is None

    def test_transform_with_simple_processors(self, sample_data_raw):
        """Test surface transformation with custom, simple processors."""
        config = ProcessingConfig(num_processes=1)

        # Create a mock processor
        def mock_processor(data):
            # Modify surface centers to verify it was called
            if data.surface_mesh_centers is not None:
                data.surface_mesh_centers = data.surface_mesh_centers * 1.5
            return data

        transform = ExternalAerodynamicsSurfaceTransformation(
            config,
            surface_variables={
                "wallShearStressMeanTrim": "vector",
                "pMeanTrim": "scalar",
            },
            surface_processors=(mock_processor,),
        )

        result = transform.transform(sample_data_raw)

        # The processor should have been applied after default processing
        assert result.surface_mesh_centers is not None

    def test_transform_with_default_processor_and_float32_conversion(
        self, sample_data_raw
    ):
        """Test surface transformation with float32 conversion on top of the default processor."""
        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsSurfaceTransformation(
            config,
            surface_variables={
                "wallShearStressMeanTrim": "vector",
                "pMeanTrim": "scalar",
            },
            surface_processors=(update_surface_data_to_float32,),
        )
        result = transform.transform(sample_data_raw)

        # Check data types
        assert result.surface_mesh_centers.dtype == np.float32
        assert result.surface_normals.dtype == np.float32
        assert result.surface_areas.dtype == np.float32
        assert result.surface_fields.dtype == np.float32

    def test_transform_with_default_processor_and_normalization(self, sample_data_raw):
        """Test surface transformation with normalization on top of the default processor."""
        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsSurfaceTransformation(
            config,
            surface_variables={
                "wallShearStressMeanTrim": "vector",
                "pMeanTrim": "scalar",
            },
            surface_processors=(normalize_surface_normals,),
        )
        result = transform.transform(sample_data_raw)

        # Check that the normals were normalized
        assert result.surface_normals.shape == sample_data_raw.surface_normals.shape
        assert np.all(
            np.abs(np.linalg.norm(result.surface_normals, axis=1) - 1.0) < 1e-6
        )

    def test_transform_with_default_processor_and_non_dimensionalization(
        self, sample_data_raw
    ):
        """Test surface transformation with non-dimensionalization on top of the default processor."""
        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsSurfaceTransformation(
            config,
            surface_variables={
                "wallShearStressMeanTrim": "vector",
                "pMeanTrim": "scalar",
            },
            surface_processors=(
                partial(
                    non_dimensionalize_surface_fields,
                    air_density=1.00,
                    stream_velocity=10.00,
                ),
            ),
        )
        result = transform.transform(sample_data_raw)

        # Check that the fields were non-dimensionalized
        assert result.surface_fields.shape == sample_data_raw.surface_fields.shape
        # Verify non-dimensionalization: result = original / (rho * V^2)
        dynamic_pressure_factor = 1.00 * 10.00**2  # air_density * stream_velocity^2
        expected = np.array([[1.0, 0.5, 0.2, 101325.0]]) / dynamic_pressure_factor
        np.testing.assert_allclose(result.surface_fields, expected, rtol=1e-5)

    def test_transform_with_default_processor_and_filter_invalid_surface_cells(
        self, sample_data_raw
    ):
        """Test surface transformation with filter invalid surface cells on top of the default processor."""
        # Start with sample_data_raw which has 1 valid triangle
        surface_polydata = sample_data_raw.surface_polydata

        # Get existing points and faces
        surface_points = surface_polydata.points  # Shape: (3, 3) - 3 points
        surface_faces = surface_polydata.faces  # [3, 0, 1, 2] - 1 triangle

        # Add 3 collinear points for a degenerate triangle (zero area, zero normal)
        # These points lie on a line, so they won't form a valid triangle
        additional_points = np.array(
            [
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],  # All on x-axis -> collinear
            ],
            dtype=np.float64,
        )
        updated_points = np.vstack([surface_points, additional_points])

        # Add the degenerate triangle face
        # Format: [num_vertices, vertex_idx_0, vertex_idx_1, vertex_idx_2]
        updated_faces = np.concatenate(
            [
                surface_faces,  # Original valid triangle: [3, 0, 1, 2]
                np.array([3, 3, 4, 5], dtype=np.int32),  # Degenerate triangle
            ]
        )

        # Create new polydata with updated geometry
        sample_data_raw.surface_polydata = pv.PolyData(
            updated_points, faces=updated_faces
        )

        # Update cell data fields for 2 cells (1 valid + 1 invalid)
        sample_data_raw.surface_polydata["pMeanTrim"] = np.array(
            [101325.0, 101330.0], dtype=np.float64
        )
        sample_data_raw.surface_polydata["wallShearStressMeanTrim"] = np.array(
            [[1.0, 0.5, 0.2], [1.1, 0.6, 0.3]], dtype=np.float64
        )

        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsSurfaceTransformation(
            config,
            surface_variables={
                "wallShearStressMeanTrim": "vector",
                "pMeanTrim": "scalar",
            },
            surface_processors=(filter_invalid_surface_cells,),
        )
        result = transform.transform(sample_data_raw)

        # Check that the invalid surface cell was filtered out
        # We started with 2 cells (1 valid + 1 invalid), should have 1 valid cell after filtering
        assert (
            result.surface_mesh_centers.shape[0] == 1
        ), "Should have 1 cell after filtering"
        assert result.surface_normals.shape[0] == 1
        assert result.surface_areas.shape[0] == 1
        assert result.surface_fields.shape[0] == 1

        # The remaining cell should have valid area (> 0)
        assert np.all(result.surface_areas > 0), "Remaining area should be positive"

        # The remaining normal should have non-zero magnitude
        normal_magnitudes = np.linalg.norm(result.surface_normals, axis=1)
        assert np.all(
            normal_magnitudes > 1e-6
        ), "Remaining normal should have non-zero magnitude"

        # Verify the fields dimension is correct (3 components for vector + 1 for scalar)
        assert result.surface_fields.shape[1] == 4, "Should have 4 field components"

    def test_validate_surface_sample_quality_with_valid_data(self, sample_data_raw):
        """Test surface sample quality validator with valid data."""
        config = ProcessingConfig(num_processes=1)

        # First process and non-dimensionalize the data
        transform = ExternalAerodynamicsSurfaceTransformation(
            config,
            surface_variables={
                "wallShearStressMeanTrim": "vector",
                "pMeanTrim": "scalar",
            },
            surface_processors=(
                partial(
                    non_dimensionalize_surface_fields,
                    air_density=sample_data_raw.metadata.physics_constants[
                        "air_density"
                    ],
                    stream_velocity=sample_data_raw.metadata.physics_constants[
                        "stream_velocity"
                    ],
                ),
            ),
        )
        processed_data = transform.transform(sample_data_raw)

        # Apply validator - should pass with reasonable data
        result = validate_surface_sample_quality(
            processed_data,
            statistical_tolerance=7.0,
            pressure_max=4.0,
        )

        # Should return unchanged data
        assert result.surface_fields is not None
        np.testing.assert_array_equal(
            result.surface_fields, processed_data.surface_fields
        )

    def test_validate_surface_sample_quality_with_extreme_values(self):
        """Test surface sample quality validator with data exceeding bounds."""
        # Create data with extreme pressure values that will exceed threshold
        metadata = ExternalAerodynamicsMetadata(
            filename="test_extreme",
            dataset_type=ModelType.SURFACE,
        )

        data = ExternalAerodynamicsExtractedDataInMemory(metadata=metadata)

        # Create surface fields with extreme values (after non-dimensionalization)
        # Pressure is first component, make it exceed 4.0
        data.surface_fields = np.array(
            [
                [5.0, 1.0, 0.5],  # Pressure = 5.0 > 4.0 (exceeds threshold)
                [5.5, 1.2, 0.6],
                [6.0, 1.1, 0.4],
            ]
        )
        data.surface_mesh_centers = np.random.rand(3, 3)
        data.surface_normals = np.random.rand(3, 3)
        data.surface_areas = np.random.rand(3)

        # Apply validator - should detect bounds violation and return None
        result = validate_surface_sample_quality(
            data,
            statistical_tolerance=7.0,
            pressure_max=4.0,
        )

        # Sample should be rejected (returns None)
        assert result is None


class TestExternalAerodynamicsVolumeTransformation:
    """Test the ExternalAerodynamicsVolumeTransformation class."""

    def test_initialization_with_variables(self):
        """Test initialization with volume variables."""
        config = ProcessingConfig(num_processes=1)
        volume_variables = {
            "UMeanTrim": "vector",
            "pMeanTrim": "scalar",
        }
        transform = ExternalAerodynamicsVolumeTransformation(
            config,
            volume_variables=volume_variables,
        )
        assert transform.volume_variables == volume_variables

    def test_initialization_with_no_variables_raises_error(self):
        """Test initialization with no volume variables."""
        config = ProcessingConfig(num_processes=1)
        with pytest.raises(ValueError):
            ExternalAerodynamicsVolumeTransformation(config)

    def test_transform_with_volume_data(self, sample_data_raw):
        """Test volume transformation with volume data."""
        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsVolumeTransformation(
            config,
            volume_variables={
                "UMeanTrim": "vector",
                "pMeanTrim": "scalar",
            },
        )

        result = transform.transform(sample_data_raw)

        # Check that raw volume data is cleaned up
        assert result.volume_unstructured_grid is None

        # Check volume data processing
        assert result.volume_mesh_centers is not None
        assert result.volume_fields is not None

    def test_transform_no_volume_data(self, sample_data_raw):
        """Test volume transformation without volume data."""
        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsVolumeTransformation(
            config,
            volume_variables={
                "UMeanTrim": "vector",
                "pMeanTrim": "scalar",
            },
        )

        # Set volume data to None.
        sample_data_raw.volume_unstructured_grid = None

        result = transform.transform(sample_data_raw)
        assert result.volume_mesh_centers is None
        assert result.volume_fields is None

    def test_transform_with_processors(self, sample_data_raw):
        """Test volume transformation with custom processors."""
        config = ProcessingConfig(num_processes=1)

        # Create a mock processor with kwargs
        def mock_processor(data, scale_factor=1.0):
            # Modify volume centers to verify it was called
            if data.volume_mesh_centers is not None:
                data.volume_mesh_centers = data.volume_mesh_centers * scale_factor
            return data

        transform = ExternalAerodynamicsVolumeTransformation(
            config,
            volume_variables={
                "UMeanTrim": "vector",
                "pMeanTrim": "scalar",
            },
            volume_processors=(partial(mock_processor, scale_factor=2.0),),
        )

        result = transform.transform(sample_data_raw)

        # The processor should have been applied after default processing
        assert result.volume_mesh_centers is not None

    def test_transform_with_default_processor_and_float32_conversion(
        self, sample_data_raw
    ):
        """Test volume transformation with float32 conversion on top of the default processor."""
        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsVolumeTransformation(
            config,
            volume_variables={
                "UMeanTrim": "vector",
                "pMeanTrim": "scalar",
            },
            volume_processors=(update_volume_data_to_float32,),
        )
        result = transform.transform(sample_data_raw)
        # Check data types
        assert result.volume_mesh_centers.dtype == np.float32
        assert result.volume_fields.dtype == np.float32

    def test_transform_with_default_processor_and_non_dimensionalization(
        self, sample_data_raw
    ):
        """Test volume transformation with non-dimensionalization on top of the default processor."""
        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsVolumeTransformation(
            config,
            volume_variables={
                "UMeanTrim": "vector",
                "pMeanTrim": "scalar",
            },
            volume_processors=(
                partial(
                    non_dimensionalize_volume_fields,
                    air_density=1.00,
                    stream_velocity=10.00,
                ),
            ),
        )
        result = transform.transform(sample_data_raw)
        # Check that the fields were non-dimensionalized
        assert result.volume_fields.shape == sample_data_raw.volume_fields.shape
        # Verify non-dimensionalization:
        # Velocity (columns 0-2): divided by stream_velocity
        # Pressure (column 3): divided by (rho * V^2)
        expected_velocity = (
            np.array([[30, 0, 0], [25, 0, 0], [28, 1, 0], [27, 0, 1]])
            / 10.00  # stream_velocity
        )
        expected_pressure = np.array([[101325], [101300], [101320], [101310]]) / (
            1.00 * 10.00**2  # air_density * stream_velocity^2
        )
        expected = np.concatenate([expected_velocity, expected_pressure], axis=-1)
        np.testing.assert_allclose(result.volume_fields, expected, rtol=1e-5)

    def test_transform_with_default_processor_and_shuffle_data(self, sample_data_raw):
        """Test volume transformation with shuffle on top of the default processor."""
        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsVolumeTransformation(
            config,
            volume_variables={
                "UMeanTrim": "vector",
                "pMeanTrim": "scalar",
            },
            volume_processors=(shuffle_volume_data,),
        )

        # Get original volume data for comparison
        original_centers, original_fields = get_volume_data(
            sample_data_raw.volume_unstructured_grid,
            {"UMeanTrim": "vector", "pMeanTrim": "scalar"},
        )
        original_fields = np.concatenate(original_fields, axis=-1)

        result = transform.transform(sample_data_raw)

        # Check that data was shuffled (should be different from original)
        assert not np.array_equal(result.volume_mesh_centers, original_centers)
        assert not np.array_equal(result.volume_fields, original_fields)

        # Check that shapes are preserved
        assert result.volume_mesh_centers.shape == original_centers.shape
        assert result.volume_fields.shape == original_fields.shape

        # Verify all data points are preserved (no loss or duplication)
        # Sort both original and shuffled data to compare
        original_sorted_idx = np.lexsort(original_centers.T)
        result_sorted_idx = np.lexsort(result.volume_mesh_centers.T)

        np.testing.assert_allclose(
            original_centers[original_sorted_idx],
            result.volume_mesh_centers[result_sorted_idx],
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            original_fields[original_sorted_idx],
            result.volume_fields[result_sorted_idx],
            rtol=1e-10,
        )

        # Check that data is coupled (same permutation applied to both)
        for i in range(len(result.volume_mesh_centers)):
            shuffled_center = result.volume_mesh_centers[i]
            shuffled_field = result.volume_fields[i]

            # Find where this center was in the original data
            original_idx = np.where(
                np.all(np.isclose(original_centers, shuffled_center), axis=1)
            )[0][0]

            # Verify the field at this position matches the original
            np.testing.assert_allclose(
                shuffled_field, original_fields[original_idx], rtol=1e-10
            )

    def test_transform_with_default_processor_and_filter_invalid_volume_cells(
        self, sample_data_raw
    ):
        """Test volume transformation with filter invalid volume cells on top of the default processor."""
        config = ProcessingConfig(num_processes=1)

        # First, process the volume data using the default processor
        transform_process = ExternalAerodynamicsVolumeTransformation(
            config,
            volume_variables={
                "UMeanTrim": "vector",
                "pMeanTrim": "scalar",
            },
        )
        processed_data = transform_process.transform(sample_data_raw)

        # Now inject some invalid data into the processed volume data
        # Original has 4 valid volume cells
        n_original = len(processed_data.volume_mesh_centers)

        # Add 2 cells: 1 with NaN in coordinates, 1 with inf in fields
        invalid_center_nan = np.array([[np.nan, 0.5, 0.5]])
        invalid_field_nan = np.array([[10.0, 0.0, 0.0, 101000.0]])

        invalid_center_inf = np.array([[0.5, 0.5, 0.5]])
        invalid_field_inf = np.array([[np.inf, 0.0, 0.0, 101000.0]])

        # Concatenate invalid data
        processed_data.volume_mesh_centers = np.vstack(
            [processed_data.volume_mesh_centers, invalid_center_nan, invalid_center_inf]
        )
        processed_data.volume_fields = np.vstack(
            [processed_data.volume_fields, invalid_field_nan, invalid_field_inf]
        )

        # Now apply the filter
        result = filter_volume_invalid_cells(processed_data)

        # Check that the invalid volume cells were filtered out
        # We started with 4 valid + 2 invalid = 6 cells, should have 4 valid cells after filtering
        assert (
            result.volume_mesh_centers.shape[0] == n_original
        ), f"Should have {n_original} cells after filtering"
        assert result.volume_fields.shape[0] == n_original

        # All remaining cells should have finite coordinates
        assert np.all(
            np.isfinite(result.volume_mesh_centers)
        ), "All coordinates should be finite"

        # All remaining cells should have finite fields
        assert np.all(np.isfinite(result.volume_fields)), "All fields should be finite"

    def test_transform_with_default_processor_and_validate_volume_sample_quality_with_valid_data(
        self, sample_data_raw
    ):
        """Test volume sample quality validator with valid data."""
        config = ProcessingConfig(num_processes=1)

        # First process and non-dimensionalize the data
        transform = ExternalAerodynamicsVolumeTransformation(
            config,
            volume_variables={
                "UMeanTrim": "vector",
                "pMeanTrim": "scalar",
            },
            volume_processors=(
                partial(
                    non_dimensionalize_volume_fields,
                    air_density=sample_data_raw.metadata.physics_constants[
                        "air_density"
                    ],
                    stream_velocity=sample_data_raw.metadata.physics_constants[
                        "stream_velocity"
                    ],
                ),
            ),
        )
        processed_data = transform.transform(sample_data_raw)

        # Apply validator - should pass with reasonable data
        # Note: Using higher pressure_max since sample data has absolute pressure values
        # that non-dimensionalize to ~93 (atmospheric pressure / dynamic pressure)
        result = validate_volume_sample_quality(
            processed_data,
            statistical_tolerance=7.0,
            velocity_max=3.5,
            pressure_max=100.0,  # Higher threshold for test with absolute pressure
        )

        # Should return unchanged data
        assert result is not None
        assert result.volume_fields is not None
        np.testing.assert_array_equal(
            result.volume_fields, processed_data.volume_fields
        )

    def test_transform_with_default_processor_and_validate_volume_sample_quality_with_outliers_velocity(
        self,
    ):
        """Test volume sample quality validator with outliers in velocity values."""
        # Create data with outliers in velocity values
        metadata = ExternalAerodynamicsMetadata(
            filename="test_outliers_velocity",
            dataset_type=ModelType.VOLUME,
        )

        data = ExternalAerodynamicsExtractedDataInMemory(metadata=metadata)

        # Create volume fields with outliers in velocity values (after non-dimensionalization)
        # Fields are [u, v, w, p, ...], make velocity exceed 3.5
        data.volume_fields = np.array(
            [
                [4.0, 2.0, 2.0, 2.0],  # u = 4.0 > 3.5 (exceeds threshold)
                [3.8, 2.5, 2.2, 2.5],  # u = 3.8 > 3.5 (exceeds threshold)
                [4.2, 1.8, 2.3, 2.3],  # u = 4.2 > 3.5 (exceeds threshold)
            ]
        )
        data.volume_mesh_centers = np.random.rand(3, 3)

        # Apply validator - should detect bounds violation and return None
        result = validate_volume_sample_quality(
            data,
            statistical_tolerance=7.0,
            velocity_max=3.5,
            pressure_max=4.0,
        )

        # Sample should be rejected (returns None)
        assert result is None

    def test_validate_volume_sample_quality_with_extreme_pressure(self):
        """Test volume sample quality validator with extreme pressure values."""
        # Create data with extreme pressure values
        metadata = ExternalAerodynamicsMetadata(
            filename="test_extreme",
            dataset_type=ModelType.VOLUME,
        )

        data = ExternalAerodynamicsExtractedDataInMemory(metadata=metadata)

        # Create volume fields with extreme pressure (4th component)
        data.volume_fields = np.array(
            [
                [2.0, 2.0, 2.0, 5.0],  # p = 5.0 > 4.0 (exceeds threshold)
                [2.5, 2.2, 1.8, 5.5],
                [2.3, 1.9, 2.1, 6.0],
            ]
        )
        data.volume_mesh_centers = np.random.rand(3, 3)

        # Apply validator - should detect bounds violation and return None
        result = validate_volume_sample_quality(
            data,
            statistical_tolerance=7.0,
            velocity_max=3.5,
            pressure_max=4.0,
        )

        # Sample should be rejected (returns None)
        assert result is None

    def test_transform_with_default_processor_and_shuffle_data_large_array(self):
        """Test shuffling with a larger array to ensure efficiency."""
        # Create a larger dataset
        n_points = 10000
        centers = np.random.rand(n_points, 3).astype(np.float64)
        fields = np.random.rand(n_points, 4).astype(np.float64)

        # Create metadata
        metadata = ExternalAerodynamicsMetadata(
            filename="test_large",
            dataset_type=ModelType.VOLUME,  # Only has volume data for this test.
        )

        # Create data container
        data = ExternalAerodynamicsExtractedDataInMemory(metadata=metadata)
        data.volume_mesh_centers = centers
        data.volume_fields = fields

        # Store original
        original_centers = centers.copy()
        original_fields = fields.copy()

        # Shuffle
        result = shuffle_volume_data(data, seed=42)

        # Verify shapes
        assert result.volume_mesh_centers.shape == original_centers.shape
        assert result.volume_fields.shape == original_fields.shape

        # Verify coupling: for a sample of points, check they moved together
        sample_indices = [0, 100, 500, 1000, 5000, 9999]
        for orig_idx in sample_indices:
            orig_center = original_centers[orig_idx]
            orig_field = original_fields[orig_idx]

            # Find where this center moved to
            new_idx = np.where(
                np.all(np.isclose(result.volume_mesh_centers, orig_center), axis=1)
            )[0][0]

            # Check the field moved with it
            np.testing.assert_allclose(
                result.volume_fields[new_idx], orig_field, rtol=1e-10
            )
