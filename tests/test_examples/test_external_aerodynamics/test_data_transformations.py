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

import shutil
import tempfile
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv
import vtk

from examples.external_aerodynamics.constants import (
    ModelType,
)
from examples.external_aerodynamics.data_transformations import (
    ExternalAerodynamicsNumpyTransformation,
    ExternalAerodynamicsSTLTransformation,
    ExternalAerodynamicsSurfaceTransformation,
    ExternalAerodynamicsVolumeTransformation,
    ExternalAerodynamicsZarrTransformation,
)
from examples.external_aerodynamics.external_aero_geometry_data_processors import (
    update_geometry_data_to_float32,
)
from examples.external_aerodynamics.external_aero_surface_data_processors import (
    non_dimensionalize_surface_fields,
    normalize_surface_normals,
    update_surface_data_to_float32,
)
from examples.external_aerodynamics.external_aero_volume_data_processors import (
    non_dimensionalize_volume_fields,
    update_volume_data_to_float32,
)
from examples.external_aerodynamics.schemas import (
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
            stream_velocity=30.0,
            air_density=1.205,
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


class TestExternalAerodynamicsZarrTransformation:
    """Test the ExternalAerodynamicsZarrTransformation class."""

    def test_initialization(self):
        """Test initialization of Zarr transformation."""
        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsZarrTransformation(config)
        assert transform.config == config
        assert transform.compressor.cname == "zstd"
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
                    air_density=sample_data_raw.metadata.air_density,
                    stream_velocity=sample_data_raw.metadata.stream_velocity,
                ),
            ),
        )
        result = transform.transform(sample_data_raw)

        # Check that the fields were non-dimensionalized
        assert result.surface_fields.shape == sample_data_raw.surface_fields.shape
        # Verify non-dimensionalization: result = original / (rho * V^2)
        dynamic_pressure_factor = (
            sample_data_raw.metadata.air_density
            * sample_data_raw.metadata.stream_velocity**2
        )
        expected = np.array([[1.0, 0.5, 0.2, 101325.0]]) / dynamic_pressure_factor
        np.testing.assert_allclose(result.surface_fields, expected, rtol=1e-5)


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
                    air_density=sample_data_raw.metadata.air_density,
                    stream_velocity=sample_data_raw.metadata.stream_velocity,
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
            / sample_data_raw.metadata.stream_velocity
        )
        expected_pressure = np.array([[101325], [101300], [101320], [101310]]) / (
            sample_data_raw.metadata.air_density
            * sample_data_raw.metadata.stream_velocity**2
        )
        expected = np.concatenate([expected_velocity, expected_pressure], axis=-1)
        np.testing.assert_allclose(result.volume_fields, expected, rtol=1e-5)
