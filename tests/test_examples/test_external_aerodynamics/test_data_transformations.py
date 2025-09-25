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
from pathlib import Path

import numpy as np
import pytest

from examples.external_aerodynamics.constants import (
    ModelType,
)
from examples.external_aerodynamics.data_sources import (
    DatasetKind,
    ExternalAerodynamicsDataSource,
)
from examples.external_aerodynamics.data_transformations import (
    ExternalAerodynamicsNumpyTransformation,
    ExternalAerodynamicsPreprocessingTransformation,
    ExternalAerodynamicsZarrTransformation,
)
from examples.external_aerodynamics.schemas import (
    ExternalAerodynamicsExtractedDataInMemory,
    ExternalAerodynamicsMetadata,
    ExternalAerodynamicsNumpyDataInMemory,
    ExternalAerodynamicsZarrDataInMemory,
    PreparedZarrArrayInfo,
)
from physicsnemo_curator.etl.processing_config import ProcessingConfig

from .utils import create_mock_stl, create_mock_surface_vtk, create_mock_volume_vtk


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_data_raw(temp_dir):
    """Create sample Raw DrivAerML data for testing."""

    case_dir = temp_dir / "run_001"
    case_dir.mkdir(parents=True, exist_ok=True)

    # Create STL file with correct name and path
    stl_path = case_dir / "drivaer_001.stl"
    stl_path.parent.mkdir(parents=True, exist_ok=True)
    create_mock_stl(stl_path)

    # Create VTK files with correct paths
    vtk_path = case_dir
    vtk_path.mkdir(parents=True, exist_ok=True)
    create_mock_surface_vtk(vtk_path / "boundary_001.vtp")  # Updated surface file name
    create_mock_volume_vtk(vtk_path / "volume_001.vtu")  # Updated volume file name

    config = ProcessingConfig(num_processes=1)
    source = ExternalAerodynamicsDataSource(
        config,
        input_dir=case_dir.parent,
        kind=DatasetKind.DRIVAERML,
        model_type=ModelType.COMBINED,
    )

    data = source.read_file("run_001")

    return data


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


class TestExternalAerodynamicsPreprocessingTransformation:
    """Test the ExternalAerodynamicsPreprocessingTransformation class."""

    def test_initialization(self):
        """Test initialization of preprocessing transformation."""
        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsPreprocessingTransformation(
            config,
            surface_variables={
                "pMeanTrim": "scalar",
                "wallShearStressMeanTrim": "vector",
            },
            volume_variables={
                "UMeanTrim": "vector",
                "pMeanTrim": "scalar",
            },
        )
        assert transform.config == config
        assert transform.surface_variables == {
            "pMeanTrim": "scalar",
            "wallShearStressMeanTrim": "vector",
        }
        assert transform.volume_variables == {
            "UMeanTrim": "vector",
            "pMeanTrim": "scalar",
        }
        assert transform.decimation_algo is None
        assert transform.target_reduction is None

    def test_initialization_with_decimation(self):
        """Test initialization with decimation parameters."""
        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsPreprocessingTransformation(
            config,
            surface_variables={"pMeanTrim": "scalar"},
            volume_variables={"UMeanTrim": "vector"},
            decimation={"algo": "decimate_pro", "reduction": 0.5},
        )
        assert transform.decimation_algo == "decimate_pro"
        assert transform.target_reduction == 0.5

    def test_initialization_invalid_decimation_algo(self):
        """Test initialization with invalid decimation algorithm."""
        config = ProcessingConfig(num_processes=1)
        with pytest.raises(ValueError, match="Unsupported decimation algo"):
            ExternalAerodynamicsPreprocessingTransformation(
                config,
                decimation={"algo": "invalid_algo", "reduction": 0.5},
            )

    def test_initialization_invalid_reduction(self):
        """Test initialization with invalid reduction value."""
        config = ProcessingConfig(num_processes=1)
        with pytest.raises(ValueError, match="Expected value in \[0, 1\)"):
            ExternalAerodynamicsPreprocessingTransformation(
                config,
                decimation={"algo": "decimate_pro", "reduction": 1.5},
            )

    def test_transform_basic(self, sample_data_raw):
        """Test basic preprocessing transformation of External Aerodynamics data."""
        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsPreprocessingTransformation(
            config,
            surface_variables={
                "pMeanTrim": "scalar",
                "wallShearStressMeanTrim": "vector",
            },
            volume_variables={
                "UMeanTrim": "vector",
                "pMeanTrim": "scalar",
            },
        )

        result = transform.transform(sample_data_raw)
        assert isinstance(result, ExternalAerodynamicsExtractedDataInMemory)

        # Check that raw data is cleaned up
        assert result.stl_polydata is None
        assert result.surface_polydata is None
        assert result.volume_unstructured_grid is None

        # Check that processed data is updated and converted to float32
        assert result.stl_coordinates.dtype == np.float32
        assert result.stl_centers.dtype == np.float32
        assert result.stl_faces.dtype == np.int32
        assert result.stl_areas.dtype == np.float32

        # Check metadata updates
        assert result.metadata.stream_velocity == 30.0  # From PhysicsConstants
        assert result.metadata.air_density == 1.205  # From PhysicsConstants
        assert result.metadata.num_points == 3  # From the test mesh
        assert result.metadata.num_faces == 3  # From the test mesh

    def test_transform_with_surface_data(self, sample_data_raw):
        """Test preprocessing transformation with surface data."""
        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsPreprocessingTransformation(
            config,
            surface_variables={
                "pMeanTrim": "scalar",
                "wallShearStressMeanTrim": "vector",
            },
            volume_variables={
                "UMeanTrim": "vector",
                "pMeanTrim": "scalar",
            },
        )

        # Set volume data to None since this is a surface data transformation only test.
        sample_data_raw.volume_unstructured_grid = None
        result = transform.transform(sample_data_raw)

        # Check surface data processing
        assert result.surface_mesh_centers.dtype == np.float32
        assert result.surface_normals.dtype == np.float32
        assert result.surface_areas.dtype == np.float32
        assert result.surface_fields.dtype == np.float32

        # Check that surface fields are non-dimensionalized
        # The normalization factor should be (air_density * stream_velocity^2)
        normalization_factor = 1.205 * 30.0**2
        np.testing.assert_array_almost_equal(
            result.surface_fields,
            np.array(
                [[1.0 / normalization_factor, 1.0 / normalization_factor, 0.0, 0.0]]
            ),
        )

        # Check that normals are normalized
        norms = np.linalg.norm(result.surface_normals, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones_like(norms))

    def test_transform_with_volume_data(self, sample_data_raw):
        """Test preprocessing transformation with volume data."""
        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsPreprocessingTransformation(
            config,
            volume_variables={
                "UMeanTrim": "vector",
                "pMeanTrim": "scalar",
            },
        )

        # Set surface data to None since this is a volume data transformation only test.
        sample_data_raw.surface_polydata = None
        result = transform.transform(sample_data_raw)

        # Check volume data processing
        assert result.volume_mesh_centers.dtype == np.float32
        assert result.volume_fields.dtype == np.float32
        assert result.volume_fields is not None

    def test_transform_no_surface_data(self, sample_data_raw):
        """Test preprocessing transformation without surface data."""
        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsPreprocessingTransformation(
            config,
            volume_variables={"UMeanTrim": "vector"},
        )

        # Set surface data to None.
        sample_data_raw.surface_polydata = None

        result = transform.transform(sample_data_raw)

        # Check that surface data is None
        assert result.surface_mesh_centers is None
        assert result.surface_normals is None
        assert result.surface_areas is None
        assert result.surface_fields is None

    def test_transform_no_volume_data(self, sample_data_raw):
        """Test preprocessing transformation without volume data."""
        config = ProcessingConfig(num_processes=1)
        transform = ExternalAerodynamicsPreprocessingTransformation(
            config,
            surface_variables={"pMeanTrim": "scalar"},
        )

        # Set volume data to None.
        sample_data_raw.volume_unstructured_grid = None

        result = transform.transform(sample_data_raw)

        # Check that volume data is None
        assert result.volume_mesh_centers is None
        assert result.volume_fields is None
