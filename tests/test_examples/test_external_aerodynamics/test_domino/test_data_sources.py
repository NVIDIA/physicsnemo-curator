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
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import pyvista as pv
import vtk
from numcodecs import Blosc

from examples.external_aerodynamics.domino.constants import (
    DatasetKind,
    ModelType,
)
from examples.external_aerodynamics.domino.data_sources import (
    DoMINODataSource,
)
from examples.external_aerodynamics.domino.paths import (
    DrivAerMLPaths,
    DriveSimPaths,
)
from examples.external_aerodynamics.domino.schemas import (
    DoMINOExtractedDataInMemory,
    DoMINOMetadata,
    DoMINONumpyDataInMemory,
    DoMINONumpyMetadata,
    DoMINOZarrDataInMemory,
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
def mock_domino_data_drivaerml(temp_dir):
    """Create mock DrivAerML DoMINO data structure."""
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

    return case_dir


@pytest.fixture
def mock_drivesim_data(temp_dir):
    """Create mock DriveSim DoMINO data structure."""
    case_dir = temp_dir / "run_001"
    case_dir.mkdir(parents=True, exist_ok=True)

    # Create STL file with DriveSim path/name
    stl_path = case_dir / "body.stl"  # Different from DrivAerML
    stl_path.parent.mkdir(parents=True, exist_ok=True)
    create_mock_stl(stl_path)

    # Create VTK files with DriveSim paths
    vtk_path = case_dir / "VTK" / "simpleFoam_steady_3000"  # Different structure
    vtk_path.mkdir(parents=True, exist_ok=True)

    # Create surface data
    surface_path = vtk_path / "boundary"
    surface_path.mkdir(parents=True, exist_ok=True)
    create_mock_surface_vtk(surface_path / "aero_suv.vtp")  # Different filename

    # Create volume data
    create_mock_volume_vtk(vtk_path / "internal.vtu")  # Different filename

    return case_dir


class TestDoMINODataSource:
    """Test the DoMINODataSource class."""

    def test_drivaerml_initialization(self, temp_dir):
        """Test data source initialization."""
        config = ProcessingConfig(num_processes=1)
        source = DoMINODataSource(
            config,
            input_dir=temp_dir,
            kind=DatasetKind.DRIVAERML,
            model_type=ModelType.SURFACE,
        )

        assert source.input_dir == temp_dir
        assert source.kind == DatasetKind.DRIVAERML
        assert source.model_type == ModelType.SURFACE
        assert isinstance(source.path_getter, DrivAerMLPaths.__class__)

    def test_drivaerml_initialization_with_invalid_directory_raises_error(self):
        """Test initialization with non-existent directory."""
        config = ProcessingConfig(num_processes=1)
        with pytest.raises(FileNotFoundError):
            DoMINODataSource(
                config, input_dir="/nonexistent/path", kind=DatasetKind.DRIVAERML
            )

    def test_drivaerml_get_file_list(self, mock_domino_data_drivaerml):
        """Test file list retrieval."""
        config = ProcessingConfig(num_processes=1)
        source = DoMINODataSource(
            config,
            input_dir=mock_domino_data_drivaerml.parent,
            kind=DatasetKind.DRIVAERML,
        )

        files = source.get_file_list()
        assert len(files) == 1
        assert files[0] == "run_001"

    def test_drivaerml_read_surface_data(self, mock_domino_data_drivaerml):
        """Test reading surface data."""
        config = ProcessingConfig(num_processes=1)
        source = DoMINODataSource(
            config,
            input_dir=mock_domino_data_drivaerml.parent,
            kind=DatasetKind.DRIVAERML,
            model_type=ModelType.SURFACE,
        )

        data = source.read_file("run_001")

        assert isinstance(data, DoMINOExtractedDataInMemory)
        assert data.surface_polydata is not None
        assert isinstance(data.surface_polydata, pv.PolyData)
        assert data.metadata.filename == "run_001"

    def test_drivaerml_read_volume_data(self, mock_domino_data_drivaerml):
        """Test reading volume data."""
        config = ProcessingConfig(num_processes=1)
        source = DoMINODataSource(
            config,
            input_dir=mock_domino_data_drivaerml.parent,
            kind=DatasetKind.DRIVAERML,
            model_type=ModelType.VOLUME,
        )

        data = source.read_file("run_001")

        assert isinstance(data, DoMINOExtractedDataInMemory)
        assert data.volume_unstructured_grid is not None
        assert isinstance(data.volume_unstructured_grid, vtk.vtkUnstructuredGrid)
        assert data.metadata.filename == "run_001"

    def test_drivaerml_write_numpy(self, temp_dir):
        """Test writing data in NumPy format."""
        config = ProcessingConfig(num_processes=1)
        source = DoMINODataSource(
            config, output_dir=temp_dir, serialization_method="numpy"
        )

        test_data = DoMINONumpyDataInMemory(
            metadata=DoMINONumpyMetadata(
                filename="test_case",
                stream_velocity=[30.0, 0.0, 0.0],
                air_density=1.225,
            ),
            stl_coordinates=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            stl_centers=np.array([[0.5, 0.5, 0.5]]),
            stl_faces=np.array([[0, 1, 2]]),
            stl_areas=np.array([1.0]),
        )
        source.write(test_data, "test_case")
        assert (temp_dir / "test_case.npz").exists()

    def test_drivaerml_write_zarr(self, temp_dir):
        """Test writing data in Zarr format."""
        config = ProcessingConfig(num_processes=1)
        source = DoMINODataSource(
            config, output_dir=temp_dir, serialization_method="zarr"
        )
        compressor_for_test = Blosc(
            cname="zstd",
            clevel=5,
            shuffle=Blosc.SHUFFLE,
        )

        test_data = DoMINOZarrDataInMemory(
            stl_coordinates=PreparedZarrArrayInfo(
                data=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
                chunks=(2, 3),
                compressor=compressor_for_test,
            ),
            stl_centers=PreparedZarrArrayInfo(
                data=np.array([[0.5, 0.5, 0.5]]),
                chunks=(1, 3),
                compressor=compressor_for_test,
            ),
            stl_faces=PreparedZarrArrayInfo(
                data=np.array([[0, 1, 2]]),
                chunks=(1, 3),
                compressor=compressor_for_test,
            ),
            stl_areas=PreparedZarrArrayInfo(
                data=np.array([1.0]),
                chunks=(1,),
                compressor=compressor_for_test,
            ),
            metadata=DoMINOMetadata(
                stream_velocity=[30.0, 0.0, 0.0],
                air_density=1.225,
                filename="test_case",
                dataset_type=ModelType.COMBINED,
                x_bound=(0.0, 1.0),
                y_bound=(0.0, 1.0),
                z_bound=(0.0, 1.0),
                num_points=3,
                num_faces=3,
                decimation_reduction=0.5,
                decimation_algo="decimate_pro",
            ),
            # Surface data
            surface_mesh_centers=PreparedZarrArrayInfo(
                data=np.array([[0.5, 0.5, 0.5]]),
                chunks=(1, 3),
                compressor=compressor_for_test,
            ),
            surface_normals=PreparedZarrArrayInfo(
                data=np.array([[0.0, 0.0, 1.0]]),
                chunks=(1, 3),
                compressor=compressor_for_test,
            ),
            surface_areas=PreparedZarrArrayInfo(
                data=np.array([1.0]),
                chunks=(1,),
                compressor=compressor_for_test,
            ),
            surface_fields=PreparedZarrArrayInfo(
                data=np.array([[1.0, 0.0, 0.0]]),
                chunks=(1, 3),
                compressor=compressor_for_test,
            ),
            # Volume data
            volume_mesh_centers=PreparedZarrArrayInfo(
                data=np.array([[0.5, 0.5, 0.5]]),
                chunks=(1, 3),
                compressor=compressor_for_test,
            ),
            volume_fields=PreparedZarrArrayInfo(
                data=np.array([[1.0, 0.0, 0.0]]),
                chunks=(1, 3),
                compressor=compressor_for_test,
            ),
        )

        source.write(test_data, "test_case")

        # Verify the Zarr store was created
        assert (temp_dir / "test_case.zarr").exists()

    def test_drivesim_initialization(self, temp_dir):
        """Test data source initialization with DriveSim dataset."""
        config = ProcessingConfig(num_processes=1)
        source = DoMINODataSource(
            config,
            input_dir=temp_dir,
            kind=DatasetKind.DRIVESIM,
            model_type=ModelType.SURFACE,
        )

        assert source.input_dir == temp_dir
        assert source.kind == DatasetKind.DRIVESIM
        assert source.model_type == ModelType.SURFACE
        assert isinstance(source.path_getter, DriveSimPaths.__class__)

    def test_drivesim_read_surface_data(self, mock_drivesim_data):
        """Test reading surface data from DriveSim dataset."""
        config = ProcessingConfig(num_processes=1)
        source = DoMINODataSource(
            config,
            input_dir=mock_drivesim_data.parent,
            kind=DatasetKind.DRIVESIM,
            model_type=ModelType.SURFACE,
        )

        data = source.read_file("run_001")

        assert isinstance(data, DoMINOExtractedDataInMemory)
        assert data.surface_polydata is not None
        assert isinstance(data.surface_polydata, pv.PolyData)
        assert data.metadata.filename == "run_001"

    def test_drivesim_read_volume_data(self, mock_drivesim_data):
        """Test reading volume data from DriveSim dataset."""
        config = ProcessingConfig(num_processes=1)
        source = DoMINODataSource(
            config,
            input_dir=mock_drivesim_data.parent,
            kind=DatasetKind.DRIVESIM,
            model_type=ModelType.VOLUME,
        )

        data = source.read_file("run_001")

        assert isinstance(data, DoMINOExtractedDataInMemory)
        assert data.volume_unstructured_grid is not None
        assert isinstance(data.volume_unstructured_grid, vtk.vtkUnstructuredGrid)
        assert data.metadata.filename == "run_001"

    def test_path_getter_selection(self):
        """Test correct path getter class is selected based on dataset kind."""
        config = ProcessingConfig(num_processes=1)

        drivesim_source = DoMINODataSource(config, kind=DatasetKind.DRIVESIM)
        assert isinstance(drivesim_source.path_getter, DriveSimPaths.__class__)

        drivaerml_source = DoMINODataSource(config, kind=DatasetKind.DRIVAERML)
        assert isinstance(drivaerml_source.path_getter, DrivAerMLPaths.__class__)

    def test_drivesim_initialization_with_invalid_directory_raises_error(self):
        """Test initialization with non-existent directory for DriveSim."""
        config = ProcessingConfig(num_processes=1)
        with pytest.raises(FileNotFoundError):
            DoMINODataSource(
                config, input_dir="/nonexistent/path", kind=DatasetKind.DRIVESIM
            )

    def test_drivesim_get_file_list(self, mock_drivesim_data):
        """Test file list retrieval for DriveSim dataset."""
        config = ProcessingConfig(num_processes=1)
        source = DoMINODataSource(
            config, input_dir=mock_drivesim_data.parent, kind=DatasetKind.DRIVESIM
        )

        files = source.get_file_list()
        assert len(files) == 1
        assert files[0] == "run_001"

    def test_should_skip_when_overwrite_existing_is_true(self, temp_dir):
        """Test that should_skip returns False when overwrite_existing is True."""

        for method in ["numpy", "zarr"]:
            data_source = DoMINODataSource(
                cfg=Mock(spec=ProcessingConfig),
                output_dir=temp_dir,
                serialization_method=method,
                overwrite_existing=True,
            )
            assert not data_source.should_skip("test_file")

    def test_should_skip_numpy_when_file_exists(self, temp_dir):
        """Test that should_skip returns True when numpy file exists and overwrite_existing is False."""

        data_source = DoMINODataSource(
            cfg=Mock(spec=ProcessingConfig),
            output_dir=temp_dir,
            serialization_method="numpy",
            overwrite_existing=False,
        )
        # Create a dummy .npz file
        test_file = data_source.output_dir / "test_file.npz"
        test_file.touch()

        assert data_source.should_skip("test_file")

    def test_should_skip_zarr_when_file_exists(self, temp_dir):
        """Test that should_skip returns True when numpy file exists and overwrite_existing is False."""

        data_source = DoMINODataSource(
            cfg=Mock(spec=ProcessingConfig),
            output_dir=temp_dir,
            serialization_method="zarr",
            overwrite_existing=False,
        )
        # Create a dummy .zarr file
        test_file = data_source.output_dir / "test_file.zarr"
        test_file.mkdir(parents=True, exist_ok=True)

        assert data_source.should_skip("test_file")

    def test_should_skip_when_file_does_not_exist(self, temp_dir):
        """Test that should_skip returns False when numpy file doesn't exist and overwrite_existing is False."""

        for method in ["numpy", "zarr"]:
            data_source = DoMINODataSource(
                cfg=Mock(spec=ProcessingConfig),
                output_dir=temp_dir,
                serialization_method=method,
                overwrite_existing=False,
            )
            assert not data_source.should_skip("test_file")
