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
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import dataset_validator
import pytest
from constants import DatasetKind, ModelType

from physicsnemo_curator.etl.dataset_validators import ValidationLevel
from physicsnemo_curator.etl.processing_config import ProcessingConfig


@pytest.fixture
def mock_input_dir(tmp_path):
    """Create a mock input directory structure."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Create a valid case directory
    case_dir = input_dir / "run_001"
    case_dir.mkdir()

    # Create mock files
    (case_dir / "drivaer_001.stl").touch()
    (case_dir / "volume_001.vtu").touch()
    (case_dir / "boundary_001.vtp").touch()

    return input_dir


@pytest.fixture
def basic_config():
    """Create a basic processing config."""
    return ProcessingConfig(num_processes=4)


def test_validator_init(basic_config):
    """Test validator initialization."""
    validator = dataset_validator.ExternalAerodynamicsDatasetValidator(
        basic_config,
        input_dir=Path("/test"),
        kind=DatasetKind.DRIVAERML,
        model_type=ModelType.SURFACE,
        validation_level="structure",
    )

    assert validator.input_dir == Path("/test")
    assert validator.kind == DatasetKind.DRIVAERML
    assert validator.model_type == ModelType.SURFACE
    assert validator.validation_level == ValidationLevel.STRUCTURE
    assert validator.num_processes == 4


def test_structure_validation(mock_input_dir, basic_config):
    """Test basic structure validation."""
    validator = dataset_validator.ExternalAerodynamicsDatasetValidator(
        basic_config,
        input_dir=mock_input_dir,
        kind=DatasetKind.DRIVAERML,
        model_type=ModelType.SURFACE,
        validation_level="structure",
    )

    errors = validator.validate()
    assert not errors, "Expected no validation errors"


def test_missing_directory(basic_config):
    """Test validation of non-existent directory."""
    validator = dataset_validator.ExternalAerodynamicsDatasetValidator(
        basic_config,
        input_dir=Path("/nonexistent"),
        kind=DatasetKind.DRIVAERML,
        model_type=ModelType.SURFACE,
    )

    errors = validator.validate()
    assert len(errors) == 1
    assert errors[0].level == ValidationLevel.STRUCTURE
    assert "does not exist" in errors[0].message


def test_invalid_case_name(mock_input_dir, basic_config):
    """Test invalid case name validation."""
    invalid_case = mock_input_dir / "invalid_name"
    invalid_case.mkdir()

    validator = dataset_validator.ExternalAerodynamicsDatasetValidator(
        basic_config,
        input_dir=mock_input_dir,
        kind=DatasetKind.DRIVAERML,
        model_type=ModelType.SURFACE,
    )

    errors = validator.validate_single_item(invalid_case)
    assert len(errors) == 1
    assert "must start with 'run_'" in errors[0].message


@patch("vtk.vtkXMLPolyDataReader")
def test_field_validation(mock_reader, mock_input_dir, basic_config):
    """Test validation of field variables."""
    # Mock VTK reader and data
    mock_data = Mock()
    mock_cell_data = Mock()
    mock_cell_data.GetArray.side_effect = lambda name: (
        None if name == "missing_field" else Mock()
    )
    mock_data.GetCellData.return_value = mock_cell_data

    mock_reader_instance = Mock()
    mock_reader_instance.GetOutput.return_value = mock_data
    mock_reader.return_value = mock_reader_instance

    validator = dataset_validator.ExternalAerodynamicsDatasetValidator(
        basic_config,
        input_dir=mock_input_dir,
        kind=DatasetKind.DRIVAERML,
        model_type=ModelType.SURFACE,
        surface_variables={"existing_field": "scalar", "missing_field": "scalar"},
        validation_level="fields",
    )

    case_dir = mock_input_dir / "run_001"
    errors = validator.validate_single_item(case_dir)
    assert any(
        error.level == ValidationLevel.FIELDS and "missing_field" in error.message
        for error in errors
    )


def test_combined_model_validation(mock_input_dir, basic_config):
    """Test validation with combined model type."""
    validator = dataset_validator.ExternalAerodynamicsDatasetValidator(
        basic_config,
        input_dir=mock_input_dir,
        kind=DatasetKind.DRIVAERML,
        model_type=ModelType.COMBINED,
    )

    # Remove volume file to trigger error
    case_dir = mock_input_dir / "run_001"
    (case_dir / "volume_001.vtu").unlink()

    errors = validator.validate_single_item(case_dir)
    assert any(
        error.level == ValidationLevel.STRUCTURE
        and "Volume data file not found" in error.message
        for error in errors
    )


def test_validation_logging(mock_input_dir, basic_config, caplog):
    """Test validation logging messages."""
    with caplog.at_level(logging.INFO):
        validator = dataset_validator.ExternalAerodynamicsDatasetValidator(
            basic_config,
            input_dir=mock_input_dir,
            kind=DatasetKind.DRIVAERML,
            model_type=ModelType.SURFACE,
        )
        validator.validate()

    assert "Starting External Aerodynamics dataset validation" in caplog.text
    assert "Found" in caplog.text and "case directories to validate" in caplog.text
    assert "Validation completed successfully" in caplog.text


@patch("dataset_validator.ProcessPoolExecutor")
@patch("dataset_validator.as_completed")
def test_parallel_validation(
    mock_as_completed, mock_executor, mock_input_dir, basic_config
):
    """Test parallel validation execution."""
    # Create mock for the executor instance and context manager
    mock_instance = MagicMock()
    mock_context = Mock()
    mock_future = Mock()

    # Set up the future's result
    mock_future.result.return_value = []

    # Set up the context manager's submit method
    mock_context.submit.return_value = mock_future

    # Set up the context manager entry/exit
    mock_instance.__enter__.return_value = mock_context

    # Set up the executor to return our mock instance
    mock_executor.return_value = mock_instance

    # Mock as_completed to return our future
    mock_as_completed.return_value = [mock_future]

    validator = dataset_validator.ExternalAerodynamicsDatasetValidator(
        basic_config,
        input_dir=mock_input_dir,
        kind=DatasetKind.DRIVAERML,
        model_type=ModelType.SURFACE,
    )

    errors = validator.validate()
    assert not errors

    # Verify parallel execution
    mock_executor.assert_called_once_with(max_workers=4)
    assert mock_context.submit.call_count == 1  # One case directory in mock_input_dir
