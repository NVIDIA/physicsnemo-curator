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

from unittest.mock import Mock, patch

import pytest
from omegaconf import OmegaConf

from physicsnemo_curator.run_etl import main


@pytest.fixture
def mock_config():
    """Create a mock Hydra config."""
    return OmegaConf.create(
        {
            "etl": {
                "processing": {"num_processes": 2},
                "source": {
                    "_target_": "examples.external_aerodynamics.domino.data_sources.DoMINODataSource",
                    "input_dir": "/test/input",
                    "kind": "DRIVESIM",
                },
                "transformations": {
                    "numpy": {
                        "_target_": "examples.external_aerodynamics.domino.transformations.DoMINONumpyTransformation",
                        "model_type": "SURFACE",
                    }
                },
                "sink": {
                    "_target_": "examples.external_aerodynamics.domino.data_sources.DoMINODataSource",
                    "output_dir": "/test/output",
                    "serialization_method": "numpy",
                },
            }
        }
    )


@patch("physicsnemo_curator.run_etl.ParallelProcessor")
@patch("physicsnemo_curator.run_etl.instantiate")
def test_main_execution(mock_instantiate, mock_processor, mock_config):
    """Test main pipeline execution."""
    # Mock components
    mock_source = Mock()
    mock_transform = Mock()
    mock_sink = Mock()
    mock_instantiate.side_effect = [mock_source, mock_sink, mock_transform]

    # Mock processor
    processor_instance = Mock()
    mock_processor.return_value = processor_instance

    # Run main
    with patch("physicsnemo_curator.run_etl.curator_utils.setup_logger"):
        main(mock_config)

    # Verify component creation
    assert mock_instantiate.call_count == 3

    # Verify processor creation and execution
    mock_processor.assert_called_once()
    processor_instance.run.assert_called_once()


def test_main_with_processing_error(mock_config):
    """Test main function handling processing errors."""
    with (
        patch("physicsnemo_curator.run_etl.curator_utils.setup_logger"),
        patch("physicsnemo_curator.run_etl.ParallelProcessor") as mock_processor,
    ):

        # Make processor raise an error
        processor_instance = Mock()
        processor_instance.run.side_effect = Exception("Test error")
        mock_processor.return_value = processor_instance

        with pytest.raises(Exception):
            main(mock_config)
