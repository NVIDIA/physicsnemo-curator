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

import pytest

from physicsnemo_curator.etl.data_transformations import DataTransformation
from physicsnemo_curator.etl.processing_config import ProcessingConfig


class MockTransformation(DataTransformation):
    """Mock implementation of DataTransformation for testing."""

    def __init__(self, config: ProcessingConfig):
        super().__init__(config)
        self.transform_called = False

    def transform(self, data):
        self.transform_called = True
        # Add a new field to demonstrate transformation
        data["transformed"] = True
        return data


def test_datatransformation_direct_instantiation_raises_error():
    """Test that DataTransformation cannot be instantiated directly."""
    with pytest.raises(TypeError):
        DataTransformation({})


def test_mock_transformation_instantiation():
    """Test that concrete implementation can be instantiated."""
    config = ProcessingConfig(num_processes=1)
    transform = MockTransformation(config)
    assert isinstance(transform, DataTransformation)
    assert transform.config == config
    assert transform.logger.name == "MockTransformation"


def test_mock_transformation_works_as_expected():
    """Test transform method modifies data as expected."""
    config = ProcessingConfig(num_processes=1)
    transform = MockTransformation(config)

    input_data = {"original": "data"}
    result = transform.transform(input_data)

    assert transform.transform_called
    assert result["transformed"] is True
    assert result["original"] == "data"
