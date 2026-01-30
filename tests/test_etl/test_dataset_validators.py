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


from pathlib import Path

from physicsnemo_curator.etl.dataset_validators import (
    DatasetValidator,
    ValidationError,
    ValidationLevel,
)
from physicsnemo_curator.etl.processing_config import ProcessingConfig


class MockValidator(DatasetValidator):
    """Mock validator for testing."""

    def validate(self) -> list[ValidationError]:
        if self.kwargs.get("should_fail", False):
            return [
                ValidationError(
                    path=Path("/mock/path"),
                    message="Mock error",
                    level=ValidationLevel.STRUCTURE,
                )
            ]
        return []

    def validate_single_item(self, item: Path) -> list[ValidationError]:
        if self.kwargs.get("should_fail", False):
            return [
                ValidationError(
                    path=item,
                    message="Mock item error",
                    level=ValidationLevel.STRUCTURE,
                )
            ]
        return []


def test_validation_level_enum():
    """Test ValidationLevel enum values."""
    assert ValidationLevel.STRUCTURE.value == "structure"
    assert ValidationLevel.FIELDS.value == "fields"
    assert len(ValidationLevel) == 2


def test_validation_error():
    """Test ValidationError dataclass."""
    error = ValidationError(
        path=Path("/test/path"), message="Test error", level=ValidationLevel.STRUCTURE
    )

    assert error.path == Path("/test/path")
    assert error.message == "Test error"
    assert error.level == ValidationLevel.STRUCTURE


def test_parallel_validator_init():
    """Test ParallelDatasetValidator initialization."""
    config = ProcessingConfig(num_processes=4)
    validator = MockValidator(config, test_arg="value")

    assert validator.config == config
    assert validator.num_processes == 4
    assert validator.kwargs == {"test_arg": "value"}


def test_parallel_validator_validate():
    """Test ParallelDatasetValidator validate method."""
    config = ProcessingConfig(num_processes=4)

    # Test successful validation
    validator = MockValidator(config)
    errors = validator.validate()
    assert len(errors) == 0

    # Test failed validation
    validator = MockValidator(config, should_fail=True)
    errors = validator.validate()
    assert len(errors) == 1
    assert errors[0].message == "Mock error"
    assert errors[0].level == ValidationLevel.STRUCTURE


def test_parallel_validator_validate_single_item():
    """Test validate_single_item method."""
    config = ProcessingConfig(num_processes=4)
    validator = MockValidator(config)

    # Test successful validation
    errors = validator.validate_single_item(Path("/test/item"))
    assert len(errors) == 0

    # Test failed validation
    validator = MockValidator(config, should_fail=True)
    errors = validator.validate_single_item(Path("/test/item"))
    assert len(errors) == 1
    assert errors[0].message == "Mock item error"
    assert errors[0].level == ValidationLevel.STRUCTURE
