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

"""Base classes for dataset validation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from physicsnemo_curator.etl.processing_config import ProcessingConfig


class ValidationLevel(Enum):
    """Level of validation to perform."""

    STRUCTURE = "structure"  # Validate directory structure and file existence
    FIELDS = "fields"  # Also validate required fields are present


@dataclass
class ValidationError:
    """Validation error details."""

    path: Path
    message: str
    level: ValidationLevel


class DatasetValidator(ABC):
    """Base class for dataset validators."""

    def __init__(self, cfg: ProcessingConfig, **kwargs):
        """Initialize validator."""
        self.config = cfg
        self.num_processes = cfg.num_processes
        self.kwargs = kwargs

    def validate(self) -> list[ValidationError]:
        """Validate the dataset structure and fields.

        Returns:
            List of validation errors. Empty list means validation passed.
        """
        raise NotImplementedError("Validation must be implemented by subclass")

    @abstractmethod
    def validate_single_item(self, item: Path) -> list[ValidationError]:
        """Validate a single item (e.g., directory, file).

        Args:
            item: Path to the item to validate

        Returns:
            List of validation errors for this item.
        """
        pass
