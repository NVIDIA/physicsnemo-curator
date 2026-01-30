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
from typing import List

import h5py

from physicsnemo_curator.etl.dataset_validators import (
    DatasetValidator,
    ValidationError,
    ValidationLevel,
)
from physicsnemo_curator.etl.processing_config import ProcessingConfig


class TutorialValidator(DatasetValidator):
    """Validator for HDF5 physics simulation dataset."""

    def __init__(
        self, cfg: ProcessingConfig, input_dir: str, validation_level: str = "fields"
    ):
        """Initialize the validator.

        Args:
            cfg: Processing configuration
            input_dir: Directory containing HDF5 files to validate
            validation_level: "structure" or "fields"
        """
        super().__init__(cfg)
        self.input_dir = Path(input_dir)
        self.validation_level = ValidationLevel(validation_level)

        # Define our expected schema
        self.required_groups = [
            "/fields",
            "/geometry",
            "/metadata",
            "/metadata/simulation_params",
        ]
        self.required_datasets = {
            "/fields/temperature": {"shape_dims": 1, "dtype": "float"},
            "/fields/velocity": {"shape_dims": 2, "expected_cols": 3, "dtype": "float"},
            "/geometry/coordinates": {
                "shape_dims": 2,
                "expected_cols": 3,
                "dtype": "float",
            },
        }
        self.required_attributes = {
            "/metadata": [
                "timestamp",
                "num_points",
                "temperature_units",
                "velocity_units",
            ],
            "/metadata/simulation_params": ["total_time"],
        }

    def validate(self) -> List[ValidationError]:
        """Validate the entire dataset.

        Returns:
            List of validation errors (empty if validation passes)
        """
        errors = []

        # Check if input directory exists
        if not self.input_dir.exists():
            errors.append(
                ValidationError(
                    path=self.input_dir,
                    message=f"Input directory does not exist: {self.input_dir}",
                    level=self.validation_level,
                )
            )
            return errors

        # Find all HDF5 files
        h5_files = list(self.input_dir.glob("*.h5"))

        if not h5_files:
            errors.append(
                ValidationError(
                    path=self.input_dir,
                    message="No HDF5 files found in input directory",
                    level=self.validation_level,
                )
            )
            return errors

        # Validate each file
        for h5_file in h5_files:
            file_errors = self.validate_single_item(h5_file)
            errors.extend(file_errors)

        return errors

    def validate_single_item(self, item: Path) -> List[ValidationError]:
        """Validate a single HDF5 file.

        Args:
            item: Path to HDF5 file to validate

        Returns:
            List of validation errors for this file
        """
        errors = []

        try:
            with h5py.File(item, "r") as f:
                # Structure validation
                errors.extend(self._validate_structure(f, item))

                # Field validation (if requested and structure is valid)
                if self.validation_level == ValidationLevel.FIELDS and not errors:
                    errors.extend(self._validate_fields(f, item))

        except Exception as e:
            errors.append(
                ValidationError(
                    path=item,
                    message=f"Failed to open HDF5 file: {str(e)}",
                    level=self.validation_level,
                )
            )

        return errors

    def _validate_structure(
        self, f: h5py.File, file_path: Path
    ) -> List[ValidationError]:
        """Validate HDF5 file structure."""
        errors = []

        # Check required groups exist
        errors.extend(
            [
                ValidationError(
                    path=file_path,
                    message=f"Missing required group: {group_path}",
                    level=self.validation_level,
                )
                for group_path in self.required_groups
                if group_path not in f
            ]
        )

        # Check required datasets exist and have correct structure
        for dataset_path, requirements in self.required_datasets.items():
            if dataset_path not in f:
                errors.append(
                    ValidationError(
                        path=file_path,
                        message=f"Missing required dataset: {dataset_path}",
                        level=self.validation_level,
                    )
                )
                continue

            dataset = f[dataset_path]

            # Check dimensions
            if len(dataset.shape) != requirements["shape_dims"]:
                errors.append(
                    ValidationError(
                        path=file_path,
                        message=f"Dataset {dataset_path} has wrong dimensions: expected {requirements['shape_dims']}D, got {len(dataset.shape)}D",
                        level=self.validation_level,
                    )
                )

            # Check column count for 2D arrays
            if "expected_cols" in requirements and len(dataset.shape) >= 2:
                if dataset.shape[1] != requirements["expected_cols"]:
                    errors.append(
                        ValidationError(
                            path=file_path,
                            message=f"Dataset {dataset_path} has wrong number of columns: expected {requirements['expected_cols']}, got {dataset.shape[1]}",
                            level=self.validation_level,
                        )
                    )

        # Check required attributes exist
        errors.extend(
            [
                ValidationError(
                    path=file_path,
                    message=f"Missing required attribute: {group_path}@{attr_name}",
                    level=self.validation_level,
                )
                for group_path, attr_list in self.required_attributes.items()
                if group_path in f
                for attr_name in attr_list
                if attr_name not in f[group_path].attrs
            ]
        )

        return errors

    def _validate_fields(self, f: h5py.File, file_path: Path) -> List[ValidationError]:
        """Validate field data content."""
        errors = []

        # Check that datasets have consistent sizes
        if "/fields/temperature" in f and "/geometry/coordinates" in f:
            temp_size = f["/fields/temperature"].shape[0]
            coord_size = f["/geometry/coordinates"].shape[0]

            if temp_size != coord_size:
                errors.append(
                    ValidationError(
                        path=file_path,
                        message=f"Inconsistent data sizes: temperature has {temp_size} points, coordinates has {coord_size} points",
                        level=self.validation_level,
                    )
                )

        # Check for reasonable data ranges
        if "/fields/temperature" in f:
            temp_data = f["/fields/temperature"][:]
            if temp_data.min() < 0 or temp_data.max() > 10000:  # Kelvin range check
                errors.append(
                    ValidationError(
                        path=file_path,
                        message=f"Temperature data out of reasonable range: [{temp_data.min():.1f}, {temp_data.max():.1f}] K",
                        level=self.validation_level,
                    )
                )

        return errors
