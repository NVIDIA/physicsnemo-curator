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

"""Validation utilities for External Aerodynamics datasets."""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import vtk
from constants import DatasetKind, ModelType
from paths import get_path_getter

from physicsnemo_curator.etl.dataset_validators import (
    DatasetValidator,
    ValidationError,
    ValidationLevel,
)
from physicsnemo_curator.etl.processing_config import ProcessingConfig


class ExternalAerodynamicsDatasetValidator(DatasetValidator):
    """Validator for External Aerodynamics datasets."""

    def __init__(
        self,
        cfg: ProcessingConfig,
        validation_level: Optional[str] = "structure",
        input_dir: Optional[Path] = None,
        kind: DatasetKind | str = DatasetKind.DRIVAERML,
        surface_variables: Optional[dict[str, str]] = None,
        volume_variables: Optional[dict[str, str]] = None,
        model_type: Optional[ModelType | str] = None,
        **kwargs,
    ):
        """Initialize External Aerodynamics validator."""
        super().__init__(cfg, **kwargs)

        # Get parameters from config
        self.input_dir = Path(input_dir) if input_dir else None
        self.kind = DatasetKind(kind) if isinstance(kind, str) else kind
        self.surface_variables = surface_variables
        self.volume_variables = volume_variables
        self.model_type = ModelType(model_type) if isinstance(model_type, str) else None
        self.validation_level = ValidationLevel(validation_level)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.path_getter = get_path_getter(kind)

    def validate(self) -> list[ValidationError]:
        """Validate the dataset.

        Returns:
            List of validation errors. Empty list means validation passed.
        """
        self.logger.info(
            f"Starting External Aerodynamics dataset validation (level: {self.validation_level.value})"
        )

        if not self.input_dir.exists():
            return [
                ValidationError(
                    self.input_dir,
                    "Input directory does not exist",
                    ValidationLevel.STRUCTURE,
                )
            ]

        # Get all case directories
        case_dirs = sorted(d for d in self.input_dir.iterdir() if d.is_dir())
        if not case_dirs:
            return [
                ValidationError(
                    self.input_dir,
                    "No case directories found",
                    ValidationLevel.STRUCTURE,
                )
            ]

        self.logger.info(f"Found {len(case_dirs)} case directories to validate")

        all_errors = []
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            future_to_dir = {
                executor.submit(self.validate_single_item, case_dir): case_dir
                for case_dir in case_dirs
            }

            # Collect results as they complete
            for future in as_completed(future_to_dir):
                case_dir = future_to_dir[future]
                try:
                    all_errors.extend(future.result())
                except Exception as e:
                    self.logger.error(f"Error validating {case_dir}: {str(e)}")
                    all_errors.append(
                        ValidationError(
                            case_dir,
                            f"Validation failed: {str(e)}",
                            ValidationLevel.STRUCTURE,
                        )
                    )

        if all_errors:
            self.logger.warning(f"Validation found {len(all_errors)} errors")
        else:
            self.logger.info("Validation completed successfully")

        return all_errors

    def validate_single_item(self, case_dir: Path) -> list[ValidationError]:
        """Validate a single case directory.

        This method runs in a separate process for parallel validation.
        """
        errors = []

        # Validate directory name format
        if self.kind in [DatasetKind.DRIVAERML, DatasetKind.AHMEDML]:
            if not case_dir.name.startswith("run_"):
                errors.append(
                    ValidationError(
                        case_dir,
                        "Directory name must start with 'run_'",
                        ValidationLevel.STRUCTURE,
                    )
                )
                return errors  # Return early since path validation will fail

        # Validate geometry file exists
        geom_path = self.path_getter.geometry_path(case_dir)
        if not geom_path.exists():
            errors.append(
                ValidationError(
                    geom_path, "Geometry file not found", ValidationLevel.STRUCTURE
                )
            )

        # Validate volume data if needed
        if self.model_type in (ModelType.VOLUME, ModelType.COMBINED):
            volume_path = self.path_getter.volume_path(case_dir)
            if not volume_path.exists():
                errors.append(
                    ValidationError(
                        volume_path,
                        "Volume data file not found",
                        ValidationLevel.STRUCTURE,
                    )
                )
            elif self.validation_level == ValidationLevel.FIELDS:
                errors.extend(self._validate_volume_fields(volume_path))

        # Validate surface data if needed
        if self.model_type in (ModelType.SURFACE, ModelType.COMBINED):
            surface_path = self.path_getter.surface_path(case_dir)
            if not surface_path.exists():
                errors.append(
                    ValidationError(
                        surface_path,
                        "Surface data file not found",
                        ValidationLevel.STRUCTURE,
                    )
                )
            elif self.validation_level == ValidationLevel.FIELDS:
                errors.extend(self._validate_surface_fields(surface_path))

        return errors

    def _validate_volume_fields(self, path: Path) -> list[ValidationError]:
        """Validate volume field names exist."""
        errors = []
        if not self.volume_variables:
            return errors

        try:
            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(str(path))
            reader.Update()
            data = reader.GetOutput()
            point_data = data.GetPointData()

            errors.extend(
                ValidationError(
                    path,
                    f"Missing volume field: {field_name}",
                    ValidationLevel.FIELDS,
                )
                for field_name in self.volume_variables
                if point_data.GetArray(field_name) is None
            )
        except Exception as e:
            errors.append(
                ValidationError(
                    path,
                    f"Could not read volume file: {str(e)}",
                    ValidationLevel.STRUCTURE,
                )
            )
        return errors

    def _validate_surface_fields(self, path: Path) -> list[ValidationError]:
        """Validate surface field names exist."""
        errors = []
        if not self.surface_variables:
            return errors

        try:
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(str(path))
            reader.Update()
            data = reader.GetOutput()
            cell_data = data.GetCellData()

            errors.extend(
                ValidationError(
                    path,
                    f"Missing surface field: {field_name}",
                    ValidationLevel.FIELDS,
                )
                for field_name in self.surface_variables
                if cell_data.GetArray(field_name) is None
            )
        except Exception as e:
            errors.append(
                ValidationError(
                    path,
                    f"Could not read surface file: {str(e)}",
                    ValidationLevel.STRUCTURE,
                )
            )
        return errors
