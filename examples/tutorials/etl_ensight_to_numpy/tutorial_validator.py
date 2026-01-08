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

from pathlib import Path
from typing import List

import numpy as np
import pyvista as pv

from physicsnemo_curator.etl.dataset_validators import (
    DatasetValidator,
    ValidationError,
    ValidationLevel,
)
from physicsnemo_curator.etl.processing_config import ProcessingConfig


class EnSightTutorialValidator(DatasetValidator):
    """Validator for EnSight Gold physics simulation dataset."""

    def __init__(
        self, cfg: ProcessingConfig, input_dir: str, validation_level: str = "fields"
    ):
        """Initialize the validator.

        Args:
            cfg: Processing configuration
            input_dir: Directory containing EnSight files to validate
            validation_level: "structure" or "fields"
        """
        super().__init__(cfg)
        self.input_dir = Path(input_dir)
        self.validation_level = ValidationLevel(validation_level)

        # Define minimum requirements
        self.min_points = 3
        self.min_cells = 1
        self.expected_spatial_dims = 3

    def validate(self) -> List[ValidationError]:
        """Validate all EnSight case files in the input directory."""
        errors = []

        if not self.input_dir.exists():
            errors.append(
                ValidationError(
                    path=self.input_dir,
                    message=f"Input directory does not exist: {self.input_dir}",
                    level=self.validation_level,
                )
            )
            return errors

        case_files = list(self.input_dir.glob("*.case"))
        if not case_files:
            errors.append(
                ValidationError(
                    path=self.input_dir,
                    message="No EnSight .case files found in input directory",
                    level=self.validation_level,
                )
            )
            return errors

        for case_file in case_files:
            file_errors = self.validate_single_item(case_file)
            errors.extend(file_errors)

        return errors

    def validate_single_item(self, item: Path) -> List[ValidationError]:
        """Validate a single EnSight .case file."""
        errors = []

        try:
            # Read EnSight file
            reader = pv.get_reader(str(item))
            
            # FIX: Use correct methods to enable all arrays instead of 'read_all_variables'
            if hasattr(reader, "enable_all_point_arrays"):
                reader.enable_all_point_arrays()
            if hasattr(reader, "enable_all_cell_arrays"):
                reader.enable_all_cell_arrays()
                
            dataset = reader.read()
            
            if not dataset or len(dataset) == 0:
                errors.append(
                    ValidationError(
                        path=item,
                        message="EnSight file contains no valid blocks",
                        level=self.validation_level,
                    )
                )
                return errors

            # Merge blocks and extract surface for validation
            full_mesh = None
            for block in dataset:
                if block and block.n_points > 0:
                    mesh = block.extract_surface().triangulate()
                    if full_mesh is None:
                        full_mesh = mesh
                    else:
                        full_mesh = full_mesh.merge(mesh)

            if full_mesh is None or full_mesh.n_points < self.min_points:
                errors.append(
                    ValidationError(
                        path=item,
                        message="Merged surface mesh has too few points",
                        level=self.validation_level,
                    )
                )
                return errors

            # Structure validation
            errors.extend(self._validate_structure(full_mesh, item))

            # Field validation
            if self.validation_level == ValidationLevel.FIELDS:
                errors.extend(self._validate_fields(full_mesh, item))

        except Exception as e:
            errors.append(
                ValidationError(
                    path=item,
                    message=f"Failed to open or process EnSight file: {str(e)}",
                    level=self.validation_level,
                )
            )

        return errors

    def _validate_structure(
        self, mesh: pv.PolyData, file_path: Path
    ) -> List[ValidationError]:
        """Validate mesh structure."""
        errors = []

        if mesh.n_points < self.min_points:
            errors.append(
                ValidationError(
                    path=file_path,
                    message=f"Mesh has too few points: {mesh.n_points}",
                    level=self.validation_level,
                )
            )

        if mesh.n_cells < self.min_cells:
            errors.append(
                ValidationError(
                    path=file_path,
                    message=f"Mesh has too few cells: {mesh.n_cells}",
                    level=self.validation_level,
                )
            )

        if mesh.points.shape[1] != self.expected_spatial_dims:
            errors.append(
                ValidationError(
                    path=file_path,
                    message=f"Mesh points have wrong dimensions: expected {self.expected_spatial_dims}D",
                    level=self.validation_level,
                )
            )

        try:
            surf = mesh.extract_surface().triangulate()
            if surf.n_points == 0:
                errors.append(
                    ValidationError(
                        path=file_path,
                        message="Extracted surface mesh has no points",
                        level=self.validation_level,
                    )
                )
        except Exception as e:
            errors.append(
                ValidationError(
                    path=file_path,
                    message=f"Failed to extract surface mesh: {str(e)}",
                    level=self.validation_level,
                )
            )

        return errors

    def _validate_fields(self, mesh: pv.PolyData, file_path: Path) -> List[ValidationError]:
        """Validate point data fields."""
        errors = []

        if mesh.cell_data:
            mesh = mesh.cell_data_to_point_data()

        if not mesh.point_data:
            errors.append(
                ValidationError(
                    path=file_path,
                    message="Mesh has no point data fields",
                    level=self.validation_level,
                )
            )
            return errors

        for name, data in mesh.point_data.items():
            if len(data) != mesh.n_points:
                errors.append(
                    ValidationError(
                        path=file_path,
                        message=f"Field '{name}' length does not match number of points",
                        level=self.validation_level,
                    )
                )
            if np.isnan(data).any():
                errors.append(
                    ValidationError(
                        path=file_path,
                        message=f"Field '{name}' contains NaN values",
                        level=self.validation_level,
                    )
                )
            if np.isinf(data).any():
                errors.append(
                    ValidationError(
                        path=file_path,
                        message=f"Field '{name}' contains infinite values",
                        level=self.validation_level,
                    )
                )

        return errors