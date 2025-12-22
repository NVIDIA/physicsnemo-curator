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


class TutorialValidator(DatasetValidator):
    """Validator for CGNS physics simulation dataset."""

    def __init__(
        self, cfg: ProcessingConfig, input_dir: str, validation_level: str = "fields"
    ):
        """Initialize the validator.

        Args:
            cfg: Processing configuration
            input_dir: Directory containing CGNS files to validate
            validation_level: "structure" or "fields"
        """
        super().__init__(cfg)
        self.input_dir = Path(input_dir)
        self.validation_level = ValidationLevel(validation_level)

        # Define minimum requirements for CGNS mesh
        self.min_points = 3  # At least 3 points to form a triangle
        self.min_cells = 1  # At least 1 cell
        self.expected_spatial_dims = 3  # 3D coordinates (x, y, z)

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

        # Find all CGNS files
        cgns_files = list(self.input_dir.glob("*.cgns"))

        if not cgns_files:
            errors.append(
                ValidationError(
                    path=self.input_dir,
                    message="No CGNS files found in input directory",
                    level=self.validation_level,
                )
            )
            return errors

        # Validate each file
        for cgns_file in cgns_files:
            file_errors = self.validate_single_item(cgns_file)
            errors.extend(file_errors)

        return errors

    def validate_single_item(self, item: Path) -> List[ValidationError]:
        """Validate a single CGNS file.

        Args:
            item: Path to CGNS file to validate

        Returns:
            List of validation errors for this file
        """
        errors = []

        try:
            # Try to read the CGNS file
            reader = pv.CGNSReader(str(item))
            reader.load_boundary_patch = False
            mesh = reader.read()

            # Check if mesh is valid
            if not mesh or not mesh[0]:
                errors.append(
                    ValidationError(
                        path=item,
                        message="CGNS file contains no valid mesh data",
                        level=self.validation_level,
                    )
                )
                return errors

            original_mesh = mesh[0][0]

            # Structure validation
            errors.extend(self._validate_structure(original_mesh, item))

            # Field validation (if requested and structure is valid)
            if self.validation_level == ValidationLevel.FIELDS and not errors:
                errors.extend(self._validate_fields(original_mesh, item))

        except Exception as e:
            errors.append(
                ValidationError(
                    path=item,
                    message=f"Failed to open CGNS file: {str(e)}",
                    level=self.validation_level,
                )
            )

        return errors

    def _validate_structure(
        self, mesh: pv.DataSet, file_path: Path
    ) -> List[ValidationError]:
        """Validate CGNS mesh structure."""
        errors = []

        # Check mesh has points
        if mesh.n_points < self.min_points:
            errors.append(
                ValidationError(
                    path=file_path,
                    message=f"Mesh has too few points: {mesh.n_points} (minimum: {self.min_points})",
                    level=self.validation_level,
                )
            )

        # Check mesh has cells
        if mesh.n_cells < self.min_cells:
            errors.append(
                ValidationError(
                    path=file_path,
                    message=f"Mesh has too few cells: {mesh.n_cells} (minimum: {self.min_cells})",
                    level=self.validation_level,
                )
            )

        # Check coordinate dimensions
        if mesh.points is not None:
            if mesh.points.shape[1] != self.expected_spatial_dims:
                errors.append(
                    ValidationError(
                        path=file_path,
                        message=f"Mesh points have wrong dimensions: expected {self.expected_spatial_dims}D, got {mesh.points.shape[1]}D",
                        level=self.validation_level,
                    )
                )

        # Check if mesh can be extracted and triangulated (needed for surface processing)
        try:
            surface_mesh = mesh.extract_surface().triangulate()
            if surface_mesh.n_points == 0:
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
                    message=f"Failed to extract and triangulate surface: {str(e)}",
                    level=self.validation_level,
                )
            )

        return errors

    def _validate_fields(self, mesh: pv.DataSet, file_path: Path) -> List[ValidationError]:
        """Validate field data content."""
        errors = []

        # Extract surface and triangulate for field validation
        try:
            surface_mesh = mesh.extract_surface().triangulate()
            
            # Convert cell data to point data if present
            if surface_mesh.cell_data:
                surface_mesh = surface_mesh.cell_data_to_point_data()

        except Exception as e:
            errors.append(
                ValidationError(
                    path=file_path,
                    message=f"Failed to process mesh for field validation: {str(e)}",
                    level=self.validation_level,
                )
            )
            return errors

        # Check that mesh has at least some point data fields
        if not surface_mesh.point_data:
            errors.append(
                ValidationError(
                    path=file_path,
                    message="Mesh has no point data fields",
                    level=self.validation_level,
                )
            )
            return errors

        # Validate each field
        for field_name, field_data in surface_mesh.point_data.items():
            # Check field size matches number of points
            if len(field_data) != surface_mesh.n_points:
                errors.append(
                    ValidationError(
                        path=file_path,
                        message=f"Field '{field_name}' size ({len(field_data)}) does not match number of points ({surface_mesh.n_points})",
                        level=self.validation_level,
                    )
                )

            # Check for NaN or infinite values
            if np.isnan(field_data).any():
                errors.append(
                    ValidationError(
                        path=file_path,
                        message=f"Field '{field_name}' contains NaN values",
                        level=self.validation_level,
                    )
                )

            if np.isinf(field_data).any():
                errors.append(
                    ValidationError(
                        path=file_path,
                        message=f"Field '{field_name}' contains infinite values",
                        level=self.validation_level,
                    )
                )

        return errors
