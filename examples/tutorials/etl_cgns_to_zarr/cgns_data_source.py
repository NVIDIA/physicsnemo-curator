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
from typing import Any, Dict, List

import numpy as np
import pyvista as pv

from physicsnemo_curator.etl.data_sources import DataSource
from physicsnemo_curator.etl.processing_config import ProcessingConfig


class CGNSDataSource(DataSource):
    """DataSource for reading CGNS physics simulation files."""

    def __init__(self, cfg: ProcessingConfig, input_dir: str):
        """Initialize the CGNS data source.

        Args:
            cfg: Processing configuration
            input_dir: Directory containing input CGNS files
        """
        super().__init__(cfg)
        self.input_dir = Path(input_dir)

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory {self.input_dir} does not exist")

    def get_file_list(self) -> List[str]:
        """Get list of CGNS files to process.

        Returns:
            List of filenames (without extension) to process
        """
        # Find all .cgns files and return their base names
        cgns_files = list(self.input_dir.glob("*.cgns"))
        filenames = [f.stem for f in cgns_files]  # Remove .cgns extension

        self.logger.info(f"Found {len(filenames)} CGNS files to process")
        return sorted(filenames)

    def _read_cgns_mesh(self, filepath: Path):
        """Read a CGNS file and extract the mesh.

        Args:
            filepath: Path to the CGNS file

        Returns:
            pyvista mesh object or None if reading fails
        """
        try:
            reader = pv.CGNSReader(str(filepath))
            # Turn off loading the interior mesh
            reader.load_boundary_patch = False
            mesh = reader.read()

            # Check if the mesh is valid and contains a block to process
            if not mesh or not mesh[0]:
                self.logger.warning(f"No valid data found in {filepath}. Skipping.")
                return None
            
            original = mesh[0][0]
            return original

        except Exception as e:
            self.logger.error(f"Error processing file {filepath}: {e}")
            return None

    def read_file(self, filename: str) -> Dict[str, Any]:
        """Read one CGNS file and extract all data.

        Args:
            filename: Base filename (without extension)

        Returns:
            Dictionary containing extracted data and metadata
        """
        filepath = self.input_dir / f"{filename}.cgns"
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        self.logger.warning(f"Reading {filepath}")

        # Read the CGNS mesh
        original_mesh = self._read_cgns_mesh(filepath)
        if original_mesh is None:
            raise ValueError(f"Failed to read CGNS file: {filepath}")

        # Extract surface and triangulate
        surface_mesh = original_mesh.extract_surface().triangulate()

        # Convert cell data to point data if present
        if surface_mesh.cell_data:
            self.logger.info("Found cell data. Converting to point data.")
            surface_mesh = surface_mesh.cell_data_to_point_data()

        # Build data dictionary
        data = {}

        # Extract coordinates (points)
        data["coordinates"] = np.array(surface_mesh.points)

        # Extract connectivity/faces information
        data["faces"] = np.array(surface_mesh.faces).reshape(-1, 4)[:, 1:]  # Remove size prefix

        # Extract all point data fields
        for field_name in surface_mesh.point_data.keys():
            data[field_name] = np.array(surface_mesh.point_data[field_name])
            self.logger.info(f"Extracted point data field: {field_name}")

        # Store metadata
        metadata = {
            "n_points": surface_mesh.n_points,
            "n_cells": surface_mesh.n_cells,
            "bounds": surface_mesh.bounds,
        }
        data["metadata"] = metadata
        data["filename"] = filename

        self.logger.warning(f"Loaded data with {surface_mesh.n_points} points and {surface_mesh.n_cells} cells")
        return data

    def _get_output_path(self, filename: str) -> Path:
        """Get the final output path for a given filename.

        Args:
            filename: Name of the file to process

        Returns:
            Path object representing the final output location
        """
        raise NotImplementedError("CGNSDataSource only supports reading")

    def _write_impl_temp_file(
        self,
        data: Dict[str, Any],
        output_path: Path,
    ) -> None:
        """Not implemented - this DataSource only reads."""
        raise NotImplementedError("CGNSDataSource only supports reading")

    def should_skip(self, filename: str) -> bool:
        """Never skip files for reading."""
        return False
