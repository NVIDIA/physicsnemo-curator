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

from typing import Any, Dict

import numpy as np

from physicsnemo_curator.etl.data_transformations import DataTransformation
from physicsnemo_curator.etl.processing_config import ProcessingConfig


class CGNSToNumpyTransformation(DataTransformation):
    """Transform CGNS data into NumPy array format."""

    def __init__(
        self, cfg: ProcessingConfig, precision: str = "float32"
    ):
        """Initialize the transformation.

        Args:
            cfg: Processing configuration
            precision: Data precision for NumPy arrays ('float32' or 'float64')
        """
        super().__init__(cfg)
        self.dtype = np.float32 if precision == "float32" else np.float64

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform CGNS data to NumPy array format.

        Args:
            data: Dictionary from CGNSDataSource.read_file()

        Returns:
            Dictionary with NumPy arrays and metadata
        """
        self.logger.info(f"Transforming {data['filename']} to NumPy arrays")

        # Get the number of points
        num_points = len(data["coordinates"])

        # Prepare arrays
        numpy_data = {}

        # Coordinates (2D array: points x 3 dimensions)
        numpy_data["coordinates"] = data["coordinates"].astype(self.dtype)

        # Faces/connectivity (2D array: cells x 3 for triangles)
        if "faces" in data:
            numpy_data["faces"] = data["faces"].astype(np.int32)

        # Process all point data fields (e.g., pressure, velocity, temperature, etc.)
        for field_name, field_data in data.items():
            if field_name in ["coordinates", "faces", "metadata", "filename"]:
                continue  # Skip already processed fields
            
            if isinstance(field_data, np.ndarray):
                self.logger.info(f"Processing field: {field_name} with shape {field_data.shape}")
                
                # Convert to specified precision
                numpy_data[field_name] = field_data.astype(self.dtype)

        # Build comprehensive metadata
        metadata = data.get("metadata", {})
        metadata["num_points"] = num_points
        metadata["precision"] = str(self.dtype)
        metadata["format"] = "numpy"

        # Add statistics for each field
        for field_name, field_data in data.items():
            if field_name in ["coordinates", "faces", "metadata", "filename"]:
                continue
            
            if isinstance(field_data, np.ndarray):
                if field_data.ndim == 1:
                    # Add statistics for scalar fields
                    metadata[f"{field_name}_min"] = float(np.min(field_data))
                    metadata[f"{field_name}_max"] = float(np.max(field_data))
                    metadata[f"{field_name}_mean"] = float(np.mean(field_data))
                    metadata[f"{field_name}_std"] = float(np.std(field_data))
                elif field_data.ndim == 2:
                    # For vector fields, compute magnitude statistics
                    magnitude = np.linalg.norm(field_data, axis=1)
                    metadata[f"{field_name}_magnitude_max"] = float(np.max(magnitude))
                    metadata[f"{field_name}_magnitude_mean"] = float(np.mean(magnitude))
                    metadata[f"{field_name}_magnitude_std"] = float(np.std(magnitude))
                    
                    # Store magnitude as a separate field
                    numpy_data[f"{field_name}_magnitude"] = magnitude.astype(self.dtype)

        numpy_data["metadata"] = metadata

        return numpy_data

