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


class EnSightToNumpyTransformation(DataTransformation):
    """Transform EnSight Gold data into NumPy array format."""

    def __init__(
        self, cfg: ProcessingConfig, precision: str = "float32"
    ):
        super().__init__(cfg)
        self.dtype = np.float32 if precision == "float32" else np.float64

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform EnSight data to NumPy array format."""
        self.logger.info(f"Transforming {data['filename']} to NumPy arrays")

        num_points = len(data["coordinates"])
        numpy_data = {}

        # Coordinates
        numpy_data["coordinates"] = data["coordinates"].astype(self.dtype)

        # Faces/connectivity
        if "faces" in data:
            numpy_data["faces"] = data["faces"].astype(np.int32)

        # Process all point data fields
        for field_name, field_data in data.items():
            if field_name in ["coordinates", "faces", "metadata", "filename"]:
                continue
            if isinstance(field_data, np.ndarray):
                self.logger.info(f"Processing field: {field_name} with shape {field_data.shape}")
                numpy_data[field_name] = field_data.astype(self.dtype)

        # Build metadata
        metadata = data.get("metadata", {})
        metadata.update({
            "num_points": num_points,
            "precision": str(self.dtype),
            "format": "numpy",
        })

        # Add statistics for each field
        for field_name, field_data in data.items():
            if field_name in ["coordinates", "faces", "metadata", "filename"]:
                continue
            if isinstance(field_data, np.ndarray):
                if field_data.ndim == 1:
                    metadata[f"{field_name}_min"] = float(np.min(field_data))
                    metadata[f"{field_name}_max"] = float(np.max(field_data))
                    metadata[f"{field_name}_mean"] = float(np.mean(field_data))
                    metadata[f"{field_name}_std"] = float(np.std(field_data))
                elif field_data.ndim == 2:
                    magnitude = np.linalg.norm(field_data, axis=1)
                    metadata[f"{field_name}_magnitude_max"] = float(np.max(magnitude))
                    metadata[f"{field_name}_magnitude_mean"] = float(np.mean(magnitude))
                    metadata[f"{field_name}_magnitude_std"] = float(np.std(magnitude))
                    numpy_data[f"{field_name}_magnitude"] = magnitude.astype(self.dtype)

        numpy_data["metadata"] = metadata
        return numpy_data


