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


from typing import Any, Dict

import numpy as np
import zarr

from physicsnemo_curator.etl.data_transformations import DataTransformation
from physicsnemo_curator.etl.processing_config import ProcessingConfig


class RstToZarrTransformation(DataTransformation):
    """
    Transform thermal simulation RST data into Zarr-optimized format.

    This transformation takes thermal analysis data (coordinates, temperature, heat_flux)
    and prepares it for efficient storage in Zarr format with:
    - Appropriate chunking for parallel access
    - Compression optimized for floating-point scientific data
    - Type conversion to float32 for storage efficiency
    """

    def __init__(
        self, cfg: ProcessingConfig, chunk_size: int = 1000, compression_level: int = 3
    ):
        """
        Initialize the transformation.

        Args:
            cfg: Processing configuration
            chunk_size: Number of points per chunk (affects I/O performance)
        """
        super().__init__(cfg)
        self.chunk_size = chunk_size
        self.compressor = zarr.codecs.BloscCodec(
            cname="zstd",
            clevel=compression_level,
            shuffle=zarr.codecs.BloscShuffle.shuffle,
        )

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform thermal RST data into Zarr-ready format.

        Args:
            data: Dictionary containing:
                - coordinates: (N, 3) array of node positions
                - temperature: (N,) array of temperature values
                - heat_flux: (N, 3) array of heat flux vectors
                - metadata: simulation metadata dict
                - filename: source filename

        Returns:
            zarr_data: Dictionary with arrays prepared for Zarr writing
        """
        self.logger.info(f"Transforming thermal data from {data['filename']}...")

        # Extract numpy arrays from the data source
        coords = data["coordinates"]
        temperature = data["temperature"]
        heat_flux = data["heat_flux"]

        num_points = len(coords)

        # Determine chunk size (do not exceed total points)
        chunk_points = min(self.chunk_size, num_points)

        self.logger.info(
            f"  Processing {num_points} nodes with chunk size {chunk_points}"
        )

        # Prepare Zarr-ready data structure
        zarr_data = {
            "coordinates": {},
            "temperature": {},
            "heat_flux": {},
            "metadata": data["metadata"].copy(),  # Copy to avoid modifying original
        }

        # 1. Transform Coordinates (N, 3)
        zarr_data["coordinates"] = {
            "data": coords.astype(np.float32),
            "chunks": (chunk_points, 3),
            "compressors": (self.compressor,),
            "dtype": np.float32,
        }

        # 2. Transform Temperature (N,) - scalar field
        zarr_data["temperature"] = {
            "data": temperature.astype(np.float32),
            "chunks": (chunk_points,),
            "compressors": (self.compressor,),
            "dtype": np.float32,
        }

        # 3. Transform Heat Flux (N, 3) - vector field
        zarr_data["heat_flux"] = {
            "data": heat_flux.astype(np.float32),
            "chunks": (chunk_points, 3),
            "compressors": (self.compressor,),
            "dtype": np.float32,
        }

        # Add curation metadata
        zarr_data["metadata"]["curator_chunk_size"] = chunk_points
        zarr_data["metadata"]["compression_type"] = "zstd"
        zarr_data["metadata"]["compression_level"] = self.compressor.clevel

        self.logger.info(f"  Transformation complete for {data['filename']}")

        return zarr_data
