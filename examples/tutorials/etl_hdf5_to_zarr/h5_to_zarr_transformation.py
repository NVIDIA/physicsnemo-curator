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


class H5ToZarrTransformation(DataTransformation):
    """Transform HDF5 data into Zarr-optimized format."""

    def __init__(
        self, cfg: ProcessingConfig, chunk_size: int = 500, compression_level: int = 3
    ):
        """Initialize the transformation.

        Args:
            cfg: Processing configuration
            chunk_size: Chunk size for Zarr arrays (number of points per chunk)
            compression_level: Compression level (1-9, higher = more compression)
        """
        super().__init__(cfg)
        self.chunk_size = chunk_size
        self.compression_level = compression_level

        # Set up Zarr 3 compression codec
        self.compressor = zarr.codecs.BloscCodec(
            cname="zstd",
            clevel=self.compression_level,
            shuffle=zarr.codecs.BloscShuffle.shuffle,
        )

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform HDF5 data to Zarr-optimized format.

        Args:
            data: Dictionary from H5DataSource.read_file()

        Returns:
            Dictionary with Zarr-optimized arrays and metadata
        """
        self.logger.info(f"Transforming {data['filename']} for Zarr storage")

        # Get the number of points to determine chunking
        num_points = len(data["temperature"])

        # Calculate optimal chunks (don't exceed chunk_size)
        chunk_points = min(self.chunk_size, num_points)

        # Prepare arrays that will be written to Zarr stores
        zarr_data = {
            "temperature": {},
            "velocity": {},
            "coordinates": {},
            "velocity_magnitude": {},
        }

        # Temperature field (1D array)
        zarr_data["temperature"] = {
            "data": data["temperature"].astype(np.float32),
            "chunks": (chunk_points,),
            "compressor": self.compressor,
            "dtype": np.float32,
        }

        # Velocity field (2D array: points x 3 components)
        zarr_data["velocity"] = {
            "data": data["velocity"].astype(np.float32),
            "chunks": (chunk_points, 3),
            "compressor": self.compressor,
            "dtype": np.float32,
        }

        # Coordinates (2D array: points x 3 dimensions)
        zarr_data["coordinates"] = {
            "data": data["coordinates"].astype(np.float32),
            "chunks": (chunk_points, 3),
            "compressor": self.compressor,
            "dtype": np.float32,
        }

        # Add some computed metadata useful for Zarr to existing metadata
        metadata = data["metadata"]
        metadata["num_points"] = num_points
        metadata["chunk_size"] = chunk_points
        metadata["compression"] = "zstd"
        metadata["compression_level"] = self.compression_level

        # Also add some simple derived fields
        # Temperature statistics
        metadata["temperature_min"] = float(np.min(data["temperature"]))
        metadata["temperature_max"] = float(np.max(data["temperature"]))
        metadata["temperature_mean"] = float(np.mean(data["temperature"]))

        # Velocity magnitude
        velocity_magnitude = np.linalg.norm(data["velocity"], axis=1)
        zarr_data["velocity_magnitude"] = {
            "data": velocity_magnitude.astype(np.float32),
            "chunks": (chunk_points,),
            "compressor": self.compressor,
            "dtype": np.float32,
        }
        metadata["velocity_max"] = float(np.max(velocity_magnitude))
        zarr_data["metadata"] = metadata

        return zarr_data
