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
import zarr

from physicsnemo_curator.etl.data_transformations import DataTransformation
from physicsnemo_curator.etl.processing_config import ProcessingConfig

class EnSightToZarrTransformation(DataTransformation):
    """Transform EnSight Gold data into Zarr-optimized format."""

    def __init__(
        self, cfg: ProcessingConfig, chunk_size: int = 500, compression_level: int = 3
    ):
        super().__init__(cfg)
        self.chunk_size = chunk_size
        self.compression_level = compression_level

        self.compressor = zarr.codecs.BloscCodec(
            cname="zstd",
            clevel=self.compression_level,
            shuffle=zarr.codecs.BloscShuffle.shuffle,
        )

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform EnSight data to Zarr-optimized format."""
        self.logger.info(f"Transforming {data['filename']} for Zarr storage")

        num_points = len(data["coordinates"])
        chunk_points = min(self.chunk_size, num_points)

        zarr_data = {
            "coordinates": {
                "data": data["coordinates"].astype(np.float32),
                "chunks": (chunk_points, 3),
                "compressor": self.compressor,
                "dtype": np.float32,
            },
            "faces": {},
        }

        if "faces" in data:
            num_cells = len(data["faces"])
            chunk_cells = min(self.chunk_size, num_cells)
            zarr_data["faces"] = {
                "data": data["faces"].astype(np.int32),
                "chunks": (chunk_cells, 3),
                "compressor": self.compressor,
                "dtype": np.int32,
            }

        # Process all point data fields
        for field_name, field_data in data.items():
            if field_name in ["coordinates", "faces", "metadata", "filename"]:
                continue

            if isinstance(field_data, np.ndarray):
                self.logger.info(f"Processing field: {field_name} with shape {field_data.shape}")

                if field_data.ndim == 1:
                    zarr_data[field_name] = {
                        "data": field_data.astype(np.float32),
                        "chunks": (chunk_points,),
                        "compressor": self.compressor,
                        "dtype": np.float32,
                    }
                elif field_data.ndim == 2:
                    zarr_data[field_name] = {
                        "data": field_data.astype(np.float32),
                        "chunks": (chunk_points, field_data.shape[1]),
                        "compressor": self.compressor,
                        "dtype": np.float32,
                    }

        # Add metadata and statistics
        metadata = data.get("metadata", {})
        metadata.update({
            "num_points": num_points,
            "chunk_size": chunk_points,
            "compression": "zstd",
            "compression_level": self.compression_level,
        })

        for field_name, field_data in data.items():
            if field_name in ["coordinates", "faces", "metadata", "filename"]:
                continue
            if isinstance(field_data, np.ndarray):
                if field_data.ndim == 1:
                    metadata[f"{field_name}_min"] = float(np.min(field_data))
                    metadata[f"{field_name}_max"] = float(np.max(field_data))
                    metadata[f"{field_name}_mean"] = float(np.mean(field_data))
                elif field_data.ndim == 2:
                    magnitude = np.linalg.norm(field_data, axis=1)
                    metadata[f"{field_name}_magnitude_max"] = float(np.max(magnitude))
                    metadata[f"{field_name}_magnitude_mean"] = float(np.mean(magnitude))
                    zarr_data[f"{field_name}_magnitude"] = {
                        "data": magnitude.astype(np.float32),
                        "chunks": (chunk_points,),
                        "compressor": self.compressor,
                        "dtype": np.float32,
                    }

        zarr_data["metadata"] = metadata
        return zarr_data
