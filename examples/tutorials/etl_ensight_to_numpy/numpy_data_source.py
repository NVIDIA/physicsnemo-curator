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

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from physicsnemo_curator.etl.data_sources import DataSource
from physicsnemo_curator.etl.processing_config import ProcessingConfig


class NumpyDataSource(DataSource):
    """DataSource for writing NumPy arrays to .npz files."""

    def __init__(self, cfg: ProcessingConfig, output_dir: str):
        """Initialize the NumPy data sink.

        Args:
            cfg: Processing configuration
            output_dir: Directory where .npz files will be written
        """
        super().__init__(cfg)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_file_list(self) -> List[str]:
        """Not used for sink-only operations."""
        return []

    def read_file(self, filename: str) -> Dict[str, Any]:
        """Not implemented - this DataSource only writes."""
        raise NotImplementedError("NumpyDataSource only supports writing")

    def _get_output_path(self, filename: str) -> Path:
        """Get the final output path for a given filename.

        Args:
            filename: Name of the file to process

        Returns:
            Path object representing the final output location
        """
        return self.output_dir / f"{filename}.npz"

    def _write_impl_temp_file(
        self,
        data: Dict[str, Any],
        output_path: Path,
    ) -> None:
        """Write data to a NumPy .npz file.

        Args:
            data: Dictionary containing NumPy arrays and metadata
            output_path: Path where the .npz file should be written
        """
        self.logger.info(f"Writing NumPy arrays to {output_path}")

        # Separate metadata from array data
        metadata = data.pop("metadata", {})
        filename = data.pop("filename", "unknown")

        # Save all arrays to .npz file (compressed)
        # FIX: Use an open file object to prevent numpy from appending .npz to the temp filename
        with open(output_path, "wb") as f:
            np.savez_compressed(f, **data)

        # Save metadata as separate JSON file
        # Note: This writes directly to the final .json path (sidecar file)
        metadata_path = output_path.with_suffix(".json")
        metadata["filename"] = filename
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(
            f"Saved {len(data)} arrays to {output_path} "
            f"and metadata to {metadata_path}"
        )

    def should_skip(self, filename: str) -> bool:
        """Check if output file already exists.

        Args:
            filename: Name of the file to check

        Returns:
            True if the output file already exists and we should skip processing
        """
        output_path = self._get_output_path(filename)
        exists = output_path.exists()
        
        if exists:
            self.logger.info(f"Skipping {filename} - output already exists")
        
        return exists