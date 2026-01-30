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

from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np

from physicsnemo_curator.etl.data_sources import DataSource
from physicsnemo_curator.etl.processing_config import ProcessingConfig


class H5DataSource(DataSource):
    """DataSource for reading HDF5 physics simulation files."""

    def __init__(self, cfg: ProcessingConfig, input_dir: str):
        """Initialize the H5 data source.

        Args:
            cfg: Processing configuration
            input_dir: Directory containing input HDF5 files
        """
        super().__init__(cfg)
        self.input_dir = Path(input_dir)

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory {self.input_dir} does not exist")

    def get_file_list(self) -> List[str]:
        """Get list of HDF5 files to process.

        Returns:
            List of filenames (without extension) to process
        """
        # Find all .h5 files and return their base names
        h5_files = list(self.input_dir.glob("*.h5"))
        filenames = [f.stem for f in h5_files]  # Remove .h5 extension

        self.logger.info(f"Found {len(filenames)} HDF5 files to process")
        return sorted(filenames)

    def read_file(self, filename: str) -> Dict[str, Any]:
        """Read one HDF5 file and extract all data.

        Args:
            filename: Base filename (without extension)

        Returns:
            Dictionary containing extracted data and metadata
        """
        filepath = self.input_dir / f"{filename}.h5"
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        self.logger.warning(f"Reading {filepath}")

        data = {}

        with h5py.File(filepath, "r") as f:
            # Read field data
            data["temperature"] = np.array(f["fields/temperature"])
            data["velocity"] = np.array(f["fields/velocity"])

            # Read geometry data
            data["coordinates"] = np.array(f["geometry/coordinates"])

            # Read metadata
            metadata = dict(f["metadata"].attrs.items())

            data["metadata"] = metadata
            data["filename"] = filename

        self.logger.warning(f"Loaded data with {len(data['temperature'])} points")
        return data

    def _get_output_path(self, filename: str) -> Path:
        """Get the final output path for a given filename.

        Args:
            filename: Name of the file to process

        Returns:
            Path object representing the final output location
        """
        return NotImplementedError("H5DataSource only supports reading")

    def _write_impl_temp_file(
        self,
        data: Dict[str, Any],
        output_path: Path,
    ) -> None:
        """Not implemented - this DataSource only reads."""
        raise NotImplementedError("H5DataSource only supports reading")

    def should_skip(self, filename: str) -> bool:
        """Never skip files for reading."""
        return False
