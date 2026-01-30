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

import zarr
from zarr.storage import LocalStore

from physicsnemo_curator.etl.data_sources import DataSource
from physicsnemo_curator.etl.processing_config import ProcessingConfig


class ZarrDataSource(DataSource):
    """DataSource for writing to Zarr stores."""

    def __init__(self, cfg: ProcessingConfig, output_dir: str):
        """Initialize the Zarr data source.

        Args:
            cfg: Processing configuration
            output_dir: Directory to write Zarr stores
        """
        super().__init__(cfg)
        self.output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_file_list(self) -> List[str]:
        """Not implemented - this DataSource only writes."""
        raise NotImplementedError("ZarrDataSource only supports writing")

    def read_file(self, filename: str) -> Dict[str, Any]:
        """Not implemented - this DataSource only writes."""
        raise NotImplementedError("ZarrDataSource only supports writing")

    def _get_output_path(self, filename: str) -> Path:
        """Get the output path for a given filename.

        Args:
            filename: Name of the file to process

        Returns:
            Path object representing the output location.
        """
        return self.output_dir / f"{filename}.zarr"

    def _write_impl_temp_file(self, data: Dict[str, Any], output_path: Path) -> None:
        """
        Implement actual data writing logic to a temporary Zarr store.

        This method is called by the write() method to write the data to a temporary Zarr store.
        The data is written to a temporary Zarr store and then renamed to the final output path.
        This is to improve the robustness of the write operation.

        Args:
            data: Transformed data from H5ToZarrTransformation
            output_path: Path where data should be written (may be temporary)
        """
        # Create Zarr store
        self.logger.info(f"Creating Zarr store: {output_path}")
        store = LocalStore(output_path)
        root = zarr.open_group(store=store, mode="w")

        # Store metadata as root attributes
        if "metadata" in data:
            for key, value in data["metadata"].items():
                # Convert numpy types to Python types for JSON serialization
                if hasattr(value, "item"):  # numpy scalar
                    value = value.item()
                root.attrs[key] = value
            data.pop("metadata")

        # Write all arrays from the transformation
        for array_name, array_info in data.items():
            root.create_array(
                name=array_name,
                data=array_info["data"],
                chunks=array_info["chunks"],
                compressors=(
                    array_info["compressor"] if array_info["compressor"] else None
                ),
                dtype=array_info["dtype"],
            )

        # Add some store-level metadata
        root.attrs["zarr_format"] = 3
        root.attrs["created_by"] = "physicsnemo-curator-tutorial"

        # Something weird is happening here.
        # If this error occurs, the stores are created and we move to the next one.
        # If this error does NOT occur, we seem to skip all the remaining files.
        # Debug with Alexey.
        self.logger.info("Successfully created Zarr store")

    def should_skip(self, filename: str) -> bool:
        """Check if we should skip writing this store.

        Args:
            filename: Base filename to check

        Returns:
            True if store should be skipped (already exists)
        """
        store_path = self.output_dir / f"{filename}.zarr"
        exists = store_path.exists()

        if exists:
            self.logger.info(f"Skipping {filename} - Zarr store already exists")
            return True

        return False
