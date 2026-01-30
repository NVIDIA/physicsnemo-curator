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

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .processing_config import ProcessingConfig


class DataSource(ABC):
    """Abstract base class for data sources."""

    @abstractmethod
    def __init__(self, cfg: ProcessingConfig):
        self.config = cfg
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def get_file_list(self) -> list[str]:
        """Get list of files to process."""
        pass

    @abstractmethod
    def read_file(self, filename: str) -> dict[str, Any]:
        """Read a single file and return its data."""
        pass

    @abstractmethod
    def _get_output_path(self, filename: str) -> Path:
        """Get the final output path for a given filename.

        Args:
            filename: Name of the file to process

        Returns:
            Path object representing the final output location
        """
        pass

    @abstractmethod
    def _write_impl_temp_file(self, data: dict[str, Any], output_path: Path) -> None:
        """Implement actual data writing logic to a temporary file.

        Args:
            data: Transformed data to write
            output_path: Path where data should be written (may be temporary)
        """
        pass

    def write(self, data: dict[str, Any], filename: str) -> None:
        """Write transformed data to storage with atomic temp-then-rename.

        This method handles the temp-then-rename pattern for robust writes.
        It ensures that output files are either complete or don't exist,
        preventing partial files from interrupted writes.

        Subclasses should implement _write_impl_temp_file() for actual serialization.

        Args:
            data: Transformed data to write
            filename: Name of the file being processed
        """
        import shutil

        final_path = self._get_output_path(filename)
        temp_path = self._get_temporary_output_path(final_path)

        # Write to temporary location
        self._write_impl_temp_file(data, temp_path)

        # Remove destination if it exists (for overwrite case)
        # This is necessary because rename() won't replace non-empty directories
        if final_path.exists():
            if final_path.is_dir():
                shutil.rmtree(final_path)
            else:
                final_path.unlink()

        # Atomic rename to final location
        temp_path.rename(final_path)
        self.logger.debug(f"Successfully wrote {final_path}")

    def _get_temporary_output_path(self, final_path: Path) -> Path:
        """Get temporary path for a final output path.

        Args:
            final_path: The final destination path

        Returns:
            Temporary path with _temp suffix before the extension
        """
        return final_path.with_name(f"{final_path.name}_temp")

    def should_skip(self, filename: str) -> bool:
        """Check if processing should be skipped for this file.

        Default implementation checks if the output file exists.
        Subclasses can override for custom skip logic (e.g., overwrite flags).

        Args:
            filename: Name of the file to check

        Returns:
            True if processing should be skipped, False otherwise
        """
        return self._get_output_path(filename).exists()

    def cleanup_temp_files(self) -> None:
        """Clean up orphaned temporary files from interrupted runs.

        This method should be called at the start of processing to remove
        any temporary files left over from previous interrupted runs.
        Subclasses can override to implement cleanup logic specific to
        their output format and directory structure.
        """
        pass
