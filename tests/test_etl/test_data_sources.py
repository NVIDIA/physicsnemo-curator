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

import pytest

from physicsnemo_curator.etl.data_sources import DataSource
from physicsnemo_curator.etl.processing_config import ProcessingConfig


class MockDataSource(DataSource):
    """Mock implementation of DataSource for testing."""

    def __init__(self, config: ProcessingConfig, output_dir: Path = None):
        super().__init__(config)
        self.files = ["file1.txt", "file2.txt"]
        self.data = {"key": "value"}
        self.written_data = {}
        self.write_calls = []  # Track all write calls with paths
        self.output_dir = output_dir or Path("/mock/output")
        self.existing_files = set()  # Track which files "exist"

    def get_file_list(self) -> List[str]:
        return self.files

    def read_file(self, filename: str) -> Dict[str, Any]:
        if filename not in self.files:
            raise FileNotFoundError(f"File not found: {filename}")
        return self.data

    def _get_output_path(self, filename: str) -> Path:
        """Get the final output path for a given filename."""
        return self.output_dir / filename

    def _write_impl_temp_file(self, data: Dict[str, Any], output_path: Path) -> None:
        """Write data to the specified path (may be temporary)."""
        self.write_calls.append({"path": output_path, "data": data})
        self.written_data[str(output_path)] = data

        # Actually create the file on disk to support the rename operation
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(str(data))

    def should_skip(self, filename: str) -> bool:
        """Check if file should be skipped based on existing_files."""
        return self._get_output_path(filename) in self.existing_files


def test_datasource_direct_instantiation_raises_error():
    """Test that DataSource cannot be instantiated directly."""
    config = ProcessingConfig(num_processes=1)
    with pytest.raises(TypeError):
        DataSource(config)


def test_mock_datasource_instantiation():
    """Test that concrete implementation can be instantiated."""
    config = ProcessingConfig(num_processes=1)
    source = MockDataSource(config)
    assert isinstance(source, DataSource)
    assert source.config == config
    assert source.logger.name == "MockDataSource"


def test_get_file_list():
    """Test get_file_list returns expected files."""
    config = ProcessingConfig(num_processes=1)
    source = MockDataSource(config)
    files = source.get_file_list()
    assert files == ["file1.txt", "file2.txt"]


def test_read_file():
    """Test read_file returns expected data."""
    config = ProcessingConfig(num_processes=1)
    source = MockDataSource(config)
    data = source.read_file("file1.txt")
    assert data == {"key": "value"}

    with pytest.raises(FileNotFoundError):
        source.read_file("nonexistent.txt")


def test_write(tmp_path):
    """Test write stores data correctly using template method."""
    config = ProcessingConfig(num_processes=1)
    source = MockDataSource(config, output_dir=tmp_path)
    test_data = {"test": "data"}
    source.write(test_data, "output.txt")

    # Verify data was written
    assert len(source.write_calls) == 1
    assert source.write_calls[0]["data"] == test_data

    # Verify temp path was used
    temp_path = source.write_calls[0]["path"]
    assert temp_path.name == "output.txt_temp"

    # Verify final file exists after rename
    final_path = tmp_path / "output.txt"
    assert final_path.exists()


def test_logger_creation():
    """Test that logger is created with correct name."""
    config = ProcessingConfig(num_processes=1)
    source = MockDataSource(config)
    assert source.logger.name == "MockDataSource"


def test_get_output_path():
    """Test _get_output_path returns correct path."""
    config = ProcessingConfig(num_processes=1)
    output_dir = Path("/test/output")
    source = MockDataSource(config, output_dir=output_dir)

    output_path = source._get_output_path("test_file.txt")
    assert output_path == Path("/test/output/test_file.txt")


def test_get_temporary_output_path():
    """Test _get_temporary_output_path generates correct temp path."""
    config = ProcessingConfig(num_processes=1)
    source = MockDataSource(config)

    # Test with file extension
    final_path = Path("/output/file.npz")
    temp_path = source._get_temporary_output_path(final_path)
    assert temp_path == Path("/output/file.npz_temp")

    # Test with directory (no extension)
    final_path = Path("/output/file.zarr")
    temp_path = source._get_temporary_output_path(final_path)
    assert temp_path == Path("/output/file.zarr_temp")


def test_write_uses_temp_then_rename_pattern(tmp_path):
    """Test that write uses temp file before final file."""
    config = ProcessingConfig(num_processes=1)
    MockDataSource(config, output_dir=tmp_path)

    # Mock the write to actually create files
    final_path = tmp_path / "output.txt"
    temp_path = tmp_path / "output.txt_temp"

    # Manually create temp file to simulate write
    temp_path.write_text("temp data")

    # Verify temp exists and final doesn't
    assert temp_path.exists()
    assert not final_path.exists()

    # Simulate rename
    temp_path.rename(final_path)

    # Verify final exists and temp doesn't
    assert final_path.exists()
    assert not temp_path.exists()


def test_should_skip_default_implementation():
    """Test should_skip checks if output path exists."""
    config = ProcessingConfig(num_processes=1)
    source = MockDataSource(config)

    # File doesn't exist, should not skip
    assert not source.should_skip("new_file.txt")

    # Mark file as existing
    source.existing_files.add(Path("/mock/output/existing_file.txt"))

    # File exists, should skip
    assert source.should_skip("existing_file.txt")


def test_cleanup_temp_files_base_implementation():
    """Test cleanup_temp_files base implementation does nothing."""
    config = ProcessingConfig(num_processes=1)
    source = MockDataSource(config)

    # Should not raise any errors
    source.cleanup_temp_files()


def test_write_impl_temp_file_called_with_temp_path(tmp_path):
    """Test that _write_impl_temp_file is called with temporary path."""
    config = ProcessingConfig(num_processes=1)
    source = MockDataSource(config, output_dir=tmp_path)
    test_data = {"key": "value"}

    source.write(test_data, "myfile.txt")

    # Verify _write_impl_temp_file was called with temp path
    assert len(source.write_calls) == 1
    written_path = source.write_calls[0]["path"]
    assert str(written_path).endswith("_temp")
    assert "myfile.txt_temp" in str(written_path)
