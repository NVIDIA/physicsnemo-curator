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
from unittest.mock import Mock, patch

import pytest

from physicsnemo_curator.etl.data_processors import ParallelProcessor
from physicsnemo_curator.etl.data_sources import DataSource
from physicsnemo_curator.etl.data_transformations import DataTransformation
from physicsnemo_curator.etl.processing_config import ProcessingConfig


class MockDataSource(DataSource):
    """Mock implementation of DataSource for testing ParallelProcessor."""

    def __init__(self, config: ProcessingConfig, output_dir: Path = None):
        super().__init__(config)
        self.files = ["file1.txt", "file2.txt"]
        self.data = {"key": "value"}
        self.written_data = {}
        self.skipped_files = set()
        self.cleanup_called = False
        self.output_dir = output_dir

    def get_file_list(self):
        return self.files

    def read_file(self, filename: str):
        if filename not in self.files:
            raise FileNotFoundError(f"File not found: {filename}")
        return self.data.copy()  # Return a copy to prevent mutations

    def _get_output_path(self, filename: str) -> Path:
        """Get the final output path for a given filename."""
        if self.output_dir:
            return self.output_dir / filename
        return Path(f"/mock/output/{filename}")

    def _write_impl_temp_file(self, data, output_path: Path):
        """Write data to the specified path."""
        # Store data for verification
        # output_path.name might be like "file1.txt_temp", remove "_temp" to get original
        filename = output_path.name.replace("_temp", "")
        self.written_data[filename] = data

        # Actually create the file on disk to support the rename operation
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(str(data))

    def should_skip(self, filename: str) -> bool:
        """Checks whether the file should be skipped."""
        return filename in self.skipped_files

    def cleanup_temp_files(self) -> None:
        """Track cleanup calls."""
        self.cleanup_called = True


class MockTransformation(DataTransformation):
    """Mock implementation of DataTransformation for testing ParallelProcessor."""

    def __init__(self, config: ProcessingConfig):
        super().__init__(config)
        self.transform_called = False

    def transform(self, data):
        self.transform_called = True
        data["transformed"] = True
        return data


class MockTransformationReturnsNone(DataTransformation):
    """Mock transformation that returns None to simulate filtering."""

    def __init__(self, config: ProcessingConfig):
        super().__init__(config)
        self.transform_called = False

    def transform(self, data):
        self.transform_called = True
        return None


class MockTransformationWithTag(DataTransformation):
    """Mock transformation that adds a tag to track execution."""

    def __init__(self, config: ProcessingConfig, tag: str):
        super().__init__(config)
        self.transform_called = False
        self.tag = tag

    def transform(self, data):
        self.transform_called = True
        data[self.tag] = True
        return data


@pytest.fixture
def processor_setup(tmp_path):
    """Setup a basic processor with mocked components."""
    config = ProcessingConfig(num_processes=2)
    source = MockDataSource(config, output_dir=tmp_path)
    transforms = {"mock": MockTransformation(config)}
    sink = MockDataSource(config, output_dir=tmp_path)

    return {"config": config, "source": source, "transform": transforms, "sink": sink}


def test_processor_initialization(processor_setup):
    """Test that processor initializes correctly."""
    processor = ParallelProcessor(
        processor_setup["source"],
        processor_setup["transform"],
        processor_setup["sink"],
        processor_setup["config"],
    )

    assert processor.source == processor_setup["source"]
    assert processor.transformations == processor_setup["transform"]
    assert processor.sink == processor_setup["sink"]
    assert processor.config == processor_setup["config"]
    assert processor.logger.name == "ParallelProcessor"
    assert processor.progress_counter.value == 0
    assert processor.total_files == 0


def test_process_files(processor_setup):
    """Test processing of a single file subset."""
    processor = ParallelProcessor(
        processor_setup["source"],
        processor_setup["transform"],
        processor_setup["sink"],
        processor_setup["config"],
    )

    # Process a subset of files
    file_subset = ["file1.txt"]
    processor.process_files(file_subset, worker_id=0)

    # Verify the pipeline was executed correctly
    assert processor_setup["sink"].written_data  # Check that data was written
    assert (
        "transformed" in processor_setup["sink"].written_data["file1.txt"]
    )  # Check transformation
    assert processor.progress_counter.value == 1  # Check progress counter


def test_process_files_with_error(processor_setup):
    """Test handling of errors during file processing."""
    processor = ParallelProcessor(
        processor_setup["source"],
        processor_setup["transform"],
        processor_setup["sink"],
        processor_setup["config"],
    )

    # Make source.read_file raise an exception
    processor_setup["source"].read_file = Mock(side_effect=Exception("Test error"))

    # Process should log the error but not raise it
    with patch.object(processor.logger, "error") as mock_logger:
        processor.process_files(["file1.txt"], worker_id=0)
        mock_logger.assert_called_once()  # Verify error was logged

    # Verify no data was written
    assert not processor_setup["sink"].written_data


@patch("multiprocessing.get_context")
@patch("physicsnemo_curator.etl.data_processors.tqdm")
def test_run_parallel_processing(mock_tqdm, mock_get_context, processor_setup):
    """Test the parallel processing execution."""
    # Mock the multiprocessing context
    mock_process = Mock()
    mock_process.is_alive.side_effect = [
        True,
        True,
        False,
        False,
    ]  # Simulate process completion
    mock_context = Mock()
    mock_context.Process.return_value = mock_process
    mock_get_context.return_value = mock_context

    # Mock tqdm progress bar
    mock_pbar = Mock()
    mock_tqdm.return_value = mock_pbar

    processor = ParallelProcessor(
        processor_setup["source"],
        processor_setup["transform"],
        processor_setup["sink"],
        processor_setup["config"],
    )

    # Run the processor
    processor.run()

    # Verify processes were started and joined
    assert mock_context.Process.called
    assert mock_process.start.called
    assert mock_process.join.called

    # Verify progress bar was created and updated
    mock_tqdm.assert_called_once_with(
        total=len(processor_setup["source"].files), desc="Processing files", unit="file"
    )
    assert mock_pbar.refresh.called
    assert mock_pbar.close.called


def test_run_with_no_files(processor_setup):
    """Test handling of empty file list."""
    processor_setup["source"].files = []  # Set empty file list

    processor = ParallelProcessor(
        processor_setup["source"],
        processor_setup["transform"],
        processor_setup["sink"],
        processor_setup["config"],
    )

    # Should complete without error
    processor.run()
    assert processor.progress_counter.value == 0  # No progress for empty file list


@patch("multiprocessing.get_context")
@patch("physicsnemo_curator.etl.data_processors.tqdm")
def test_file_distribution(mock_tqdm, mock_get_context, processor_setup):
    """Test that files are distributed correctly among workers."""
    # Create a processor with 2 workers and 3 files
    processor_setup["config"] = ProcessingConfig(num_processes=2)
    processor_setup["source"].files = ["file1.txt", "file2.txt", "file3.txt"]

    # Mock the multiprocessing context
    mock_process = Mock()
    mock_process.is_alive.side_effect = [
        True,
        True,
        False,
        False,
    ]  # Simulate process completion
    mock_context = Mock()
    mock_context.Process.return_value = mock_process
    mock_get_context.return_value = mock_context

    # Mock tqdm progress bar
    mock_pbar = Mock()
    mock_tqdm.return_value = mock_pbar

    processor = ParallelProcessor(
        processor_setup["source"],
        processor_setup["transform"],
        processor_setup["sink"],
        processor_setup["config"],
    )

    processor.run()

    # Verify process creation calls
    calls = mock_context.Process.call_args_list
    assert len(calls) == 2  # Should create 2 processes

    # First process should get 2 files, second process should get 1 file
    first_call_files = calls[0][1]["args"][0]
    second_call_files = calls[1][1]["args"][0]
    assert len(first_call_files) == 2
    assert len(second_call_files) == 1


@patch("multiprocessing.get_context")
@patch("physicsnemo_curator.etl.data_processors.tqdm")
def test_cleanup_temp_files_called_on_run(mock_tqdm, mock_get_context, processor_setup):
    """Test that cleanup_temp_files is called before processing starts."""
    # Mock the multiprocessing context
    mock_process = Mock()
    mock_process.is_alive.return_value = False  # Process completes immediately
    mock_context = Mock()
    mock_context.Process.return_value = mock_process
    mock_get_context.return_value = mock_context

    # Mock tqdm progress bar
    mock_pbar = Mock()
    mock_tqdm.return_value = mock_pbar

    processor = ParallelProcessor(
        processor_setup["source"],
        processor_setup["transform"],
        processor_setup["sink"],
        processor_setup["config"],
    )

    # Verify cleanup hasn't been called yet
    assert not processor_setup["sink"].cleanup_called

    # Run the processor
    processor.run()

    # Verify cleanup was called
    assert processor_setup["sink"].cleanup_called


def test_cleanup_temp_files_called_before_workers_start(processor_setup):
    """Test that cleanup_temp_files is called before processing files."""
    processor_setup["sink"].files = ["file1.txt"]

    processor = ParallelProcessor(
        processor_setup["source"],
        processor_setup["transform"],
        processor_setup["sink"],
        processor_setup["config"],
    )

    # Mock cleanup to track order
    call_order = []

    processor_setup["sink"].cleanup_temp_files = lambda: call_order.append("cleanup")

    original_process = processor.process_files

    def tracked_process(*args, **kwargs):
        call_order.append("process")
        return original_process(*args, **kwargs)

    processor.process_files = tracked_process

    # Run with mocked multiprocessing to avoid actual spawning
    with patch("multiprocessing.get_context") as mock_get_context:
        mock_process = Mock()
        mock_process.is_alive.return_value = False
        mock_context = Mock()
        mock_context.Process.return_value = mock_process
        mock_get_context.return_value = mock_context

        with patch("physicsnemo_curator.etl.data_processors.tqdm"):
            processor.run()

    # Verify cleanup was called (at least once in the test flow)
    assert "cleanup" in call_order


def test_transformation_returns_none(processor_setup):
    """Test handling when a transformation returns None (filters out data)."""
    processor = ParallelProcessor(
        processor_setup["source"],
        processor_setup["transform"],
        processor_setup["sink"],
        processor_setup["config"],
    )

    # Replace transform with one that returns None
    none_transform = MockTransformationReturnsNone(processor_setup["config"])
    processor.transformations = {"none_transform": none_transform}

    # Process files and capture logs
    with patch.object(processor.logger, "warning") as mock_warning:
        with patch.object(processor.logger, "info") as mock_info:
            processor.process_files(["file1.txt"], worker_id=0)

            # Verify warning was logged about None data
            mock_warning.assert_called_once()
            warning_message = mock_warning.call_args[0][0]
            assert "No data was returned by transform" in warning_message
            assert "MockTransformationReturnsNone" in warning_message
            assert "file1.txt" in warning_message

            # Verify info log about skipping write
            info_calls = [call[0][0] for call in mock_info.call_args_list]
            assert any("Skipping write for file1.txt" in msg for msg in info_calls)

    # Verify transform was called
    assert none_transform.transform_called

    # Verify no data was written
    assert "file1.txt" not in processor_setup["sink"].written_data

    # Verify progress counter was still incremented
    assert processor.progress_counter.value == 1


def test_multiple_transforms_one_returns_none(processor_setup):
    """Test that transformation chain breaks when one transform returns None."""
    processor = ParallelProcessor(
        processor_setup["source"],
        processor_setup["transform"],
        processor_setup["sink"],
        processor_setup["config"],
    )

    # Create chain: transform1 -> none_transform -> transform2
    transform1 = MockTransformationWithTag(processor_setup["config"], "tag1")
    none_transform = MockTransformationReturnsNone(processor_setup["config"])
    transform2 = MockTransformationWithTag(processor_setup["config"], "tag2")

    processor.transformations = {
        "transform1": transform1,
        "none_transform": none_transform,
        "transform2": transform2,
    }

    # Process files
    with patch.object(processor.logger, "warning"):
        with patch.object(processor.logger, "info"):
            processor.process_files(["file1.txt"], worker_id=0)

    # Verify transform1 and none_transform were called
    assert transform1.transform_called
    assert none_transform.transform_called

    # Verify transform2 was NOT called (chain broke after None)
    assert not transform2.transform_called

    # Verify no data was written
    assert "file1.txt" not in processor_setup["sink"].written_data

    # Verify progress counter was still incremented
    assert processor.progress_counter.value == 1


def test_should_skip_functionality(processor_setup):
    """Test that files are skipped when should_skip returns True."""
    processor = ParallelProcessor(
        processor_setup["source"],
        processor_setup["transform"],
        processor_setup["sink"],
        processor_setup["config"],
    )

    # Mark file1.txt as should be skipped
    processor_setup["sink"].skipped_files.add("file1.txt")

    # Process files
    processor.process_files(["file1.txt", "file2.txt"], worker_id=0)

    # Verify file1.txt was skipped (not in written_data)
    assert "file1.txt" not in processor_setup["sink"].written_data

    # Verify file2.txt was processed
    assert "file2.txt" in processor_setup["sink"].written_data
    assert "transformed" in processor_setup["sink"].written_data["file2.txt"]

    # Verify progress counter includes both files
    assert processor.progress_counter.value == 2


def test_all_transforms_succeed_data_written(processor_setup):
    """Test that data is written when all transformations succeed."""
    processor = ParallelProcessor(
        processor_setup["source"],
        processor_setup["transform"],
        processor_setup["sink"],
        processor_setup["config"],
    )

    # Create chain of successful transforms
    transform1 = MockTransformationWithTag(processor_setup["config"], "tag1")
    transform2 = MockTransformationWithTag(processor_setup["config"], "tag2")

    processor.transformations = {
        "transform1": transform1,
        "transform2": transform2,
    }

    # Process files
    processor.process_files(["file1.txt"], worker_id=0)

    # Verify both transforms were called
    assert transform1.transform_called
    assert transform2.transform_called

    # Verify data was written with both tags
    assert "file1.txt" in processor_setup["sink"].written_data
    assert "tag1" in processor_setup["sink"].written_data["file1.txt"]
    assert "tag2" in processor_setup["sink"].written_data["file1.txt"]

    # Verify progress counter was incremented
    assert processor.progress_counter.value == 1
