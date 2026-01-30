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
import multiprocessing
import time
from collections.abc import Mapping

from tqdm import tqdm

from .data_sources import DataSource
from .data_transformations import DataTransformation
from .processing_config import ProcessingConfig


class ParallelProcessor:
    """Base class for parallel data processing."""

    def __init__(
        self,
        source: DataSource,
        transformations: Mapping[str, DataTransformation],
        sink: DataSource,
        config: ProcessingConfig,
    ):
        self.source = source
        self.transformations = transformations
        self.sink = sink
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        # Shared counter for tracking progress.
        self.progress_counter = multiprocessing.Value("i", 0)
        self.total_files = 0

    def process_files(self, file_subset: list[str], worker_id: int) -> None:
        """Process a subset of files.

        Args:
            file_subset: List of files to process
            worker_id: ID of the worker process
        """
        self.logger.info(f"Worker {worker_id} starting")

        for filename in file_subset:
            try:
                if not self.sink.should_skip(filename):
                    data = self.source.read_file(filename)
                    for tf in self.transformations.values():
                        data = tf.transform(data)
                        # Check if the data is None after each transformation
                        # Sometimes, transforms might filter out the data.
                        if data is None:
                            self.logger.warning(
                                f"No data was returned by transform: {tf.__class__.__name__} "
                                f"for file {filename}. Skipping."
                            )
                            break  # Stop processing this file

                    # Only write if data survived all transformations
                    if data is not None:
                        self.sink.write(data, filename)
                    else:
                        self.logger.info(f"Skipping write for {filename}")

                # Update progress counter.
                with self.progress_counter.get_lock():
                    self.progress_counter.value += 1
            except Exception as e:  # noqa: PERF203
                self.logger.error(
                    f"Error processing file {filename} in worker {worker_id}: {str(e)}"
                )
                continue  # Skip to next file on error

        self.logger.info(f"Worker {worker_id} completed")

    def run(self) -> None:
        """Run the parallel processing pipeline."""
        files = self.source.get_file_list()
        num_files = len(files)
        self.total_files = num_files

        if num_files == 0:
            self.logger.warning("No files found to process")
            return

        # Clean up orphaned temp files from previous interrupted runs
        self.sink.cleanup_temp_files()

        files_per_worker = (
            num_files + self.config.num_processes - 1
        ) // self.config.num_processes

        processes = []
        ctx = multiprocessing.get_context("spawn")

        # Create progress bar
        pbar = tqdm(total=num_files, desc="Processing files", unit="file")

        for i in range(self.config.num_processes):
            start_idx = i * files_per_worker
            end_idx = min(start_idx + files_per_worker, num_files)
            file_subset = files[start_idx:end_idx]

            if not file_subset:
                continue

            process = ctx.Process(target=self.process_files, args=(file_subset, i))
            process.start()
            processes.append(process)

        # Update progress bar while processes are running.
        while any(p.is_alive() for p in processes):
            pbar.n = self.progress_counter.value
            pbar.refresh()
            time.sleep(0.1)  # Delay to prevent excessive updates.

        # Ensure progress bar shows final state.
        pbar.n = self.progress_counter.value
        pbar.refresh()
        pbar.close()

        for process in processes:
            process.join()
