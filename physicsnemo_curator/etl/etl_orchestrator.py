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
import time
from collections.abc import Mapping
from typing import Optional

from physicsnemo_curator.etl.data_processors import ParallelProcessor
from physicsnemo_curator.etl.data_sources import DataSource
from physicsnemo_curator.etl.data_transformations import DataTransformation
from physicsnemo_curator.etl.dataset_validators import DatasetValidator
from physicsnemo_curator.etl.processing_config import ProcessingConfig
from physicsnemo_curator.utils import utils as curator_utils


class ETLOrchestrator:
    """Orchestrates the ETL pipeline execution.

    This class manages the complete ETL workflow including:
    - Logging configuration
    - Optional dataset validation
    - Parallel processing execution
    - Performance tracking

    The orchestrator accepts already-instantiated components (source, sink,
    transformations), which allows each example to handle its own component
    instantiation.

    The class is designed to be extensible - subclass and override methods
    to customize behavior for specific use cases.
    """

    def __init__(
        self,
        source: DataSource,
        sink: DataSource,
        transformations: Mapping[str, DataTransformation],
        processing_config: ProcessingConfig,
        validator: Optional[DatasetValidator] = None,
    ):
        """Initialize the orchestrator with instantiated components.

        Args:
            source: Instantiated data source for reading input data
            sink: Instantiated data sink for writing output data
            transformations: Dictionary of instantiated transformation objects
            processing_config: Processing configuration with settings like num_processes
            validator: Optional instantiated validator for dataset validation
        """
        self.source = source
        self.sink = sink
        self.transformations = transformations
        self.processing_config = processing_config
        self.validator = validator
        self.logger = None
        self.processor = None

    def setup_logging(self) -> logging.Logger:
        """Set up and return logger instance.

        Override this method to customize logging configuration.

        Returns:
            Configured logger instance
        """
        return curator_utils.setup_logger()

    def run_validation(self) -> None:
        """Run dataset validation if validator is configured.

        Override this method to customize validation behavior.

        Raises:
            ValueError: If dataset validation fails
        """
        if self.validator is not None:
            errors = self.validator.validate()
            if errors:
                for error in errors:
                    self.logger.error(f"{error.path}: {error.message}")
                raise ValueError("Dataset validation failed")

    def create_processor(self) -> ParallelProcessor:
        """Create the parallel processor.

        Override this method to use a custom processor implementation.

        Returns:
            ParallelProcessor instance
        """
        return ParallelProcessor(
            source=self.source,
            transformations=self.transformations,
            sink=self.sink,
            config=self.processing_config,
        )

    def log_summary(self, wall_clock_time: float) -> None:
        """Log processing summary statistics.

        Override this method to customize summary logging.

        Args:
            wall_clock_time: Total wall clock time in seconds
        """
        self.logger.info("\nProcessing Summary:")
        self.logger.info(f"Number of processes: {self.processing_config.num_processes}")
        self.logger.info(f"Total wall clock time: {wall_clock_time:.2f} seconds")

    def run(self) -> None:
        """Execute the complete ETL pipeline.

        This is the main entry point that orchestrates the entire workflow:
        1. Setup logging
        2. Optional dataset validation
        3. Parallel processing execution
        4. Summary reporting

        Raises:
            ValueError: If validation fails
            Exception: If processing fails
        """
        # Setup phase
        self.logger = self.setup_logging()
        self.logger.info("Starting ETL pipeline")

        # Validation phase (optional)
        self.run_validation()

        # Processor creation and execution phase
        self.processor = self.create_processor()

        wall_clock_start = time.time()

        try:
            self.processor.run()

            wall_clock_time = time.time() - wall_clock_start
            self.log_summary(wall_clock_time)

        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            raise
