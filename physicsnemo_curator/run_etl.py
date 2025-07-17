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

import multiprocessing
import time

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from physicsnemo_curator.etl.data_processors import ParallelProcessor
from physicsnemo_curator.etl.processing_config import ProcessingConfig
from physicsnemo_curator.utils import utils as curator_utils


@hydra.main(version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main ETL pipeline execution.

    Can be run with a config dir and a config name:
    python run_etl.py --config-dir /path/to/config/dir --config-name name-of-config

    Users can also override any config parameters by passing them on the command line.
    For example:
    python run_etl.py --config-dir /path/to/config/dir --config-name name-of-config etl.processing.num_processes=16
    """

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        # Start method has already been set, skip.
        pass

    logger = curator_utils.setup_logger()
    logger.info("Starting ETL pipeline")

    if not cfg:  # Check for None or empty config
        logger.error("No configuration provided or empty configuration")
        logger.error("Please run with --config-dir and --config-name")
        return

    logger.info(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")

    # Create processing config with common settings
    processing_config = ProcessingConfig(**cfg.etl.processing)

    # Create and run validator
    if "validator" in cfg.etl:
        validator = instantiate(
            cfg.etl.validator,
            processing_config,
            **{k: v for k, v in cfg.etl.source.items() if not k.startswith("_")},
        )
        errors = validator.validate()
        if errors:
            for error in errors:
                logger.error(f"{error.path}: {error.message}")
            raise ValueError("Dataset validation failed")

    # Create source
    source = instantiate(cfg.etl.source, processing_config)
    # Create sink
    sink = instantiate(cfg.etl.sink, processing_config)
    # Create transformations
    # Need to pass processing_config to each transformation, see:
    # https://hydra.cc/docs/advanced/instantiate_objects/overview/#recursive-instantiation
    cfgs = {k: {"_args_": [processing_config]} for k in cfg.etl.transformations.keys()}
    transformations = instantiate(cfg.etl.transformations, **cfgs)

    # Create and run processor
    processor = ParallelProcessor(
        source=source,
        transformations=transformations,
        sink=sink,
        config=processing_config,
    )

    wall_clock_start = time.time()

    try:
        processor.run()

        wall_clock_time = time.time() - wall_clock_start
        logger.info("\nProcessing Summary:")
        logger.info(f"Number of processes: {processing_config.num_processes}")
        logger.info(f"Total wall clock time: {wall_clock_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
