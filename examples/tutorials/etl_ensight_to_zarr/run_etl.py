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

"""EnSight to Zarr Tutorial ETL pipeline runner.

Usage:
    python run_etl.py \
        etl.source.input_dir=/path/to/ensight_data \
        etl.sink.output_dir=/path/to/zarr_output
"""

import os
import sys

# Add current directory to sys.path so Hydra can import tutorial modules
sys.path.insert(0, os.path.dirname(__file__))

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from physicsnemo_curator.etl.etl_orchestrator import ETLOrchestrator
from physicsnemo_curator.etl.processing_config import ProcessingConfig
from physicsnemo_curator.utils import utils as curator_utils


@hydra.main(version_base="1.3", config_path=".", config_name="tutorial_config")
def main(cfg: DictConfig) -> None:
    """Run the EnSight Gold to Zarr tutorial ETL pipeline."""
    # Setup multiprocessing
    curator_utils.setup_multiprocessing()

    # Create processing config
    processing_config = ProcessingConfig(**cfg.etl.processing)

    # Create validator if configured
    validator = None
    if "validator" in cfg.etl:
        validator = instantiate(
            cfg.etl.validator,
            processing_config,
            **{k: v for k, v in cfg.etl.source.items() if not k.startswith("_")},
        )

    # Instantiate source (now EnSightDataSource)
    source = instantiate(cfg.etl.source, processing_config)

    # Instantiate sink
    sink = instantiate(cfg.etl.sink, processing_config)

    # Instantiate transformations
    cfgs = {k: {"_args_": [processing_config]} for k in cfg.etl.transformations.keys()}
    transformations = instantiate(cfg.etl.transformations, **cfgs)

    # Create and run orchestrator
    orchestrator = ETLOrchestrator(
        source=source,
        sink=sink,
        transformations=transformations,
        processing_config=processing_config,
        validator=validator,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()

