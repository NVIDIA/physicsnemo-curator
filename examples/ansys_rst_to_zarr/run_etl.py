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

"""
Ansys Thermal RST to Zarr ETL Pipeline

This script orchestrates the ETL pipeline to convert Ansys thermal simulation
results (.rst files) into training-ready Zarr format.

Usage:
    python run_etl.py --config-dir ./config \
                      --config-name st_pydpf_config \
                      etl.source.input_dir=path/to/rst/files \
                      etl.sink.output_dir=path/to/output

Example:
    python run_etl.py --config-dir ./config \
                      --config-name st_pydpf_config \
                      etl.source.input_dir=mock_thermal_data \
                      etl.sink.output_dir=output_zarr
"""

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

# PhysicsNeMo Curator Imports
from physicsnemo_curator.etl.etl_orchestrator import ETLOrchestrator
from physicsnemo_curator.etl.processing_config import ProcessingConfig
from physicsnemo_curator.utils import utils as curator_utils


@hydra.main(version_base="1.3", config_path="./config", config_name="st_pydpf_config")
def main(cfg: DictConfig) -> None:
    """
    Main ETL pipeline execution.

    Args:
        cfg: Hydra configuration object loaded from YAML
    """
    print("=" * 70)
    print("Ansys Thermal RST to Zarr ETL Pipeline")
    print("=" * 70)

    # 1. Setup Multiprocessing
    curator_utils.setup_multiprocessing()

    # 2. Parse Processing Config
    processing_config = ProcessingConfig(**cfg.etl.processing)
    print(f"Processing with {processing_config.num_processes} processes")

    # 3. Instantiate ETL Components
    print("\nInitializing ETL components...")

    # Source: Reads thermal RST files
    source = instantiate(cfg.etl.source, processing_config)
    print(f"  Source: {cfg.etl.source._target_}")
    print(f"  Input directory: {cfg.etl.source.input_dir}")

    # Sink: Writes Zarr stores
    sink = instantiate(cfg.etl.sink, processing_config)
    print(f"  Sink: {cfg.etl.sink._target_}")
    print(f"  Output directory: {cfg.etl.sink.output_dir}")

    # Transformations: RST to Zarr conversion
    cfgs = {k: {"_args_": [processing_config]} for k in cfg.etl.transformations.keys()}
    transformations = instantiate(cfg.etl.transformations, **cfgs)
    print(f"  Transformations: {list(cfg.etl.transformations.keys())}")

    # 4. Run ETL Orchestrator
    print("\n" + "=" * 70)
    print("Starting ETL Pipeline...")
    print("=" * 70 + "\n")

    orchestrator = ETLOrchestrator(
        source=source,
        sink=sink,
        transformations=transformations,
        processing_config=processing_config,
    )
    orchestrator.run()

    print("\n" + "=" * 70)
    print("ETL Pipeline Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
