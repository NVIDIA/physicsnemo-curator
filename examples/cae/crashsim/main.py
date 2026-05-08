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

"""Crash Simulation ETL Pipeline."""

import argparse
from pathlib import Path

from physicsnemo_curator.core.logging import get_logger
from physicsnemo_curator.domains.mesh.filters.edge_compute import EdgeComputeFilter
from physicsnemo_curator.domains.mesh.filters.mesh_info import MeshInfoFilter
from physicsnemo_curator.domains.mesh.filters.precision import PrecisionFilter
from physicsnemo_curator.domains.mesh.filters.wall_node import WallNodeFilter
from physicsnemo_curator.domains.mesh.sinks.mesh_zarr import MeshZarrSink
from physicsnemo_curator.domains.mesh.sources.d3plot import D3PlotSource
from physicsnemo_curator.run import run_pipeline

logger = get_logger("CrashSimETL")


def main() -> None:
    """Run the Crash Simulation ETL pipeline."""
    parser = argparse.ArgumentParser(description="Crash Simulation ETL Pipeline")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("input/crashsim"),
        help="Path to LS-DYNA d3plot run directories (default: input/crashsim)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/crashsim"),
        help="Output directory for processed meshes (default: output/crashsim)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2)",
    )
    parser.add_argument(
        "--wall-threshold",
        type=float,
        default=1.0,
        help="Wall node displacement threshold (default: 1.0)",
    )
    args = parser.parse_args()

    input_dir: Path = args.input.resolve()
    output_dir: Path = args.output.resolve()

    # Configure the source to read LS-DYNA d3plot files with stress and
    # thickness data from companion .k keyword files.
    source = D3PlotSource(
        input_dir=str(input_dir),
        read_stress=True,
        read_k_file=True,
    )

    n_runs = len(source)
    logger.info("Total runs available: %d", n_runs)
    logger.info("Input: %s", input_dir)
    logger.info("Output: %s", output_dir)

    # Build the pipeline:
    # 1. WallNodeFilter — remove non-deforming wall nodes
    # 2. EdgeComputeFilter — extract unique edges from cell connectivity
    # 3. MeshInfoFilter — log mesh metadata to JSON-lines
    # 4. PrecisionFilter — cast float64 → float32
    # 5. MeshZarrSink — write compressed Zarr stores
    pipeline = (
        source.filter(WallNodeFilter(threshold=args.wall_threshold))
        .filter(EdgeComputeFilter())
        .filter(MeshInfoFilter(output=str(output_dir / "mesh_info.jsonl")))
        .filter(PrecisionFilter(target_dtype="float32"))
        .write(MeshZarrSink(output_dir=str(output_dir)))
    )

    # Run the pipeline
    results = run_pipeline(pipeline, n_jobs=args.workers, backend="process_pool")

    logger.info("Processed %d runs", len(results))
    for i, paths in enumerate(results):
        logger.info("  Run %d: %s", i, paths)


if __name__ == "__main__":
    main()
