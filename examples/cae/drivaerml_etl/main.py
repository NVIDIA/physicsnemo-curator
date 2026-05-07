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

"""DrivAerML End-to-End ETL Pipeline."""

import argparse
from pathlib import Path

from physicsnemo_curator.domains.mesh.filters.random_permutation import RandomPermutationFilter
from physicsnemo_curator.domains.mesh.filters.stats import MeshStatsFilter
from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink
from physicsnemo_curator.domains.mesh.sources.drivaerml import DrivAerMLSource
from physicsnemo_curator.run import gather_pipeline, run_pipeline


def main() -> None:
    """Run the DrivAerML ETL pipeline."""
    parser = argparse.ArgumentParser(description="DrivAerML End-to-End ETL Pipeline")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("input/drivaerml"),
        help="Path to downloaded DrivAerML dataset (default: input/drivaerml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/drivaerml"),
        help="Output directory for processed meshes (default: output/drivaerml)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    args = parser.parse_args()

    input_dir: Path = args.input.resolve()
    output_dir: Path = args.output.resolve()

    # Configure the source to read from local files.
    # DrivAerMLSource in mesh_type="multi" reads all mesh representations:
    #   - domain: DomainMesh combining volume interior + boundary surface
    #   - stl: vehicle geometry from the STL file
    #   - single_solid: same STL merged into one contiguous solid
    source = DrivAerMLSource(
        url=f"file://{input_dir}",
        mesh_type="multi",
        manifold_dim="auto",
        point_source="vertices",
        backend="rust",
    )

    n_runs = len(source)
    print(f"Total runs available: {n_runs}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    # Build the pipeline:
    # 1. MeshStatsFilter — per-field statistics with Welford accumulators
    # 2. RandomPermutationFilter — shuffle point/cell order
    # 3. MeshSink — write to per-run subdirectories
    # Note: PrecisionFilter is not needed since DrivAerMLSource already
    # downcasts to float32 internally.
    pipeline = (
        source.filter(MeshStatsFilter(output=str(output_dir / "stats.parquet")))
        .filter(RandomPermutationFilter(seed=42))
        .write(
            MeshSink(
                output_dir=str(output_dir),
                naming_template="run_{run_id}/{mesh_name}",
            )
        )
    )

    # Run the pipeline
    results = run_pipeline(pipeline, n_jobs=args.workers, backend="process_pool")

    print(f"\nProcessed {len(results)} runs")
    for i, paths in enumerate(results):
        print(f"  Run {i}: {paths}")

    # Merge per-worker statistics shards into a single Parquet file
    merged = gather_pipeline(pipeline)
    for path in merged:
        print(f"Merged statistics: {path}")


if __name__ == "__main__":
    main()
