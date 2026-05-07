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

"""AhmedML End-to-End ETL Pipeline.

Curates the AhmedML dataset — 500 geometric variations of the Ahmed Car
Body with transient hybrid RANS-LES CFD (OpenFOAM v2212, ~20 M cells
per case).

The pipeline reads meshes from HuggingFace Hub (or a local mirror),
computes per-field statistics, converts to float32, and writes the
processed meshes to disk in PhysicsNeMo's native format.

Supported modes:

* **boundary** — surface mesh with flow fields
* **volume** — volumetric mesh (single VTU per run)
* **slices** — x/y/z-normal slice planes
* **multi** — DomainMesh combining interior + boundary + STL

All modes attach CSV metadata (force/moment coefficients and geometric
parameters) as ``global_data`` on every mesh.
"""

import argparse
from pathlib import Path

from physicsnemo_curator.domains.mesh.filters.random_permutation import RandomPermutationFilter
from physicsnemo_curator.domains.mesh.filters.stats import MeshStatsFilter
from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink
from physicsnemo_curator.domains.mesh.sources.ahmedml import AhmedMLSource
from physicsnemo_curator.run import gather_pipeline, run_pipeline


def main() -> None:
    """Run the AhmedML ETL pipeline."""
    parser = argparse.ArgumentParser(description="AhmedML End-to-End ETL Pipeline")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("input/ahmedml"),
        help="Path to downloaded AhmedML dataset (default: input/ahmedml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/ahmedml"),
        help="Output directory for processed meshes (default: output/ahmedml)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2)",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=None,
        help="Limit processing to first N runs (default: all)",
    )
    args = parser.parse_args()

    input_dir: Path = args.input.resolve()
    output_dir: Path = args.output.resolve()

    # Configure the source to read from local files.
    # AhmedMLSource reads boundary, volume, slice, or multi meshes from the
    # AhmedML dataset. Each run_<i>/ directory contains the VTP/VTU files.
    # CSV metadata (force coefficients, geometry params) is attached as
    # global_data on every yielded mesh.
    source = AhmedMLSource(
        url=f"file://{input_dir}",
        mesh_type="multi",
        backend="rust",
    )

    n_runs = len(source)
    print(f"Total runs available: {n_runs}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    # Build the pipeline:
    # 1. MeshStatsFilter — per-field statistics with Welford accumulators
    # 2. RandomPermutationFilter — shuffle point order for training
    # 3. MeshSink — write to per-run subdirectories
    # Note: PrecisionFilter is not needed since AhmedMLSource already
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
    indices = range(min(args.n_runs, n_runs)) if args.n_runs else None
    results = run_pipeline(
        pipeline,
        n_jobs=args.workers,
        backend="loky",
        indices=indices,
        progress="log",
    )

    print(f"\nProcessed {len(results)} runs")
    for i, paths in enumerate(results):
        print(f"  Run {i}: {paths}")

    # Merge per-worker statistics shards into a single Parquet file
    merged = gather_pipeline(pipeline)
    for path in merged:
        print(f"Merged statistics: {path}")


if __name__ == "__main__":
    main()
