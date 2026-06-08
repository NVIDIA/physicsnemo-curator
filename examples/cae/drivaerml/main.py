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

from physicsnemo_curator.domains.mesh.boundaries import BoxTunnelBoundaries
from physicsnemo_curator.domains.mesh.filters.boundary_injection import BoundaryInjectionFilter
from physicsnemo_curator.domains.mesh.filters.random_permutation import RandomPermutationFilter
from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink
from physicsnemo_curator.domains.mesh.sources.drivaerml import DrivAerMLSource
from physicsnemo_curator.run import run_pipeline

# DrivAerML rectangular wind-tunnel domain (arXiv:2408.11969 v2, Sect. C / Fig. 11).
# All in metres in the DrivAerML coordinate system; z_floor is inferred per
# sample from the vehicle surface (tire contact patch).
_X_MIN, _X_MAX = -40.0, 80.0
_Y_MIN, _Y_MAX = -22.0, 22.0
_Z_HEIGHT = 20.0
_X_BL = -2.339  # slip-to-noslip floor transition


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
    parser.add_argument(
        "--check-watertight",
        action="store_true",
        help="Verify (and log) DomainMesh.is_boundary_watertight after injecting boundaries",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Index of the first run to process (0-based offset, for chunking across jobs)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N runs starting at --start (default: all remaining)",
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

    # Synthesize the rectangular wind-tunnel boundaries (inlet/outlet/slip/
    # no_slip) the dataset ships without, inferring the floor height per
    # sample from the vehicle surface.
    boundary_generator = BoxTunnelBoundaries(
        x_min=_X_MIN,
        x_max=_X_MAX,
        y_min=_Y_MIN,
        y_max=_Y_MAX,
        z_height=_Z_HEIGHT,
        x_bl=_X_BL,
        vehicle_key="vehicle",
    )

    # Build the pipeline:
    # 1. BoundaryInjectionFilter — add inlet/outlet/slip/no_slip to each
    #    DomainMesh (STL meshes pass through unchanged).
    # 2. RandomPermutationFilter — shuffle point/cell order.
    # 3. MeshSink — write to per-run subdirectories.
    # Note: PrecisionFilter is not needed since DrivAerMLSource already
    # downcasts to float32 internally.
    pipeline = (
        source.filter(BoundaryInjectionFilter(boundary_generator, check_watertight=args.check_watertight))
        .filter(RandomPermutationFilter(seed=42))
        .write(
            MeshSink(
                output_dir=str(output_dir),
                naming_template="run_{run_id}/{mesh_name}",
            )
        )
    )

    # Select a contiguous chunk of runs [start, start + limit) for this job.
    start = max(0, args.start)
    stop = n_runs if args.limit is None else min(start + args.limit, n_runs)
    indices: range | None
    if start > 0 or args.limit is not None:
        indices = range(start, stop)
        print(f"Processing runs [{start}, {stop}) -> {len(indices)} run(s)")
    else:
        indices = None

    # Run the pipeline. With a single worker use the sequential backend so a
    # node only ever holds one (large) volume mesh in memory at a time.
    backend = "process_pool" if args.workers > 1 else "sequential"
    results = run_pipeline(pipeline, n_jobs=args.workers, backend=backend, indices=indices)

    print(f"\nProcessed {len(results)} runs")
    for i, paths in enumerate(results):
        print(f"  Run {i}: {paths}")


if __name__ == "__main__":
    main()
