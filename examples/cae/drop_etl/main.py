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

"""Drop Test ETL Pipeline."""

import argparse
from pathlib import Path

from physicsnemo_curator.domains.mesh.filters.edge_compute import EdgeComputeFilter
from physicsnemo_curator.domains.mesh.filters.wall_node import WallNodeFilter
from physicsnemo_curator.domains.mesh.sinks.mesh_vtu import MeshVTUSink
from physicsnemo_curator.domains.mesh.sources.openradioss import OpenRadiossSource
from physicsnemo_curator.run import run_pipeline


def main() -> None:
    """Run the Drop Test ETL pipeline."""
    parser = argparse.ArgumentParser(description="Drop Test ETL Pipeline")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("input/drop"),
        help="Path to OpenRadioss VTK data (default: input/drop)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/drop"),
        help="Output directory for VTU files (default: output/drop)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--wall-threshold",
        type=float,
        default=1e-5,
        help="Wall node displacement threshold (default: 1e-5)",
    )
    args = parser.parse_args()

    input_dir: Path = args.input.resolve()
    output_dir: Path = args.output.resolve()

    source = OpenRadiossSource(
        input_dir=str(input_dir),
        vtk_glob="*.vtk",
        read_stress=True,
        read_velocity=True,
        read_acceleration=True,
        cell_type="mixed",
    )

    n_runs = len(source)
    print(f"Total runs available: {n_runs}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    pipeline = (
        source.filter(WallNodeFilter(threshold=args.wall_threshold))
        .filter(EdgeComputeFilter())
        .write(MeshVTUSink(output_dir=str(output_dir), naming_template="{run_id}"))
    )

    results = run_pipeline(pipeline, n_jobs=args.workers, backend="process_pool")

    print(f"\nProcessed {len(results)} runs")
    for i, paths in enumerate(results):
        print(f"  Run {i}: {paths}")


if __name__ == "__main__":
    main()
