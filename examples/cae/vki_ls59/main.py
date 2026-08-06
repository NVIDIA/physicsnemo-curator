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

"""VKI-LS59 PLAID ETL Pipeline.

Curates the VKI-LS59 turbine cascade dataset from the PLAID library:
839 2-D compressible RANS samples (Spalart-Allmaras) with surface and
volume meshes.

The pipeline:

1. Reads each sample from the PLAID HuggingFace disk format as a
   :class:`~physicsnemo.mesh.domain_mesh.DomainMesh` with:

   * ``interior`` — 2-D volume point cloud / triangulated mesh
     (``Base_2_2``, QUAD_4 elements split into triangles)
   * ``boundaries["surface"]`` — 1-D blade surface edges (``Base_1_2``,
     BAR_2 connectivity)
   * ``global_data`` — 8 scalar quantities (Mach, angle of attack, …)

2. Computes per-field statistics via :class:`MeshStatsFilter`.
3. Writes processed meshes to disk via :class:`MeshSink`.

If the dataset is not already present locally it is downloaded
automatically from ``PLAID-datasets/VKI-LS59`` on HuggingFace Hub.
"""

import argparse
import logging
from pathlib import Path

from physicsnemo_curator.core.logging import configure_logging
from physicsnemo_curator.domains.mesh.filters.stats import MeshStatsFilter
from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink
from physicsnemo_curator.domains.mesh.sources.plaid import PlaidSource
from physicsnemo_curator.run import gather_pipeline, run_pipeline

_VKI_LS59_REPO_ID = "PLAID-datasets/VKI-LS59"


def main() -> None:
    """Run the VKI-LS59 PLAID ETL pipeline."""
    parser = argparse.ArgumentParser(description="VKI-LS59 PLAID ETL Pipeline")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("input/vki_ls59"),
        help="Local PLAID dataset root (downloaded automatically if absent)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/vki_ls59"),
        help="Output directory for processed meshes (default: output/vki_ls59)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="Dataset split to process (default: train)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Limit processing to first N samples (default: all)",
    )
    parser.add_argument(
        "--point-cloud",
        action="store_true",
        help="Disable connectivity — produce point clouds instead of meshes",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    args = parser.parse_args()

    configure_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    input_dir: Path = args.input.resolve()
    output_dir: Path = args.output.resolve()

    # PlaidSource auto-downloads the dataset when local_folder is absent.
    element_type_surface = None if args.point_cloud else "BAR_2"
    element_type_volume = None if args.point_cloud else "QUAD_4"

    source = PlaidSource(
        local_folder=str(input_dir),
        split=args.split,
        repo_id=_VKI_LS59_REPO_ID,
        surface_element_type=element_type_surface,
        volume_element_type=element_type_volume,
    )

    n_samples = len(source)
    print(f"Split '{args.split}': {n_samples} samples")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    pipeline = (
        source.filter(MeshStatsFilter(output=str(output_dir / "stats.parquet")))
        .write(
            MeshSink(
                output_dir=str(output_dir),
                naming_template="sample_{index}",
            )
        )
    )

    indices = range(min(args.n_samples, n_samples)) if args.n_samples else None
    results = run_pipeline(
        pipeline,
        n_jobs=args.workers,
        backend="loky",
        indices=indices,
        use_tui=False,
    )

    print(f"\nProcessed {len(results)} samples")
    for i, paths in enumerate(results):
        print(f"  Sample {i}: {paths}")

    merged = gather_pipeline(pipeline)
    for path in merged:
        print(f"Merged statistics: {path}")


if __name__ == "__main__":
    main()
