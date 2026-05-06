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

"""OMol25 Atomic Data ETL Pipeline."""

import argparse
from pathlib import Path

from physicsnemo_curator.domains.atm.filters.stats import AtomicStatsFilter
from physicsnemo_curator.domains.atm.sinks.zarr_writer import AtomicDataZarrSink
from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource
from physicsnemo_curator.run import gather_pipeline, run_pipeline


def main() -> None:
    """Run the OMol25 ETL pipeline."""
    parser = argparse.ArgumentParser(description="OMol25 Atomic Data ETL Pipeline")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("input/omol25/val"),
        help="Path to downloaded OMol25 LMDB directory (default: input/omol25/val)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/omol25"),
        help="Output directory (default: output/omol25)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--n-indices",
        type=int,
        default=64,
        help="Number of LMDB files to process (default: 64)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Zarr write batch size (default: 500)",
    )
    args = parser.parse_args()

    input_dir: Path = args.input.resolve()
    output_dir: Path = args.output.resolve()

    # Configure the source to read ASE LMDB files.
    # Each .aselmdb file is one source index containing many atomic structures.
    source = ASELMDBSource(
        data_dir=str(input_dir),
        metadata_path=str(input_dir / "metadata.npz"),
    )

    n_files = len(source)
    print(f"OMol25 ETL: {input_dir}")
    print(f"  LMDB files discovered: {n_files}")
    print(f"  Indices to process: {args.n_indices}")
    print(f"  Workers: {args.workers}")
    print(f"  Output: {output_dir}")

    # Build the pipeline:
    # 1. AtomicStatsFilter — per-field statistics with Welford accumulators
    # 2. AtomicDataZarrSink — write per-LMDB-file Zarr stores
    pipeline = source.filter(AtomicStatsFilter(output=str(output_dir / "stats.parquet"))).write(
        AtomicDataZarrSink(
            output_path=str(output_dir),
            naming_template="{stem}.zarr",
            batch_size=args.batch_size,
        )
    )

    # Run the pipeline
    results = run_pipeline(
        pipeline,
        n_jobs=args.workers,
        backend="process_pool",
        indices=range(args.n_indices),
        progress=True,
    )

    print(f"\nProcessed {len(results)} LMDB files")
    for i, paths in enumerate(results):
        print(f"  File {i}: {paths}")

    # Merge per-worker statistics shards into a single Parquet file
    merged = gather_pipeline(pipeline)
    for path in merged:
        print(f"Merged statistics: {path}")


if __name__ == "__main__":
    main()
