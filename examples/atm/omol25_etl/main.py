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

"""OMol25 Atomic Data ETL Pipeline.

Demonstrates the pre-allocated parallel Zarr write mode for large-scale
atomic datasets.  The sink pre-allocates all arrays upfront using the
per-structure atom counts from ``metadata.npz``, then workers fill in
non-overlapping regions concurrently without synchronization.
"""

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
        help="Number of structures to process (default: 64, -1 for all)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Atoms per Zarr chunk — controls parallel partitioning (default: 2048)",
    )
    args = parser.parse_args()

    input_dir: Path = args.input.resolve()
    output_dir: Path = args.output.resolve()

    # Configure the source to read ASE LMDB files.
    source = ASELMDBSource(
        data_dir=str(input_dir),
        metadata_path=str(input_dir / "metadata.npz"),
    )

    n_total = len(source)
    n_indices = n_total if args.n_indices == -1 else min(args.n_indices, n_total)

    print(f"OMol25 ETL: {input_dir}")
    print(f"  Total structures: {n_total}")
    print(f"  Indices to process: {n_indices}")
    print(f"  Workers: {args.workers}")
    print(f"  Chunk size: {args.chunk_size} atoms")
    print(f"  Output: {output_dir}")

    # Get schema from first sample and per-structure atom counts.
    natoms = source.metadata["natoms"][:n_indices]
    sample = next(source[0])

    # Build the pipeline with pre-allocated parallel sink.
    pipeline = source.filter(AtomicStatsFilter(output=str(output_dir / "stats.parquet"))).write(
        AtomicDataZarrSink(
            output_path=str(output_dir / "dataset.zarr"),
            natoms=natoms,
            schema=sample,
            chunk_size=args.chunk_size,
        )
    )

    # Run the pipeline — workers write to non-overlapping regions.
    results = run_pipeline(
        pipeline,
        n_jobs=args.workers,
        backend="process_pool",
        indices=range(n_indices),
        progress=True,
    )

    print(f"\nProcessed {len(results)} structures")

    # Merge per-worker statistics shards into a single Parquet file
    merged = gather_pipeline(pipeline)
    for path in merged:
        print(f"Merged statistics: {path}")


if __name__ == "__main__":
    main()
