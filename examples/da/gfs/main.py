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

"""GFS Global Weather Analysis ETL Pipeline.

See README.md for detailed usage instructions.
"""

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

from physicsnemo_curator.domains.da.filters.stats import DataArrayStatsFilter
from physicsnemo_curator.domains.da.sinks.zarr_writer import ZarrSink
from physicsnemo_curator.domains.da.sources.gfs import GFSSource
from physicsnemo_curator.run import gather_pipeline, run_pipeline

# Load .env from this directory (does not overwrite existing env vars)
load_dotenv(Path(__file__).parent / ".env")

os.environ["LOGURU_LEVEL"] = "ERROR"

# Fixed time range for the full dataset (2021-01-01 to 2027-01-01)
# This is ~8760 6-hourly timestamps (6 years)
DATASET_START = datetime(2021, 1, 1, 0, 0)
DATASET_END = datetime(2027, 1, 1, 0, 0)
STATS_OUTPUT_PATH = Path("outputs/stats.zarr")
CHECKPOINT_DB_DIR = Path("outputs/checkpoint/")

PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
PRESSURE_LEVEL_VARS = ["t", "u", "v", "z", "q"]
SURFACE_VARS = ["u10m", "v10m", "u100m", "v100m", "t2m", "sp", "msl", "tcwv"]


def build_variable_list() -> list[str]:
    """Build the full list of earth2studio variable identifiers.

    Returns
    -------
    list[str]
        Variable IDs in earth2studio lexicon format (e.g., "t850", "u500").
    """
    variables: list[str] = []

    # Add pressure-level variables: var@level format (e.g., t850, u500)
    for level in PRESSURE_LEVELS:
        for var in PRESSURE_LEVEL_VARS:
            variables.append(f"{var}{level}")

    # Add surface variables
    variables.extend(SURFACE_VARS)

    return variables


def generate_6hourly_times(start: datetime, end: datetime) -> list[datetime]:
    """Generate 6-hourly timestamps from start to end (exclusive).

    GFS analysis data is available at 00, 06, 12, 18 UTC.

    Parameters
    ----------
    start : datetime
        Start time (should be at 00, 06, 12, or 18 UTC).
    end : datetime
        End time (exclusive).

    Returns
    -------
    list[datetime]
        All 6-hourly timestamps from start up to (but not including) end.
    """
    times: list[datetime] = []
    current = start
    while current < end:
        times.append(current)
        current += timedelta(hours=6)
    return times


def _build_s3_storage_options(anon: bool = False) -> dict[str, object]:
    """Build S3 storage options from environment variables.

    Reads credentials from ZARR_S3_* environment variables to avoid
    conflicting with GFS source (which uses anonymous S3 access):

    - ZARR_S3_ACCESS_KEY_ID: Access key ID
    - ZARR_S3_SECRET_ACCESS_KEY: Secret access key
    - ZARR_S3_REGION: AWS region (e.g., us-east-1)
    - ZARR_S3_ENDPOINT_URL: Custom endpoint URL (for S3-compatible stores)

    Parameters
    ----------
    anon : bool
        If True, use anonymous access (no credentials).

    Returns
    -------
    dict[str, object]
        Storage options dict for s3fs/fsspec.
    """
    if anon:
        return {"anon": True}

    options: dict[str, object] = {}

    # Read credentials from ZARR_S3_* environment variables
    # (using custom prefix to avoid conflict with GFS anonymous S3 access)
    access_key = os.environ.get("ZARR_S3_ACCESS_KEY_ID")
    secret_key = os.environ.get("ZARR_S3_SECRET_ACCESS_KEY")
    region = os.environ.get("ZARR_S3_REGION")
    endpoint_url = os.environ.get("ZARR_S3_ENDPOINT_URL")

    # s3fs expects these as top-level kwargs
    if access_key:
        options["key"] = access_key
    if secret_key:
        options["secret"] = secret_key
    if endpoint_url:
        options["endpoint_url"] = endpoint_url

    # Region goes in client_kwargs for boto3
    if region:
        options["client_kwargs"] = {"region_name": region}

    return options


def main() -> None:
    """Run the GFS global weather data ETL pipeline."""
    # Generate full time range (fixed for the dataset)
    all_times = generate_6hourly_times(DATASET_START, DATASET_END)
    total_indices = len(all_times)

    parser = argparse.ArgumentParser(
        description="GFS Global Weather Analysis ETL Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--zarr-path",
        type=str,
        default="s3://gfs",
        help="Zarr output directory. Supports S3 URLs (e.g., s3://bucket/path)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help=f"Start index (inclusive, default: 0). Total indices: {total_indices}",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=10,
        help=f"End index (exclusive, -1 = all, default: 10). Total indices: {total_indices}",
    )
    parser.add_argument(
        "--s3-anon",
        action="store_true",
        help="Use anonymous S3 credentials (for public buckets)",
    )
    args = parser.parse_args()

    # Determine indices to process
    start_idx = max(0, args.start_index)
    end_idx = total_indices if args.end_index == -1 else min(args.end_index, total_indices)
    indices_to_process = list(range(start_idx, end_idx))

    if not indices_to_process:
        print(f"No indices to process (start={start_idx}, end={end_idx}, total={total_indices})")
        return

    # Build variable list
    variables = build_variable_list()

    # Determine if output is remote (S3/GCS/Azure)
    is_remote = "://" in args.zarr_path
    storage_options: dict[str, object] | None = None
    if is_remote and args.zarr_path.startswith("s3://"):
        # Build S3 storage options from environment variables
        storage_options = _build_s3_storage_options(anon=args.s3_anon)
    elif is_remote:
        # Other remote stores (GCS, Azure) - use empty options (ambient credentials)
        storage_options = {}

    print("GFS Global Weather Analysis ETL Pipeline")
    print("=" * 50)
    print(f"  Dataset time range: {DATASET_START} to {DATASET_END}")
    print(f"  Total timestamps: {total_indices} (6-hourly)")
    print(f"  Processing indices: {start_idx} to {end_idx} ({len(indices_to_process)} indices)")
    print(f"  Time range for this run: {all_times[start_idx]} to {all_times[end_idx - 1]}")
    print(f"  Variables: {len(variables)} total")
    print(f"    - Pressure levels: {PRESSURE_LEVELS}")
    print(f"    - Level vars: {PRESSURE_LEVEL_VARS}")
    print(f"    - Surface vars: {SURFACE_VARS}")
    print(f"  Workers: {args.workers}")
    print(f"  Zarr path: {args.zarr_path}")
    if is_remote:
        print(f"  Storage: remote ({args.zarr_path.split(':')[0]})")
    print()

    # Configure the GFS source with all times (full dataset)
    source = GFSSource(
        times=all_times,
        variables=variables,
        source="aws",
        cache=False,
    )

    # Build the pipeline:
    # 1. DataArrayStatsFilter — compute running statistics
    # 2. ZarrSink — write to Zarr store (local or remote)
    #
    # GFS grid is 721 x 1440 (0.25 degree global)
    # n_indices is the full time range so store is properly sized
    stats_filter = DataArrayStatsFilter(output=STATS_OUTPUT_PATH, dims=("time",), keep_shards=True)
    pipeline = source.filter(stats_filter).write(
        ZarrSink(
            output_path=f"{args.zarr_path}/data.zarr",
            chunks={"time": 1, "lat": 721, "lon": 1440},
            n_indices=total_indices,
            variables=variables,
            overwrite=False,
            storage_options=storage_options,
        )
    )

    # Run the pipeline with parallel workers (only process selected indices)
    print(f"Processing {len(indices_to_process)} timestamps with {args.workers} workers...")
    print(f"  Indices range: {indices_to_process[0]} to {indices_to_process[-1]}")
    results = run_pipeline(
        pipeline,
        n_jobs=args.workers,
        backend="process_pool",
        indices=indices_to_process,
        db_dir=CHECKPOINT_DB_DIR,
        resume=True,
        use_tui=False,
    )

    print(f"\nProcessed {len(results)} timestamps")

    # Merge statistics from parallel workers
    gathered = gather_pipeline(pipeline)
    if gathered:
        print(f"Merged {len(gathered)} statistic shards")
    print(f"Statistics written to: {STATS_OUTPUT_PATH}")
    print(f"Dataset written to: {args.zarr_path}/data.zarr")


if __name__ == "__main__":
    main()
