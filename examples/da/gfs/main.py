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

Curate GFS 0.25-degree global weather analysis data into a Zarr store,
fetching pressure-level variables (t, u, v, z, q) at multiple levels
plus surface variables (u10m, v10m, u100m, v100m, t2m, sp, msl, tcwv).

GFS data is available from AWS at 6-hour intervals starting from 2021-01-01.

Remote Zarr Output
------------------
To write to a remote Zarr store (e.g., S3), pass an S3 URL as the output path::

    python main.py --output s3://my-bucket/gfs/dataset.zarr

S3 credentials can be configured via environment variables::

    export AWS_ACCESS_KEY_ID="your-access-key"
    export AWS_SECRET_ACCESS_KEY="your-secret-key"
    export AWS_REGION="us-east-1"  # Optional
    export AWS_ENDPOINT_URL="https://s3.amazonaws.com"  # Optional, for S3-compatible stores

For other cloud providers (GCS, Azure), use the appropriate URL scheme
(gs://, az://) and install the corresponding fsspec backend.
"""

import argparse
import os
from datetime import datetime, timedelta

from physicsnemo_curator.domains.da.filters.stats import DataArrayStatsFilter
from physicsnemo_curator.domains.da.sinks.zarr_writer import ZarrSink
from physicsnemo_curator.domains.da.sources.gfs import GFSSource
from physicsnemo_curator.run import gather_pipeline, run_pipeline

os.environ["LOGURU_LEVEL"] = "ERROR"


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

    Reads credentials from standard AWS environment variables:
    - AWS_ACCESS_KEY_ID: Access key ID
    - AWS_SECRET_ACCESS_KEY: Secret access key
    - AWS_REGION: AWS region (e.g., us-east-1)
    - AWS_ENDPOINT_URL: Custom endpoint URL (for S3-compatible stores)

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

    options: dict[str, object] = {"anon": False}

    # Read credentials from environment
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    region = os.environ.get("AWS_REGION")
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL")

    if access_key and secret_key:
        options["key"] = access_key
        options["secret"] = secret_key

    # Build client_kwargs for region and endpoint
    client_kwargs: dict[str, str] = {}
    if region:
        client_kwargs["region_name"] = region
    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url

    if client_kwargs:
        options["client_kwargs"] = client_kwargs

    return options


def main() -> None:
    """Run the GFS global weather data ETL pipeline."""
    parser = argparse.ArgumentParser(
        description="GFS Global Weather Analysis ETL Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/gfs_global",
        help="Output directory for Zarr store. Supports S3 URLs (e.g., s3://bucket/path)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2021-01-01T00:00",
        help="Start datetime in ISO format (default: 2021-01-01T00:00)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2027-01-01T00:00",
        help="End datetime in ISO format, exclusive (default: 2027-01-08T00:00)",
    )
    parser.add_argument(
        "--n-indices",
        type=int,
        default=None,
        help="Limit to first N timestamps (default: all)",
    )
    parser.add_argument(
        "--s3-anon",
        action="store_true",
        help="Use anonymous S3 credentials (for public buckets)",
    )
    args = parser.parse_args()

    # Parse time range
    start_time = datetime.fromisoformat(args.start)
    end_time = datetime.fromisoformat(args.end)
    times = generate_6hourly_times(start_time, end_time)

    if args.n_indices is not None:
        times = times[: args.n_indices]

    # Build variable list
    variables = build_variable_list()

    # Determine if output is remote (S3/GCS/Azure)
    is_remote = "://" in args.output
    storage_options: dict[str, object] | None = None
    if is_remote and args.output.startswith("s3://"):
        # Build S3 storage options from environment variables
        storage_options = _build_s3_storage_options(anon=args.s3_anon)
    elif is_remote:
        # Other remote stores (GCS, Azure) - use empty options (ambient credentials)
        storage_options = {}

    print("GFS Global Weather Analysis ETL Pipeline")
    print("=" * 50)
    print(f"  Time range: {start_time} to {end_time}")
    print(f"  Timestamps: {len(times)} (6-hourly)")
    print(f"  Variables: {len(variables)} total")
    print(f"    - Pressure levels: {PRESSURE_LEVELS}")
    print(f"    - Level vars: {PRESSURE_LEVEL_VARS}")
    print(f"    - Surface vars: {SURFACE_VARS}")
    print(f"  Workers: {args.workers}")
    print(f"  Output: {args.output}")
    if is_remote:
        print(f"  Storage: remote ({args.output.split(':')[0]})")
    print()

    # Configure the GFS source
    source = GFSSource(
        times=times,
        variables=variables,
        source="aws",
        cache=True,
    )

    # Build the pipeline:
    # 1. DataArrayStatsFilter — compute running statistics
    # 2. ZarrSink — write to Zarr store (local or remote)
    #
    # GFS grid is 721 x 1440 (0.25 degree global)
    stats_path = f"{args.output}/stats.zarr"
    stats_filter = DataArrayStatsFilter(output=stats_path, dims=("time",))
    pipeline = source.filter(stats_filter).write(
        ZarrSink(
            output_path=f"{args.output}/dataset.zarr",
            chunks={"time": 1, "lat": 721, "lon": 1440},
            n_indices=len(times),
            variables=variables,
            overwrite=True,
            storage_options=storage_options,
        )
    )

    # Run the pipeline with parallel workers
    print(f"Processing {len(times)} timestamps with {args.workers} workers...")
    results = run_pipeline(
        pipeline,
        n_jobs=args.workers,
        backend="process_pool",
        indices=range(len(times)),
    )

    print(f"\nProcessed {len(results)} timestamps")

    # Merge statistics from parallel workers
    gathered = gather_pipeline(pipeline)
    if gathered:
        print(f"Merged {len(gathered)} statistic shards")
    print(f"Statistics written to: {stats_path}")
    print(f"Dataset written to: {args.output}/dataset.zarr")


if __name__ == "__main__":
    main()
