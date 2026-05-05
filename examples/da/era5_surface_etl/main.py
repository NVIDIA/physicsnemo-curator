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

"""ERA5 Surface Data ETL Pipeline.

Curate ERA5 surface reanalysis data for January 2020 into a Zarr store,
computing running statistics along the way.
"""

import argparse
from datetime import datetime, timedelta

from physicsnemo_curator.domains.da.filters.moments import MomentsFilter
from physicsnemo_curator.domains.da.sinks.zarr_writer import ZarrSink
from physicsnemo_curator.domains.da.sources.era5 import ERA5Source
from physicsnemo_curator.run import run_pipeline


def _generate_hourly_times(year: int, month: int) -> list[datetime]:
    """Generate hourly timestamps for an entire month.

    Parameters
    ----------
    year : int
        Year of the target month.
    month : int
        Month number (1-12).

    Returns
    -------
    list[datetime]
        All hourly timestamps from the 1st 00:00 to end of month.
    """
    start = datetime(year, month, 1, 0)
    end = datetime(year + 1, 1, 1, 0) if month == 12 else datetime(year, month + 1, 1, 0)

    times: list[datetime] = []
    current = start
    while current < end:
        times.append(current)
        current += timedelta(hours=1)
    return times


def main() -> None:
    """Run the ERA5 surface ETL pipeline."""
    parser = argparse.ArgumentParser(description="ERA5 Surface Data ETL Pipeline")
    parser.add_argument(
        "--output",
        type=str,
        default="output/era5_surface",
        help="Output directory for Zarr store (default: output/era5_surface)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2020,
        help="Year to fetch (default: 2020)",
    )
    parser.add_argument(
        "--month",
        type=int,
        default=1,
        help="Month to fetch (default: 1 = January)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="arco",
        choices=["arco", "wb2", "ncar", "cds"],
        help="ERA5 data backend (default: arco)",
    )
    args = parser.parse_args()

    # Surface variables: 2m temperature, 10m wind (u/v), surface pressure, mean sea level pressure
    variables = ["t2m", "u10m", "v10m", "sp", "msl"]

    # Generate all hourly timestamps for the target month
    times = _generate_hourly_times(args.year, args.month)
    print(f"ERA5 surface ETL: {args.year}-{args.month:02d}")
    print(f"  Timestamps: {len(times)} (hourly)")
    print(f"  Variables: {variables}")
    print(f"  Backend: {args.backend}")
    print(f"  Workers: {args.workers}")
    print(f"  Output: {args.output}")

    # Configure the ERA5 source
    source = ERA5Source(
        times=times,
        variables=variables,
        backend=args.backend,
        cache=True,
    )

    # Build the pipeline:
    # 1. MomentsFilter — compute running statistics (mean, variance, skewness, min, max)
    # 2. ZarrSink — write each timestep to a Zarr v3 store
    stats_path = f"{args.output}/stats.zarr"
    stats_filter = MomentsFilter(output=stats_path, dims=("time",))
    pipeline = source.filter(stats_filter).write(
        ZarrSink(
            output_path=f"{args.output}/dataset.zarr",
            chunks={"time": 1, "lat": 721, "lon": 1440},
        )
    )

    # Run the pipeline with parallel workers.
    # Use thread_pool backend since ERA5 fetching is I/O-bound (network downloads).
    # Use progress="log" for simple timestamped output that coexists with
    # earth2studio's loguru logging (the default TUI can conflict with it).
    results = run_pipeline(pipeline, n_jobs=args.workers, backend="thread_pool", progress="log")

    print(f"\nProcessed {len(results)} timesteps")

    # Flush accumulated statistics to disk
    stats_filter.flush()
    print(f"Statistics written to: {stats_path}")


if __name__ == "__main__":
    main()
