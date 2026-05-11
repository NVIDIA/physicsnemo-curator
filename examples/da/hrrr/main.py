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

"""HRRR Analysis Data ETL Pipeline.

Curate HRRR 3 km analysis data for 3 days into a Zarr store,
computing running statistics along the way.
"""

import argparse
import os
from datetime import datetime, timedelta

from physicsnemo_curator.domains.da.filters.stats import DataArrayStatsFilter
from physicsnemo_curator.domains.da.sinks.zarr_writer import ZarrSink
from physicsnemo_curator.domains.da.sources.hrrr import HRRRSource
from physicsnemo_curator.run import gather_pipeline, run_pipeline

os.environ["LOGURU_LEVEL"] = "ERROR"


def _generate_hourly_times(start: datetime, days: int) -> list[datetime]:
    """Generate hourly timestamps for a given number of days.

    Parameters
    ----------
    start : datetime
        Start timestamp (inclusive).
    days : int
        Number of days to generate timestamps for.

    Returns
    -------
    list[datetime]
        All hourly timestamps from *start* through *start + days*.
    """
    end = start + timedelta(days=days)

    times: list[datetime] = []
    current = start
    while current < end:
        times.append(current)
        current += timedelta(hours=1)
    return times


def main() -> None:
    """Run the HRRR analysis ETL pipeline."""
    parser = argparse.ArgumentParser(description="HRRR Analysis Data ETL Pipeline")
    parser.add_argument(
        "--output",
        type=str,
        default="output/hrrr_analysis",
        help="Output directory for Zarr store (default: output/hrrr_analysis)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2024-01-01",
        help="Start date in YYYY-MM-DD format (default: 2024-01-01)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=3,
        help="Number of days to fetch (default: 3)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="aws",
        choices=["aws", "google", "nomads"],
        help="HRRR cloud data source (default: aws)",
    )
    parser.add_argument(
        "--n-indices",
        type=int,
        default=72,
        help="Number of source indices to process (default: 72 = 3 days)",
    )
    args = parser.parse_args()

    # Variables: 2m temperature, 2m specific humidity, total column water vapour
    variables = ["t2m", "q2m", "tcwv"]

    # Generate all hourly timestamps for the target period
    start = datetime.fromisoformat(args.start_date)
    times = _generate_hourly_times(start, args.days)
    print(f"HRRR analysis ETL: {args.start_date} ({args.days} days)")
    print(f"  Timestamps: {len(times)} (hourly)")
    print(f"  Variables: {variables}")
    print(f"  Source: {args.source}")
    print(f"  Workers: {args.workers}")
    print(f"  Indices: {args.n_indices}")
    print(f"  Output: {args.output}")

    # Configure the HRRR source
    source = HRRRSource(
        times=times,
        variables=variables,
        source=args.source,
        cache=True,
    )

    # Build the pipeline:
    # 1. DataArrayStatsFilter — compute running statistics (mean, variance, skewness, min, max)
    # 2. ZarrSink — write each timestep to a Zarr v3 store
    #    n_indices + variables enables pre-allocated store with concurrent-safe region writes.
    stats_path = f"{args.output}/stats.zarr"
    stats_filter = DataArrayStatsFilter(output=stats_path, dims=("time",))
    zarr_path = f"{args.output}/dataset.zarr"
    chunks = {"time": 1, "hrrr_x": 1799, "hrrr_y": 1059}
    pipeline = source.filter(stats_filter).write(
        ZarrSink(
            output_path=zarr_path,
            chunks=chunks,
            n_indices=args.n_indices,
            variables=variables,
            overwrite=True,
        )
    )

    # Run the pipeline with parallel workers.
    # Each forked process gets its own copy of the pipeline, so earth2studio
    # backends are isolated (no async event-loop conflicts).
    results = run_pipeline(pipeline, n_jobs=args.workers, backend="process_pool", indices=range(args.n_indices))

    print(f"\nProcessed {len(results)} timesteps")

    # Statistics are auto-flushed after each index.
    # gather_pipeline merges per-worker shards when using process_pool backend.
    gathered = gather_pipeline(pipeline)
    if gathered:
        print(f"Merged {len(gathered)} statistic shards")
    print(f"Statistics written to: {stats_path}")


if __name__ == "__main__":
    main()
