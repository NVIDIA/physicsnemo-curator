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

"""
ERA5 Reanalysis ETL Pipeline
==============================

This example demonstrates a complete **Source → Filter → Sink** pipeline
for curating `ERA5 <https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5>`_
global reanalysis data.

ERA5 provides hourly estimates of atmospheric variables on a 0.25° global
latitude–longitude grid (721 × 1440).  The pipeline fetches data via
`earth2studio <https://github.com/NVIDIA/earth2studio>`_ backends (ARCO,
WB2, NCAR, CDS), computes running temporal statistics with Welford's
online algorithm, and writes the results to a Zarr store — a common
preprocessing step for training weather and climate ML models (e.g.
FourCastNet, GraphCast, Pangu-Weather).

We fetch only 3 hourly snapshots and 3 surface variables to keep the
example fast.
"""

# %%
# Imports
# -------
#
# Import the core pipeline components: a **Source** to fetch ERA5 fields,
# a **Filter** to compute temporal statistics, a **Sink** to write
# outputs, and :func:`~physicsnemo.curator.run.run_pipeline` for
# parallel execution.

from datetime import datetime

from physicsnemo.curator.da.filters.moments import MomentsFilter
from physicsnemo.curator.da.sinks.zarr_writer import ZarrSink
from physicsnemo.curator.da.sources.era5 import ERA5Source
from physicsnemo.curator.run import gather_pipeline, run_pipeline

# %%
# Configure the Source
# --------------------
#
# :class:`~physicsnemo.curator.da.sources.era5.ERA5Source` connects to a
# cloud-hosted ERA5 mirror and discovers available timestamps.  Each
# index yields one hourly snapshot as an :class:`xarray.DataArray` with
# dimensions ``(time, variable, lat, lon)``.
#
# We select three surface variables:
#
# - ``t2m`` — 2-metre temperature (K)
# - ``u10m`` — 10-metre U-component of wind (m/s)
# - ``v10m`` — 10-metre V-component of wind (m/s)
#
# The default ``"arco"`` backend streams data from Google's
# `Analysis-Ready, Cloud-Optimized ERA5 <https://cloud.google.com/storage/docs/public-datasets/era5>`_
# Zarr store — no API key required.

times = [
    datetime(2020, 6, 1, 0),
    datetime(2020, 6, 1, 6),
    datetime(2020, 6, 1, 12),
]

variables = ["t2m", "u10m", "v10m"]

source = ERA5Source(
    times=times,
    variables=variables,
    backend="arco",
    cache=True,
)

print(f"Timestamps to fetch: {len(source)}")
print(f"Variables: {source.variables}")
print(f"Backend: {source.active_backend}")

# %%
# Build the Pipeline
# ------------------
#
# The fluent API chains **Source → Filter → Sink** into a lazy
# :class:`~physicsnemo.curator.core.base.Pipeline`.  Nothing is
# executed until we explicitly process indices.
#
# - :class:`~physicsnemo.curator.da.filters.moments.MomentsFilter`
#   computes running temporal statistics (mean, variance, skewness,
#   min, max) using Welford's numerically stable online algorithm.
#   The filter is **pass-through** — each DataArray is yielded
#   unchanged while accumulators update in the background.
# - :class:`~physicsnemo.curator.da.sinks.zarr_writer.ZarrSink`
#   writes each DataArray to a Zarr store, with one group per
#   variable and automatic append along the ``time`` dimension.

pipeline = source.filter(MomentsFilter(output="outputs/era5/moments.zarr", dims=("time",))).write(
    ZarrSink(output_path="outputs/era5/data.zarr")
)

# %%
# Run the Pipeline
# ----------------
#
# :func:`~physicsnemo.curator.run.run_pipeline` dispatches work
# sequentially here (ERA5 backends are I/O-bound and share an
# in-process cache, so parallelism offers limited benefit for small
# fetches).  For large time ranges, use ``backend="process_pool"``
# with ``n_jobs=4``.

results = run_pipeline(
    pipeline,
    n_jobs=1,
    backend="sequential",
    indices=range(len(source)),
    progress=True,
)

print(f"Processed {len(results)} snapshots")
for i, paths in enumerate(results):
    print(f"  Snapshot {i} ({times[i]}): {paths}")

# %%
# Gather Statistics
# ------------------
#
# Merge per-index shard files produced by the MomentsFilter into a
# single Zarr store with global temporal statistics.

merged = gather_pipeline(pipeline)
for path in merged:
    print(f"Merged moments: {path}")

# %%
# Inspect Outputs
# ----------------
#
# The ``outputs/era5/`` directory now contains:
#
# .. code-block:: text
#
#     outputs/era5/
#     ├── data.zarr/                  # Raw ERA5 DataArrays
#     │   ├── t2m/                    # 2-metre temperature
#     │   │   └── data.zarr           # (time, lat, lon)
#     │   ├── u10m/                   # U-wind at 10 m
#     │   │   └── data.zarr
#     │   └── v10m/                   # V-wind at 10 m
#     │       └── data.zarr
#     └── moments.zarr/               # Temporal statistics (merged)
#         ├── t2m/
#         │   ├── mean                # Temporal mean field
#         │   ├── variance            # Temporal variance
#         │   ├── skewness            # Temporal skewness
#         │   ├── min                 # Temporal minimum
#         │   └── max                 # Temporal maximum
#         ├── u10m/
#         │   └── ...
#         └── v10m/
#             └── ...
#
# .. note::
#
#    **Backend selection.** ERA5Source supports multiple earth2studio
#    backends (``"arco"``, ``"wb2"``, ``"ncar"``, ``"cds"``).  Pass a
#    list like ``backend=["arco", "ncar"]`` for automatic fallback —
#    variables not found in the first backend's lexicon are automatically
#    routed to the next available backend.  Use the
#    ``variable_routing`` property to inspect the resulting mapping.
