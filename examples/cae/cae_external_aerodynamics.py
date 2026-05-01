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
External Aerodynamics ETL Pipeline
===================================

This example demonstrates a **multi-pipeline ETL workflow** for curating
automotive CFD simulation data from the
`DrivAerML <https://huggingface.co/datasets/neashton/drivaerml>`_ dataset.

DrivAerML contains 500 parametrically morphed variants of the DrivAer
notchback vehicle with high-fidelity scale-resolving CFD (OpenFOAM).
Each run provides **boundary** (surface) and **volume** mesh data with
flow fields such as velocity, pressure, and wall shear stress.

The pipeline processes both surface and volume meshes through a chain of
filters that log metadata, compute field statistics, and convert field
precision to float32 — a common preprocessing step for ML training
(e.g. DoMINO, Transolver).

We process only the first 3 runs to keep the example fast.
"""

# %%
# Imports
# -------
#
# Import the core pipeline components: a **Source** to read meshes,
# **Filters** for metadata logging, statistics, and precision conversion,
# a **Sink** to write outputs, and
# :func:`~physicsnemo_curator.run.run_pipeline` for parallel execution.

from physicsnemo_curator.domains.mesh.filters.mesh_info import MeshInfoFilter
from physicsnemo_curator.domains.mesh.filters.precision import PrecisionFilter
from physicsnemo_curator.domains.mesh.filters.stats import MeshStatsFilter
from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink
from physicsnemo_curator.domains.mesh.sources.drivaerml import DrivAerMLSource
from physicsnemo_curator.run import gather_pipeline, run_pipeline

# %%
# Surface (Boundary) Mesh Pipeline
# ---------------------------------
#
# The surface pipeline reads the boundary VTP files that contain flow fields
# on the vehicle surface (pressure coefficient, wall shear stress, etc.).
#
# We chain four stages:
#
# 1. :class:`~physicsnemo_curator.domains.mesh.sources.drivaerml.DrivAerMLSource`
#    discovers runs and reads surface meshes from HuggingFace Hub.
# 2. :class:`~physicsnemo_curator.domains.mesh.filters.mesh_info.MeshInfoFilter`
#    logs metadata (point/cell counts, field shapes) and writes structured
#    records to a JSON-lines file for post-analysis.
# 3. :class:`~physicsnemo_curator.domains.mesh.filters.stats.MeshStatsFilter` computes
#    comprehensive per-field statistics (mean, std, skewness, kurtosis) and
#    stores Welford accumulator state for cross-file aggregation.
# 4. :class:`~physicsnemo_curator.domains.mesh.filters.precision.PrecisionFilter`
#    casts all float64 fields to float32, reducing memory and storage by
#    half — a standard step before ML training.

surface_source = DrivAerMLSource(
    mesh_type="boundary",
    manifold_dim="auto",
    point_source="vertices",
)

print(f"Total DrivAerML runs available: {len(surface_source)}")

surface_pipeline = (
    surface_source.filter(MeshInfoFilter(output="output/aero/surface_info.jsonl"))
    .filter(MeshStatsFilter(output="output/aero/surface_stats.parquet"))
    .filter(PrecisionFilter(target_dtype="float32"))
    .write(MeshSink(output_dir="output/aero/surface_meshes/"))
)

# %%
# Volume Mesh Pipeline
# ---------------------
#
# The volume pipeline reads the volumetric VTU files containing the full
# 3-D flow field (velocity, pressure, turbulent kinetic energy).  Volume
# meshes are typically much larger than surface meshes, so we read cell
# centroids rather than raw vertices.
#
# Here we use :class:`~physicsnemo_curator.domains.mesh.filters.stats.MeshStatsFilter`
# for volume field statistics.  This filter includes Welford accumulators
# that can be merged across parallel workers for exact global statistics.

volume_source = DrivAerMLSource(
    mesh_type="volume",
    manifold_dim="auto",
    point_source="cell_centroids",
)

volume_pipeline = (
    volume_source.filter(MeshInfoFilter(output="output/aero/volume_info.jsonl"))
    .filter(MeshStatsFilter(output="output/aero/volume_stats.parquet"))
    .filter(PrecisionFilter(target_dtype="float32"))
    .write(MeshSink(output_dir="output/aero/volume_meshes/"))
)

# %%
# Run the Surface Pipeline
# -------------------------
#
# :func:`~physicsnemo_curator.run.run_pipeline` dispatches work to a
# ``process_pool`` backend with 4 workers.  Each worker gets an
# independent copy of the pipeline, so meshes are read, filtered, and
# written concurrently.

surface_results = run_pipeline(
    surface_pipeline,
    n_jobs=4,
    backend="process_pool",
    indices=range(3),
    progress=True,
)

print(f"Surface meshes processed: {len(surface_results)}")
for i, paths in enumerate(surface_results):
    print(f"  Run {i}: {paths}")

# %%
# Run the Volume Pipeline
# ------------------------
#
# Volume meshes are significantly larger (millions of cells), so consider
# reducing ``n_jobs`` if memory is constrained.

volume_results = run_pipeline(
    volume_pipeline,
    n_jobs=2,
    backend="process_pool",
    indices=range(3),
    progress=True,
)

print(f"Volume meshes processed: {len(volume_results)}")
for i, paths in enumerate(volume_results):
    print(f"  Run {i}: {paths}")

# %%
# Gather Statistics
# ------------------
#
# Each worker writes per-index shard files for stateful filters.
# :func:`~physicsnemo_curator.run.gather_pipeline` merges them into
# single output files and removes the temporary shards.

for pipe in (surface_pipeline, volume_pipeline):
    merged = gather_pipeline(pipe)
    for path in merged:
        print(f"Merged: {path}")

# %%
# Inspect Outputs
# ----------------
#
# The ``output/aero/`` directory now contains:
#
# .. code-block:: text
#
#     output/aero/
#     ├── surface_info.jsonl         # Mesh metadata (JSON-lines)
#     ├── surface_stats.parquet      # Per-field statistics (merged)
#     ├── surface_meshes/
#     │   ├── mesh_0000_0/           # Run 0 surface in tensordict format
#     │   ├── mesh_0001_0/           # Run 1 surface
#     │   └── mesh_0002_0/           # Run 2 surface
#     ├── volume_info.jsonl          # Volume mesh metadata
#     ├── volume_stats.parquet       # Volume field statistics (merged)
#     └── volume_meshes/
#         ├── mesh_0000_0/           # Run 0 volume
#         ├── mesh_0001_0/           # Run 1 volume
#         └── mesh_0002_0/           # Run 2 volume
#
# .. note::
#
#    **Relationship to the legacy pipeline.** This example replaces the
#    Hydra-based ``external_aerodynamics`` pipeline from ``examples-old/``,
#    which required ~2,800 lines across 10+ files with YAML configuration.
#    The modern fluent API provides the same core workflow — read, filter,
#    write — in a fraction of the code, with built-in parallel execution.
