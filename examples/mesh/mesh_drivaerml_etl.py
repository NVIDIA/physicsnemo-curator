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
DrivAerML End-to-End ETL Pipeline.
==================================

This example demonstrates a complete **Source → Filter → Sink** pipeline
that reads `DrivAerML <https://huggingface.co/datasets/neashton/drivaerml>`_
boundary meshes from HuggingFace Hub, computes spatial field statistics,
and writes the processed meshes to disk — all in parallel using a process
pool.

DrivAerML contains 500 parametrically morphed variants of the DrivAer
notchback vehicle with high-fidelity scale-resolving CFD (OpenFOAM).
We process the first 3 runs to keep the example fast.
"""

# %%
# Imports
# -------
#
# Import the core pipeline components: a **Source** to read meshes, a
# **Filter** to compute statistics, a **Sink** to write outputs, and
# :func:`~physicsnemo_curator.run.run_pipeline` for parallel execution.

from physicsnemo_curator.mesh.filters.mean import MeanFilter
from physicsnemo_curator.mesh.sinks.mesh_writer import MeshSink
from physicsnemo_curator.mesh.sources.drivaerml import DrivAerMLSource
from physicsnemo_curator.run import run_pipeline

# %%
# Configure the Source
# --------------------
#
# :class:`~physicsnemo_curator.mesh.sources.drivaerml.DrivAerMLSource`
# connects to the HuggingFace Hub dataset and discovers available runs.
# We select ``mesh_type="boundary"`` to read the surface VTP files
# which contain the flow fields on the vehicle boundary.

source = DrivAerMLSource(
    mesh_type="boundary",
    manifold_dim="auto",
    point_source="vertices",
)

print(f"Total runs available: {len(source)}")

# %%
# Build the Pipeline
# ------------------
#
# The fluent API chains **Source → Filter → Sink** into a lazy
# :class:`~physicsnemo_curator.core.base.Pipeline`.  Nothing is
# executed until we explicitly process indices.
#
# - :class:`~physicsnemo_curator.mesh.filters.mean.MeanFilter` computes
#   per-field spatial means and accumulates them into a Parquet summary.
# - :class:`~physicsnemo_curator.mesh.sinks.mesh_writer.MeshSink` writes
#   each mesh in PhysicsNeMo's native tensordict memory-mapped format.
#
# We keep a reference to the filter so we can call :meth:`flush` after
# the pipeline finishes.

mean_filter = MeanFilter(output="outputs/mean_stats.parquet")

pipeline = source.filter(mean_filter).write(MeshSink(output_dir="outputs/meshes/"))

# %%
# Run the Pipeline
# ----------------
#
# :func:`~physicsnemo_curator.run.run_pipeline` processes the first 3
# runs sequentially.  Sequential execution ensures the stateful
# :class:`~physicsnemo_curator.mesh.filters.mean.MeanFilter` accumulates
# statistics across all items in a single process.

results = run_pipeline(
    pipeline,
    n_jobs=1,
    backend="sequential",
    indices=range(3),
    progress=True,
)

# %%
# Gather Statistics
# -----------------
#
# Stateful filters like :class:`~physicsnemo_curator.mesh.filters.mean.MeanFilter`
# accumulate results in memory during the pipeline run.  Call
# :meth:`~physicsnemo_curator.mesh.filters.mean.MeanFilter.flush` to write
# the accumulated per-field spatial means to a Parquet file.

parquet_path = mean_filter.flush()
print(f"Statistics written to: {parquet_path}")

# %%
# Inspect Results
# ---------------
#
# ``results`` is a list of lists — one entry per processed index,
# each containing the file paths written by the sink.

print(f"Processed {len(results)} runs")
for i, paths in enumerate(results):
    print(f"  Run {i}: {paths}")

# %%
# The ``outputs/`` directory now contains:
#
# .. code-block:: text
#
#     outputs/
#     ├── mean_stats.parquet      # Per-field spatial means (from flush)
#     └── meshes/
#         ├── mesh_0000_0/        # Run 0 in tensordict format
#         ├── mesh_0001_0/        # Run 1
#         └── mesh_0002_0/        # Run 2
#
# .. note::
#
#    Stateful filters like
#    :class:`~physicsnemo_curator.mesh.filters.mean.MeanFilter`
#    accumulate results in memory.  You must call :meth:`flush` after the
#    pipeline finishes to write the Parquet file.  When using parallel
#    backends (``n_jobs > 1``), each worker gets an independent copy of
#    the filter, so use :meth:`MeanFilter.merge` to combine per-worker
#    outputs afterwards.
