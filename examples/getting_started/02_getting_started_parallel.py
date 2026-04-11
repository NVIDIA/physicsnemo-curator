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
Running a Pipeline in Parallel
==============================

This example demonstrates :func:`~physicsnemo_curator.run.run_pipeline`
to execute a pipeline across multiple source indices using parallel
workers.

Building on the :doc:`Creating a Pipeline </auto_examples/getting_started/01_getting_started_pipeline>`
example, we process multiple DrivAerML CFD meshes concurrently with a
``process_pool`` backend and then merge per-worker statistics using
:func:`~physicsnemo_curator.run.gather_pipeline`.

.. note::

   Install the mesh extras before running::

       pip install physicsnemo-curator[mesh]
"""

# %%
# Imports
# -------

from physicsnemo_curator.domains.mesh.filters.mean import MeanFilter
from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink
from physicsnemo_curator.domains.mesh.sources.drivaerml import DrivAerMLSource
from physicsnemo_curator.run import gather_pipeline, run_pipeline

# %%
# Build the Pipeline
# ------------------
#
# :class:`~physicsnemo_curator.domains.mesh.sources.drivaerml.DrivAerMLSource`
# provides 500 DrivAerML automotive CFD meshes from HuggingFace Hub.
# We attach a :class:`~physicsnemo_curator.domains.mesh.filters.mean.MeanFilter`
# for spatial statistics and a
# :class:`~physicsnemo_curator.domains.mesh.sinks.mesh_writer.MeshSink` for
# output.

pipeline = (
    DrivAerMLSource(mesh_type="boundary")
    .filter(MeanFilter(output="outputs/parallel/stats.parquet"))
    .write(MeshSink(output_dir="outputs/parallel/meshes/"))
)

print(f"Total runs available: {len(pipeline)}")

# %%
# Run in Parallel
# ---------------
#
# :func:`~physicsnemo_curator.run.run_pipeline` dispatches indices to
# parallel workers.  Key parameters:
#
# - ``n_jobs`` — number of workers (``-1`` = all CPUs)
# - ``backend`` — ``"process_pool"``, ``"thread_pool"``, ``"loky"``,
#   ``"dask"``, ``"prefect"``, or ``"auto"``
# - ``indices`` — which source indices to process (default: all)
# - ``progress`` — show a progress bar
#
# Each worker receives an independent copy of the pipeline, so data is
# read, filtered, and written concurrently.

results = run_pipeline(
    pipeline,
    n_jobs=4,
    backend="process_pool",
    indices=range(3),
    progress=True,
)

# %%
# Inspect Results
# ---------------
#
# ``results`` is a list of lists — one entry per processed index, each
# containing the file paths returned by the sink.

print(f"\nProcessed {len(results)} runs")
for i, paths in enumerate(results):
    print(f"  Run {i}: {paths}")

# %%
# Gather Statistics
# -----------------
#
# When running in parallel, stateful filters (like
# :class:`~physicsnemo_curator.domains.mesh.filters.mean.MeanFilter`) produce
# per-index shard files.
# :func:`~physicsnemo_curator.run.gather_pipeline` discovers those
# shards, calls the filter's ``merge()`` method to combine them into
# a single output file, and cleans up the temporaries.

merged = gather_pipeline(pipeline)
for path in merged:
    print(f"Merged statistics: {path}")

# %%
# Available Backends
# ------------------
#
# +------------------+---------------------+-----------------------------------+
# | Backend          | Install extra       | Best for                          |
# +==================+=====================+===================================+
# | ``sequential``   | *(built-in)*        | Debugging, small datasets         |
# +------------------+---------------------+-----------------------------------+
# | ``thread_pool``  | *(built-in)*        | I/O-bound tasks                   |
# +------------------+---------------------+-----------------------------------+
# | ``process_pool`` | *(built-in)*        | CPU-bound tasks (default)         |
# +------------------+---------------------+-----------------------------------+
# | ``loky``         | ``pip install .[loky]``    | Robust multi-process       |
# +------------------+---------------------+-----------------------------------+
# | ``dask``         | ``pip install .[dask]``    | Distributed clusters       |
# +------------------+---------------------+-----------------------------------+
# | ``prefect``      | ``pip install .[prefect]`` | Orchestrated workflows     |
# +------------------+---------------------+-----------------------------------+
#
# Use ``backend="auto"`` to let the framework pick the best available
# backend for your system.
