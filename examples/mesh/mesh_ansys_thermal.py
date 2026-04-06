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
Ansys Thermal Simulation ETL Pipeline
======================================

This example demonstrates a **Source → Filter → Sink** pipeline for
curating Ansys thermal simulation data.

Ansys solvers produce ``.rst`` result files containing mesh coordinates,
temperature distributions, heat flux vectors, and other physics fields.
The pipeline reads these files with
:class:`~physicsnemo_curator.mesh.sources.ansys_rst.AnsysRSTSource`,
logs mesh metadata, computes summary statistics, converts fields to
single precision, and writes the processed meshes to disk.

.. note::

   This example requires the ``ansys-dpf-core`` package for reading
   ``.rst`` files.  Install it with ``pip install ansys-dpf-core``.
   Reading real ``.rst`` files additionally requires an Ansys
   installation (2021 R1+) with a valid license.
"""

# %%
# Imports
# -------
#
# Import the pipeline building blocks: a **Source** for Ansys ``.rst``
# files, informational / statistics / precision filters, a **Sink** for
# writing outputs, and :func:`~physicsnemo_curator.run.run_pipeline` for
# parallel execution.

from physicsnemo_curator.mesh.filters.mesh_info import MeshInfoFilter
from physicsnemo_curator.mesh.filters.precision import PrecisionFilter
from physicsnemo_curator.mesh.filters.stats import StatsFilter
from physicsnemo_curator.mesh.sinks.mesh_writer import MeshSink
from physicsnemo_curator.mesh.sources.ansys_rst import AnsysRSTSource
from physicsnemo_curator.run import run_pipeline

# %%
# Configure the Source
# --------------------
#
# :class:`~physicsnemo_curator.mesh.sources.ansys_rst.AnsysRSTSource`
# scans ``input_dir`` for files matching ``*.rst``.  Each file
# corresponds to one simulation case (e.g. a different thermal scenario).
#
# The source auto-discovers available result types in each file.  To
# limit extraction to specific fields, pass a list of result type names
# such as ``["temperature", "heat_flux"]`` via the ``result_types``
# parameter.  Leaving ``result_types`` empty (the default) extracts
# every field the solver wrote.

INPUT_DIR = "/data/ansys_thermal"

source = AnsysRSTSource(
    input_dir=INPUT_DIR,
    result_types=["temperature", "heat_flux"],
)

# %%
# Build the Pipeline
# ------------------
#
# Chain several filters in order:
#
# 1. **MeshInfoFilter** — Logs mesh metadata (node and element counts,
#    field names and shapes) and writes a JSON-lines summary.
#
# 2. **StatsFilter** — Computes per-field statistics (mean, standard
#    deviation, min, max) and writes them to a Parquet file.
#
# 3. **PrecisionFilter** — Converts floating-point fields from float64
#    to float32 to halve memory and storage requirements.
#
# Finally a **MeshSink** writes each processed mesh as a TensorDict
# memory-mapped directory.

OUTPUT_DIR = "/data/ansys_thermal_processed"

pipeline = (
    source.filter(MeshInfoFilter(output=f"{OUTPUT_DIR}/mesh_info.jsonl"))
    .filter(StatsFilter(output=f"{OUTPUT_DIR}/stats.parquet"))
    .filter(PrecisionFilter(target_dtype="float32"))
    .write(MeshSink(output_dir=OUTPUT_DIR))
)

# %%
# Run the Pipeline
# ----------------
#
# Process the first 3 simulation files in parallel using a process pool
# with 4 workers.  Thermal simulations are typically smaller than crash
# or CFD datasets, so more workers can be used.

results = run_pipeline(
    pipeline,
    n_jobs=4,
    backend="process_pool",
    indices=range(min(3, len(source))),
    progress=True,
)

# %%
# Inspect Results
# ---------------
#
# ``run_pipeline`` returns a list of output paths per index.  Each entry
# is the list of files written by the sink for that simulation.

for idx, paths in enumerate(results):
    print(f"Simulation {idx}: {len(paths)} output(s)")
    for p in paths:
        print(f"  {p}")

# %%
# Summary
# -------
#
# This example showed how to:
#
# - Read Ansys ``.rst`` thermal simulation results with
#   :class:`~physicsnemo_curator.mesh.sources.ansys_rst.AnsysRSTSource`.
# - Auto-discover or explicitly select result fields (temperature,
#   heat flux, displacement, stress, etc.).
# - Log mesh metadata, compute statistics, and convert precision in a
#   composable filter chain.
# - Write processed meshes in parallel with ``run_pipeline``.
#
# The same source works for structural analyses — just point it at
# ``.rst`` files from a structural solver and the source will
# auto-discover displacement, stress, and strain fields.
