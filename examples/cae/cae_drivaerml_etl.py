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

The output is written in PhysicsNeMo's native ``.pmsh`` format with file
names that match the ``MeshReader`` glob pattern used by downstream
training recipes (e.g. ``boundary_{index}.vtp.pmsh``).  A train/val
split is demonstrated so the output directory can be pointed at directly
by the ``drivaer_ml_surface.yaml`` dataset config.
"""

# %%
# Imports
# -------
#
# Import the core pipeline components: a **Source** to read meshes, a
# **Filter** to compute statistics, a **Sink** to write outputs, and
# :func:`~physicsnemo_curator.run.run_pipeline` for parallel execution.

from physicsnemo_curator.domains.mesh.filters.mean import MeanFilter
from physicsnemo_curator.domains.mesh.filters.precision import PrecisionFilter
from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink
from physicsnemo_curator.domains.mesh.sources.drivaerml import DrivAerMLSource
from physicsnemo_curator.run import gather_pipeline, run_pipeline

# %%
# Configure the Source
# --------------------
#
# :class:`~physicsnemo_curator.domains.mesh.sources.drivaerml.DrivAerMLSource`
# connects to the HuggingFace Hub dataset and discovers available runs.
# We select ``mesh_type="boundary"`` to read the surface VTP files
# which contain the flow fields on the vehicle boundary.

source = DrivAerMLSource(
    mesh_type="boundary",
    manifold_dim="auto",
    point_source="vertices",
)

n_runs = len(source)
print(f"Total runs available: {n_runs}")

# %%
# Define a Train / Val Split
# ---------------------------
#
# Reserve the last 20% of runs for validation.  The indices map directly
# to the DrivAerML run list, so ``boundary_{index}.vtp.pmsh`` names stay
# consistent across the split.

val_frac = 0.2
n_val = max(1, int(n_runs * val_frac))
n_train = n_runs - n_val
train_indices = range(n_train)
val_indices = range(n_train, n_runs)

print(f"Train: {len(train_indices)} runs, Val: {len(val_indices)} runs")

# %%
# Build the Training Pipeline
# ----------------------------
#
# The fluent API chains **Source → Filter → Sink** into a lazy
# :class:`~physicsnemo_curator.core.base.Pipeline`.  Nothing is
# executed until we explicitly process indices.
#
# - :class:`~physicsnemo_curator.domains.mesh.filters.mean.MeanFilter` computes
#   per-field spatial means and accumulates them into a Parquet summary.
# - :class:`~physicsnemo_curator.domains.mesh.filters.precision.PrecisionFilter`
#   converts to float32 for consistency with training.
# - :class:`~physicsnemo_curator.domains.mesh.sinks.mesh_writer.MeshSink` writes
#   each mesh in PhysicsNeMo's native ``.pmsh`` format.
#
# The ``naming_template`` produces output names like
# ``boundary_0.vtp.pmsh``, ``boundary_1.vtp.pmsh``, etc. — matching the
# glob pattern ``**/boundary*.vtp.pmsh`` expected by
# ``MeshReader`` in the ``drivaer_ml_surface.yaml`` dataset config.

train_pipeline = (
    source.filter(MeanFilter(output="outputs/drivaerml/train/mean_stats.parquet"))
    .filter(PrecisionFilter(target_dtype="float32"))
    .write(
        MeshSink(
            output_dir="outputs/drivaerml/train/",
            naming_template="boundary_{index}.vtp.pmsh",
        )
    )
)

# %%
# Build the Validation Pipeline
# ------------------------------
#
# The validation pipeline uses the same source and filters, with a
# separate output directory.

val_pipeline = (
    source.filter(MeanFilter(output="outputs/drivaerml/val/mean_stats.parquet"))
    .filter(PrecisionFilter(target_dtype="float32"))
    .write(
        MeshSink(
            output_dir="outputs/drivaerml/val/",
            naming_template="boundary_{index}.vtp.pmsh",
        )
    )
)

# %%
# Run in Parallel
# ---------------
#
# :func:`~physicsnemo_curator.run.run_pipeline` dispatches work to a
# ``process_pool`` backend with 4 workers.  We process the training
# split first, then the validation split.
#
# Each worker gets an independent copy of the pipeline, so meshes are
# read, filtered, and written concurrently.

train_results = run_pipeline(
    train_pipeline,
    n_jobs=4,
    backend="process_pool",
    indices=train_indices,
    progress=True,
)

val_results = run_pipeline(
    val_pipeline,
    n_jobs=4,
    backend="process_pool",
    indices=val_indices,
    progress=True,
)

# %%
# Inspect Results
# ---------------
#
# ``results`` is a list of lists — one entry per processed index,
# each containing the file paths written by the sink.

print(f"Processed {len(train_results)} training runs")
print(f"Processed {len(val_results)} validation runs")
for i, paths in enumerate(train_results[:3]):
    print(f"  Train run {i}: {paths}")

# %%
# Gather Statistics
# -----------------
#
# When running in parallel, each worker writes per-index shard files for
# stateful filters.  :func:`~physicsnemo_curator.run.gather_pipeline`
# discovers those shards, merges them into a single output file, and
# cleans up the temporary shard files.

for pipe in (train_pipeline, val_pipeline):
    merged = gather_pipeline(pipe)
    for path in merged:
        print(f"Merged statistics: {path}")

# %%
# Using the Output with ``MeshReader``
# -------------------------------------
#
# The output directory structure is directly compatible with
# ``MeshReader`` from `PhysicsNeMo PR #1512
# <https://github.com/NVIDIA/physicsnemo/pull/1512>`_:
#
# .. code-block:: text
#
#     outputs/drivaerml/
#     ├── train/
#     │   ├── mean_stats.parquet              # Per-field spatial means
#     │   ├── boundary_0.vtp.pmsh/            # Run 0 in .pmsh format
#     │   ├── boundary_1.vtp.pmsh/            # Run 1
#     │   └── ...
#     └── val/
#         ├── mean_stats.parquet
#         ├── boundary_387.vtp.pmsh/           # First val run
#         └── ...
#
# Point the ``drivaer_ml_surface.yaml`` config at the output:
#
# .. code-block:: yaml
#
#     train_datadir: outputs/drivaerml/train/
#     val_datadir: outputs/drivaerml/val/
#
#     pipeline:
#       reader:
#         _target_: ${dp:MeshReader}
#         path: ${train_datadir}
#         pattern: "**/boundary*.vtp.pmsh"
