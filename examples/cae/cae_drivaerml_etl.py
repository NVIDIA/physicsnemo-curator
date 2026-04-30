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
DrivAerML End-to-End ETL Pipeline
==================================

This example demonstrates a complete **Source → Filter → Sink** pipeline
that reads `DrivAerML <https://huggingface.co/datasets/neashton/drivaerml>`_
meshes from HuggingFace Hub in **multi** mode, producing:

- A :class:`~physicsnemo.mesh.domain_mesh.DomainMesh` per run — combining
  the volumetric interior (point-cloud from VTU cell centroids), boundary
  surface (triangulated VTP with flow fields), and global reference data
  (``U_inf``, ``rho_inf``).
- STL geometry meshes (with and without solid merging) for reference.

The pipeline showcases:

- **MeshInfoFilter** — logs structured metadata (point/cell counts, field
  shapes) to JSON-lines for post-hoc analysis.
- **StatsFilter** — computes per-field statistics (mean, std, skewness,
  kurtosis) using numerically stable Welford accumulators that merge across
  parallel workers.
- **PrecisionFilter** — casts float64 fields to float32, halving storage
  and matching training precision.
- **MeshSink with ``{mesh_name}`` naming** — leverages the source's
  :meth:`~DrivAerMLSource.mesh_name` method to produce output names that
  match the canonical DrivAerML structure (e.g.
  ``domain_1.pdmsh``, ``drivaer_1.stl.pmsh``).

A train/val split is demonstrated so the output directory can be pointed at
directly by downstream training configs.
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
from physicsnemo_curator.domains.mesh.filters.stats import StatsFilter
from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink
from physicsnemo_curator.domains.mesh.sources.drivaerml import DrivAerMLSource
from physicsnemo_curator.run import gather_pipeline, run_pipeline

# %%
# Configure the Source
# --------------------
#
# :class:`~physicsnemo_curator.domains.mesh.sources.drivaerml.DrivAerMLSource`
# connects to the HuggingFace Hub dataset and discovers available runs.
# We select ``mesh_type="multi"`` to read all mesh representations:
#
# - **domain** — DomainMesh combining volume interior + boundary surface
# - **stl** — vehicle geometry from the STL file
# - **single_solid** — same STL merged into one contiguous solid

source = DrivAerMLSource(
    mesh_type="multi",
    manifold_dim="auto",
    point_source="vertices",
)

n_runs = len(source)
print(f"Total runs available: {n_runs}")

# %%
# Define a Train / Val Split
# ---------------------------
#
# Reserve the last 20% of runs for validation.  The ``{mesh_name}`` naming
# template ensures output file names encode the canonical DrivAerML mesh
# name (e.g. ``domain_1``, ``drivaer_1.stl``) regardless of split.

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
# 1. :class:`~physicsnemo_curator.domains.mesh.filters.mesh_info.MeshInfoFilter`
#    logs structured metadata for each mesh (point/cell counts, field
#    names and shapes) to a JSON-lines file.
# 2. :class:`~physicsnemo_curator.domains.mesh.filters.stats.StatsFilter`
#    computes per-field statistics (mean, std, skewness, kurtosis) using
#    numerically stable Welford accumulators that merge across workers.
# 3. :class:`~physicsnemo_curator.domains.mesh.filters.precision.PrecisionFilter`
#    casts all float64 fields to float32 for training consistency.
# 4. :class:`~physicsnemo_curator.domains.mesh.sinks.mesh_writer.MeshSink`
#    writes each mesh in PhysicsNeMo's native format — ``.pdmsh`` for
#    :class:`DomainMesh` and ``.pmsh`` for plain :class:`Mesh`.
#
# The ``naming_template`` uses ``{mesh_name}`` which is resolved via the
# source's :meth:`~DrivAerMLSource.mesh_name` method — producing output
# names like ``domain_1.pdmsh``, ``drivaer_1.stl.pmsh``, etc.

train_pipeline = (
    source.filter(MeshInfoFilter(output="outputs/drivaerml/train/mesh_info.jsonl"))
    .filter(StatsFilter(output="outputs/drivaerml/train/stats.parquet"))
    .filter(PrecisionFilter(target_dtype="float32"))
    .write(
        MeshSink(
            output_dir="outputs/drivaerml/train/",
            naming_template="{mesh_name}",
        )
    )
)

# %%
# Build the Validation Pipeline
# ------------------------------
#
# The validation pipeline uses the same source and filter chain, with a
# separate output directory.  The ``{mesh_name}`` template still resolves
# correctly because :meth:`~DrivAerMLSource.mesh_name` is index-based.

val_pipeline = (
    source.filter(MeshInfoFilter(output="outputs/drivaerml/val/mesh_info.jsonl"))
    .filter(StatsFilter(output="outputs/drivaerml/val/stats.parquet"))
    .filter(PrecisionFilter(target_dtype="float32"))
    .write(
        MeshSink(
            output_dir="outputs/drivaerml/val/",
            naming_template="{mesh_name}",
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
# stateful filters (StatsFilter and MeshInfoFilter).
# :func:`~physicsnemo_curator.run.gather_pipeline` discovers those
# shards, merges them into single output files, and cleans up the
# temporary shard files.
#
# After gathering, the statistics Parquet contains the exact global
# mean, std, skewness, and kurtosis computed across all processed meshes.

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
#     │   ├── mesh_info.jsonl                  # Mesh metadata (JSON-lines)
#     │   ├── stats.parquet                    # Per-field statistics (merged)
#     │   ├── domain_1.pdmsh/                  # DomainMesh: interior + surface
#     │   ├── drivaer_1.stl.pmsh/              # STL geometry
#     │   ├── drivaer_1_single_solid.stl.pmsh/ # Merged STL
#     │   ├── domain_2.pdmsh/                  # Run 2
#     │   └── ...
#     └── val/
#         ├── mesh_info.jsonl
#         ├── stats.parquet
#         ├── domain_388.pdmsh/                 # First val run
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
#         pattern: "**/*.pdmsh"
