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
Creating a Pipeline
===================

This example shows how to build a data curation pipeline using the
**Source → Filter → Sink** pattern.  We read meshes from the
`Navier-Stokes Cylinder
<https://huggingface.co/datasets/SISSAmathLab/navier-stokes-cylinder>`_
dataset, apply a statistics filter, and write the outputs to disk.

A pipeline is *lazy* — nothing is executed until you index into it
with ``pipeline[i]``.  This makes it easy to build, inspect, and
compose pipelines before running them.

.. note::

   Install the mesh extras before running::

       pip install physicsnemo-curator[mesh]
"""

# %%
# Imports
# -------
#
# Every pipeline has three ingredients: a **Source** (data reader), zero
# or more **Filters** (transforms or analytics), and a **Sink** (writer).

from physicsnemo.curator.mesh.filters.mean import MeanFilter
from physicsnemo.curator.mesh.filters.precision import PrecisionFilter
from physicsnemo.curator.mesh.sinks.mesh_writer import MeshSink
from physicsnemo.curator.mesh.sources.ns_cylinder import NavierStokesCylinderSource

# %%
# Step 1: Create a Source
# -----------------------
#
# A :class:`~physicsnemo.curator.core.base.Source` is an indexed
# collection of data items.  Here we use
# :class:`~physicsnemo.curator.mesh.sources.ns_cylinder.NavierStokesCylinderSource`
# which provides 500 Navier-Stokes flow simulations as
# :class:`~physicsnemo.mesh.Mesh` objects.

source = NavierStokesCylinderSource()

print(f"Source: {source.name}")
print(f"Items available: {len(source)}")

# %%
# Step 2: Add Filters
# --------------------
#
# Filters transform or inspect items as they flow through the pipeline.
# The fluent ``.filter()`` method chains multiple filters together:
#
# - :class:`~physicsnemo.curator.mesh.filters.mean.MeanFilter`
#   computes spatial means and writes a Parquet summary.
# - :class:`~physicsnemo.curator.mesh.filters.precision.PrecisionFilter`
#   converts floating-point fields to ``float32``.

pipeline = source.filter(MeanFilter(output="outputs/getting_started/stats.parquet")).filter(
    PrecisionFilter(target_dtype="float32")
)

print(f"Filters: {[f.name for f in pipeline.filters]}")
print(f"Sink: {pipeline.sink}")  # None — no sink yet

# %%
# Step 3: Attach a Sink
# ---------------------
#
# A :class:`~physicsnemo.curator.core.base.Sink` persists items to
# storage.  The ``.write()`` method attaches a sink and returns a
# complete pipeline.

pipeline = pipeline.write(MeshSink(output_dir="outputs/getting_started/meshes/"))

assert pipeline.sink is not None
print(f"Sink: {pipeline.sink.name}")
print(f"Pipeline length: {len(pipeline)}")

# %%
# Step 4: Execute One Index
# -------------------------
#
# Indexing into a pipeline runs the full **Source → Filters → Sink**
# chain for a single source item and returns the file paths written
# by the sink.

paths = pipeline[0]
print(f"Index 0 wrote: {paths}")

# %%
# The fluent API also supports building in one expression:
#
# .. code-block:: python
#
#     pipeline = (
#         NavierStokesCylinderSource()
#         .filter(MeanFilter(output="stats.parquet"))
#         .filter(PrecisionFilter(target_dtype="float32"))
#         .write(MeshSink(output_dir="meshes/"))
#     )
#
# Each call returns a new immutable
# :class:`~physicsnemo.curator.core.base.Pipeline` — the original
# source, filters, and sink are never modified.
