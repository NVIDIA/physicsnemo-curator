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
meshes in **multi** mode, producing:

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

Prerequisites
-------------

Download sample data from HuggingFace using the ``huggingface-cli``:

.. code-block:: bash

    pip install huggingface_hub[cli]

    # Download run_1 and run_2 (boundary, volume, STL, and metadata)
    cd examples/cae
    huggingface-cli download neashton/drivaerml \\
        --repo-type dataset \\
        --include "run_1/*" "run_2/*" \\
        --local-dir data/drivaerml

This creates ``examples/cae/data/drivaerml/run_1/`` and
``examples/cae/data/drivaerml/run_2/`` with the VTP, VTU, STL, and CSV
files needed by the pipeline.
"""

# %%
# Imports
# -------
#
# Import the core pipeline components: a **Source** to read meshes,
# **Filters** for metadata logging, statistics, and precision conversion,
# a **Sink** to write outputs, and
# :func:`~physicsnemo_curator.run.run_pipeline` for parallel execution.

from pathlib import Path

from physicsnemo_curator.domains.mesh.filters.mesh_info import MeshInfoFilter
from physicsnemo_curator.domains.mesh.filters.precision import PrecisionFilter
from physicsnemo_curator.domains.mesh.filters.stats import MeshStatsFilter
from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink
from physicsnemo_curator.domains.mesh.sources.drivaerml import DrivAerMLSource
from physicsnemo_curator.run import gather_pipeline, run_pipeline

# %%
# Resolve Data Directory
# -----------------------
#
# Resolve the path to the downloaded data relative to this example file.
# This ensures the example works regardless of the working directory.

_HERE = Path(__file__).resolve().parent
_DATA_DIR = _HERE / "data" / "drivaerml"
_OUTPUT_DIR = _HERE / "outputs" / "drivaerml"

# %%
# Configure the Source
# --------------------
#
# :class:`~physicsnemo_curator.domains.mesh.sources.drivaerml.DrivAerMLSource`
# reads from the locally downloaded dataset.  We pass a ``file://`` URL
# pointing to the data directory resolved relative to this example file.
#
# We select ``mesh_type="multi"`` to read all mesh representations:
#
# - **domain** — DomainMesh combining volume interior + boundary surface
# - **stl** — vehicle geometry from the STL file
# - **single_solid** — same STL merged into one contiguous solid

source = DrivAerMLSource(
    url=f"file://{_DATA_DIR}",
    mesh_type="multi",
    manifold_dim="auto",
    point_source="vertices",
)

n_runs = len(source)
print(f"Total runs available: {n_runs}")

# %%
# Build the Pipeline
# -------------------
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

pipeline = (
    source.filter(MeshInfoFilter(output=str(_OUTPUT_DIR / "mesh_info.jsonl")))
    .filter(MeshStatsFilter(output=str(_OUTPUT_DIR / "stats.parquet")))
    .filter(PrecisionFilter(target_dtype="float32"))
    .write(
        MeshSink(
            output_dir=str(_OUTPUT_DIR),
            naming_template="{mesh_name}",
        )
    )
)

# %%
# Run in Parallel
# ---------------
#
# :func:`~physicsnemo_curator.run.run_pipeline` dispatches work to a
# ``process_pool`` backend with 4 workers.  Each worker gets an
# independent copy of the pipeline, so meshes are read, filtered, and
# written concurrently.

results = run_pipeline(
    pipeline,
    n_jobs=4,
    backend="process_pool",
    progress=True,
)

# %%
# Inspect Results
# ---------------
#
# ``results`` is a list of lists — one entry per processed index,
# each containing the file paths written by the sink.

print(f"Processed {len(results)} runs")
for i, paths in enumerate(results[:3]):
    print(f"  Run {i}: {paths}")

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

merged = gather_pipeline(pipeline)
for path in merged:
    print(f"Merged statistics: {path}")

# %%
# Output Structure
# ----------------
#
# The output directory structure is directly compatible with
# ``MeshReader`` from `PhysicsNeMo PR #1512
# <https://github.com/NVIDIA/physicsnemo/pull/1512>`_:
#
# .. code-block:: text
#
#     outputs/drivaerml/
#     ├── mesh_info.jsonl                  # Mesh metadata (JSON-lines)
#     ├── stats.parquet                    # Per-field statistics (merged)
#     ├── domain_1.pdmsh/                  # DomainMesh: interior + surface
#     ├── drivaer_1.stl.pmsh/              # STL geometry
#     ├── drivaer_1_single_solid.stl.pmsh/ # Merged STL
#     ├── domain_2.pdmsh/                  # Run 2
#     ├── drivaer_2.stl.pmsh/
#     ├── drivaer_2_single_solid.stl.pmsh/
#     └── ...
#
# Point a training config at the output:
#
# .. code-block:: yaml
#
#     datadir: outputs/drivaerml/
#
#     pipeline:
#       reader:
#         _target_: ${dp:MeshReader}
#         path: ${datadir}
#         pattern: "**/*.pdmsh"
