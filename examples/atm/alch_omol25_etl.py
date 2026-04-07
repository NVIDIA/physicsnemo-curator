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
OMol25 Atomic Data ETL Pipeline
================================

This example demonstrates a complete **Source → Filter → Sink** pipeline
for curating atomic/molecular data from the `Open Molecules 2025 (OMol25)
<https://huggingface.co/facebook/OMol25>`_ dataset.

OMol25 contains over 100 million DFT calculations at the
ωB97M-V/def2-TZVPD level of theory, covering 83 elements and ~83 million
unique molecular systems.  The dataset is stored as ASE LMDB
(``.aselmdb``) files — each file holds thousands of atomic structures
with positions, forces, energies, and other computed properties.

The pipeline reads the raw LMDB files, computes per-field statistics
(mean, std, min, max, skewness, kurtosis) using numerically stable
Welford accumulators, and writes the processed structures to a Zarr store
in the `nvalchemi <https://nvidia.github.io/nvalchemi-toolkit/>`_ format — a
common preprocessing step for training machine-learned interatomic
potentials (MLIPs).

We process only the first 2 LMDB files to keep the example fast.

References
----------
- OMol25 dataset: https://huggingface.co/facebook/OMol25
- OMol25 paper: Levine et al., arXiv:2505.08762 (2025)
- nvalchemi toolkit: https://nvidia.github.io/nvalchemi-toolkit/
- ASE database backends: https://github.com/NVIDIA/ase-db-backends
"""

# %%
# Imports
# -------
#
# Import the core pipeline components: a **Source** to read ASE LMDB
# files, a **Filter** to compute field statistics, a **Sink** to write
# AtomicData to a Zarr store, and
# :func:`~physicsnemo_curator.run.run_pipeline` for parallel execution.

import pyarrow.parquet as pq

from physicsnemo_curator.atm.filters.stats import AtomicStatsFilter
from physicsnemo_curator.atm.sinks.zarr_writer import AtomicDataZarrSink
from physicsnemo_curator.atm.sources.aselmdb import ASELMDBSource
from physicsnemo_curator.run import gather_pipeline, run_pipeline

# %%
# Configure the Source
# --------------------
#
# :class:`~physicsnemo_curator.atm.sources.aselmdb.ASELMDBSource`
# discovers all ``.aselmdb`` files in a directory, sorted
# lexicographically.  Each file corresponds to one source index and
# may contain thousands of atomic structures.
#
# The optional *metadata_path* parameter points to a NumPy ``.npz``
# file containing ``natoms`` and ``data_ids`` arrays, which the source
# loads eagerly for downstream reference.

source = ASELMDBSource(
    data_dir="./val/",
    metadata_path="./val/metadata.npz",
)

print(f"LMDB files discovered: {len(source)}")
if source.metadata is not None:
    natoms = source.metadata.get("natoms")
    if natoms is not None:
        print(f"Total structures in metadata: {len(natoms):,}")

# %%
# Build the Pipeline
# ------------------
#
# The fluent API chains **Source → Filter → Sink** into a lazy
# :class:`~physicsnemo_curator.core.base.Pipeline`.  Nothing is
# executed until we explicitly process indices.
#
# - :class:`~physicsnemo_curator.atm.filters.stats.AtomicStatsFilter`
#   examines each :class:`~nvalchemi.data.AtomicData` and accumulates
#   per-field, per-component statistics using Welford's online algorithm.
#   Fields are grouped by level (node, edge, system) and include
#   ``positions``, ``forces``, ``energies``, ``atomic_numbers``, and
#   any extra data attached to the structures.  The filter is
#   **pass-through** — each item is yielded unchanged.
# - :class:`~physicsnemo_curator.atm.sinks.zarr_writer.AtomicDataZarrSink`
#   collects items into batches (default 1000) and writes them to a
#   structured Zarr store using ``AtomicDataZarrWriter``.  Multiple
#   pipeline indices append to the **same** store.

pipeline = source.filter(AtomicStatsFilter(output="outputs/omol25/stats.parquet")).write(
    AtomicDataZarrSink(output_path="outputs/omol25/dataset.zarr", batch_size=500)
)

# %%
# Run in Parallel
# ---------------
#
# :func:`~physicsnemo_curator.run.run_pipeline` dispatches work to a
# ``process_pool`` backend.  We pass ``indices=range(2)`` to process
# only the first 2 LMDB files (each containing many structures).
#
# Each worker gets an independent copy of the pipeline, so LMDB files
# are read, statistics are accumulated, and structures are written
# concurrently.

results = run_pipeline(
    pipeline,
    n_jobs=2,
    backend="process_pool",
    indices=range(2),
    progress=True,
)

# %%
# Inspect Results
# ---------------
#
# ``results`` is a list of lists — one entry per processed index,
# each containing the file paths written by the sink.

print(f"\nProcessed {len(results)} LMDB files")
for i, paths in enumerate(results):
    print(f"  File {i}: {paths}")

# %%
# Gather Statistics
# -----------------
#
# When running in parallel, each worker writes per-index shard files for
# the stateful statistics filter.
# :func:`~physicsnemo_curator.run.gather_pipeline` discovers those shards,
# merges them using the parallel Welford algorithm (Chan et al., 1979),
# and writes a single consolidated Parquet file.

merged = gather_pipeline(pipeline)
for path in merged:
    print(f"Merged statistics: {path}")

# %%
# Explore the Statistics
# ----------------------
#
# The merged Parquet file contains one row per (field, component) pair
# with columns for mean, std, variance, min, max, median, skewness,
# kurtosis, and the full Welford accumulator state.

table = pq.read_table("outputs/omol25/stats.parquet")
print(f"\nStatistics table: {table.num_rows} rows, {table.num_columns} columns")
print(f"Fields tracked: {table.column('field_key').to_pylist()[:10]}...")
print(f"Levels: {set(table.column('level').to_pylist())}")

# %%
# The ``outputs/omol25/`` directory now contains:
#
# .. code-block:: text
#
#     outputs/omol25/
#     ├── stats.parquet              # Per-field statistics (merged)
#     └── dataset.zarr/              # AtomicData Zarr store
#         ├── meta/                  # Pointer arrays (atoms_ptr, edges_ptr)
#         ├── core/                  # Core fields (positions, forces, ...)
#         ├── custom/                # User-defined fields
#         └── .zattrs                # Root metadata (num_samples, fields)
#
# The Zarr store follows the nvalchemi
# :class:`~nvalchemi.data.datapipes.backends.zarr.AtomicDataZarrWriter`
# layout with CSR-style pointer arrays for variable-size systems,
# enabling efficient random access for training loops.

# %%
# Adding Checkpointing
# ---------------------
#
# For large-scale runs (all 80 LMDB files), wrap the pipeline with
# :class:`~physicsnemo_curator.core.checkpoint.CheckpointedPipeline`
# to enable restart from where you left off.  Create a checkpoint with
# ``CheckpointedPipeline(pipeline, db_path="outputs/omol25/etl.db")``,
# then pass it to ``run_pipeline`` as usual.  On restart, completed
# LMDB files are skipped automatically.
# See :doc:`/user-guide/checkpointing` for the full guide.
