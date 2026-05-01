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
VTK Backend Profiling: PyVista vs Rust
=======================================

This example demonstrates the built-in
:class:`~physicsnemo_curator.core.profiling.ProfiledPipeline` utility and
compares the two VTK reading backends available in
:class:`~physicsnemo_curator.domains.mesh.sources.vtk.VTKSource`:

- **PyVista** (default): full-featured Python reader supporting all VTK
  formats, manifold dimensions, and point-source modes.
- **Rust**: native reader built with PyO3 for faster I/O on ASCII
  VTU/VTP files.

We download DrivAerML boundary files, then run the same pipeline twice
— once per backend — and print the per-stage timing breakdown so you
can see exactly where time is spent.

.. note::

   The Rust backend requires the native extension to be built
   (``maturin develop``).  It currently supports ASCII VTU/VTP files
   only and does not apply ``manifold_dim`` or ``point_source``
   conversion.
"""

# %%
# Imports
# -------
#
# We use :class:`~physicsnemo_curator.domains.mesh.sources.drivaerml.DrivAerMLSource`
# to download DrivAerML boundary files, then
# :class:`~physicsnemo_curator.domains.mesh.sources.vtk.VTKSource` (which exposes the
# ``backend`` parameter) for local VTK reading, a filter, a sink, and the
# :class:`~physicsnemo_curator.core.profiling.ProfiledPipeline` wrapper.

from physicsnemo_curator.core.profiling import ProfiledPipeline

from physicsnemo_curator.domains.mesh.filters.precision import PrecisionFilter
from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink
from physicsnemo_curator.domains.mesh.sources.vtk import VTKSource
from physicsnemo_curator.run import run_pipeline

# %%
# Download DrivAerML Files
# -------------------------
#
# We use ``fsspec`` to download a small subset of DrivAerML boundary VTP files
# from HuggingFace Hub to a local cache directory.  Subsequent runs read from cache.

N_RUNS = 3
N_JOBS = 1  # Sequential for fair comparison (no scheduling noise)

DRIVAERML_URL = "hf://datasets/neashton/drivaerml"
CACHE_DIR = "output/profiling/drivaerml_cache"

import pathlib

import fsspec

fs, root_path = fsspec.core.url_to_fs(DRIVAERML_URL)
glob_expr = f"{root_path}/**/boundary*.vtp"
all_files = fs.glob(glob_expr)
files = sorted(f for f in all_files if f.endswith(".vtp") and not fs.isdir(f))

# Force download of the files we need
for remote_path in files[:N_RUNS]:
    local_path = pathlib.Path(CACHE_DIR) / remote_path.lstrip("/")
    if not local_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        fs.get(remote_path, str(local_path))

print(f"VTP files cached: {len(files)}")

# %%
# PyVista Backend
# ---------------
#
# The default backend reads VTK files through
# `PyVista <https://docs.pyvista.org/>`_ and converts to
# :class:`~physicsnemo.mesh.Mesh` via
# :func:`~physicsnemo.mesh.io.from_pyvista`.

pyvista_source = VTKSource(CACHE_DIR, backend="pyvista")

print(f"VTK files discovered locally: {len(pyvista_source)}")

pyvista_pipeline = pyvista_source.filter(PrecisionFilter(target_dtype="float32")).write(
    MeshSink(output_dir="output/profiling/pyvista_meshes/")
)

# %%
# Wrap with ProfiledPipeline
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# :class:`~physicsnemo_curator.core.profiling.ProfiledPipeline` is a
# transparent proxy — it passes through to the real pipeline while
# recording wall-clock time, memory usage, and optional GPU metrics for
# every stage.

profiled_pyvista = ProfiledPipeline(pyvista_pipeline)

results_pyvista = run_pipeline(
    profiled_pyvista,
    n_jobs=N_JOBS,
    backend="sequential",
    indices=range(N_RUNS),
)

print("=== PyVista Backend ===")
profiled_pyvista.metrics.to_console()

# %%
# Rust Backend
# ------------
#
# The Rust backend parses VTK XML directly using ``quick-xml`` and
# returns raw NumPy arrays.  It skips the PyVista/VTK library stack
# entirely, trading feature completeness for speed.
#
# We point to the same cached files, so the comparison measures only
# parse time — not download time.
#
# .. note::
#
#    The Rust backend only supports ASCII VTU/VTP files and does not
#    apply ``manifold_dim`` or ``point_source`` conversion.

rust_source = VTKSource(CACHE_DIR, backend="rust")

rust_pipeline = rust_source.filter(PrecisionFilter(target_dtype="float32")).write(
    MeshSink(output_dir="output/profiling/rust_meshes/")
)

profiled_rust = ProfiledPipeline(rust_pipeline)

results_rust = run_pipeline(
    profiled_rust,
    n_jobs=N_JOBS,
    backend="sequential",
    indices=range(N_RUNS),
)

print("=== Rust Backend ===")
profiled_rust.metrics.to_console()

# %%
# Compare Results
# ---------------
#
# Print a side-by-side summary of the two backends.  The table shows
# total wall-clock time and mean per-index time for each backend.

pyvista_metrics = profiled_pyvista.metrics
rust_metrics = profiled_rust.metrics

pyvista_total_ms = pyvista_metrics.total_wall_time_ns / 1e6
rust_total_ms = rust_metrics.total_wall_time_ns / 1e6

print("\n=== Comparison ===\n")
print(f"  {'Backend':<12s} {'Total (ms)':>12s} {'Mean/index (ms)':>17s}")
print("  " + "-" * 43)
print(f"  {'PyVista':<12s} {pyvista_total_ms:>12,.2f} {pyvista_metrics.mean_index_time_ns / 1e6:>17,.2f}")
print(f"  {'Rust':<12s} {rust_total_ms:>12,.2f} {rust_metrics.mean_index_time_ns / 1e6:>17,.2f}")

if rust_total_ms > 0:
    speedup = pyvista_total_ms / rust_total_ms
    print(f"\n  Rust speedup: {speedup:.1f}x")

# %%
# Export Metrics
# --------------
#
# :class:`~physicsnemo_curator.core.profiling.PipelineMetrics` supports
# three export formats for further analysis:
#
# - **JSON**: full per-index, per-stage breakdown.
# - **CSV**: tabular format for spreadsheets or plotting.
# - **Console**: human-readable summary (shown above).

pyvista_metrics.to_json("output/profiling/pyvista_profile.json")
pyvista_metrics.to_csv("output/profiling/pyvista_profile.csv")
rust_metrics.to_json("output/profiling/rust_profile.json")
rust_metrics.to_csv("output/profiling/rust_profile.csv")

print("\nMetrics exported to output/profiling/")

# %%
# Cleanup
# -------
#
# Remove the temporary metric directories created by each
# ``ProfiledPipeline`` instance.

profiled_pyvista.cleanup()
profiled_rust.cleanup()

# %%
# .. note::
#
#    **Typical results** on an x86-64 workstation show the Rust backend
#    reads VTK files 2–5x faster than PyVista for ASCII VTU/VTP files.
#    The speedup comes from direct XML parsing (``quick-xml``) and
#    zero-copy NumPy conversion, bypassing the VTK C++ library and
#    PyVista wrapper layers.
#
#    Actual results depend on file size, disk speed, and CPU.  Use this
#    example as a starting point for profiling your own pipelines.
