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

"""Crash Simulation ETL."""

# Import the pipeline building blocks: a Source for LS-DYNA d3plot data,
# the WallNodeFilter for removing non-deforming boundary nodes,
# informational and precision filters, a Sink for writing outputs, and
# run_pipeline for parallel execution.

from physicsnemo_curator.domains.mesh.filters.mesh_info import MeshInfoFilter
from physicsnemo_curator.domains.mesh.filters.precision import PrecisionFilter
from physicsnemo_curator.domains.mesh.filters.wall_node import WallNodeFilter
from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink
from physicsnemo_curator.domains.mesh.sources.d3plot import D3PlotSource
from physicsnemo_curator.run import run_pipeline

# Configure the Source
#
# D3PlotSource scans input_dir for subdirectories containing a d3plot
# file. Each subdirectory corresponds to one crash simulation run.
#
# Set read_stress=True to include von Mises stress and effective plastic
# strain as cell data fields. Set read_k_file=True to parse companion
# .k keyword files for per-node shell thickness.

INPUT_DIR = "input/crash_simulations"

source = D3PlotSource(
    input_dir=INPUT_DIR,
    read_stress=True,
    read_k_file=True,
)

# Build the Pipeline
#
# Chain several filters in order:
#
# 1. WallNodeFilter — Removes non-deforming "wall" nodes whose maximum
#    displacement variation across all timesteps falls below a threshold.
#    This typically removes 30-60% of nodes, significantly reducing
#    dataset size while preserving the structural response.
#
# 2. MeshInfoFilter — Logs mesh metadata (node counts, cell counts,
#    field names) and writes a JSON-lines summary.
#
# 3. PrecisionFilter — Converts floating-point fields from float64
#    to float32 to halve memory and storage requirements.
#
# Finally a MeshSink writes each processed mesh as a TensorDict
# memory-mapped directory.

OUTPUT_DIR = "output/crash_simulations"

pipeline = (
    source.filter(WallNodeFilter(threshold=1.0))
    .filter(MeshInfoFilter(output=f"{OUTPUT_DIR}/mesh_info.jsonl"))
    .filter(PrecisionFilter(target_dtype="float32"))
    .write(MeshSink(output_dir=OUTPUT_DIR))
)

# Run the Pipeline
#
# Process the first 3 runs in parallel using a process pool with 2
# workers. Crash simulations can be memory-intensive, so a modest
# worker count helps avoid out-of-memory conditions.

results = run_pipeline(
    pipeline,
    n_jobs=2,
    backend="process_pool",
    indices=range(min(3, len(source))),
    use_tui=True,
)

# Inspect Results
#
# run_pipeline returns a list of output paths per index. Each entry
# is the list of files written by the sink for that run.

for idx, paths in enumerate(results):
    print(f"Run {idx}: {len(paths)} output(s)")
    for p in paths:
        print(f"  {p}")

# Summary
#
# This example showed how to:
#
# - Read LS-DYNA d3plot crash simulation data with D3PlotSource.
# - Remove non-deforming wall nodes with WallNodeFilter.
# - Log mesh metadata and convert precision in a composable filter chain.
# - Write processed meshes in parallel with run_pipeline.
#
# For production workloads, increase n_jobs and remove the
# indices limit to process the full dataset.
