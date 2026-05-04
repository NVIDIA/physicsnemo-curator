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

"""DrivAerML End-to-End ETL."""

# Import the core pipeline components: a Source to read meshes, Filters
# for metadata logging, statistics, and precision conversion, a Sink to
# write outputs, and run_pipeline for parallel execution.

from pathlib import Path

from physicsnemo_curator.domains.mesh.filters.precision import PrecisionFilter
from physicsnemo_curator.domains.mesh.filters.random_permutation import RandomPermutationFilter
from physicsnemo_curator.domains.mesh.filters.stats import MeshStatsFilter
from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink
from physicsnemo_curator.domains.mesh.sources.drivaerml import DrivAerMLSource
from physicsnemo_curator.run import gather_pipeline, run_pipeline

# Configure the Source
#
# DrivAerMLSource streams data from HuggingFace Hub by default. Files are
# cached locally after first download.
#
# We select mesh_type="multi" to read all mesh representations:
#
# - domain — DomainMesh combining volume interior + boundary surface
# - stl — vehicle geometry from the STL file
# - single_solid — same STL merged into one contiguous solid

_HERE = Path(__file__).resolve().parent.parent
_INPUT_DIR = _HERE / "input" / "drivaerml"
_OUTPUT_DIR = _HERE / "output" / "drivaerml"

source = DrivAerMLSource(
    url=f"file://{_INPUT_DIR}",
    mesh_type="multi",
    manifold_dim="auto",
    point_source="vertices",
    backend="rust",
)

n_runs = len(source)
print(f"Total runs available: {n_runs}")

# Build the Pipeline
#
# The fluent API chains Source -> Filter -> Sink into a lazy Pipeline.
# Nothing is executed until we explicitly process indices.
#
# 1. MeshStatsFilter computes per-field statistics (mean, std, skewness,
#    kurtosis) using numerically stable Welford accumulators that merge
#    across workers.
# 2. PrecisionFilter casts all float64 fields to float32 for training
#    consistency.
# 3. RandomPermutationFilter randomly shuffles point and cell ordering
#    to remove spatial bias.
# 4. MeshSink writes each mesh in PhysicsNeMo's native format — .pdmsh
#    for DomainMesh and .pmsh for plain Mesh.
#
# The naming_template uses run_{run_id}/{mesh_name} which groups outputs
# into per-run subdirectories. {run_id} is resolved via the source's
# run_id method and {mesh_name} via mesh_name — producing paths like
# run_1/domain_1.pdmsh, run_1/drivaer_1.stl.pmsh, etc.

pipeline = (
    source.filter(MeshStatsFilter(output=str(_OUTPUT_DIR / "stats.parquet")))
    .filter(PrecisionFilter(target_dtype="float32"))
    .filter(RandomPermutationFilter(seed=42))
    .write(
        MeshSink(
            output_dir=str(_OUTPUT_DIR),
            naming_template="run_{run_id}/{mesh_name}",
        )
    )
)

# Run in Parallel
#
# run_pipeline dispatches work to a process_pool backend with 2 workers.
# Each worker gets an independent copy of the pipeline, so meshes are
# read, filtered, and written concurrently.

results = run_pipeline(pipeline, n_jobs=1, backend="process_pool")

# Inspect Results
#
# results is a list of lists — one entry per processed index, each
# containing the file paths written by the sink.

print(f"Processed {len(results)} runs")
for i, paths in enumerate(results):
    print(f"  Run {i}: {paths}")

# Gather Statistics
#
# When running in parallel, each worker writes per-worker shard files for
# stateful filters (e.g. MeshStatsFilter). gather_pipeline discovers those
# shards, merges them into single output files, and cleans up the temporary
# shard files.
#
# After gathering, the statistics Parquet contains the exact global mean,
# std, skewness, and kurtosis computed across all processed meshes.

merged = gather_pipeline(pipeline)
print(merged)
for path in merged:
    print(f"Merged statistics: {path}")

# Output Structure
#
#     output/drivaerml/
#     ├── mesh_info.jsonl                          # Mesh metadata (JSON-lines)
#     ├── stats.parquet                            # Per-field statistics (merged)
#     ├── run_1/
#     │   ├── domain_1.pdmsh/                      # DomainMesh: interior + surface
#     │   ├── drivaer_1.stl.pmsh/                  # STL geometry
#     │   └── drivaer_1_single_solid.stl.pmsh/     # Merged STL
#     ├── run_2/
#     │   ├── domain_2.pdmsh/
#     │   ├── drivaer_2.stl.pmsh/
#     │   └── drivaer_2_single_solid.stl.pmsh/
#     └── ...
