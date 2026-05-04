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

"""Profiling a Pipeline."""

from physicsnemo_curator.core.profiling import ProfiledPipeline

from physicsnemo_curator.domains.mesh.filters.mean import MeanFilter
from physicsnemo_curator.domains.mesh.filters.precision import PrecisionFilter
from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink
from physicsnemo_curator.domains.mesh.sources.ns_cylinder import NavierStokesCylinderSource
from physicsnemo_curator.run import run_pipeline

# Build and Wrap the Pipeline
#
# First build a normal pipeline, then wrap it with ProfiledPipeline.
# The wrapper records timing for each stage: source, every filter, and sink.

pipeline = (
    NavierStokesCylinderSource()
    .filter(MeanFilter(output="output/profiling/stats.parquet"))
    .filter(PrecisionFilter(target_dtype="float32"))
    .write(MeshSink(output_dir="output/profiling/meshes/"))
)

profiled = ProfiledPipeline(pipeline)

# Run with Profiling
#
# Pass the ProfiledPipeline to run_pipeline exactly as you would a normal
# pipeline. Timing data is collected transparently.

results = run_pipeline(
    profiled,
    n_jobs=1,
    backend="sequential",
    indices=range(3),
    progress=True,
)

print(f"Processed {len(results)} indices")

# View Timing Summary
#
# The metrics property returns a PipelineMetrics object with aggregated
# timing data.

metrics = profiled.metrics

print("\n--- Console Summary ---")
metrics.to_console()

# Export Metrics
#
# Metrics can be exported to JSON or CSV for further analysis.

metrics.to_json("output/profiling/timing.json")
metrics.to_csv("output/profiling/timing.csv")

print("\nExported timing.json and timing.csv")

# Inspect Per-Index Breakdown
#
# Each IndexMetrics entry contains per-stage durations.

summary = metrics.summary()
print(f"\nTotal wall time: {summary['total_wall_time_ns'] / 1e9:.2f}s")
print(f"Mean time per index: {summary['mean_index_time_ns'] / 1e9:.2f}s")
print(f"Indices profiled: {summary['n_indices']}")

# Cleanup
#
# Remove temporary metrics files created during profiling.

profiled.cleanup()
