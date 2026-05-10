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

"""Launching the Metrics Dashboard.

See README.md for a full walkthrough.
"""

from pathlib import Path

from physicsnemo_curator import Pipeline
from physicsnemo_curator.domains.mesh.filters.mean import MeanFilter
from physicsnemo_curator.domains.mesh.filters.precision import PrecisionFilter
from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink
from physicsnemo_curator.domains.mesh.sources.random import RandomMeshSource
from physicsnemo_curator.run import run_pipeline

# Step 1: Run a Pipeline with Metrics
# Every pipeline automatically collects timing and memory metrics in a SQLite
# database. Set `db_dir` to control where the database file is stored.

pipeline = Pipeline(
    source=RandomMeshSource(n_samples=10, n_points=100, n_cells=50),
    filters=[
        MeanFilter(output="output/dashboard/stats.parquet"),
        PrecisionFilter(target_dtype="float32"),
    ],
    sink=MeshSink(output_dir="output/dashboard/meshes/"),
    resume=True,
    db_dir=Path("output/dashboard/"),
)

results = run_pipeline(
    pipeline,
    n_jobs=1,
    backend="sequential",
    indices=range(len(pipeline)),
    use_tui=True,
)

print(f"\nProcessed {len(results)} items")
print(f"Database: {pipeline.db_path}")

# Step 2: Inspect Metrics Programmatically
metrics = pipeline.metrics
metrics.to_console()

# Step 3: Launch the Dashboard
# Uncomment the lines below to open the interactive web dashboard in your
# browser. The dashboard provides three tabs: Overview, Pipeline, and
# Performance — with charts for timing, memory, progress, and artifacts.

# from physicsnemo_curator.dashboard import launch
# launch(str(pipeline.db_path), port=5006, open_browser=True)

print(f"\nTo launch the dashboard manually, run:")
print(f"  from physicsnemo_curator.dashboard import launch")
print(f"  launch('{pipeline.db_path}')")
