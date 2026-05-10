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

"""Running a Pipeline in Parallel.

See README.md for a full walkthrough.
"""

from physicsnemo_curator.domains.mesh.filters.mean import MeanFilter
from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink
from physicsnemo_curator.domains.mesh.sources.random import RandomMeshSource
from physicsnemo_curator.run import gather_pipeline, run_pipeline

if __name__ == "__main__":
    # Build the Pipeline
    pipeline = (
        RandomMeshSource(n_samples=10, n_points=100, n_cells=50)
        .filter(MeanFilter(output="output/parallel/stats.parquet"))
        .write(MeshSink(output_dir="output/parallel/meshes/"))
    )

    print(f"Total runs available: {len(pipeline)}")

    # Run in Parallel
    results = run_pipeline(
        pipeline,
        n_jobs=4,
        backend="process_pool",
        indices=range(3),
        use_tui=True,
    )

    # Inspect Results
    print(f"\nProcessed {len(results)} runs")
    for i, paths in enumerate(results):
        print(f"  Run {i}: {paths}")

    # Gather Statistics
    merged = gather_pipeline(pipeline)
    for path in merged:
        print(f"Merged statistics: {path}")
