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

"""Checkpointing a Pipeline.

See README.md for a full walkthrough.
"""

from pathlib import Path

from physicsnemo_curator import Pipeline
from physicsnemo_curator.domains.mesh.filters.precision import PrecisionFilter
from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink
from physicsnemo_curator.domains.mesh.sources.ns_cylinder import NavierStokesCylinderSource
from physicsnemo_curator.run import run_pipeline

# Build a Resumable Pipeline
resumable = Pipeline(
    source=NavierStokesCylinderSource(),
    filters=[PrecisionFilter(target_dtype="float32")],  # ty: ignore[invalid-argument-type]
    sink=MeshSink(output_dir="output/checkpoint/meshes/"),
    resume=True,
    db_dir=Path("output/checkpoint/"),
)

print(f"Resume enabled: {resumable.resume}")
print(f"Database dir: {resumable.db_dir}")

# First Run — Process 5 Indices
results = run_pipeline(
    resumable,
    n_jobs=1,
    backend="sequential",
    indices=range(5),
    use_tui=True,
)

print(f"\nFirst run processed {len(results)} indices")
print(f"Completed: {resumable.completed_indices}")
print(f"Database: {resumable.db_path}")
print(f"Summary: {resumable.summary()}")

# Second Run — Resume from Checkpoint
results_resumed = run_pipeline(
    resumable,
    n_jobs=1,
    backend="sequential",
    indices=range(8),  # 0-4 cached, 5-7 new
    use_tui=True,
)

print(f"\nSecond run returned {len(results_resumed)} results")
print(f"Completed: {resumable.completed_indices}")
print(f"Remaining (of {len(resumable)}): {len(resumable.remaining_indices())}")

# Query Checkpoint State
print(f"\nCompleted indices: {resumable.completed_indices}")
print(f"Failed indices: {resumable.failed_indices}")
print(f"Remaining indices: {resumable.remaining_indices()}")
print(f"Summary: {resumable.summary()}")

# Individual Index Lookup
paths_for_0 = resumable.output_paths_for_index(0)
print(f"\nPaths for index 0: {paths_for_0}")

if paths_for_0:
    idx = resumable.index_for_path(paths_for_0[0])
    print(f"Reverse lookup: {paths_for_0[0]} → index {idx}")

# Reset Checkpoint
resumable.reset()
print(f"\nAfter reset: {resumable.summary()}")
