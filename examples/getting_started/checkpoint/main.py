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

"""Checkpointing a Pipeline."""

from physicsnemo_curator.core.checkpoint import CheckpointedPipeline

from physicsnemo_curator.domains.mesh.filters.precision import PrecisionFilter
from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink
from physicsnemo_curator.domains.mesh.sources.ns_cylinder import NavierStokesCylinderSource
from physicsnemo_curator.run import run_pipeline

# Build and Wrap the Pipeline
#
# First build a normal pipeline, then wrap it with CheckpointedPipeline.
# The db_path argument specifies where the SQLite checkpoint file is stored.

pipeline = (
    NavierStokesCylinderSource()
    .filter(PrecisionFilter(target_dtype="float32"))
    .write(MeshSink(output_dir="output/checkpoint/meshes/"))
)

checkpointed = CheckpointedPipeline(
    pipeline,
    db_path="output/checkpoint/pipeline.db",
)

# First Run — Process 5 Indices
#
# On the first run, all indices are new and will be fully executed.

results = run_pipeline(
    checkpointed,
    n_jobs=1,
    backend="sequential",
    indices=range(5),
    progress=True,
)

print(f"First run processed {len(results)} indices")
print(f"Checkpoint summary: {checkpointed.summary()}")

# Second Run — Resume from Checkpoint
#
# If we run the same pipeline again (even with overlapping indices),
# completed indices are skipped. Their cached output paths are returned
# from the database without re-executing the pipeline.

results_resumed = run_pipeline(
    checkpointed,
    n_jobs=1,
    backend="sequential",
    indices=range(8),  # 0-4 cached, 5-7 new
    progress=True,
)

print(f"\nSecond run returned {len(results_resumed)} results")
print(f"Checkpoint summary: {checkpointed.summary()}")

# Query Checkpoint State
#
# The checkpoint database tracks which indices have been completed,
# which failed, and which remain.

print(f"\nCompleted indices: {checkpointed.completed_indices}")
print(f"Remaining (of 500): {len(checkpointed.remaining_indices)}")
print(f"Config hash: {checkpointed.config_hash[:16]}...")
print(f"Database: {checkpointed.db_path}")

# Composing with ProfiledPipeline
#
# CheckpointedPipeline composes with ProfiledPipeline — you can profile
# and checkpoint at the same time:
#
#     from physicsnemo_curator.core.profiling import ProfiledPipeline
#
#     profiled = ProfiledPipeline(pipeline)
#     checkpointed = CheckpointedPipeline(profiled, db_path="ckpt.db")
#     run_pipeline(checkpointed, n_jobs=4)
#
# The checkpoint wraps the profiled pipeline, so skipped indices bypass
# both profiling and execution.

# Reset Checkpoint
#
# To re-process all indices from scratch, call reset():

checkpointed.reset()
print(f"\nAfter reset: {checkpointed.summary()}")
