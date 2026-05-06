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

"""Creating a Pipeline.

See README.md for a full walkthrough.
"""

from physicsnemo_curator.domains.mesh.filters.mean import MeanFilter
from physicsnemo_curator.domains.mesh.filters.precision import PrecisionFilter
from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink
from physicsnemo_curator.domains.mesh.sources.ns_cylinder import NavierStokesCylinderSource

# Step 1: Create a Source
source = NavierStokesCylinderSource()

print(f"Source: {source.name}")
print(f"Items available: {len(source)}")

# Step 2: Add Filters
pipeline = source.filter(MeanFilter(output="output/getting_started/stats.parquet")).filter(
    PrecisionFilter(target_dtype="float32")
)

print(f"Filters: {[f.name for f in pipeline.filters]}")
print(f"Sink: {pipeline.sink}")  # None — no sink yet

# Step 3: Attach a Sink
pipeline = pipeline.write(MeshSink(output_dir="output/getting_started/meshes/"))

assert pipeline.sink is not None
print(f"Sink: {pipeline.sink.name}")
print(f"Pipeline length: {len(pipeline)}")

# Step 4: Execute One Index
paths = pipeline[0]
print(f"Index 0 wrote: {paths}")
