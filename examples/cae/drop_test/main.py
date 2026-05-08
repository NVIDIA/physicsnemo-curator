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

"""Drop Test ETL Pipeline.

This example demonstrates processing OpenRadioss drop test simulations
using the PhysicsNeMo Curator pipeline.  The pipeline:

1. Reads per-timestep VTK files from OpenRadioss simulations
2. Filters out rigid wall nodes using displacement analysis
3. Computes edge connectivity for downstream ML workflows
4. Writes output to Zarr format

The output Zarr stores contain:
- mesh_pos: (T, N, 3) node positions across all timesteps
- edges: (E, 2) edge connectivity
- thickness: (N,) per-node thickness (zeros for solid elements)
- displacement fields and optional stress/velocity data
"""

from physicsnemo_curator.domains.mesh.filters.edge_compute import EdgeComputeFilter
from physicsnemo_curator.domains.mesh.filters.wall_node import WallNodeFilter
from physicsnemo_curator.domains.mesh.sinks.mesh_zarr import MeshZarrSink
from physicsnemo_curator.domains.mesh.sources.openradioss import OpenRadiossSource
from physicsnemo_curator.run import run_pipeline

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Input directory containing run subdirectories, each with per-timestep VTK
# files produced by OpenRadioss anim_to_vtk converter.
#
# Expected structure:
#   input/
#   ├── run0001/
#   │   ├── Cell_Phone_DropA001.vtk
#   │   ├── Cell_Phone_DropA002.vtk
#   │   └── ...
#   ├── run0002/
#   │   └── ...

INPUT_DIR = "input/drop_test_runs"

# Glob pattern for VTK files within each run directory
VTK_GLOB = "*.vtk"

# Output directory for Zarr stores
OUTPUT_DIR = "output/drop_test"

# Wall node threshold: nodes with max displacement variation below this
# value are considered rigid (wall) and filtered out.  For drop test
# simulations, use a small value like 1e-5 to identify truly stationary
# boundary nodes.
WALL_THRESHOLD = 1e-5

# -----------------------------------------------------------------------------
# Pipeline Definition
# -----------------------------------------------------------------------------

# Configure the Source
#
# OpenRadiossSource scans input_dir for subdirectories containing VTK
# files matching vtk_glob.  Each subdirectory corresponds to one drop
# test simulation run.
#
# Set read_stress=True to include von Mises stress as cell data fields.
# Set read_velocity=True to include velocity as point data fields.

source = OpenRadiossSource(
    input_dir=INPUT_DIR,
    vtk_glob=VTK_GLOB,
    read_stress=True,
)

# Build the Pipeline
#
# Chain several filters in order:
#
# 1. WallNodeFilter — Removes non-deforming "wall" nodes whose maximum
#    displacement variation across all timesteps falls below the threshold.
#    This removes rigid boundary conditions from the mesh, keeping only
#    the structural response.
#
# 2. EdgeComputeFilter — Computes unique edge connectivity from cell
#    connectivity.  Edges are stored in global_data["edges"] for use
#    by downstream sinks and ML models.
#
# Finally, MeshZarrSink writes each processed mesh to a Zarr store
# with zstd compression.

pipeline = (
    source.filter(WallNodeFilter(threshold=WALL_THRESHOLD))
    .filter(EdgeComputeFilter())
    .write(MeshZarrSink(output_dir=OUTPUT_DIR, compression_level=3))
)

# -----------------------------------------------------------------------------
# Execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Run the Pipeline
    #
    # Process runs in parallel using a process pool.  Drop test simulations
    # can be memory-intensive due to large meshes and many timesteps, so
    # adjust n_jobs based on available memory.

    results = run_pipeline(
        pipeline,
        n_jobs=2,
        backend="process_pool",
        indices=range(min(3, len(source))),
        use_tui=True,
    )

    # Inspect Results
    #
    # run_pipeline returns a list of output paths per index.  Each entry
    # is the list of files written by the sink for that run.

    for idx, paths in enumerate(results):
        print(f"Run {idx}: {len(paths)} output(s)")
        for p in paths:
            print(f"  {p}")

    # -----------------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------------
    #
    # After running this pipeline, output/drop_test/ will contain Zarr
    # stores like:
    #
    #   output/drop_test/
    #   ├── mesh_0000.zarr/
    #   │   ├── mesh_pos          # (T, N, 3) positions
    #   │   ├── edges             # (E, 2) connectivity
    #   │   ├── thickness         # (N,) zeros for solid elements
    #   │   ├── displacement_t000 # (N, 3) displacement at t=0
    #   │   ├── displacement_t001 # (N, 3) displacement at t=1
    #   │   └── ...
    #   ├── mesh_0001.zarr/
    #   └── ...
    #
    # These Zarr stores can be loaded for physics-informed ML training.
