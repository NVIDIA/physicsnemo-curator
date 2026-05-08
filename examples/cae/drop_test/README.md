# Drop Test ETL

Process OpenRadioss drop test simulations through a complete
Source → Filter → Sink pipeline for deep learning.

The pipeline:

1. **OpenRadiossSource** — reads per-timestep VTK files produced by
   OpenRadioss `anim_to_vtk` converter, storing positions and optional
   fields (velocity, acceleration, stress).
2. **WallNodeFilter** — identifies and removes rigid wall nodes using
   displacement variation analysis across timesteps.
3. **EdgeComputeFilter** — computes edge connectivity from mesh cells
   for graph neural network workflows.
4. **MeshZarrSink** — writes each mesh to a Zarr store with zstd
   compression, optimized for deep learning dataloaders.

## Prerequisites

```bash
uv sync --extra mesh --extra loky

# or with pip
pip install physicsnemo-curator[mesh,loky]
```

## Input Data

The pipeline expects OpenRadioss simulation output converted to VTK
format using the `anim_to_vtk` tool. Each simulation run should be in
its own subdirectory containing one VTK file per timestep:

```text
input/drop_test_runs/
├── run0001/
│   ├── Cell_Phone_DropA001.vtk
│   ├── Cell_Phone_DropA002.vtk
│   └── ...
├── run0002/
│   └── ...
└── ...
```

## Usage

```bash
# Basic usage (reads from ./input, writes to ./output)
python main.py

# Custom input/output directories
python main.py --input /path/to/runs --output /path/to/output

# Limit to specific number of workers
python main.py --workers 4
```

## Output Structure

```text
output/drop_test/
├── mesh_0000.zarr/
│   ├── mesh_pos           # (T, N, 3) - positions per timestep
│   ├── edges              # (E, 2) - edge connectivity
│   ├── thickness          # (N,) - zeros for solid elements
│   ├── displacement_t000  # (N, 3) - displacement at t=0
│   ├── displacement_t001  # (N, 3) - displacement at t=1
│   ├── stress_vm_t000     # (E,) - von Mises stress (optional)
│   └── ...
├── mesh_0001.zarr/
│   └── ...
└── ...
```

Each Zarr store includes metadata attributes:

- `num_timesteps`: Number of timesteps in the simulation
- `num_nodes`: Number of nodes after wall filtering
- `num_edges`: Number of edges in the mesh
- `thickness_min/max/mean`: Thickness statistics (zeros for solids)
