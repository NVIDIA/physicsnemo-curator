# Drop Test ETL

Process OpenRadioss drop test simulations through a complete
Source → Filter → Sink pipeline for deep learning.

## Pipeline Overview

1. **OpenRadiossSource** — reads per-timestep VTK files produced by
   OpenRadioss `anim_to_vtk` converter, storing positions and optional
   fields (velocity, acceleration, stress).
2. **WallNodeFilter** — identifies and removes rigid wall nodes using
   displacement variation analysis across timesteps.
3. **EdgeComputeFilter** — computes edge connectivity from mesh cells
   for graph neural network workflows.
4. **MeshVTUSink** — writes each mesh to a VTU (VTK UnstructuredGrid)
   file compatible with the PhysicsNeMo drop_test recipe.

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
input/drop/
├── run_0001/
│   ├── Cell_Phone_DropA001.vtk
│   ├── Cell_Phone_DropA002.vtk
│   └── ...
├── run_0002/
│   └── ...
└── ...
```

## Usage

```bash
cd examples/cae/drop_etl

# Basic usage
uv run python main.py

# Custom paths and workers
uv run python main.py --input /path/to/runs --output /path/to/output --workers 4

# Adjust wall node threshold (default: 1e-5)
uv run python main.py --wall-threshold 1e-6
```

## Output Structure

```text
target/drop/
├── run_0001.vtu          # VTK UnstructuredGrid with timestep data
├── run_0002.vtu
└── ...
```

Each VTU file contains:

### Points (Reference Coordinates)

- Mesh vertices at t=0 (reference frame)

### Point Arrays

- `thickness` — (N,) zeros for solid elements
- `displacement_t0000`, `displacement_t0001`, ... — (N, 3) displacement per timestep
- `Von_Mises_t0000`, `Von_Mises_t0001`, ... — (N,) von Mises stress per timestep (optional)

### Cell Arrays (if present)

- Per-element stress and strain fields

## Wall Node Filtering

The WallNodeFilter removes rigid boundary nodes by analyzing displacement
variation across all timesteps. Nodes whose maximum displacement variation
falls below the threshold (default: 1e-5) are considered "wall" nodes and
filtered out. This keeps only the structural response for ML training.

## Solid Element Thickness

For solid elements (tetrahedra, hexahedra, etc.), thickness is set to zero
since 3D volumetric elements have no meaningful thickness attribute. This
differs from shell elements where thickness represents material thickness.

## VTU Format Details

The VTU output format stores:

- **Points**: Reference positions (t=0), enabling reconstruction of any
  timestep via `points + displacement_tNNNN`
- **Cells**: VTK cell types (TETRA=10, HEXA=12, WEDGE=13, PYRAMID=14, TRI=5)
- **4-digit timestep indexing**: `tNNNN` format for compatibility with
  PhysicsNeMo recipes

This format is consumed directly by the `physicsnemo/examples/structural_mechanics/drop_test`
recipe for physics-informed machine learning.
