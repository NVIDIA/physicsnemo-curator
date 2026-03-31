# Datasets

PhysicsNeMo Curator provides built-in sources for several curated CFD
benchmark datasets hosted on [HuggingFace Hub](https://huggingface.co).
Each source maps integer indices to individual simulation runs and handles
file discovery, caching, and format conversion transparently.

## Available Datasets

| Source | Dataset | Samples | Mesh Types | Size |
|--------|---------|---------|------------|------|
| {py:class}`~curator.mesh.sources.drivaerml.DrivAerMLSource` | [DrivAerML](https://huggingface.co/datasets/neashton/drivaerml) | ~484 | boundary, volume, slices | ~31 TB |
| {py:class}`~curator.mesh.sources.ahmedml.AhmedMLSource` | [AhmedML](https://huggingface.co/datasets/neashton/ahmedml) | 500 | boundary, volume, slices | ~2 TB |
| {py:class}`~curator.mesh.sources.windsorml.WindsorMLSource` | [WindsorML](https://huggingface.co/datasets/neashton/windsorml) | 355 | boundary, volume | ~8 TB |
| {py:class}`~curator.mesh.sources.windtunnel.WindTunnelSource` | [WindTunnel-20k](https://huggingface.co/datasets/inductiva/windtunnel-20k) | 19,812 | pressure_field | ~300 GB |

All datasets require the `mesh` dependency group:

```bash
pip install physicsnemo-curator[mesh]
```

## Quick Start

### DrivAerML — Boundary Meshes

```python
from curator.mesh.sources.drivaerml import DrivAerMLSource

# Load boundary (surface) meshes
source = DrivAerMLSource(mesh_type="boundary")
print(f"Found {len(source)} runs")

mesh = next(source[0])
print(f"Points: {mesh.n_points}, Cells: {mesh.n_cells}")
print(f"Cell data fields: {list(mesh.cell_data.keys())}")
```

### DrivAerML — Slice Planes

```python
source = DrivAerMLSource(mesh_type="slices")

# Each index yields multiple slice plane meshes
for mesh in source[0]:
    print(f"Slice: {mesh.n_points} points, {mesh.n_cells} cells")
```

### AhmedML — Full Pipeline

```python
from curator.mesh.sources.ahmedml import AhmedMLSource
from curator.mesh.filters.mean import MeanFilter
from curator.mesh.sinks.mesh_writer import MeshSink
from curator import run_pipeline

pipeline = (
    AhmedMLSource(mesh_type="boundary")
    .filter(MeanFilter(output="stats.parquet"))
    .write(MeshSink(output_dir="./output/"))
)

# Process first 10 runs in parallel
results = run_pipeline(pipeline, n_jobs=4, indices=list(range(10)))
print(f"Wrote {sum(len(r) for r in results)} files")
```

### WindsorML

```python
from curator.mesh.sources.windsorml import WindsorMLSource

source = WindsorMLSource(mesh_type="boundary")
print(f"Found {len(source)} runs (run_0 through run_354)")

mesh = next(source[0])
```

### WindTunnel-20k

```python
from curator.mesh.sources.windtunnel import WindTunnelSource

# Load pressure field meshes from the training split
source = WindTunnelSource(split="train")
print(f"Found {len(source)} simulations")

# Or load all splits at once
source_all = WindTunnelSource(split="all")
print(f"Total: {len(source_all)} simulations")
```

## Mesh Types

Each dataset organises its data into different mesh types:

### Boundary (Surface)

Surface meshes with flow field data (pressure, velocity, Reynolds stress).
These are the most commonly used for ML training. File sizes range from
~83 MB (AhmedML) to ~660 MB (DrivAerML) per run.

```python
source = DrivAerMLSource(mesh_type="boundary")
```

### Volume

Full volumetric field data. These are large — from ~5 GB (AhmedML) to
~50 GB (DrivAerML) per run. DrivAerML volume files are split into parts
on HuggingFace and are automatically concatenated on download.

```python
source = AhmedMLSource(mesh_type="volume")
```

### Slices

x/y/z-normal slice planes with flow fields. Available for DrivAerML and
AhmedML. Each run yields multiple slice meshes.

```python
source = DrivAerMLSource(mesh_type="slices")
for mesh in source[0]:  # yields multiple VTP files
    print(mesh.n_points)
```

## Caching

All dataset sources cache downloaded files locally to avoid repeated
downloads. By default, a temporary directory is created. Pass
`cache_storage` to control the cache location:

```python
source = DrivAerMLSource(
    mesh_type="boundary",
    cache_storage="/data/cache/drivaerml",
)
```

## RunIndexedFileStore

Under the hood, the run-indexed datasets (DrivAerML, AhmedML, WindsorML)
use {py:class}`~curator.core.store.RunIndexedFileStore`, which discovers
`run_<i>/` directories and resolves per-run file templates:

```python
from curator.core.store import RunIndexedFileStore

store = RunIndexedFileStore(
    url="hf://datasets/neashton/drivaerml",
    file_template="boundary_{i}.vtp",
)
print(f"Found {len(store)} runs")
print(f"Run indices: {store.run_indices[:5]}...")
```

## Dataset Details

### DrivAerML

- **Paper**: [arXiv:2408.11969](https://arxiv.org/abs/2408.11969)
- **License**: CC-BY-SA 4.0
- **Solver**: OpenFOAM v2212
- **Samples**: 500 parametrically morphed DrivAer notchback variants
  (~16 runs missing from HuggingFace)
- **Files per run**: `boundary_{i}.vtp`, `volume_{i}.vtu` (split),
  `slices/*.vtp`, `drivaer_{i}.stl`, metadata CSVs

### AhmedML

- **Paper**: [arXiv:2407.20801](https://arxiv.org/abs/2407.20801)
- **License**: CC-BY-SA 4.0
- **Solver**: OpenFOAM v2212, transient hybrid RANS-LES
- **Samples**: 500 Ahmed Car Body variants, ~20 M cells each
- **Files per run**: `boundary_{i}.vtp`, `volume_{i}.vtu`,
  `slices/*.vtp`, `ahmed_{i}.stl`, metadata CSVs

### WindsorML Details

- **Paper**: [arXiv:2407.19320](https://arxiv.org/abs/2407.19320)
- **License**: CC-BY-SA 4.0
- **Solver**: Volcano Platforms (GPU-native Cartesian immersed-boundary WMLES)
- **Samples**: 355 Windsor body variants, ~280-300 M cells each
- **Files per run**: `boundary_{i}.vtu`, `volume_{i}.vtu`,
  `windsor_{i}.stl`, `windsor_{i}.stp`, metadata CSVs
- **Note**: No slice plane meshes (images only)

### WindTunnel-20k Details

- **Source**: [inductiva/windtunnel-20k](https://huggingface.co/datasets/inductiva/windtunnel-20k)
- **Solver**: OpenFOAM
- **Samples**: 19,812 simulations (1,000 unique meshes × 20 conditions)
- **Splits**: train (13,900), validation (3,970), test (1,980)
- **Files per sim**: `pressure_field_mesh.vtk`, `input_mesh.obj`,
  `openfoam_mesh.obj`, `streamlines_mesh.ply`, `simulation_metadata.json`
- **Note**: Only `pressure_field_mesh.vtk` is currently supported
