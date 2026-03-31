# PhysicsNeMo Curator

ETL toolkit for deep-learning data curation with Python APIs.

PhysicsNeMo Curator provides a composable **Source → Filter → Sink** pipeline
for reading, transforming, and writing scientific data.  Each domain vertical
(meshes, xarray datasets, molecular dynamics tensors) communicates through a
single data structure, and pipelines are executed lazily on a per-item basis.

## Key Features

- **Fluent pipeline API** — build pipelines with `Source().filter(F()).write(S())`
- **Lazy evaluation** — `pipeline[i]` processes only the *i*-th item
- **Generator semantics** — sources and filters can yield zero, one, or many items
- **Pluggable submodules** — `mesh`, `xr`, `mdt` with independent dependency groups
- **Interactive CLI** — guided pipeline builder powered by Click + Questionary
- **Component registry** — automatic discovery of sources, filters, and sinks

## Quick Install

```bash
# Core package (no domain-specific dependencies)
pip install physicsnemo-curator

# With mesh support (physicsnemo, pyvista, pyarrow, torch)
pip install physicsnemo-curator[mesh]

# With CLI support (click, questionary)
pip install physicsnemo-curator[cli]
```

## Minimal Example

```python
from curator.mesh.sources.vtk import VTKSource
from curator.mesh.filters.mean import MeanFilter
from curator.mesh.sinks.mesh_writer import MeshSink

# Build the pipeline
pipeline = (
    VTKSource.from_path("./cfd_results/")
    .filter(MeanFilter(output="stats.parquet"))
    .write(MeshSink(output_dir="./output/"))
)

# Process lazily
for i in range(len(pipeline)):
    paths = pipeline[i]
    print(f"Item {i}: {paths}")

# Flush stateful filters
pipeline.filters[0].flush()
```

## Contents

```{toctree}
:maxdepth: 2

user-guide/installation
user-guide/quickstart
user-guide/architecture
user-guide/mesh
user-guide/cli
developer-guide/extending
autoapi/index
```
