# PhysicsNeMo Curator

ETL toolkit for deep-learning data curation with Python APIs.

PhysicsNeMo Curator provides a composable **Source → Filter → Sink** pipeline
for reading, transforming, and writing scientific data.  Each domain vertical
(meshes, xarray datasets, molecular dynamics tensors) communicates through a
single data structure, and pipelines are executed lazily on a per-item basis.

## Key Features

- **Fluent pipeline API** — build pipelines with `Source(store).filter(F()).write(S())`
- **Lazy evaluation** — `pipeline[i]` processes only the *i*-th item
- **Generator semantics** — sources and filters can yield zero, one, or many items
- **FileStore abstraction** — decouple file discovery from reading; local dirs, S3, HuggingFace Hub, or custom backends
- **Pluggable submodules** — `mesh`, `xr`, `mdt` with independent dependency groups
- **Interactive CLI** — guided pipeline builder powered by Click + Questionary
- **Component registry** — automatic discovery of sources, filters, sinks, and stores

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
from curator.core.store import LocalFileStore
from curator.mesh.sources.vtk import VTKSource
from curator.mesh.filters.mean import MeanFilter
from curator.mesh.sinks.mesh_writer import MeshSink

# Create a file store for local VTK data
store = LocalFileStore("./cfd_results/", extensions=frozenset({".vtk", ".vtu"}))

# Build the pipeline
pipeline = (
    VTKSource(store=store)
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
developer-guide/benchmarking
autoapi/index
```
