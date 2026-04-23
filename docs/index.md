# PhysicsNeMo Curator

ETL toolkit for deep-learning data curation with Python APIs.

PhysicsNeMo Curator provides a composable **Source → Filter → Sink** pipeline
for reading, transforming, and writing scientific data.  Each domain vertical
(meshes, xarray DataArrays, molecular dynamics tensors) communicates through a
single data structure, and pipelines are executed lazily on a per-item basis.

## Key Features

- **Fluent pipeline API** — build pipelines with `Source(...).filter(F()).write(S())`
- **Lazy evaluation** — `pipeline[i]` processes only the *i*-th item
- **Parallel execution** — `run_pipeline(pipeline, n_jobs=-1)` processes all items across multiple backends
- **Generator semantics** — sources and filters can yield zero, one, or many items
- **Built-in dataset sources** — DrivAerML, AhmedML, WindsorML, WindTunnel-20k from HuggingFace Hub
- **Remote data support** — sources handle local dirs, S3, HuggingFace Hub via fsspec with transparent caching
- **Pluggable submodules** — `mesh`, `da`, `atm` with independent dependency groups
- **Interactive TUI** — full-screen pipeline wizard powered by Textual
- **Component registry** — automatic discovery of sources, filters, and sinks

## Quick Install

```bash
# Core package (no domain-specific dependencies)
pip install physicsnemo-curator

# With mesh support (physicsnemo, pyvista, pyarrow, torch)
pip install physicsnemo-curator[mesh]

# With parallel backends (joblib, dask)
pip install physicsnemo-curator[parallel]

# Wizard is included by default (textual is a core dependency)
psnc
```

## Minimal Example

```python
from physicsnemo_curator import run_pipeline
from physicsnemo_curator.domains.mesh.sources.vtk import VTKSource
from physicsnemo_curator.domains.mesh.filters.mean import MeanFilter
from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink

# Build the pipeline
pipeline = (
    VTKSource("./cfd_results/")
    .filter(MeanFilter(output="stats.parquet"))
    .write(MeshSink(output_dir="./output/"))
)

# Process all items (sequentially, with progress bar)
results = run_pipeline(pipeline)

# Or process in parallel across 8 workers
results = run_pipeline(pipeline, n_jobs=8, backend="process_pool")

# Flush stateful filters (only needed for sequential runs)
pipeline.filters[0].flush()
```

## User Guide

```{toctree}
:maxdepth: 3

user-guide/index
```

## Domains

```{toctree}
:maxdepth: 3

domains/index
```

## Examples

```{toctree}
:maxdepth: 2

auto_examples/index
```

## Benchmarks

```{toctree}
:hidden:

Benchmarks <https://nvidia.github.io/physicsnemo-curator/benchmarks/>
```

## API

```{toctree}
:maxdepth: 3

api/index
```
