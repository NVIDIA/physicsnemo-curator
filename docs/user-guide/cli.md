# Interactive CLI

PhysicsNeMo Curator includes an interactive command-line tool that guides
you through building and executing a pipeline without writing code.

## Installation

```bash
pip install physicsnemo-curator[cli]
```

You also need the domain submodule installed (e.g. `pip install physicsnemo-curator[mesh]`).

## Usage

```bash
curator
```

The CLI walks through six steps:

### 1. Select Submodule

The CLI discovers registered submodules and shows which have their
dependencies installed:

```text
? Select a submodule:
  ▸ mesh — Mesh processing (physicsnemo.mesh.Mesh) [available]
    xr — XArray processing [not installed]
    mdt — Molecular dynamics tensors [not installed]
```

### 2. Select Data Store

Choose where the input data lives.  Built-in stores are registered per
submodule (see {ref}`store-registration`):

```text
? Select a data store:
  ▸ Local directory
    Remote (fsspec)
```

You are then prompted for store-specific inputs:

**Local directory:**

```text
  Path to file or directory: ./cfd_results/
  Glob pattern [*]: *.vtk
  Found 42 file(s) in store.
```

**Remote (fsspec):**

```text
  Remote URL (s3://, hf://, https://): hf://datasets/neashton/drivaerml/run_1/slices
  Glob pattern [**]: *.vtp
  Local cache directory (leave empty for temp):
  Found 68 file(s) in store.
```

### 3. Select Source

Choose from the registered sources for the selected submodule:

```text
? Select a source/reader:
  ▸ VTK Reader — Read VTK files (.vtk, .vtp, .vtu, .vts, .vtm)
```

You are then prompted for source-specific parameters (conversion options):

```text
Configure VTK Reader:
? manifold_dim (Target manifold dimension) [auto]: auto
? point_source (Point source mode) [vertices]: vertices
? warn_on_lost_data (Warn when data arrays are discarded) [True]:
```

### 4. Select Filters

Choose zero or more filters (multi-select with checkboxes):

```text
? Select filters (space to toggle, enter to confirm):
  ▸ ☑ Mean Statistics — Compute spatial means and save to Parquet
```

Each selected filter's parameters are prompted in order.

### 5. Select Sink

Choose the output writer:

```text
? Select a sink:
  ▸ PhysicsNeMo Mesh Writer — Save in native tensordict format
```

### 6. Execute

The CLI builds the pipeline and processes all items with a progress
indicator.  Stateful filters are flushed automatically after execution.

```text
Processing 42 items...
  Item 0 → ['./output/mesh_0000_0']
  Item 1 → ['./output/mesh_0001_0']
  ...
Done. Processed 42 items.
Flushed Mean Statistics → stats.parquet
```

## Programmatic Equivalent

Everything the CLI does can be done in Python:

```python
from physicsnemo_curator import run_pipeline
from physicsnemo_curator.core.store import LocalFileStore
from physicsnemo_curator.mesh.sources.vtk import VTKSource
from physicsnemo_curator.mesh.filters.mean import MeanFilter
from physicsnemo_curator.mesh.sinks.mesh_writer import MeshSink

store = LocalFileStore("./cfd_results/")
pipeline = (
    VTKSource(store=store)
    .filter(MeanFilter(output="stats.parquet"))
    .write(MeshSink(output_dir="./output/"))
)

# Sequential with progress bar (equivalent to CLI behaviour)
results = run_pipeline(pipeline)

# Or parallel across multiple cores
results = run_pipeline(pipeline, n_jobs=-1, backend="process_pool")

# Flush stateful filters (sequential only)
pipeline.filters[0].flush()
```

See {doc}`parallel` for details on `run_pipeline` and available backends.
