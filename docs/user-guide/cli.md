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

The CLI walks through five steps:

### 1. Select Submodule

The CLI discovers registered submodules and shows which have their
dependencies installed:

```
? Select a submodule:
  ▸ mesh — Mesh processing (physicsnemo.mesh.Mesh) [available]
    xr — XArray processing [not installed]
    mdt — Molecular dynamics tensors [not installed]
```

### 2. Select Source

Choose from the registered sources for the selected submodule:

```
? Select a source:
  ▸ VTK Reader — Read VTK files (.vtk, .vtp, .vtu, .vts, .vtm)
```

You are then prompted for each parameter:

```
? input_path (Path to VTK file or directory): ./cfd_results/
? file_pattern (Glob pattern for filtering files) [*]:
? manifold_dim (Target manifold dimension) [auto]: auto
? point_source (Point source mode) [vertices]: vertices
? warn_on_lost_data (Warn when data arrays are discarded) [True]:
```

### 3. Select Filters

Choose zero or more filters (multi-select with checkboxes):

```
? Select filters (space to toggle, enter to confirm):
  ▸ ☑ Mean Statistics — Compute spatial means and save to Parquet
```

Each selected filter's parameters are prompted in order.

### 4. Select Sink

Choose the output writer:

```
? Select a sink:
  ▸ PhysicsNeMo Mesh Writer — Save in native tensordict format
```

### 5. Execute

The CLI builds the pipeline and processes all items with a progress
indicator.  Stateful filters are flushed automatically after execution.

```
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
from curator.mesh.sources.vtk import VTKSource
from curator.mesh.filters.mean import MeanFilter
from curator.mesh.sinks.mesh_writer import MeshSink

pipeline = (
    VTKSource(input_path="./cfd_results/")
    .filter(MeanFilter(output="stats.parquet"))
    .write(MeshSink(output_dir="./output/"))
)

for i in range(len(pipeline)):
    pipeline[i]

pipeline.filters[0].flush()
```
