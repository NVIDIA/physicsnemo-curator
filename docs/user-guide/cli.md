# Interactive CLI

PhysicsNeMo Curator includes an interactive command-line tool that guides
you through building and executing a pipeline without writing code.

## Installation

```bash
pip install physicsnemo-curator[cli]
```

This installs the required dependencies:

- `click` — CLI framework
- `questionary` — interactive prompts
- `rich` — colored output and progress bars

You also need the domain submodule installed (e.g. `pip install physicsnemo-curator[mesh]`).

## Usage

```bash
curator
```

The CLI displays a styled welcome banner and walks through six steps with
colored output and progress indicators.

### 1. Select Submodule

The CLI discovers registered submodules and shows which have their
dependencies installed:

```text
╭─────────────────────────────────────╮
│   PhysicsNeMo Curator               │
│   Interactive ETL Pipeline Builder  │
╰─────────────────────────────────────╯

Step 1/6: Select Submodule
? Select a submodule:
  ▸ mesh — Mesh data curation (physicsnemo.mesh.Mesh)
    da — DataArray data curation (xarray.DataArray) (not installed)
    atm — Atomic data curation (nvalchemi.data.AtomicData) (not installed)
```

### 2. Select Data Store

Choose where the input data lives.  Built-in stores are registered per
submodule (see {ref}`store-registration`):

```text
Step 2/6: Configure Data Store
? Select a data store:
  ▸ LocalFileStore
    FsspecFileStore
```

You are then prompted for store-specific inputs:

**Local directory:**

```text
  Configure LocalFileStore:
  Path to file or directory: ./cfd_results/
  Glob pattern [*]: *.vtk
  ✓ Found 42 file(s) in store
```

**Remote (fsspec):**

```text
  Configure FsspecFileStore:
  Remote URL (s3://, hf://, https://): hf://datasets/neashton/drivaerml/run_1/slices
  Glob pattern [**]: *.vtp
  Local cache directory (leave empty for temp):
  ✓ Found 68 file(s) in store
```

### 3. Select Source

Choose from the registered sources for the selected submodule:

```text
Step 3/6: Select Source/Reader
? Select a source/reader:
  ▸ VTK Reader — Read VTK files (.vtk, .vtp, .vtu, .vts, .vtm)
```

You are then prompted for source-specific parameters (conversion options):

```text
  Configure VTK Reader:
  ? manifold_dim (Target manifold dimension) [auto]: auto
  ? point_source (Point source mode) [vertices]: vertices
  ? warn_on_lost_data (Warn when data arrays are discarded) [True]:
  ✓ Found 42 item(s) in source
```

### 4. Select Filters

Choose zero or more filters (multi-select with checkboxes):

```text
Step 4/6: Select Filters
? Select filters (space to toggle, enter to confirm):
  ▸ ☑ Mean Statistics — Compute spatial means and save to Parquet
  ✓ Selected 1 filter(s)
```

Each selected filter's parameters are prompted in order.

### 5. Select Sink

Choose the output writer:

```text
Step 5/6: Select Sink/Writer
? Select a sink:
  ▸ PhysicsNeMo Mesh Writer — Save in native tensordict format
  ✓ Configured sink: PhysicsNeMo Mesh Writer
```

### 6. Execute

The CLI builds the pipeline, displays a summary, and processes all items
with an animated progress bar.  Stateful filters are flushed automatically
after execution.

```text
Step 6/6: Execute Pipeline

╭──────────────────── Pipeline ────────────────────╮
│ VTK Reader → Mean Statistics → Mesh Writer       │
╰──────────────────────────────────────────────────╯

⠋ Processing... ━━━━━━━━━━━━━━━━━━━━ 100% 42/42 ./output/mesh_0041_0

• Statistics saved to stats.parquet

╭─────────────── ✓ Complete ───────────────╮
│ Source items processed:        42        │
│ Outputs written:               42        │
╰──────────────────────────────────────────╯
```

## Color Scheme

The CLI uses a consistent color scheme throughout:

| Element | Color |
|---------|-------|
| Branding | NVIDIA green |
| Step headers | Blue |
| Highlights | Cyan |
| Success (✓) | Green |
| Warnings (⚠) | Yellow |
| Errors (✗) | Red |

## Programmatic Equivalent

Everything the CLI does can be done in Python:

```python
from physicsnemo.curator import run_pipeline
from physicsnemo.curator.core.store import LocalFileStore
from physicsnemo.curator.mesh.sources.vtk import VTKSource
from physicsnemo.curator.mesh.filters.mean import MeanFilter
from physicsnemo.curator.mesh.sinks.mesh_writer import MeshSink

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
