# Pipeline Wizard

PhysicsNeMo Curator includes an interactive command-line wizard that guides
you through building and executing a pipeline without writing code.

## Installation

```bash
pip install physicsnemo-curator[wiz]
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

The wizard displays a styled welcome banner and walks through the pipeline
configuration with colored output and progress indicators.  You can either
build a new pipeline interactively or load a previously saved one from YAML
or JSON.

### 1. Select Submodule

The CLI discovers registered submodules and shows which have their
dependencies installed:

```text
╭─────────────────────────────────────╮
│   PhysicsNeMo Curator               │
│   Interactive ETL Pipeline Wizard   │
╰─────────────────────────────────────╯

Step 1/5: Select Submodule
? Select a submodule:
  ▸ mesh — Mesh data curation (physicsnemo.mesh.Mesh)
    da — DataArray data curation (xarray.DataArray) (not installed)
    atm — Atomic data curation (nvalchemi.data.AtomicData) (not installed)
```

### 2. Select Source

Choose from the registered sources for the selected submodule:

```text
Step 2/5: Select Source/Reader
? Select a source/reader:
  ▸ VTK Reader — Read VTK files (.vtk, .vtp, .vtu, .vts, .vtm)
```

You are then prompted for source-specific parameters (data location,
conversion options):

```text
  Configure VTK Reader:
  ? input_path (Path to file or directory): ./cfd_results/
  ? manifold_dim (Target manifold dimension) [auto]: auto
  ? point_source (Point source mode) [vertices]: vertices
  ? warn_on_lost_data (Warn when data arrays are discarded) [True]:
  ✓ Found 42 item(s) in source
```

### 3. Select Filters

Choose zero or more filters (multi-select with checkboxes):

```text
Step 3/5: Select Filters
? Select filters (space to toggle, enter to confirm):
  ▸ ☑ Mean Statistics — Compute spatial means and save to Parquet
  ✓ Selected 1 filter(s)
```

Each selected filter's parameters are prompted in order.

### 4. Select Sink

Choose the output writer:

```text
Step 4/5: Select Sink/Writer
? Select a sink:
  ▸ PhysicsNeMo Mesh Writer — Save in native tensordict format
  ✓ Configured sink: PhysicsNeMo Mesh Writer
```

### 5. Execute

The CLI builds the pipeline, displays a summary, and processes all items
with an animated progress bar.  Stateful filters are flushed automatically
after execution.

```text
Step 5/5: Execute Pipeline

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
from physicsnemo_curator import run_pipeline
from physicsnemo_curator.domains.mesh.sources.vtk import VTKSource
from physicsnemo_curator.domains.mesh.filters.mean import MeanFilter
from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink

pipeline = (
    VTKSource("./cfd_results/")
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
