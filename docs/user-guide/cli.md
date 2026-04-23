# Pipeline Wizard

PhysicsNeMo Curator includes an interactive full-screen wizard that guides
you through building and executing a pipeline without writing code.  The
wizard is a Textual TUI application with multi-screen navigation, keyboard
shortcuts, and real-time progress tracking.

## Installation

The wizard depends on [Textual](https://textual.textualize.io/), which is
included as a core dependency — no extra installation step is needed:

```bash
pip install physicsnemo-curator
```

You also need the domain submodule installed (e.g. `pip install physicsnemo-curator[mesh]`).

## Usage

```bash
psnc
```

The wizard launches a full-screen TUI and walks through the pipeline
configuration across multiple screens.  You can navigate forward and backward
with on-screen buttons or keyboard shortcuts.  You can either build a new
pipeline interactively or load a previously saved one from YAML or JSON.

### Welcome Screen

The landing screen displays the PhysicsNeMo Curator branding and offers two
entry paths:

- **New Pipeline** — start building a pipeline from scratch
- **Load Config** — load a previously saved pipeline from a YAML or JSON file

### 1. Select Submodule

The wizard discovers registered submodules and shows which have their
dependencies installed:

```text
┌─────────────────────────────────────────────┐
│  Select Submodule                           │
│                                             │
│  ● mesh — Mesh data curation               │
│  ○ da — DataArray data curation             │
│  ○ atm — Atomic data curation (unavailable) │
└─────────────────────────────────────────────┘
```

Submodules whose dependencies are not installed are shown as unavailable.

### 2. Select Source

Choose from the registered sources for the selected submodule:

```text
┌─────────────────────────────────────────────┐
│  Select Source/Reader                       │
│                                             │
│  ● VTK Reader                               │
│  ○ DrivAerML                                │
│  ○ WindsorML                                │
└─────────────────────────────────────────────┘
```

You are then prompted for source-specific parameters (data location,
conversion options) through form fields:

```text
  Configure VTK Reader:
  input_path:     ./cfd_results/
  manifold_dim:   auto
  point_source:   vertices
  ✓ Found 42 item(s) in source
```

### 3. Select Filters

Choose zero or more filters with toggle switches:

```text
┌─────────────────────────────────────────────┐
│  Select Filters                             │
│                                             │
│  [✓] Mean Statistics                        │
│  [ ] Precision Cast                         │
│  [ ] Normalization                          │
└─────────────────────────────────────────────┘
```

Each selected filter's parameters are prompted in order through inline forms.

### 4. Select Sink

Choose the output writer:

```text
┌─────────────────────────────────────────────┐
│  Select Sink/Writer                         │
│                                             │
│  ● PhysicsNeMo Mesh Writer                  │
└─────────────────────────────────────────────┘
```

### 5. Summary & Execute

The summary screen displays the full pipeline configuration for review
before execution:

```text
  Pipeline Summary
  ────────────────
  Source:  VTK Reader (./cfd_results/)
  Filter:  Mean Statistics → stats.parquet
  Sink:    PhysicsNeMo Mesh Writer → ./output/

  42 items to process
```

Pressing **Execute** launches the pipeline with a full-screen Textual
progress display showing per-worker status, elapsed time, and a progress
bar.

### Result Screen

After execution completes, the result screen displays:

- Items processed, outputs written, errors encountered
- Database path for the pipeline run
- Options to open the dashboard, save the config, or exit

## Cache Management

Pipeline databases are stored in `~/.cache/psnc/` by default (see
{doc}`checkpointing` for how to change the location).  The wizard includes
a **Cache** screen accessible from the welcome screen that provides tools
for inspecting and managing these databases.

The `psnc cache` command group also provides CLI access:

### Show cache directory

```bash
psnc cache path
# ~/.cache/psnc
```

### List databases

```bash
psnc cache list
```

Displays a table of all databases with their hash prefix, creation time,
pipeline components, progress, and file size.

### Inspect a database

```bash
psnc cache info a1b2
```

Shows detailed metadata for a single database identified by hash prefix.

### Remove databases

```bash
# Remove by hash prefix
psnc cache rm a1b2

# Remove databases older than 7 days
psnc cache rm --older-than 7d

# Remove all databases (with confirmation)
psnc cache rm --all

# Skip confirmation prompt
psnc cache rm --all --yes
```

### Duration format

The `--older-than` flag accepts human-readable durations:

| Suffix | Meaning |
|--------|---------|
| `s`    | seconds |
| `m`    | minutes |
| `h`    | hours   |
| `d`    | days    |
| `w`    | weeks   |

Examples: `30m`, `12h`, `7d`, `2w`.

## Programmatic Equivalent

Everything the wizard does can be done in Python:

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

# Sequential with progress display (equivalent to wizard behaviour)
results = run_pipeline(pipeline)

# Or parallel across multiple cores
results = run_pipeline(pipeline, n_jobs=-1, backend="process_pool")

# Flush stateful filters (sequential only)
pipeline.filters[0].flush()
```

See {doc}`parallel` for details on `run_pipeline` and available backends.
