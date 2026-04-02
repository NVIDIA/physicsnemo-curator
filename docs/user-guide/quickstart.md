# Quickstart

This guide walks through building your first ETL pipeline with PhysicsNeMo
Curator using the mesh submodule.

## Prerequisites

Install the package with mesh support:

```bash
pip install physicsnemo-curator[mesh]
# or with uv:
uv sync --group mesh
```

## Step 1: Create a FileStore

A {class}`~physicsnemo_curator.core.store.FileStore` maps integer indices to local file
paths.  It decouples *where* data lives from *how* it's read.

```python
from physicsnemo_curator.core.store import LocalFileStore, FsspecFileStore

# Local directory
store = LocalFileStore("./cfd_results/", extensions=frozenset({".vtk", ".vtu"}))
print(f"Found {len(store)} files")

# Or a remote dataset (HuggingFace Hub, S3, HTTPS, ...)
store = FsspecFileStore(
    "hf://datasets/neashton/drivaerml/run_1/slices",
    extensions=frozenset({".vtp"}),
)
print(f"Found {len(store)} remote files")
```

## Step 2: Read VTK Files

The {class}`~physicsnemo_curator.mesh.sources.vtk.VTKSource` accepts a
{class}`~physicsnemo_curator.core.store.FileStore` and converts each VTK file to a
{class}`physicsnemo.mesh.Mesh` using
{func}`physicsnemo.mesh.io.from_pyvista`.

```python
from physicsnemo_curator.mesh.sources.vtk import VTKSource

source = VTKSource(store=store)
print(f"Source has {len(source)} items")

# Access a single mesh (lazy — returns a generator)
mesh = next(source[0])
print(f"Points: {mesh.n_points}, Cells: {mesh.n_cells}")
```

You can also use convenience classmethods that create the store internally:

```python
# Quick one-liners
source = VTKSource.from_path("./cfd_results/")
source = VTKSource.from_url("hf://datasets/neashton/drivaerml/run_1/slices")
```

### Conversion Options

`VTKSource` exposes the full `from_pyvista` conversion interface.  These
options apply regardless of which store is used:

```python
# Read as point cloud (no cell topology)
source = VTKSource(store=store, manifold_dim=0)

# Read volume meshes (tetrahedralize)
source = VTKSource(store=store, manifold_dim=3)

# Use cell centroids as points (avoids tetrahedralization for CFD)
source = VTKSource(
    store=store,
    point_source="cell_centroids",
    warn_on_lost_data=False,
)
```

| Parameter | Values | Description |
|-----------|--------|-------------|
| `manifold_dim` | `"auto"`, `0`, `1`, `2`, `3` | Target topology dimension |
| `point_source` | `"vertices"`, `"cell_centroids"` | What becomes mesh points |
| `warn_on_lost_data` | `True` / `False` | Warn on discarded data arrays |

## Step 3: Add Filters

Filters transform the data stream.  The
{class}`~physicsnemo_curator.mesh.filters.mean.MeanFilter` computes per-field spatial
means and accumulates them into a Parquet summary table:

```python
from physicsnemo_curator.mesh.filters.mean import MeanFilter

mean_filter = MeanFilter(output="stats.parquet")
```

Filters are generators — they can yield zero, one, or many items per input.
`MeanFilter` is a pass-through: it computes statistics and yields the mesh
unchanged.

## Step 4: Write Output

The {class}`~physicsnemo_curator.mesh.sinks.mesh_writer.MeshSink` saves meshes in the
physicsnemo native tensordict format:

```python
from physicsnemo_curator.mesh.sinks.mesh_writer import MeshSink

sink = MeshSink(output_dir="./output/")
```

## Step 5: Build and Run the Pipeline

Chain the components together using the fluent API:

```python
from physicsnemo_curator.core.store import LocalFileStore
from physicsnemo_curator.mesh.sources.vtk import VTKSource
from physicsnemo_curator.mesh.filters.mean import MeanFilter
from physicsnemo_curator.mesh.sinks.mesh_writer import MeshSink

store = LocalFileStore("./cfd_results/", extensions=frozenset({".vtk", ".vtu"}))
pipeline = (
    VTKSource(store=store)
    .filter(MeanFilter(output="stats.parquet"))
    .write(MeshSink(output_dir="./output/"))
)

print(f"Pipeline has {len(pipeline)} items")

# Process each item lazily
for i in range(len(pipeline)):
    paths = pipeline[i]
    print(f"  Item {i} → {paths}")

# Flush stateful filters (writes Parquet)
mean_filter = pipeline.filters[0]
mean_filter.flush()
```

Each call to `pipeline[i]` processes only that item through the full
Source → Filter → Sink chain and returns the output file path(s).

### Using `run_pipeline` (recommended)

For batch execution, use {func}`~physicsnemo_curator.core.parallel.run_pipeline`
instead of a manual loop.  It handles progress bars, index management,
and supports parallel backends:

```python
from physicsnemo_curator import run_pipeline

# Sequential with progress bar
results = run_pipeline(pipeline)
print(f"Wrote {sum(len(r) for r in results)} files")

# Parallel across 4 worker processes
results = run_pipeline(pipeline, n_jobs=4, backend="process_pool")

# Use all CPUs with automatic backend selection
results = run_pipeline(pipeline, n_jobs=-1)

# Process a subset
results = run_pipeline(pipeline, indices=[0, 1, 2])
```

```{note}
Stateful filter side-effects (like `MeanFilter.flush()`) are **not** merged
across processes when running in parallel.  Use sequential execution
(`n_jobs=1`) when you need to collect filter state, or see
{doc}`parallel` for details.
```

## Step 6: Inspect Results

```python
import pyarrow.parquet as pq

# Read the statistics table
table = pq.read_table("stats.parquet")
print(table.to_pandas())

# Load a saved mesh
from physicsnemo.mesh import Mesh
mesh = Mesh.load("./output/mesh_0000_0")
print(f"Loaded mesh: {mesh.n_points} points, {mesh.n_cells} cells")
```

## Chaining Multiple Filters

Filters compose naturally:

```python
store = LocalFileStore("./data/")
pipeline = (
    VTKSource(store=store)
    .filter(FilterA())
    .filter(FilterB())
    .filter(FilterC())
    .write(MySink(output_dir="./out/"))
)
```

Each filter receives the output generator of the previous one, forming a
lazy processing chain.

## Using the CLI

If you have the CLI extra installed (`pip install physicsnemo-curator[cli]`),
you can build pipelines interactively:

```bash
curator
```

This launches a guided workflow that prompts you to select a submodule,
source, filters, and sink, then executes the pipeline.  See {doc}`cli` for
details.
