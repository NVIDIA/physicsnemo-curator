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

## Step 1: Read VTK Files

The {class}`~curator.mesh.sources.vtk.VTKSource` reads VTK files (`.vtk`,
`.vtp`, `.vtu`, `.vts`, `.vtm`) from a directory and converts each to a
{class}`physicsnemo.mesh.Mesh` using
{func}`physicsnemo.mesh.io.from_pyvista`.

```python
from curator.mesh.sources.vtk import VTKSource

source = VTKSource(input_path="./cfd_results/")
print(f"Found {len(source)} VTK files")

# Access a single mesh (lazy — returns a generator)
mesh = next(source[0])
print(f"Points: {mesh.n_points}, Cells: {mesh.n_cells}")
```

### Conversion Options

`VTKSource` exposes the full `from_pyvista` conversion interface:

```python
# Read as point cloud (no cell topology)
source = VTKSource(input_path="./data/", manifold_dim=0)

# Read volume meshes (tetrahedralize)
source = VTKSource(input_path="./volumes/", manifold_dim=3)

# Use cell centroids as points (avoids tetrahedralization for CFD)
source = VTKSource(
    input_path="./cfd/",
    point_source="cell_centroids",
    warn_on_lost_data=False,
)
```

| Parameter | Values | Description |
|-----------|--------|-------------|
| `manifold_dim` | `"auto"`, `0`, `1`, `2`, `3` | Target topology dimension |
| `point_source` | `"vertices"`, `"cell_centroids"` | What becomes mesh points |
| `warn_on_lost_data` | `True` / `False` | Warn on discarded data arrays |

## Step 2: Add Filters

Filters transform the data stream.  The
{class}`~curator.mesh.filters.mean.MeanFilter` computes per-field spatial
means and accumulates them into a Parquet summary table:

```python
from curator.mesh.filters.mean import MeanFilter

mean_filter = MeanFilter(output="stats.parquet")
```

Filters are generators — they can yield zero, one, or many items per input.
`MeanFilter` is a pass-through: it computes statistics and yields the mesh
unchanged.

## Step 3: Write Output

The {class}`~curator.mesh.sinks.mesh_writer.MeshSink` saves meshes in the
physicsnemo native tensordict format:

```python
from curator.mesh.sinks.mesh_writer import MeshSink

sink = MeshSink(output_dir="./output/")
```

## Step 4: Build and Run the Pipeline

Chain the components together using the fluent API:

```python
pipeline = (
    VTKSource(input_path="./cfd_results/")
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

## Step 5: Inspect Results

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
pipeline = (
    VTKSource(input_path="./data/")
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
