# Mesh Submodule

The `curator.mesh` submodule provides pipeline components for reading,
transforming, and writing {class}`physicsnemo.mesh.Mesh` objects.

## Installation

```bash
pip install physicsnemo-curator[mesh]
# or
uv sync --group mesh
```

This installs: physicsnemo, pyvista, pyarrow, torch, and their transitive
dependencies.

## Components

### VTKSource

{class}`~physicsnemo_curator.mesh.sources.vtk.VTKSource` reads VTK files via a
{class}`~physicsnemo_curator.core.store.FileStore` and converts each to a
{class}`physicsnemo.mesh.Mesh` using {func}`physicsnemo.mesh.io.from_pyvista`.

The primary constructor accepts a `FileStore` directly — this is the
recommended pattern for programmatic use:

```python
from physicsnemo_curator.core.store import LocalFileStore, FsspecFileStore
from physicsnemo_curator.mesh.sources.vtk import VTKSource

# Local directory
store = LocalFileStore("./data/", extensions=frozenset({".vtk", ".vtu"}))
source = VTKSource(store=store, manifold_dim=2)

# Remote (HuggingFace Hub, S3, HTTPS, ...)
store = FsspecFileStore(
    "hf://datasets/neashton/drivaerml/run_1/slices",
    extensions=frozenset({".vtp"}),
)
source = VTKSource(store=store)

# Custom store
source = VTKSource(store=my_database_store, point_source="cell_centroids")
```

Convenience classmethods are also available for quick one-liners:

- {meth}`~physicsnemo_curator.mesh.sources.vtk.VTKSource.from_path` — wraps a
  {class}`~physicsnemo_curator.core.store.LocalFileStore`
- {meth}`~physicsnemo_curator.mesh.sources.vtk.VTKSource.from_url` — wraps a
  {class}`~physicsnemo_curator.core.store.FsspecFileStore`

```python
# One-liner shortcuts
source = VTKSource.from_path("./data/")
source = VTKSource.from_url("hf://datasets/neashton/drivaerml/run_1/slices")
```

**Supported formats:** `.vtk`, `.vtp`, `.vtu`, `.vts`, `.vtm`

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store` | `FileStore` | *required* | File store providing indexed access to VTK files |
| `manifold_dim` | `int \| "auto"` | `"auto"` | Target manifold dimension (0–3) |
| `point_source` | `str` | `"vertices"` | `"vertices"` or `"cell_centroids"` |
| `warn_on_lost_data` | `bool` | `True` | Warn when data arrays are discarded |

**Convenience classmethod parameters (``from_path`` / ``from_url``):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` / `url` | `str` | *required* | Local path or fsspec URL |
| `file_pattern` | `str` | `"*"` / `"**"` | Glob pattern for filtering files |
| `storage_options` | `dict` | `None` | fsspec auth/config (``from_url`` only) |
| `cache_storage` | `str` | `None` | Local cache dir (``from_url`` only) |

All conversion parameters from the constructor table above (``manifold_dim``,
``point_source``, ``warn_on_lost_data``) are also accepted.

**Manifold dimensions:**

| Dim | Topology | Cell Shape | Notes |
|-----|----------|------------|-------|
| 0 | Point cloud | None | Vertices only, no connectivity |
| 1 | Line mesh | Edges (2 vertices) | Extracted from mesh topology |
| 2 | Surface mesh | Triangles (3 vertices) | Auto-triangulated if needed |
| 3 | Volume mesh | Tetrahedra (4 vertices) | Auto-tetrahedralized if needed |

**Point source modes:**

- `"vertices"` (default): Mesh vertices become points. `point_data` is
  preserved. Cell topology is determined by `manifold_dim`.
- `"cell_centroids"`: Cell centroids become points. `cell_data` is mapped
  to `point_data`. Only `manifold_dim` 0 and 1 are valid. Avoids expensive
  tetrahedralization for large polyhedral CFD meshes.

**Examples:**

```python
from physicsnemo_curator.core.store import LocalFileStore, FsspecFileStore
from physicsnemo_curator.mesh.sources.vtk import VTKSource

# --- Primary pattern: construct store + inject ---

# Local directory with extension filter
store = LocalFileStore("./data/", extensions=frozenset({".vtk", ".vtu"}))
source = VTKSource(store=store)

# Read as volume mesh
store = LocalFileStore("./volumes/")
source = VTKSource(store=store, manifold_dim=3)

# Use cell centroids for CFD polyhedral meshes
store = LocalFileStore("./cfd/")
source = VTKSource(
    store=store,
    point_source="cell_centroids",
    warn_on_lost_data=False,
)

# HuggingFace Hub dataset
store = FsspecFileStore(
    "hf://datasets/neashton/drivaerml/run_1/slices",
    extensions=frozenset({".vtp"}),
)
source = VTKSource(store=store)

# S3 (public bucket)
store = FsspecFileStore(
    "s3://my-bucket/cfd-data/",
    storage_options={"anon": True},
)
source = VTKSource(store=store, manifold_dim=2)

# --- Convenience classmethods (one-liners) ---

source = VTKSource.from_path("./data/")
source = VTKSource.from_path("./data/", file_pattern="timestep_*")
source = VTKSource.from_url("hf://datasets/neashton/drivaerml/run_1/slices")
source = VTKSource.from_url("s3://my-bucket/cfd-data/", storage_options={"anon": True})
```

### MeanFilter

{class}`~physicsnemo_curator.mesh.filters.mean.MeanFilter` computes the spatial mean
of every field in `point_data` and `cell_data` for each mesh, accumulates
the results in memory, and writes them to a Parquet file on `flush()`.

The mesh is yielded unchanged (pass-through).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output` | `str` | *required* | Parquet file path for statistics |

**Output columns:**

| Column | Description |
|--------|-------------|
| `n_points` | Number of points in the mesh |
| `n_cells` | Number of cells in the mesh |
| `point_data/{field}` | Mean of each point data field |
| `cell_data/{field}` | Mean of each cell data field |

Meshes with different field names are handled gracefully — missing columns
are filled with `NULL` in the Parquet output.

**Example:**

```python
from physicsnemo_curator.mesh.filters.mean import MeanFilter

filt = MeanFilter(output="stats.parquet")

# Use in a pipeline
pipeline = source.filter(filt).write(sink)
for i in range(len(pipeline)):
    pipeline[i]

# Write accumulated statistics
filt.flush()

# Read results
import pyarrow.parquet as pq
table = pq.read_table("stats.parquet")
print(table.to_pandas())
```

### MeshSink

{class}`~physicsnemo_curator.mesh.sinks.mesh_writer.MeshSink` saves
{class}`physicsnemo.mesh.Mesh` objects using the native tensordict
memory-mapped format ({meth}`Mesh.save`).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | `str` | *required* | Directory for output files |

**Output naming:** Each mesh is saved to
`{output_dir}/mesh_{index:04d}_{seq}` where `index` is the source item
index and `seq` is the sequence number within that item (for sources that
yield multiple meshes).

Saved meshes can be loaded back with:

```python
from physicsnemo.mesh import Mesh
mesh = Mesh.load("./output/mesh_0000_0")
```

**Example:**

```python
from physicsnemo_curator.mesh.sinks.mesh_writer import MeshSink

sink = MeshSink(output_dir="./output/")

# Use in a pipeline
pipeline = source.filter(filt).write(sink)
paths = pipeline[0]  # ['./output/mesh_0000_0']
```

## Full Pipeline Example

```python
from physicsnemo_curator.core.store import LocalFileStore, FsspecFileStore
from physicsnemo_curator.mesh.sources.vtk import VTKSource
from physicsnemo_curator.mesh.filters.mean import MeanFilter
from physicsnemo_curator.mesh.sinks.mesh_writer import MeshSink

# Local data with explicit store
store = LocalFileStore("./cfd_results/", extensions=frozenset({".vtk", ".vtu"}))
pipeline = (
    VTKSource(store=store, manifold_dim=2)
    .filter(MeanFilter(output="stats.parquet"))
    .write(MeshSink(output_dir="./output/"))
)

# Or remote data from HuggingFace with explicit store
store = FsspecFileStore("hf://datasets/neashton/drivaerml/run_1/slices")
pipeline = (
    VTKSource(store=store)
    .filter(MeanFilter(output="stats.parquet"))
    .write(MeshSink(output_dir="./output/"))
)

# Or use convenience classmethods for quick pipelines
pipeline = (
    VTKSource.from_path("./cfd_results/", manifold_dim=2)
    .filter(MeanFilter(output="stats.parquet"))
    .write(MeshSink(output_dir="./output/"))
)

# Execute
for i in range(len(pipeline)):
    paths = pipeline[i]
    print(f"Item {i}: {paths}")

# Finalize statistics
pipeline.filters[0].flush()
```
