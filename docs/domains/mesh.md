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

{class}`~physicsnemo_curator.domains.mesh.sources.vtk.VTKSource` reads VTK files from a
local directory and converts each to a
{class}`physicsnemo.mesh.Mesh` using {func}`physicsnemo.mesh.io.from_pyvista`.

The constructor takes a path string directly:

```python
from physicsnemo_curator.domains.mesh.sources.vtk import VTKSource

# Local directory (discovers VTK files automatically)
source = VTKSource("./data/", manifold_dim=2)

# With a custom glob pattern
source = VTKSource("./data/", file_pattern="**", manifold_dim="auto")

# Cell centroid mode for CFD polyhedral meshes
source = VTKSource("./cfd/", point_source="cell_centroids")
```

For remote datasets (HuggingFace Hub), use purpose-built dataset sources
such as {class}`~physicsnemo_curator.domains.mesh.sources.drivaerml.DrivAerMLSource`:

```python
from physicsnemo_curator.domains.mesh.sources.drivaerml import DrivAerMLSource

source = DrivAerMLSource(mesh_type="boundary")
```

**Supported formats:** `.vtk`, `.vtp`, `.vtu`, `.vts`, `.vtm`

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` | *required* | Path to local directory containing VTK files |
| `file_pattern` | `str` | `"**"` | Glob pattern for filtering files |
| `manifold_dim` | `int \| "auto"` | `"auto"` | Target manifold dimension (0–3) |
| `point_source` | `str` | `"vertices"` | `"vertices"` or `"cell_centroids"` |
| `warn_on_lost_data` | `bool` | `True` | Warn when data arrays are discarded |

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
from physicsnemo_curator.domains.mesh.sources.vtk import VTKSource

# Local directory (auto-discovers VTK files)
source = VTKSource("./data/")

# Read as volume mesh
source = VTKSource("./volumes/", manifold_dim=3)

# Use cell centroids for CFD polyhedral meshes
source = VTKSource(
    "./cfd/",
    point_source="cell_centroids",
    warn_on_lost_data=False,
)

# Custom glob pattern to select a subset of files
source = VTKSource("./data/", file_pattern="timestep_*")
```

For remote datasets from HuggingFace Hub, use the dedicated dataset sources:

```python
from physicsnemo_curator.domains.mesh.sources.drivaerml import DrivAerMLSource

source = DrivAerMLSource(mesh_type="boundary")
```

### MeanFilter

{class}`~physicsnemo_curator.domains.mesh.filters.mean.MeanFilter` computes the spatial mean
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
from physicsnemo_curator.domains.mesh.filters.mean import MeanFilter

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

{class}`~physicsnemo_curator.domains.mesh.sinks.mesh_writer.MeshSink` saves
{class}`physicsnemo.mesh.Mesh` objects using the native tensordict
memory-mapped format ({meth}`Mesh.save`).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | `str` | *required* | Directory for output files |
| `naming_template` | `str \| None` | `None` | Format string for output names (see below) |

**Output naming:** By default each mesh is saved to
`{output_dir}/mesh_{index:04d}_{seq}` where `index` is the source item
index and `seq` is the sequence number within that item (for sources that
yield multiple meshes).

A custom `naming_template` can be provided using Python format-string
syntax with the following placeholders:

| Placeholder | Description |
|-------------|-------------|
| `{index}` | Source item index |
| `{seq}` | Sequence number within that item (for multi-mesh sources) |
| `{relpath}` | Relative path of the source file (from the source directory) |
| `{stem}` | File stem (name without extension) of the source file |

Standard format specs are supported (e.g. `{index:04d}`).  The template
is used literally — include any file extension you need.

Saved meshes can be loaded back with:

```python
from physicsnemo.mesh import Mesh
mesh = Mesh.load("./output/mesh_0000_0")
```

**Examples:**

```python
from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink

# Default naming
sink = MeshSink(output_dir="./output/")
pipeline = source.filter(filt).write(sink)
paths = pipeline[0]  # ['./output/mesh_0000_0']

# Custom naming for MeshReader compatibility
sink = MeshSink(
    output_dir="./output/",
    naming_template="boundary_{index}.vtp.pmsh",
)
paths = pipeline[0]  # ['./output/boundary_0.vtp.pmsh']
```

## Dependencies

The `mesh` domain depends on:

| Package | Purpose |
|---------|---------|
| [physicsnemo](https://github.com/NVIDIA/PhysicsNeMo) | `Mesh` tensorclass and I/O utilities |
| [pyvista](https://docs.pyvista.org/) | VTK file reading and mesh manipulation |
| [pyarrow](https://arrow.apache.org/docs/python/) | Parquet I/O for statistics and metadata |
| [torch](https://pytorch.org/) | Tensor operations (required by physicsnemo) |
