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

**Supported formats:** `.vtk`, `.vtp`, `.vtu`, `.vts`, `.vtm`, `.stl`

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` | *required* | Path to local directory containing VTK files |
| `file_pattern` | `str` | `"**"` | Glob pattern for filtering files |
| `manifold_dim` | `int \| "auto" \| list[dict]` | `"auto"` | Target manifold dimension (0–3), or per-path rule list |
| `point_source` | `str \| list[dict]` | `"vertices"` | `"vertices"` or `"cell_centroids"` (or per-path rule list) |
| `warn_on_lost_data` | `bool` | `True` | Warn when data arrays are discarded |
| `backend` | `str` | `"pyvista"` | Reading backend: `"pyvista"` or `"rust"` (faster VTU/VTP) |
| `key_filters` | `list[dict] \| None` | `None` | Per-path data-array include/exclude rules (reader-level) |
| `volume_pattern` | `str \| None` | `None` | Filename glob for volume files (domain-mesh mode) |
| `boundary_pattern` | `str \| None` | `None` | Filename glob for boundary files (domain-mesh mode) |
| `boundary_name` | `str` | `"vehicle"` | Boundary key for the paired DomainMesh boundary |
| `boundary_generator` | object \| `None` | `None` | Optional BC generator applied to each paired DomainMesh |

**Reading backends:**

- `"pyvista"` (default): full-featured reading via `from_pyvista`.
- `"rust"`: native reader for `.vtu` / `.vtp` (much faster I/O), with a
  transparent fallback to PyVista for unsupported files/configs. Builds the
  `Mesh` (points, cells, point/cell data, or cell centroids) directly from
  raw arrays.

**Per-path conversion and array filtering:**

`manifold_dim` and `point_source` accept either a scalar (applied to every
file) or a list of `{"pattern": glob, "value": ...}` rules selected per file
(longest matching pattern wins). `key_filters` drops/keeps named data arrays
**at the reader level** (so filtered fields are never materialised — critical
for very large volume `.vtu` files):

```python
source = VTKSource(
    "./dataset/",
    backend="rust",
    manifold_dim=[
        {"pattern": "**/volume_*", "value": 0},     # volumes -> point cloud
        {"pattern": "**/boundary_*", "value": 2},   # surfaces -> triangulated
    ],
    point_source=[{"pattern": "**/volume_*", "value": "cell_centroids"}],
    key_filters=[
        {"path_pattern": "**/volume_*.vtu", "mode": "exclude", "keys": ["NodeID"]},
    ],
)
```

**Domain-mesh mode (volume + boundary -> DomainMesh):**

When both `volume_pattern` and `boundary_pattern` are set, files are paired
**by parent directory** into a {class}`physicsnemo.mesh.domain_mesh.DomainMesh`
per index (one volume interior + one boundary). Unpaired files (e.g. STLs)
fall back to standalone `Mesh`:

```python
source = VTKSource(
    "./dataset/",
    volume_pattern="volume_*.vtu",
    boundary_pattern="boundary_*.vtp",
    boundary_name="vehicle",
)
domain = next(source[0])  # DomainMesh(interior=..., boundaries={"vehicle": ...})
```

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
| `group_readable` | `bool` | `False` | `chmod g+r` the written output |

Writes are **atomic**: each mesh is written to a temporary directory in the
same parent and then renamed into place, so an interrupted run never leaves a
partial/corrupt output directory behind. `Mesh` objects are saved as `.pmsh`
and `DomainMesh` objects as `.pdmsh`.

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

### Transform filters

Several generic, in-pipeline transform filters complement the statistics
filters above. All are importable from
`physicsnemo_curator.domains.mesh.filters` and apply to both `Mesh` and
`DomainMesh` (interior + every boundary).

| Filter | Purpose |
|--------|---------|
| `PrecisionFilter` | Convert field precision (e.g. fp64 → fp32 / fp16 / bf16) |
| `CleanFilter` | `Mesh.clean`: merge duplicate points, drop duplicate cells / unused points |
| `PointDataToCellDataFilter` | Move `point_data` onto cells by per-cell averaging |
| `GlobalDataFilter` | Inject constant `global_data` (e.g. `U_inf`, `rho_inf`, …) |
| `RandomPermutationFilter` | Shuffle point/cell ordering reproducibly |
| `FieldSelectFilter` | Keep/drop fields after conversion (post-hoc) |
| `BoundaryInjectionFilter` | Synthesize + inject CFD-domain boundaries (see below) |

```python
from physicsnemo_curator.domains.mesh.filters import (
    CleanFilter, GlobalDataFilter, PointDataToCellDataFilter,
)

pipeline = (
    source
    .filter(CleanFilter())
    .filter(PointDataToCellDataFilter())  # surface point_data -> cell_data
    .filter(GlobalDataFilter(values={"U_inf": [30.0, 0.0, 0.0], "rho_inf": 1.225}))
    .write(sink)
)
```

### VTISource and GridSidecarSink (structured grids)

VTK ImageData (`.vti`) describes a uniform rectilinear grid (origin + spacing
+ dimensions) and does not fit the unstructured `Mesh` model.
{class}`~physicsnemo_curator.domains.mesh.sources.vti.VTISource` reads each
`.vti` file into a {class}`tensordict.TensorDict` of **dense N-D field
tensors** instead:

- `point_data` — sub-TensorDict with `batch_size = [Nz, Ny, Nx]`; scalar
  fields are `(Nz, Ny, Nx)`, vector fields `(Nz, Ny, Nx, C)` (VTK x-fastest
  ordering).
- `cell_data` — sub-TensorDict with `batch_size = [Cz, Cy, Cx]`.
- `grid` — non-batched metadata: `origin`, `spacing`, `dimensions`,
  `direction`.

{class}`~physicsnemo_curator.domains.mesh.sinks.grid_sidecar.GridSidecarSink`
writes the grid as a tensordict memmap **sidecar** beside the mesh outputs
(default `{relpath}/{stem}.grid`), reloadable with
`TensorDict.load_memmap`:

```python
from physicsnemo_curator.domains.mesh.sources.vti import VTISource
from physicsnemo_curator.domains.mesh.sinks.grid_sidecar import GridSidecarSink

pipeline = VTISource("./grids/").write(GridSidecarSink(output_dir="./out/"))
```

### Boundary-condition injection

Curated `DomainMesh` files often carry only the geometry surface (`vehicle`)
plus interior + global data, lacking the CFD-domain outer boundaries
(inlet / outlet / walls / symmetry). The
`physicsnemo_curator.domains.mesh.boundaries` subsystem synthesizes those
boundaries from the known domain geometry and injects them, preserving
`interior` / `vehicle` / `global_data`.

Datasets are specialized purely by choosing a `BoundaryGenerator` and its
constants:

- `BoxTunnelBoundaries` — rectangular wind tunnel (inlet/outlet/slip/no_slip);
  `z_floor` inferred per sample from the geometry boundary.
- `HemisphereBoundaries` — hemispherical open-road domain (inlet/outlet split
  by freestream direction + a constrained-Delaunay symmetry disk).

Use it either as a standalone filter on any `DomainMesh` stream (including
reading existing `.pdmsh`), or as the `boundary_generator` hook on
`VTKSource` domain-mesh mode:

```python
from physicsnemo_curator.domains.mesh.boundaries import HemisphereBoundaries
from physicsnemo_curator.domains.mesh.filters import BoundaryInjectionFilter

gen = HemisphereBoundaries(freestream_key="U_inf")
pipeline = source.filter(BoundaryInjectionFilter(gen, check_watertight=True)).write(sink)
```

## Dependencies

The `mesh` domain depends on:

| Package | Purpose |
|---------|---------|
| [physicsnemo](https://github.com/NVIDIA/PhysicsNeMo) | `Mesh` tensorclass and I/O utilities |
| [pyvista](https://docs.pyvista.org/) | VTK file reading and mesh manipulation |
| [pyarrow](https://arrow.apache.org/docs/python/) | Parquet I/O for statistics and metadata |
| [torch](https://pytorch.org/) | Tensor operations (required by physicsnemo) |
