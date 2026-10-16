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

### NastranSource

{class}`~physicsnemo_curator.domains.mesh.sources.nastran.NastranSource` reads
Nastran `.deck`, `.bdf`, `.dat`, `.nas`, and `.pch` models with optional paired
`.op2` real normal-mode results. Install the additional reader dependencies
with Python 3.12 (pyNastran 1.4 requires NumPy 1.x):

```bash
pip install 'physicsnemo-curator[mesh,nastran]'
# From a source checkout:
uv sync --python 3.12 --extra mesh --extra nastran
```

```python
from physicsnemo_curator.domains.mesh import NastranSource, MeshSink

# Each subdirectory contains a deck and its OP2 result file.
source = NastranSource("./modal_cases/")
mesh = next(source[0])
pipeline = source.write(MeshSink(output_dir="./curated/"))
paths = pipeline[0]
```

Directory discovery is recursive and deterministic. A single deck file can
also be passed directly. Each source index is one deck and yields one mesh
per real eigenvector result table, usually one per subcase, with all modes
stored together. For geometry alone use `read_results=False`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_path` | required | Deck file or directory |
| `file_pattern` | `"**/*"` | Deck discovery glob within a directory |
| `op2_pattern` | `""` | Relative OP2 filename template with `{stem}` |
| `read_results` | `True` | Require paired OP2 modal results |
| `encoding` | `"latin1"` | Deck text encoding; use `"cp932"` for Japanese text |
| `punch` | `False` | Bulk-data-only input without executive/case control; automatic for `.pch` |

An empty `op2_pattern` first selects a same-stem OP2, then a unique sibling
OP2. Thus a directory containing `Input_model.deck` and `Output_model.op2`
works directly. Multiple candidate OP2 files require an explicit template
such as `op2_pattern="{stem}.op2"` or a relative filename. Missing results
raise an error. `relative_path(index)` supports sink naming templates.

**Geometry and fields:** CTETRA4 and CTETRA10 elements with PSOLID/MAT1 are
supported. A 10-node tetrahedron becomes eight linear tetrahedra, retaining
all midside nodes. This is a piecewise linear approximation of curved
quadratic elements. Parent cell IDs and material fields are repeated on the
eight children. Coordinates use the Nastran basic frame; translational and
rotational modes are transformed from rectangular GRID CD frames to basic
axes. Results are matched by GRID ID, independently of deck/OP2 ordering.
Incomplete, duplicate, or mismatched GRID result tables raise an error.

| Location | Fields | Shape |
|----------|--------|-------|
| `point_data` | `node_id`, `input_coordinate_id`, `output_coordinate_id` | `(N,)` |
| `point_data` | `output_coordinate_basis` (local-to-basic rotation, row vectors) | `(N, 3, 3)` |
| `point_data` | `permanent_spc` (GRID PS, six DOF mask) | `(N, 6)` |
| `point_data` | `mode_shape`, `mode_rotation` | `(N, M, 3)` |
| `cell_data` | `element_id`, `property_id`, `material_id` | `(E,)` |
| `cell_data` | `young_modulus`, `poisson_ratio`, `density` | `(E,)` |
| `global_data` | `mode_numbers`, `eigenvalues`, `frequencies` | `(M,)` |
| `global_data` | `subcase_id`, `spc_id`, `mpc_id` | scalar |

Units and solver eigenvector normalization are preserved. `frequencies`
contains solver-reported cycles per model time unit (Hz if time is seconds).
Signed eigenvalues are retained, including slightly negative rigid-body
modes. No extra normalization, deformation, or unit conversion is applied.

**Constraints:** `global_data["constraints"]` holds numeric tables in
original Nastran IDs. Definitions are retained separately from the selected
subcase's `spc_id`/`mpc_id`; inactive sets are not merged into active ones.

| Key | Columns / meaning |
|-----|-------------------|
| `spc` | `(set_id, node_id, component)` for each SPC/SPC1 DOF |
| `spc_value` | Enforced displacement aligned with `spc` rows |
| `spcadds`, `mpcadds` | `(combined_set_id, child_set_id)` |
| `mpc_set_id` | Set ID per MPC equation |
| `mpc_offsets` | Offsets into the term arrays, including final end offset |
| `mpc_terms` | `(node_id, component)` per MPC term |
| `mpc_coefficient` | Coefficient per MPC term |
| `rbe2` | `(element_id, independent_node_id, dependent_node_id, components)` |
| `rbe2_alpha` | Thermal expansion coefficient per RBE2 dependent-node row |
| `rbe2_tref` | Reference temperature per RBE2 dependent-node row |

SPC/MPC components refer to each node's original output coordinate system;
its ID and basis are retained in `output_coordinate_id` and
`output_coordinate_basis`. RBE2 components use the Nastran
component code (e.g. `123456`). These tables preserve definitions; the
reader does not solve or enforce the constraints.

The initial reader supports tetrahedral real modal analysis. Other element
and material families, RBE3, cylindrical/spherical output frames, complex
modes, transient results, stress/strain, and element energies are outside
this reader's scope. Unsupported mesh/constraint cards raise an error;
only eigenvector tables are requested from OP2.

See `examples/cae/nastran/` for a command-line curation pipeline.

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

### MeshVTKSink

{class}`~physicsnemo_curator.domains.mesh.sinks.mesh_vtk.MeshVTKSink` exports
meshes for ParaView and VTK-based interchange. Select `file_format="vtu"`
(default) or `"vtk"` for a complete unstructured volume, or `"vtp"` to
extract its boundary surface. VTP omits interior nodes and volume cells.

```python
from physicsnemo_curator.domains.mesh import MeshVTKSink, NastranSource

pipeline = NastranSource("./modal_cases/").write(
    MeshVTKSink("./vtk/", file_format="vtu", naming_template="{relpath}/{stem}_{seq}")
)
paths = pipeline[0]
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `output_dir` | required | Output directory |
| `file_format` | `"vtu"` | `"vtu"`, `"vtk"`, or boundary-surface `"vtp"` |
| `naming_template` | `"mesh_{index:04d}_{seq}"` | Supports index/seq and source relpath/stem |
| `binary` | `True` | Use `False` for ASCII |

Each mesh/subcase gets its own file. Nastran modes become separate
three-component vectors such as `mode_shape_m0001`, `mode_rotation_m0001`,
and `mode_shape_m0007`, using original mode numbers. Select the appropriate
vector in ParaView's **Warp By Vector** to inspect a mode; modes are not
time steps. Other tensor components are flattened into VTK arrays. Integer
IDs and float64 precision are retained; bool and half-precision fields are
stored as uint8 and float32, respectively.

Global tensors, including constraint tables, are written to VTK Field Data.
Nested paths use `/`, e.g. `constraints/mpc_terms` (`~` and `/` in individual
keys are escaped as `~0` and `~1`). The JSON `physicsnemo_schema` field
records original paths, shapes and dtypes, and describes empty arrays that
VTK cannot store directly. For VTP, point/cell fields and their shapes refer
to the surface subset; global constraints still use the full model's
original IDs. VTK readers do not automatically reconstruct these fields as
a native TensorDict or apply the constraint equations. Use `.pmsh` for the
native structured training representation.

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

VTK ImageData (`.vti`) describes a uniform rectilinear grid (origin + spacing +
dimensions) and does not fit the unstructured `Mesh` model.
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
