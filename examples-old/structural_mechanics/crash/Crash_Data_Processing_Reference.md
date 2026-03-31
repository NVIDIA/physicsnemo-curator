# Crash Simulation Data Processing Reference

This is a reference for the entire data processing pipeline for crash simulation models in [PhysicsNeMo](https://github.com/NVIDIA/physicsnemo/blob/main/examples/structural_mechanics/crash/README.md).

## Input Data Structure

The module supports LS-DYNA crash simulation data with the following structure:

### LS-DYNA Dataset

```bash
crash_sims/
├── Run100/
│   ├── d3plot          # Required: binary mesh/displacement data
│   └── run100.k        # Optional: part thickness definitions
├── Run101/
│   ├── d3plot
│   └── run101.k
└── ...
```

#### d3plot File (Required)

LS-DYNA binary format containing:

- Node coordinates (reference configuration at t=0)
- Node displacements over time (temporal trajectory)
- Shell element connectivity (triangles and/or quads)
- Part IDs for each element
- Multiple timesteps of deformation data

#### .k File (Optional)

LS-DYNA keyword format containing:

- `*PART` definitions linking parts to sections
- `*SECTION_SHELL` definitions with thickness values

If no `.k` file is found, node thickness defaults to 0 for all nodes.

## Output Data Structure

The processed data is stored in either VTP or Zarr format:

### 1. VTP Format (Visualization Toolkit PolyData)

```bash
output/
├── Run100.vtp
├── Run101.vtp
└── ...
```

Each `.vtp` file contains:

**Point coordinates** (`mesh.points`):

- Reference coordinates at t=0 (undisplaced configuration)
- Shape: `[N, 3]` where N is number of nodes

**Point data arrays** (`mesh.point_data`):

- `thickness`: `[N]` per-node thickness values (float32)
- `displacement_t0.000`: `[N, 3]` displacement at t=0 (zeros)
- `displacement_t0.005`: `[N, 3]` displacement at t=0.005 seconds
- `displacement_t0.010`: `[N, 3]` displacement at t=0.010 seconds
- ... (one displacement field per timestep)

**Cell connectivity** (`mesh.faces`):

- Triangle and quad face connectivity
- Used for visualization and edge derivation

Notes:

- All numerical data is stored as float32
- Time step naming reflects the actual simulation time (configurable, default: 0.005s)
- Actual positions at timestep t: `mesh.points + mesh.point_data[f'displacement_t{t:.3f}']`

### 2. Zarr Format

Zarr is an open source project to develop specifications and software for storage of large N-dimensional typed arrays.
Zarr provides support for storage using distributed systems like cloud object stores,
and enables efficient I/O for parallel computing applications.

Key features relevant for AI model training:

- **Chunk multi-dimensional arrays** along any dimension
- **Store arrays** in memory, on disk, inside a Zip file, on S3, etc.
- **Read and write arrays concurrently** from multiple threads or processes
- **Organize arrays into hierarchies** via annotatable groups
- **Built-in compression** reduces storage requirements while maintaining fast access
- **Language agnostic**: Accessible from Python, C, C++, Rust, Javascript, Java, and other frameworks

For more information, see the [Zarr project website](https://zarr.dev/).

```bash
output/
├── Run100.zarr/
│   ├── mesh_pos            # Temporal positions
│   ├── thickness           # Node thickness
│   ├── edges               # Pre-computed edges
│   ├── .zattrs             # Dataset attributes
│   └── .zgroup
├── Run101.zarr/
└── ...
```

Each `.zarr` store contains:

**Arrays:**

- `mesh_pos`: `[T, N, 3]` (float32) - Temporal node positions (absolute coordinates, not displacements)
- `thickness`: `[N]` (float32) - Per-node thickness values
- `edges`: `[E, 2]` (int64) - Pre-computed edge connectivity (undirected, no self-loops)

**Attributes** (`.zattrs`):

- `filename`: Original run directory name
- `num_timesteps`: Number of timesteps (T)
- `num_nodes`: Number of nodes after filtering (N)
- `num_edges`: Number of unique edges (E)

**Compression:**

- All arrays use Blosc compression with zstd codec
- Compression level configurable (1-9, default: 3)
- Typical compression ratio: 3-5x for crash simulation data

**Chunking:**

- Adaptive chunking based on `chunk_size_mb` parameter (default: 1.0 MB)
- Chunks calculated to balance I/O efficiency and metadata overhead
- Temporal dimension typically chunked to enable efficient time-slice access

Notes:

- VTP stores **reference coordinates + displacements**; Zarr stores **absolute positions** directly
- VTP stores mesh connectivity for visualization; Zarr stores **pre-computed edges** for GNN training
- All heavy preprocessing (node filtering, edge building) is done once during curation
- Both formats store the same filtered, remapped node set

## Processing Pipeline

The ETL pipeline consists of three stages:

### 1. Data Extraction (Source)

Component: `CrashD3PlotDataSource`

Reads LS-DYNA simulation outputs:

- Opens `d3plot` binary files using `lasso.dyna.D3plot`
- Extracts node coordinates and time-varying displacements
- Reads shell element connectivity (triangles and quads)
- Extracts part IDs for each element
- Optionally parses `.k` files for thickness definitions

### 2. Data Transformation

Component: `CrashDataTransformation`

Applies the following transformations in sequence:

#### a) Wall Node Filtering

- Computes displacement variation for each node: `max_disp - min_disp` over all timesteps
- Identifies "wall" nodes with variation below `wall_threshold` (default: 1.0)
- Wall nodes are considered rigid structures and filtered out to reduce dataset size
- Typical filtering: removes 30-60% of nodes depending on simulation

#### b) Node Index Remapping

- Creates compact, contiguous indices for retained nodes
- Original indices → Filtered indices (0, 1, 2, ...)
- Ensures efficient memory layout and indexing

#### c) Thickness Computation

- Parses `.k` files to extract part-to-section mappings
- Parses section definitions to get per-part thickness values
- Maps element part IDs to thickness values
- Computes per-node thickness by averaging incident element thicknesses
- Handles nodes connected to multiple parts gracefully

#### d) Edge Building

- Derives unique edges from mesh connectivity
- For each shell element (triangle/quad), extracts all boundary pairs
- Collects unique undirected edges (no duplicates)
- Does **not** add self-loops (added later by datapipe if needed for GNN)

#### e) Connectivity Remapping

- Remaps mesh connectivity to use compacted node indices
- Filters edges to retain only those between kept nodes
- Updates all arrays to match filtered node set

### 3. Data Loading (Sink)

Two sink options available:

#### VTP Sink

Component: `CrashVTPDataSource`

- Creates one VTP file per run
- Stores reference coordinates as mesh points
- Stores displacement fields for all timesteps
- Stores thickness as point data
- Uses temp-then-rename pattern for atomic writes

#### Zarr Sink

Component: `CrashZarrDataSource`

- Creates one Zarr store per run
- Stores absolute temporal positions (`mesh_pos`)
- Stores thickness and pre-computed edges
- Applies adaptive chunking based on `chunk_size_mb`
- Uses Blosc/zstd compression
- Uses temp-then-rename pattern for atomic writes

## Field Variables

### Node Features

The pipeline supports arbitrary per-node features. Currently implemented:

#### Thickness

- **Source:** Parsed from `.k` files (`*PART` and `*SECTION_SHELL` keywords)
- **Computation:** Per-node thickness = average of incident element thicknesses
- **Units:** Original units from LS-DYNA (typically millimeters)
- **Default:** 0.0 if no `.k` file found
- **Shape:** `[N]` (one value per node)
- **Usage:** Structural property used by ML models to predict deformation response

#### Extensibility

The pipeline is designed to support additional features:

- Material properties (Young's modulus, Poisson's ratio)
- Initial stress state
- Contact/boundary conditions
- Part/component labels

To add new features:

1. Extract data in `CrashD3PlotDataSource.read_file()`
2. Store in `CrashExtractedDataInMemory` schema
3. Add to sink write logic
4. Configure in PhysicsNeMo datapipe `features` list

### Temporal Data

**Positions:**

- **VTP:** Reference coordinates + per-timestep displacement fields
- **Zarr:** Direct storage of absolute positions at all timesteps
- **Shape:** `[T, N, 3]` where T is number of timesteps

**Timesteps:**

- Number of timesteps extracted from d3plot file
- Time step size configurable in VTP output (default: 0.005 seconds)
- Typical crash simulations: 10-100 timesteps over 0.05-0.5 seconds

### Graph Structure

**Edges:**

- Derived from shell element connectivity (triangles and quads)
- Stored as `[E, 2]` array of node index pairs
- Undirected (no self-loops in stored format)
- Self-loops and symmetrization added by PhysicsNeMo datapipe if needed for GNN

**Node Filtering Impact:**

- Edge building occurs **after** node filtering
- Only edges between retained (non-wall) nodes are kept
- Ensures graph connectivity matches the filtered node set

## Integration with PhysicsNeMo

The processed data integrates with PhysicsNeMo training pipelines through three readers:

### Readers

1. **d3plot reader** (`d3plot_reader.py`):
   - Reads raw d3plot files directly
   - Performs on-the-fly filtering and edge building
   - Useful for prototyping without preprocessing

2. **VTP reader** (`vtp_reader.py`):
   - Reads pre-processed VTP files from Curator
   - Reconstructs positions from reference + displacements
   - Dynamically extracts all point data fields
   - Builds edges from cell connectivity

3. **Zarr reader** (`zarr_reader.py`):
   - Reads pre-processed Zarr stores from Curator
   - Loads positions and edges directly (no reconstruction)
   - Fastest option for large-scale training
   - Leverages Zarr's chunking for efficient I/O

### Data Contract

All readers return the same data structure to PhysicsNeMo:

**Inputs `x` (dictionary):**

- `x['coords']`: `[N, 3]` positions at t=0
- `x['features']`: `[N, F]` concatenated node features in config-specified order

**Targets `y`:**

- `[N, (T-1)*3]` positions from t1..tT flattened along feature dimension

**Features:**

- Feature order determined by PhysicsNeMo `conf/datapipe/*.yaml` configuration
- Features concatenated in the order specified
- Arrays can be `[N]` or `[N, K]`; datapipe promotes and concatenates automatically

**Recommended workflow:**

1. Use Curator to preprocess raw d3plot → VTP or Zarr once
2. Use corresponding reader for all training/validation
3. Optionally use d3plot reader for quick prototyping on raw data
