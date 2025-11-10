# External Aerodynamics Data Processing Reference

This is a reference for the entire data processing pipeline for the following external aerodynamics models in PhysicsNeMo:

1. [DoMINO](https://github.com/NVIDIA/physicsnemo/blob/main/examples/cfd/external_aerodynamics/domino/README.md)
2. [Transolver](https://github.com/NVIDIA/physicsnemo/blob/main/examples/cfd/external_aerodynamics/transolver/README.md)

## Input Data Structure

The module supports three datasets (formats + schema combinations):

### 1. DriveSim Dataset

The dataset should contain at least the following files.

```bash
run_001/
├── body.stl                       # Geometry file
└── VTK/
    └── simpleFoam_steady_3000/    # Solution data
        ├── internal.vtu           # Volume mesh data
        └── boundary/              # Surface data
            └── aero_suv.vtp       # Surface mesh data
```

### 2. DrivAerML Dataset

The dataset should contain at least the following files.

```bash
run_1/
├── drivaer_1.stl                  # Geometry file
├── volume_1.vtu                   # Volume mesh data
└── boundary_1.vtp                 # Surface mesh data
```

### 3. AhmedML Dataset

The dataset should contain at least the following files.

```bash
run_1/
├── ahmed_1.stl                    # Geometry file
├── volume_1.vtu                   # Volume mesh data
└── boundary_1.vtp                 # Surface mesh data
```

## Output Data Structure

The processed data is stored in either numpy or zarr format:

### 1. Numpy Format

```bash
output/
└── run_1.npz
```

The `npz` file contains a dictionary with the following keys:

- `stl_coordinates`: (float32) Vertex coordinates from STL geometry
- `stl_centers`: (float32) Cell center coordinates from STL
- `stl_faces`: (float32) Face connectivity from STL
- `stl_areas`: (float32) Face areas from STL
- `surface_mesh_centers`: (float32, optional) Surface mesh cell centers
- `surface_normals`: (float32, optional) Surface mesh cell normals
- `surface_areas`: (float32, optional) Surface mesh cell areas
- `surface_fields`: (float32, optional) Surface field data
- `volume_fields`: (float32, optional) Volume field data
- `volume_mesh_centers`: (float32, optional) Volume mesh cell centers
- `filename`: (str) Original case directory name
- `stream_velocity`: (float) Reference velocity
- `air_density`: (float) Reference density

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
└── run_1.zarr
    ├── stl_areas                # Face areas
    ├── stl_centers              # Cell centers
    ├── stl_coordinates          # Vertex coordinates
    ├── stl_faces                # Face connectivity
    ├── surface_areas            # Cell areas (surface, optional)
    ├── surface_fields           # Field data (surface, optional)
    ├── surface_mesh_centers     # Cell centers (surface, optional)
    ├── surface_normals          # Cell normals (surface, optional)
    ├── volume_fields            # Field data (volume, optional)
    ├── volume_mesh_centers      # Cell centers (volume, optional)
    ├── .zattrs                  # Dataset attributes
    └── .zgroup
```

Notes:

- All numerical data is stored as float32
- Surface and volume data are optional based on the model type
- Zarr format includes automatic compression (using Blosc/zstd) and chunking
- Chunk sizes are optimized for ~1MB per chunk

## Processing Pipeline

1. **Data Extraction**
   - Reads VTK files (VTU for volume, VTP for surface), and STL files
   - Extracts specified field variables (pressure, velocity, etc.)
   - Processes geometry data from STL files
   - Optionally reduces output mesh size (mesh decimation).

2. **Data Transformation**
   - Normalizes field variables
   - Computes derived quantities
   - Validates data consistency

3. **Data Loading**
   - Serializes processed data to numpy arrays or zarr store
   - Stores metadata and processing parameters

## Field Variables

### Surface Data

Surface fields are non-dimensionalized by (density * stream_velocity^2):

- Pressure coefficient (Cp)
- Wall shear stress components
- Surface normals (unit vectors)
- Surface areas

### Volume Data

Fields are non-dimensionalized as follows:

- Velocity components (U): normalized by stream_velocity
- Pressure (p): normalized by (density * stream_velocity^2)
- Turbulent quantities (k, omega, etc.): normalized by (stream_velocity * length_scale)

Note: The exact fields processed depend on the `surface_variables` and
`volume_variables` configurations provided to the data source.
See `examples/external_aerodynamics/config/external_aero_etl_drivaerml.yaml`
and `examples/external_aerodynamics/config/variables/*.yaml` files for examples.

### Reference Values

- stream_velocity: Free-stream velocity
- density: Air density
- length_scale: Characteristic length (maximum dimension of the geometry)

For more details, see the configuration examples in `examples/external_aerodynamics/config/external_aero_etl_drivaerml.yaml`.
