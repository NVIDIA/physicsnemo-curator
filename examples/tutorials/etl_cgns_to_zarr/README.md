# PhysicsNeMo-Curator Tutorial: CGNS to Zarr

## Overview

This tutorial demonstrates how to use the PhysicsNeMo-Curator ETL pipeline to:

1. Extract physics simulation data from **CGNS (CFD General Notation System)** files.
2. Transform the data into an optimized, AI-model-ready format (Zarr).
3. Write the transformed data to disk efficiently.

## 1. Create a Dataset

PhysicsNeMo-Curator works with well-defined formats and schemas. For this tutorial, we define a custom simulation dataset using:

* **Format**: CGNS (Computational Fluid Dynamics General Notation System)
* **Storage**: Local filesystem
* **Schema**: Each simulation run contains a mesh with the following fields:

| Field Name | Type | Description |
| --- | --- | --- |
| `coordinates` | `(N, 3) float32` | Spatial coordinates (x, y, z) of mesh points |
| `faces` | `(M, 4) int32` | Mesh connectivity information |
| `Temperature` | `(N,) float32` | Scalar temperature field |
| `Pressure` | `(N,) float32` | Scalar pressure field |
| `Velocity` | `(N, 3) float32` | 3D velocity vector field |
| `Density` | `(N,) float32` | Scalar density field |
| `Vorticity` | `(N,) float32` | Scalar vorticity field |

### Generate Sample Data

We have provided a script to generate 5 simulation runs with random physics-like data on a spherical mesh.

To generate the data:

```bash
python generate_sample_data.py

```

This will create a `tutorial_data/` directory containing 5 `.cgns` files (e.g., `run_001.cgns`).

## 2. The ETL Pipeline

The pipeline consists of four main components orchestrated to process files in parallel.

### A. Source: `CGNSDataSource`

* **File**: `cgns_data_source.py`
* **Function**: Reads CGNS files using `pyvista`. It extracts the mesh geometry (`coordinates`, `faces`) and all point data fields (`Temperature`, `Velocity`, etc.).

### B. Transformation: `CGNSToZarrTransformation`

* **File**: `cgns_to_zarr_transformation.py`
* **Function**: Converts the raw mesh data into a Zarr-optimized format.
* Applies chunking and compression (Zstd).
* Calculates derived statistics (min, max, mean) for all scalar fields.
* Calculates magnitude arrays and statistics for vector fields (e.g., `Velocity_magnitude`).



### C. Sink: `ZarrDataSource`

* **File**: `zarr_data_source.py`
* **Function**: Writes the transformed data into individual Zarr stores (directories). It preserves the metadata and structure defined by the transformation.

### D. Validator: `TutorialValidator`

* **File**: `tutorial_validator.py`
* **Function**: Validates the input CGNS files before processing. It checks for:
* Valid mesh structure (minimum points and cells).
* Correct spatial dimensions (3D).
* Presence of required field data.
* Data integrity (no NaNs or infinite values).



## 3. Configuration

The pipeline is configured using Hydra via `tutorial_config.yaml`.

```yaml
etl:
  processing:
    num_processes: 2  # Parallel execution

  validator:
    _target_: tutorial_validator.TutorialValidator
    validation_level: "fields"

  source:
    _target_: cgns_data_source.CGNSDataSource

  transformations:
    cgns_to_zarr:
      _target_: cgns_to_zarr_transformation.CGNSToZarrTransformation
      chunk_size: 500
      compression_level: 3

  sink:
    _target_: zarr_data_source.ZarrDataSource

```

## 4. Run the Pipeline

To run the ETL pipeline, use the `run_etl.py` script. You must specify the input and output directories.

```bash
python run_etl.py \
  etl.source.input_dir=tutorial_data \
  etl.sink.output_dir=output_zarr

```

## 5. Output

After execution, the `output_zarr/` directory will contain a separate Zarr store for each run (e.g., `run_001.zarr`). Each store will contain:

* **Arrays**:
* `coordinates`
* `faces`
* `Temperature`, `Pressure`, `Density`, `Vorticity`
* `Velocity`
* `Velocity_magnitude` (derived)


* **Metadata (`.zattrs`)**:
* Simulation statistics (e.g., `Temperature_mean`, `Velocity_magnitude_max`)
* Technical details (`chunk_size`, `compression`)
