# PhysicsNeMo-Curator Tutorial: EnSight Gold to Zarr

## Overview

This tutorial demonstrates how to use the PhysicsNeMo-Curator ETL pipeline to:

1. Extract physics simulation data from **EnSight Gold** (`.case`) files.
2. Transform the data into an optimized, AI-model-ready format (Zarr).
3. Write the transformed data to disk efficiently.

## 1. Create a Dataset

PhysicsNeMo-Curator works with well-defined formats and schemas. For this tutorial, we define a custom simulation dataset using:

* **Format**: EnSight Gold (a common CFD post-processing format).
* **Storage**: Local filesystem (consisting of a `.case` file and associated geometry/variable files).
* **Schema**: Each simulation run contains a mesh with the following fields:

| Field Name | Type | Description |
| --- | --- | --- |
| `coordinates` | `(N, 3)` | Spatial coordinates (x, y, z) of mesh points |
| `faces` | `(M, 4)` | Mesh connectivity (triangulated surface) |
| `Temperature` | `(N,)` | Scalar temperature field |
| `Pressure` | `(N,)` | Scalar pressure field |
| `Velocity` | `(N, 3)` | 3D velocity vector field |
| `Density` | `(N,)` | Scalar density field |
| `Vorticity` | `(N,)` | Scalar vorticity field |

### Generate Sample Data

We have provided a script to generate 5 simulation runs with random physics-like data on a spherical mesh.

To generate the data:

```bash
python generate_sample_data.py

```

This will create a `tutorial_data/` directory containing 5 sets of EnSight Gold files (e.g., `run_001.case` and its data files).

## 2. The ETL Pipeline

The pipeline consists of four main components orchestrated to process files in parallel.

### A. Source: `EnSightDataSource`

* **File**: `ensight_data_source.py`
* **Function**: Reads EnSight Gold files using `vtk` and `pyvista`.
* **MultiBlock Handling**: Automatically flattens complex MultiBlock datasets often found in EnSight exports.
* **Surface Extraction**: Extracts and merges surfaces from multiple parts.
* **Variable Handling**: Handles inconsistent variables across blocks by padding missing fields with zeros during the merge process.



### B. Transformation: `EnSightToZarrTransformation`

* **File**: `ensight_to_zarr_transformation.py`
* **Function**: Converts the raw mesh data into a Zarr-optimized format.
* **Chunking & Compression**: Applies chunks and Zstd compression for efficient storage.
* **Scalar Statistics**: Computes min, max, and mean for scalar fields (e.g., `Temperature`).
* **Vector Statistics**: Computes magnitudes and statistics for vector fields (e.g., `Velocity` -> `Velocity_magnitude`).



### C. Sink: `ZarrDataSource`

* **File**: `zarr_data_source.py`
* **Function**: Writes the transformed data into individual Zarr stores (directories). It preserves the metadata and structure defined by the transformation.

### D. Validator: `EnSightTutorialValidator`

* **File**: `tutorial_validator.py`
* **Function**: Validates the input EnSight files before processing. It checks for:
* **Readability**: Ensures the `.case` file and its dependencies can be opened.
* **Structure**: Verifies minimum point/cell counts and 3D spatial dimensions.
* **Data Integrity**: Checks for the presence of point data and ensures no NaNs or infinite values exist.



## 3. Configuration

The pipeline is configured using Hydra via `tutorial_config.yaml`.

```yaml
etl:
  processing:
    num_processes: 2  # Parallel execution

  validator:
    _target_: tutorial_validator.EnSightTutorialValidator
    validation_level: "fields"

  source:
    _target_: ensight_data_source.EnSightDataSource

  transformations:
    ensight_to_zarr:
      _target_: ensight_to_zarr_transformation.EnSightToZarrTransformation
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

**Note:** If you are processing a very large dataset where validation takes too long, you can skip the validation step by adding `~etl.validator` to the command.

## 5. Output

After execution, the `output_zarr/` directory will contain a separate Zarr store for each run (e.g., `run_001.zarr`). Each store will contain:

* **Arrays**:
* `coordinates`
* `faces`
* `Temperature`, `Pressure`, `Density`, `Vorticity`
* `Velocity`
* `Velocity_magnitude` (derived)


* **Metadata (`.zattrs`)**:
* Simulation statistics (e.g., `Temperature_max`, `Velocity_magnitude_mean`)
* Technical details (`num_points`, `chunk_size`, `compression`)