# PhysicsNeMo-Curator Tutorial: CGNS to NumPy

## Overview

This tutorial demonstrates how to use the PhysicsNeMo-Curator ETL pipeline to:

1. Extract physics simulation data from **CGNS (CFD General Notation System)** files.
2. Transform the data into standard **NumPy arrays** with configurable precision.
3. Write the processed data to efficient, compressed `.npz` files with sidecar metadata.

## 1. Create a Dataset

PhysicsNeMo-Curator works with well-defined formats and schemas. For this tutorial, we define a custom simulation dataset using:

* **Format**: CGNS (Computational Fluid Dynamics General Notation System)
* **Storage**: Local filesystem
* **Schema**: Each simulation run contains a mesh with the following fields:

| Field Name | Type | Description |
| --- | --- | --- |
| `coordinates` | `(N, 3)` | Spatial coordinates (x, y, z) of mesh points |
| `faces` | `(M, 4)` | Mesh connectivity information (triangulated) |
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

This will create a `tutorial_data/` directory containing 5 `.cgns` files (e.g., `run_001.cgns`).

## 2. The ETL Pipeline

The pipeline consists of four main components orchestrated to process files in parallel.

### A. Source: `CGNSDataSource`

* **File**: `cgns_data_source.py`
* **Function**: Reads CGNS files using `pyvista`. It extracts the mesh geometry (`coordinates`, `faces`) and all point data fields (`Temperature`, `Velocity`, etc.).

### B. Transformation: `CGNSToNumpyTransformation`

* **File**: `cgns_to_numpy_transformation.py`
* **Function**: Converts raw CGNS data into standard NumPy arrays.
* **Precision Control**: Configurable to output `float32` (default) or `float64`.
* **Vector Handling**: Automatically computes magnitude arrays for 2D/vector fields (e.g., `Velocity` -> `Velocity_magnitude`).
* **Statistics**: Calculates comprehensive statistics (min, max, mean, std) for all fields.



### C. Sink: `NumpyDataSource`

* **File**: `numpy_data_source.py`
* **Function**: Writes the transformed data to disk.
* **Data**: Saved as compressed `.npz` files (using `np.savez_compressed`).
* **Metadata**: Saved as separate `.json` sidecar files containing file info and calculated statistics.



### D. Validator: `TutorialValidator`

* **File**: `tutorial_validator.py`
* **Function**: Validates the input CGNS files before processing. It checks for valid mesh structure (points, cells), dimensions, and data integrity (NaN/Inf checks).

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
    cgns_to_numpy:
      _target_: cgns_to_numpy_transformation.CGNSToNumpyTransformation
      precision: "float32"  # or "float64"

  sink:
    _target_: numpy_data_source.NumpyDataSource

```

## 4. Run the Pipeline

To run the ETL pipeline, use the `run_etl.py` script. You must specify the input and output directories.

```bash
python run_etl.py \
  etl.source.input_dir=tutorial_data \
  etl.sink.output_dir=output_numpy

```

## 5. Output

After execution, the `output_numpy/` directory will contain paired files for each run:

1. **`run_001.npz`**: A compressed archive containing the NumPy arrays (`coordinates`, `faces`, `Temperature`, `Velocity`, etc.).
2. **`run_001.json`**: A JSON file containing metadata, such as:
```json
{
  "num_points": 1024,
  "precision": "<class 'numpy.float32'>",
  "Temperature_mean": 300.5,
  "Temperature_std": 12.1,
  "Velocity_magnitude_max": 15.2
}

```
