# PhysicsNeMo-Curator Tutorial: EnSight Gold to NumPy

## Overview

This tutorial demonstrates how to use the PhysicsNeMo-Curator ETL pipeline to:

1. Extract physics simulation data from **EnSight Gold** (`.case`) files.
2. Transform the data into standard **NumPy arrays** with configurable precision.
3. Write the processed data to efficient, compressed `.npz` files with sidecar metadata.

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



### B. Transformation: `EnSightToNumpyTransformation`

* **File**: `ensight_to_numpy_transformation.py`
* **Function**: Converts raw EnSight data into standard NumPy arrays.
* **Precision Control**: Configurable to output `float32` (default) or `float64`.
* **Vector Handling**: Automatically computes magnitude arrays for 2D/vector fields (e.g., `Velocity` -> `Velocity_magnitude`).
* **Statistics**: Calculates comprehensive statistics (min, max, mean, std) for all fields.



### C. Sink: `NumpyDataSource`

* **File**: `numpy_data_source.py`
* **Function**: Writes the transformed data to disk.
* **Data**: Saved as compressed `.npz` files (using `np.savez_compressed`).
* **Metadata**: Saved as separate `.json` sidecar files containing file info and calculated statistics.



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
    ensight_to_numpy:
      _target_: ensight_to_numpy_transformation.EnSightToNumpyTransformation
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

**Note:** If you are processing a large dataset where validation takes too long, you can skip the validation step by adding `~etl.validator` to the command:

```bash
python run_etl.py etl.source.input_dir=... etl.sink.output_dir=... ~etl.validator

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