# PhysicsNeMo-Curator Ansys Tutorial

## Overview

This tutorial demonstrates how to use **PhysicsNeMo-Curator** to prepare training-ready datasets from Ansys Solvers using the [Ansys DPF library](https://github.com/ansys/pydpf-core) from [PyAnsys libraries](https://github.com/ansys) The pipeline implements an ETL (Extract, Transform, Load) process to:

1.  **Extract** data from Ansys `.rst` files.
2.  **Transform** the data into an optimized, AI-ready format (Zarr).
3.  **Write** the transformed data to disk efficiently.

## Prerequisites

To run this tutorial, you must install the `physicsnemo-curator` package and `ansys-dpf-core`.

```bash
pip install -e "../../[dev]" ansys-dpf-core --user
````

**Note on Real vs. Mock Data:**
This tutorial uses a script to generate **mock data** (pickled python files mimicking `.rst` structure). To use real Ansys `.rst` files, you require:

  * Ansys installation (2021 R1+).
  * A valid Ansys license.
  * PyDPF-Core configured with a DPF Server.

## Pipeline Components

The ETL pipeline consists of three main components:

### 1\. Source: `RstDataSource`

Reads the input simulation files.

  * **Input:** Directory containing `.rst` files.
  * **Function:** extracts coordinates, temperature fields, and heat flux data.

### 2\. Transformation: `RstToZarrTransformation`

Prepares the data for high-performance I/O and AI training.

  * **Type Conversion:** Converts arrays to `float32` for storage efficiency.
  * **Chunking:** Splits data into manageable chunks (e.g., 1000 points) for parallel access
  * **Compression:** Applies Blosc/Zstd compression (Level 5).

### 3\. Sink: `ZarrDataSource`

Writes the processed data to disk.

  * **Output:** Creates `.zarr` stores containing the array data and metadata attributes.
  * **Features:** Checks for existing stores to avoid redundant writes.

## Usage Guide

### Step 1: Generate Mock Data

Run the provided generation script to create sample thermal simulation files.

```python
# Run the mock generation script included in the tutorial
# Generates 5 files in ./mock_thermal_data/
```

### Step 2: Configuration

Create a configuration file (e.g., `rst_to_zarr.yaml`) to define the pipeline parameters.

```yaml
etl:
  processing:
    num_processes: 4  # Adjust based on CPU cores
  
  source:
    _target_: rst_data_source.RstDataSource
    input_dir: ???
  
  transformations:
    rst_to_zarr:
      _target_: rst_to_zarr_transformation.RstToZarrTransformation
      chunk_size: 1000
      compression_level: 5
  
  sink:
    _target_: zarr_data_source.ZarrDataSource
    output_dir: ???
```


### Step 3: Run the Pipeline

Execute the pipeline using the orchestrator script (`run_etl.py`). This sets up multiprocessing and instantiates the components.

```bash
python run_etl.py --config-dir ./ \
  --config-name rst_to_zarr \
  etl.source.input_dir=mock_thermal_data \
  etl.sink.output_dir=output_zarr
```


## Output Structure

The pipeline generates a Zarr store for each input simulation file.

```text
output_zarr/
├── thermal_sim_001.zarr/
│   ├── coordinates/      # (N, 3) float32
│   ├── temperature/      # (N,) float32
│   ├── heat_flux/        # (N, 3) float32
│   └── .zattrs           # Metadata (solver info, units, etc.)
├── thermal_sim_002.zarr/
...
```

## Customizing to Ansys Solver Outputs

To process real Ansys `.rst` files, modify the `RstDataSource.read_file` method to use `ansys.dpf.core. To process other Ansys solver output formats, you can extend and customize this recipe to the appropriate format such as xxx.

```python
from ansys.dpf import core as dpf

def read_file(self, filename: str) -> Dict[str, Any]:
    filepath = self.input_dir / f"{filename}.rst"
    model = dpf.Model(str(filepath))
    
    # Extract data using DPF operators
    mesh = model.metadata.meshed_region
    coords = np.array(mesh.nodes.coordinates_field.data)
    
    # Extract results (Temperature, etc.)
    temp_op = model.results.temperature()
    temperature = np.array(temp_op.outputs.fields_container()[0].data)
    
    return {
        "coordinates": coords,
        "temperature": temperature,
        "metadata": {...}
    }
```
