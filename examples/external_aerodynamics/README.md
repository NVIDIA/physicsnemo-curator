# How to curate data for External Aerodynamics model with PhysicsNeMo-Curator

This document describes the External Aerodynamics data processing pipeline in PhysicsNeMo Curator.
**NOTE** This can be used as is for the DoMINO and Transolver models.

## Overview

The External Aerodynamics ETL pipeline processes automotive aerodynamics simulation data for machine learning training. It:

**Reads:** CFD simulation data in multiple schemas (DriveSim, DrivAerML, AhmedML)

- Geometry files (STL format)
- Volume mesh data (VTU format)
- Surface mesh data (VTP format)
- Flow field variables (pressure, velocity, turbulence)

**Transforms:** Raw simulation data into ML-optimized format (Zarr or NumPy)

- Extracts and normalizes field variables (pressure coefficients, wall shear stress)
- Processes geometry data (coordinates, face connectivity, areas)
- Computes derived quantities and reference values
- Applies mesh decimation for efficiency (optional)

**Outputs:** Training-ready datasets in NumPy or Zarr format

- Optimized for DoMINO and Transolver model training workflows
- Compressed and chunked for efficient data loading
- Preserves metadata and processing parameters

This pipeline handles the complex data engineering required to convert raw CFD outputs into
datasets suitable for training AI models for external aerodynamics applications.

## Download DrivAerML Dataset

Here, we're providing examples on how to download the DrivAerML dataset from different sources.
[DrivAerML](https://caemldatasets.org/drivaerml/) is the dataset that the DoMINO model was trained on.

This high-fidelity, open-source (CC-BY-SA) public dataset is specifically designed for automotive aerodynamics research.
It comprises 500 parametrically morphed variants of the widely utilized DrivAer notchback generic vehicle.
For more technical details about this dataset, please refer to their [paper](https://arxiv.org/pdf/2408.11969).

Download the DrivAer ML dataset using the provided [download_hugging_face_dataset.sh](./download_hugging_face_dataset.sh) script:

```bash
# Download a few runs (1-5) to default directory
./download_hugging_face_dataset.sh

# Download specific runs to a custom directory
./download_hugging_face_dataset.sh -d ./my_data -s 1 -e 100

# Get help
./download_hugging_face_dataset.sh --help
```

**Note**: The default for both options downloads a subset of runs (1-5),
while the full version contain 500 runs.

## Running the Curator

Example of the command line that launches Curator configured for DrivAerML dataset:

```bash
export PYTHONPATH=$PYTHONPATH:examples &&
physicsnemo-curator-etl                         \
    --config-dir=examples/config                \
    --config-name=external_aero_etl_drivaerml   \
    etl.source.input_dir=/data/drivaerml/       \
    etl.sink.output_dir=/data/drivaerml.processed.surface \
    etl.common.model_type=surface
```

To run on AhmedML dataset:

```bash
export PYTHONPATH=$PYTHONPATH:examples &&
physicsnemo-curator-etl                     \
    --config-dir=examples/config            \
    --config-name=external_aero_etl_ahmedml \
    etl.source.input_dir=/data/ahmedml/     \
    etl.sink.output_dir=/data/ahmedml.processed.surface \
    etl.common.model_type=surface
```

### Some useful configuration options

- `etl.sink.overwrite_existing`: can be either `true` (default) or `false`.
    When set to `false`, the Curator will not overwrite existing files.
    This is useful for incremental runs of the Curator.
- `etl.common.model_type`: can be `surface` (default) or `volume` or `combined`.
    This option is used to specify the type of model that will be trained on the
    dataset.
- **Output format**: To switch from Zarr (default) to NumPy, please use the `serialization_format=numpy` flag.

Please refer to the [config file](../../../examples/config/external_aero_etl_drivaerml.yaml) for more
options.

### Injecting and chaining transformations

The pipeline provides a flexible processor injection system that allows you to
customize and extend the data transformation pipeline. Each transformation class (STL,
Surface, Volume) supports injecting custom processors that are applied **after** the
default processing.

#### How Processors Work

1. **Default Processing**: Each transformation always applies required default processing
   first (e.g., extracting mesh centers, computing normals)
2. **Custom Processors**: Additional processors are then applied in sequence, allowing
   you to chain multiple operations
3. **Function-Based**: Processors are Python functions that take data as input and
   return modified data

#### Available Processors

**Geometry Processors** (for STL transformation):

- `update_geometry_data_to_float32`: Convert geometry data to float32 precision

**Surface Processors** (for surface transformation):

- `decimate_mesh`: Reduce mesh resolution (see Mesh Decimation Options below)
- `normalize_surface_normals`: Normalize surface normal vectors to unit length
- `non_dimensionalize_surface_fields`: Convert fields to non-dimensional form
- `update_surface_data_to_float32`: Convert surface data to float32 precision

**Volume Processors** (for volume transformation):

- `non_dimensionalize_volume_fields`: Convert volume fields to non-dimensional form
- `update_volume_data_to_float32`: Convert volume data to float32 precision

#### Configuration Examples

**Single Processor (no parameters):**

```yaml
surface_preprocessing:
  _target_: examples.external_aerodynamics.data_transformations.ExternalAerodynamicsSurfaceTransformation
  surface_processors:
    - _target_: examples.external_aerodynamics.external_aero_surface_data_processors.normalize_surface_normals
      _partial_: true
```

**Single Processor (with parameters):**

```yaml
surface_preprocessing:
  surface_processors:
    - _target_: examples.external_aerodynamics.external_aero_surface_data_processors.non_dimensionalize_surface_fields
      _partial_: true
      air_density: 1.205  # kg/m³
      stream_velocity: 30.0  # m/s
```

**Chaining Multiple Processors:**

```yaml
surface_preprocessing:
  surface_processors:
    # 1. Normalize normals
    - _target_: examples.external_aerodynamics.external_aero_surface_data_processors.normalize_surface_normals
      _partial_: true
    # 2. Then non-dimensionalize fields
    - _target_: examples.external_aerodynamics.external_aero_surface_data_processors.non_dimensionalize_surface_fields
      _partial_: true
      air_density: 1.205
      stream_velocity: 30.0
    # 3. Finally convert to float32
    - _target_: examples.external_aerodynamics.external_aero_surface_data_processors.update_surface_data_to_float32
      _partial_: true
```

#### Creating Custom Processors

You can create your own processors by writing functions that follow this signature:

```python
from examples.external_aerodynamics.schemas import ExternalAerodynamicsExtractedDataInMemory

def my_custom_processor(
    data: ExternalAerodynamicsExtractedDataInMemory,
    my_param: float = 1.0,  # Optional parameters
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Custom processing logic."""
    # Modify data as needed
    data.surface_fields = data.surface_fields * my_param
    return data
```

Then reference it in your config:

```yaml
surface_processors:
  - _target_: my_module.my_custom_processor
    _partial_: true
    my_param: 2.0
```

**Key Points:**

- Always use `_partial_: true` when configuring processors with Hydra
- Processors are applied in the order they appear in the configuration
- Each processor receives the output of the previous processor
- Processors must return the modified data object
- Default processing always runs first, before any custom processors

### Mesh Decimation Options

PhysicsNeMo-Curator supports mesh decimation with the following options:

- **Algorithms**:
  - `decimate_pro` (recommended): Advanced decimation algorithm that preserves mesh features and quality.
  Based on [PyVista's decimate_pro](https://docs.pyvista.org/api/core/_autosummary/pyvista.polydatafilters.decimate_pro).
  - `decimate`: Basic [decimation](https://docs.pyvista.org/api/core/_autosummary/pyvista.polydatafilters.decimate)
  algorithm. Note: May hang on meshes larger than 400K triangles.

- **Target Reduction**: Specify reduction ratio between 0 and 1 (a value of 0.9 will leave 10% of the original number of vertices).

Each algorithm provides an additional set of parameters that control features specific to the algorithm.
For example, `decimate_pro` supports the following parameters:

- **Additional Parameters**:
  - `preserve_topology`: Controls topology preservation. If enabled, mesh splitting and hole elimination will not occur.
  - `feature_angle`: Angle used to define what an edge is.
  - and others.

See the algorithm documentation for more information.

Example configuration:

```yaml
etl:
  source:
    decimation:
        algo: decimate_pro
        reduction: 0.5
        preserve_topology: true
```

For more details on the decimation algorithms and their parameters, refer to:

- [PyVista decimate_pro documentation](https://docs.pyvista.org/api/core/_autosummary/pyvista.polydatafilters.decimate_pro)
- [PyVista decimate documentation](https://docs.pyvista.org/api/core/_autosummary/pyvista.polydatafilters.decimate)

> **Note**: Currently, only surface meshes are supported for decimation.
Volume meshes are not supported, and in case of `combined` model type,
only the surface part will be decimated.

### Custom Transformations

This ETL pipeline is intended to be configurable. As such, you can extend it in the following ways:

- Create your own transformation (use [data_transformations.py](./data_transformations.py) as a template)
- Stack multiple transformations on top of each other.
This is already demonstrated in [external_aero_etl_drivaerml.yaml](../../../examples/config/external_aero_etl_drivaerml.yaml),
where `DoMINOPreprocessingTransformation` is applied followed by `DoMINOZarrTransformation`.

Please use these guidelines to create your own ETL pipeline to train a DoMINO or Transolver model
on a different dataset or on a non external aerodynamics application!

### Model Training

Train your External Aerodynamics Model on your own data by following the
[DoMINO example in PhysicsNeMo](https://github.com/NVIDIA/physicsnemo/tree/main/examples/cfd/external_aerodynamics/domino)
or the
[Transolver example in PhysicsNeMo](https://github.com/NVIDIA/physicsnemo/tree/main/examples/cfd/external_aerodynamics/transolver).
