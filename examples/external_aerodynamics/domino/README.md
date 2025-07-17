# How to curate data for DoMINO model with PhysicsNeMo-Curator

This document describes the DoMINO data processing pipeline in PhysicsNeMo Curator.

## What this pipeline does

The DoMINO ETL pipeline processes automotive aerodynamics simulation data for machine learning training. It:

**Reads:** CFD simulation data in multiple formats (DriveSim, DrivAerML, AhmedML)

- Geometry files (STL format)
- Volume mesh data (VTU format)
- Surface mesh data (VTP format)
- Flow field variables (pressure, velocity, turbulence)

**Transforms:** Raw simulation data into ML-optimized format

- Extracts and normalizes field variables (pressure coefficients, wall shear stress)
- Processes geometry data (coordinates, face connectivity, areas)
- Computes derived quantities and reference values
- Applies mesh decimation for efficiency (optional)

**Outputs:** Training-ready datasets in numpy or Zarr format

- Optimized for DoMINO model training workflows
- Compressed and chunked for efficient data loading
- Preserves metadata and processing parameters

This pipeline handles the complex data engineering required to convert raw CFD outputs into
datasets suitable for training AI models for external aerodynamics applications.

## Running the Curator

Example of the command line that launches Curator configured for DrivAerML dataset:

```bash
physicsnemo-curator-etl                         \
    --config-name=domino_etl                    \
    etl.source.input_dir=/data/drivaerml/       \
    etl.sink.output_dir=/data/drivaerml.processed.surface \
    etl.common.model_type=surface
```

To run on AhmedML dataset:

```bash
physicsnemo-curator-etl                     \
    --config-name=domino_etl_ahmed_ml      \
    etl.source.input_dir=/data/ahmed_ml/   \
    etl.sink.output_dir=/data/ahmed_ml.processed.surface \
    etl.common.model_type=surface
```

### Some useful configuration options

- `etl.sink.overwrite_existing`: can be either `true` (default) or `false`.
    When set to `false`, the Curator will not overwrite existing files.
    This is useful for incremental runs of the Curator.
- `etl.common.model_type`: can be `surface` (default) or `volume` or `combined`.
    This option is used to specify the type of model that will be trained on the
    dataset.
- **Output format**: To switch from Zarr (default) to NumPy, make two changes for consistency:
  1. In `etl.transformations`: uncomment the `numpy` section and comment out the `zarr` section
  2. In `etl.sink`: set `serialization_method` to `"numpy"`

Please refer to the [config file](../../../examples/config/domino_etl.yaml) for more
options.

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
