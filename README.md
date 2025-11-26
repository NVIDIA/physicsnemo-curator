# PhysicsNeMo-Curator
<!-- markdownlint-disable -->

[![Project Status: Active.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![GitHub](https://img.shields.io/github/license/NVIDIA/physicsnemo)](https://github.com/NVIDIA/physicsnemo/blob/master/LICENSE.txt)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<!-- markdownlint-enable -->
[**PhysicsNeMo Curator**](#what-is-physicsnemo-curator)
| [**Getting started**](#getting-started)
| [**Documentation**](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/index.html)
| [**Contributing Guidelines**](#contributing-to-physicsnemo-curator)
| [**Communication**](#communication)

## What is PhysicsNeMo Curator?

PhysicsNeMo Curator is a sub-module of PhysicsNeMo framework, a pythonic library
designed to streamline and accelerate the crucial process of data curation at
scale for engineering and scientific datasets for training and inference.
It accelerates data curation by leveraging GPUs.

This includes customizable interfaces and pipelines for extracting, transforming
and loading data in supported formats and schema.
Please refer to the [External Aerodynamics ETL example](examples/external_aerodynamics/README.md)
that illustrates the concept.

This package is intended to be used as part of the PhysicsNeMo [framework](https://github.com/NVIDIA/physicsnemo/blob/main/README.md).

## Installation and Usage

The recommended way of using `PhysicsNeMo-Curator` is to leverage the `PhysicsNeMo` docker image.
This can be pulled from the
[NVIDIA Container Registry](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/containers/physicsnemo).

Current limitations:

- Currently only `linux/amd64` platform is supported
- Currently we don't provide a PyPi wheel, and support installing from source

### PhysicsNeMo Container (Recommended)

The instructions to get started with `PhysicsNeMo-Curator` within the `PhysicsNeMo` docker container are shown below.

```bash
docker pull nvcr.io/nvidia/physicsnemo/physicsnemo:25.08

# Install from source
git clone git@github.com:NVIDIA/physicsnemo-curator.git && cd physicsnemo-curator

pip install --upgrade pip
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Getting Started

### New to PhysicsNeMo-Curator?

If you're new to the framework, start with our comprehensive [**Tutorial**](./examples/tutorials/etl_hdf5_to_zarr/hdf5_to_zarr.ipynb).
It walks you through building a complete ETL pipeline from scratch. You'll learn how to:

- Define data schemas
- Implement schema validation, data sources, transformations, and sinks
- Convert HDF5 data to ML-optimized Zarr format
- Configure and run parallel processing pipelines

### Working with Your CFD Data

Have CFD simulation data from a solver like Fluent?
PhysicsNeMo-Curator can process your data through the following approaches:

#### Option 1: Convert to Supported Formats (Recommended)

**Currently Supported Formats:**

- **VTK formats**: VTU (volume mesh data), VTP (surface mesh data)
- **STL**: Geometry files

**Next Steps:**

1. Organize your converted data according to one of the supported dataset formats:
   - [External Aerodynamics Data Processing](./examples/external_aerodynamics/External_Aero_Data_Processing_Reference.md#input-data-structure)
   - [Crash Data Processing](./examples/structural_mechanics/crash/Crash_Data_Processing_Reference.md)
2. Use the built-in [External Aerodynamics ETL pipeline](examples/external_aerodynamics/README.md) or [Crash ETL pipeline](./examples/structural_mechanics/crash/README.md)
to convert your data to an AI model training ready format.
This built-in pipeline produces a dataset that can be used to train both models in PhysicsNeMo!
3. Train your model on your own data by following these guides:
   - [DoMINO External Aerodynamics example in PhysicsNeMo](https://github.com/NVIDIA/physicsnemo/tree/main/examples/cfd/external_aerodynamics/domino)
   - [Transolver External Aerodynamics example in PhysicsNeMo](https://github.com/NVIDIA/physicsnemo/tree/main/examples/cfd/external_aerodynamics/transolver)
   - [Structural Mechanics Crash example in PhysicsNeMo](https://github.com/NVIDIA/physicsnemo/tree/main/examples/structural_mechanics/crash)

#### Option 2: Extend the Framework for Custom Formats

If your data is in a format not directly supported (VTU/VTP/STL), you can extend the framework.
The [Tutorial](./examples/tutorials/etl_hdf5_to_zarr/hdf5_to_zarr.ipynb)
demonstrates creating a complete pipeline that reads in HDF5 data and converts it to Zarr data.

#### Getting Help

- **Domain-Specific Examples**:
  - [External aerodynamics pipeline](./examples/external_aerodynamics/README.md)
  provides an example ETL pipeline for training DoMINO/Transolver models for automotive aerodynamics applications.
  For more questions about the formats, please refer to [Data Processing Reference](./physicsnemo_curator/examples/external_aerodynamics/External_Aero_Data_Processing_Reference.md)
  - [Structural Mechanics / Crash pipeline](./examples/structural_mechanics/crash/README.md)
  provides an example ETL pipeline for training models for structural mechanics/crash applications.
- **Architecture Questions**: See the [Tutorial](./examples/tutorials/etl_hdf5_to_zarr/hdf5_to_zarr.ipynb)
for framework concepts, and to understand how to extend the pipeline
- **Anything else**: Please open a GitHub issue and we'll engage with you to answer the questions!

## Contributing to PhysicsNeMo-Curator

PhysicsNeMo-Curator and PhysicsNeMo are open source collaborations and their
success is rooted in community contribution to further the field of Physics-ML.
Thank you for contributing to the project so others can build on top of your
contribution.

For guidance on contributing to PhysicsNeMo-Curator, please refer to the
[contributing guidelines](CONTRIBUTING.md).

## Cite PhysicsNeMo-Curator

If PhysicsNeMo-Curator helped your research and you would like to cite it,
please refer to the [guidelines](https://github.com/NVIDIA/physicsnemo/blob/main/CITATION.cff).

## Communication

- Github Discussions: Discuss new data formats, transformations, Physics-ML
research, etc.
- GitHub Issues: Bug reports, feature requests, install issues, etc.
- PhysicsNeMo Forum: The [PhysicsNeMo Forum](https://forums.developer.nvidia.com/t/welcome-to-the-physicsnemo-ml-model-framework-forum/178556)
hosts an audience of new to moderate-level users and developers for
general chat, online discussions, collaboration, etc.

## Feedback

Want to suggest some improvements to PhysicsNeMo-Curator? Use our
[feedback form](https://docs.google.com/forms/d/e/1FAIpQLSfX4zZ0Lp7MMxzi3xqvzX4IQDdWbkNh5H_a_clzIhclE2oSBQ/viewform?usp=sf_link).

## License

PhysicsNeMo-Curator is provided under the Apache License 2.0, please see
[LICENSE.txt](./LICENSE.txt) for full license text.
