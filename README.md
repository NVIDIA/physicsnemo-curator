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
Please refer to the [DoMINO ETL example](./physicsnemo_curator/examples/external_aerodynamics/domino/README.md)
that illustrates the concept.

This package is intended to be used as part of the PhysicsNeMo [framework](https://github.com/NVIDIA/physicsnemo/blob/main/README.md).

## Installation and Usage

There are several ways to install PhysicsNeMo-Curator in an isolated environment:

Requirements:

- Python 3.10 or higher
- pip
- venv or conda (for environment management)

Currently only `linux/amd64` platform is supported.

### Virtual Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/NVIDIA/physicsnemo-curator.git
cd physicsnemo-curator

# Create a virtual environment
# Use python3 to automatically select the latest available Python 3.x version
python3 -m venv venv

# Activate the virtual environment
# On Linux/Mac:
source venv/bin/activate

# Verify Python version
python --version  # Should be 3.10 or higher

# Install the package in editable mode with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Conda Environment

```bash
# Clone the repository
git clone https://github.com/NVIDIA/physicsnemo-curator.git
cd physicsnemo-curator

# Create a conda environment
conda create -n physicsnemo python=3.10
conda activate physicsnemo

# Install the package in editable mode with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Container

The recommended PhysicsNeMo docker image can be pulled from the
[NVIDIA Container Registry](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/containers/physicsnemo):

```bash
docker pull nvcr.io/nvidia/physicsnemo/physicsnemo:25.06

git clone git@github.com:NVIDIA/physicsnemo-curator.git && cd physicsnemo-curator

pip install --upgrade pip
pip install .
```

## Getting Started

### New to PhysicsNeMo-Curator?

If you're new to the framework, start with our comprehensive [**Tutorial**](./TUTORIAL.md).
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

1. Organize your converted data according to one of the [supported dataset formats](./physicsnemo_curator/examples/external_aerodynamics/domino/DoMINO_Data_Processing_Reference.md#input-data-structure)
2. Use the built-in [DoMINO pipeline](./physicsnemo_curator/examples/external_aerodynamics/domino/README.md)
to convert your data to an AI model training ready format

#### Option 2: Extend the Framework for Custom Formats

If your data is in a format not directly supported (VTU/VTP/STL), you can extend the framework:

1. **Follow the Tutorial**: The [Tutorial](./TUTORIAL.md) demonstrates creating a complete pipeline for HDF5 data
2. **Implement Custom DataSource**: Create a DataSource class to read your format
3. **Add Transformations**: Convert your data to ML-optimized formats like Zarr
4. **Leverage Existing ETL framework**: Use the built-in parallel processing capabilities of the existing framework

#### Getting Help

- **Domain-Specific Examples**: Check if your use case matches our [automotive aerodynamics pipeline](./physicsnemo_curator/examples/external_aerodynamics/domino/README.md)
- **Architecture Questions**: See the [Tutorial](./TUTORIAL.md) for framework concepts
- **Format Questions**: Check our [Data Processing Reference](./physicsnemo_curator/examples/external_aerodynamics/domino/DoMINO_Data_Processing_Reference.md)

### Domain-Specific Examples

For domain-specific use cases, we provide ready-to-use pipelines:

**External Aerodynamics (DoMINO)**: Use the built-in ETL pipeline for training DoMINO models for automotive aerodynamics.
Please refer to the [DoMINO training recipe](https://github.com/NVIDIA/physicsnemo/tree/main/examples/cfd/external_aerodynamics/domino)
first.

```bash
physicsnemo-curator-etl --config-name domino_etl
```

Configuration is handled through Hydra. Key parameters can be set in
`config/domino_etl.yaml` or overridden via command line:

```bash
physicsnemo-curator-etl --config-name domino_etl etl.processing.num_processes=4
```

For a complete reference about supported data formats, output structures, and field variables,
see the [DoMINO Data Processing Reference](./physicsnemo_curator/examples/external_aerodynamics/domino/DoMINO_Data_Processing_Reference.md).

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
