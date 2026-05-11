# PhysicsNeMo Curator

<!-- markdownlint-disable MD013 MD033 -->

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/)
[![License](https://img.shields.io/github/license/NVIDIA/physicsnemo-curator)](https://github.com/NVIDIA/physicsnemo-curator/blob/main/LICENSE.txt)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)

[![Build & Install](https://github.com/NVIDIA/physicsnemo-curator/actions/workflows/build.yml/badge.svg)](https://github.com/NVIDIA/physicsnemo-curator/actions/workflows/build.yml)
[![Docs](https://github.com/NVIDIA/physicsnemo-curator/actions/workflows/docs.yml/badge.svg)](https://github.com/NVIDIA/physicsnemo-curator/actions/workflows/docs.yml)
[![Lint](https://github.com/NVIDIA/physicsnemo-curator/actions/workflows/lint.yml/badge.svg)](https://github.com/NVIDIA/physicsnemo-curator/actions/workflows/lint.yml)
[![Test](https://github.com/NVIDIA/physicsnemo-curator/actions/workflows/test.yml/badge.svg)](https://github.com/NVIDIA/physicsnemo-curator/actions/workflows/test.yml)
[![Coverage](https://img.shields.io/codecov/c/github/nvidia/physicsnemo-curator/refactor?logo=codecov)](https://codecov.io/gh/NVIDIA/physicsnemo-curator/branch/refactor)

<!-- markdownlint-enable MD013 MD033 -->

**PhysicsNeMo Curator** is an accelerated ETL toolkit for building
AI-ready datasets across multiple scientific and engineering domains,
including CAE, weather/climate, molecular dynamics, and more. Curator is
designed to be a flexible and customizable package that provides core
pipeline components for users to create their own data processing pipelines.

[**Docs**](https://nvidia.github.io/physicsnemo-curator/)
| [**Getting Started**](#getting-started)
| [**Domains**](https://nvidia.github.io/physicsnemo-curator/domains/index.html)
| [**Extending**](https://nvidia.github.io/physicsnemo-curator/extending/index.html)
| [**Examples**](https://nvidia.github.io/physicsnemo-curator/auto_examples/index.html)
| [**Contributing**](#contributing-to-physicsnemo-curator)

---

> [!WARNING]
> This package is in beta and subject to extensive changes. There are no guarantees for API stability.

## Key Features

- **Fluent pipeline API** — chain `Source → Filter → Sink` with a single
  expression, then execute in parallel
- **Lazy generator semantics** — sources and filters yield items lazily;
  `pipeline[i]` processes only the *i*-th item
- **Multiple domains** — first-class support for unstructured meshes
  (`physicsnemo.mesh.Mesh`), gridded data arrays (`xarray.DataArray`),
  and atomic/molecular data (`nvalchemi.data.AtomicData`)
- **Pluggable execution** — sequential, thread pool, process pool, Loky,
  Dask, or Prefect backends
- **Registry & CLI** — all sources, filters, and sinks are discoverable
  via a global registry and optional interactive CLI
- **Extensible** — write custom sources, filters, and sinks with minimal
  boilerplate ([guide](https://nvidia.github.io/physicsnemo-curator/extending/index.html))

## Getting Started

### Requirements

- **Python** >= 3.11
- **OS**: Linux x86_64
- **Rust** toolchain (for building the native extension from source)

### Installation

```bash
git clone git@github.com:NVIDIA/physicsnemo-curator.git
cd physicsnemo-curator

# Install all dev dependencies and build the Rust extension
uv sync --group dev
uv run maturin develop

# (Optional) Install pre-commit hooks
uv run pre-commit install
```

### Quick Start Sample

Curate a simple global weather dataset:

```bash
# First install the data array dependency group
uv sync --extra da
```

```python
from datetime import datetime, timedelta

from physicsnemo_curator.domains.da.filters.stats import DataArrayStatsFilter
from physicsnemo_curator.domains.da.sinks.zarr_writer import ZarrSink
from physicsnemo_curator.domains.da.sources.era5 import ERA5Source
from physicsnemo_curator.run import run_pipeline

# Hourly timestamps for one day
times = [datetime(2020, 1, 1) + timedelta(hours=h) for h in range(24)]

# Source → Filter → Sink
pipeline = (
    ERA5Source(times=times, variables=["u10m", "v10m", "t2m"], backend="arco")
    .filter(DataArrayStatsFilter(output="output/stats.zarr", dims=("time",)))
    .write(ZarrSink(output_path="output/dataset.zarr"))
)

# Execute in parallel
results = run_pipeline(pipeline, n_jobs=4, backend="process_pool")
```

### Optional Dependencies

Install domain-specific extras as needed:

```bash
# Mesh domain (CAE, CFD)
pip install physicsnemo-curator[mesh]

# DataArray domain (weather/climate)
pip install physicsnemo-curator[da]

# Atomic domain (molecular dynamics)
pip install physicsnemo-curator[atm]

# Dashboard
pip install physicsnemo-curator[dashboard]
```

## CLI

PhysicsNeMo Curator includes the `psnc` command-line tool with an
interactive full-screen pipeline wizard powered by Textual.

```bash
psnc
```

### Dashboard

```bash
pip install 'physicsnemo-curator[dashboard]'
psnc dashboard pipeline.db
```

## Contributing to PhysicsNeMo Curator

PhysicsNeMo Curator is an open source project and its success is rooted in
community contributions. Thank you for contributing so others can build on
your work.

For guidance, please refer to the [contributing guidelines](CONTRIBUTING.md).
See also:

- [Extending / Customization](https://nvidia.github.io/physicsnemo-curator/extending/index.html) —
  how to write custom sources, filters, and sinks
- [Developer Guide](https://nvidia.github.io/physicsnemo-curator/developer-guide/index.html) —
  style conventions, benchmarking, and AI-assisted development

## Ecosystem

PhysicsNeMo Curator is part of NVIDIA's open-source Physics-ML ecosystem:

<!-- markdownlint-disable MD013 -->

| Package | Description |
|---|---|
| [PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) | Core framework for building, training, and fine-tuning physics-ML models |
| [PhysicsNeMo CFD](https://github.com/NVIDIA/physicsnemo-cfd) | Pretrained AI models for computational fluid dynamics |
| [Earth-2 Studio](https://github.com/NVIDIA/earth2studio) | Pretrained AI models for weather and climate |
| [ALCHEMI Toolkit](https://github.com/NVIDIA/nvalchemi-toolkit) | GPU-first framework for AI-driven atomic simulations |
| [ALCHEMI Toolkit Ops](https://github.com/NVIDIA/nvalchemi-toolkit-ops) | GPU-optimized primitives for neighbor lists, dispersion, and electrostatics |

<!-- markdownlint-enable MD013 -->

## Communication

- **GitHub Discussions** — new data formats, transformations, Physics-ML research
- **GitHub Issues** — bug reports, feature requests, installation issues

## License

PhysicsNeMo Curator is provided under the Apache License 2.0. See
[LICENSE.txt](./LICENSE.txt) for the full license text.
