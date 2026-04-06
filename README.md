# PhysicsNeMo Curator

<!-- markdownlint-disable MD013 MD033 -->

[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License](https://img.shields.io/github/license/NVIDIA/physicsnemo-curator)](https://github.com/NVIDIA/physicsnemo-curator/blob/main/LICENSE.txt)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![ty](https://img.shields.io/badge/type--checked-ty-blue?logo=python&logoColor=white)](https://github.com/astral-sh/ty)

<!-- markdownlint-enable MD013 MD033 -->

**PhysicsNeMo Curator** is a Rust-accelerated ETL toolkit for curating
engineering and scientific datasets for deep learning. It provides a
composable Python API built on a high-performance native extension (PyO3)
for reading, filtering, and writing simulation data at scale.

[**Getting Started**](#getting-started)
| [**Architecture**](#architecture)
| [**Examples**](#examples)
| [**Contributing**](#contributing-to-physicsnemo-curator)
| [**License**](#license)

---

## Key Features

- **Fluent pipeline API** — chain `Source → Filter → Sink` with a single
  expression, then execute in parallel
- **Rust-accelerated core** — native extension for I/O-bound and
  compute-heavy operations via PyO3
- **Multiple domains** — first-class support for unstructured meshes
  (`physicsnemo.mesh.Mesh`) and gridded data arrays (`xarray.DataArray`)
- **Pluggable execution** — sequential, thread pool, process pool, Loky,
  Dask, or Prefect backends
- **Registry & CLI** — all sources, filters, and sinks are discoverable
  via a global registry and optional interactive CLI

## Architecture

```text
┌─────────────────────────────────────────────────────┐
│                   Python API Layer                   │
│   Source[T] ──▶ Filter[T] ──▶ Sink[T] ──▶ Pipeline  │
│                   run_pipeline(n_jobs, backend)      │
├─────────────────────────────────────────────────────┤
│              Rust Extension (_lib.abi3.so)           │
│           VTK I/O · Mesh ops · Parallel readers     │
└─────────────────────────────────────────────────────┘
```

| Layer | Location | Purpose |
|-------|----------|---------|
| Python API | `src/physicsnemo_curator/` | Public interface — sources, filters, sinks, pipeline runner |
| Rust core | `src/rust/` | Native extension built with PyO3 + maturin |
| Tests | `test/` | Python test suite (pytest) |
| Examples | `examples/` | Sphinx Gallery scripts (runnable + notebook export) |
| Docs | `docs/` | Sphinx documentation |

## Pipeline Components

### Mesh Domain

Sources, filters, and sinks operating on `physicsnemo.mesh.Mesh` objects.

| Component | Type | Description |
|-----------|------|-------------|
| `DrivAerMLSource` | Source | DrivAer notchback variants (500 runs, HuggingFace) |
| `AhmedMLSource` | Source | Ahmed body variants (500 runs, HuggingFace) |
| `WindsorMLSource` | Source | Windsor body variants (355 runs, HuggingFace) |
| `WindTunnelSource` | Source | WindTunnel-20k automobile simulations |
| `NavierStokesCylinderSource` | Source | 2D cylinder flow (Parquet-based) |
| `VTKSource` | Source | Generic VTK reader (.vtk/.vtp/.vtu/.vts/.vtm) |
| `D3PlotSource` | Source | LS-DYNA crash simulation d3plot files |
| `AnsysRSTSource` | Source | Ansys .rst result files via DPF |
| `MeanFilter` | Filter | Per-field spatial means (Parquet output) |
| `StatsFilter` | Filter | Comprehensive statistics with Welford accumulators |
| `MeshInfoFilter` | Filter | Mesh metadata logging (JSON-lines output) |
| `PrecisionFilter` | Filter | In-place floating-point conversion |
| `WallNodeFilter` | Filter | Wall-node removal for crash meshes |
| `MeshSink` | Sink | Writes meshes in PhysicsNeMo tensordict format |

### DataArray Domain

Sources, filters, and sinks operating on `xarray.DataArray` objects.

| Component | Type | Description |
|-----------|------|-------------|
| `ERA5Source` | Source | ERA5 reanalysis via earth2studio (ARCO, WB2, NCAR, CDS) |
| `MomentsFilter` | Filter | Running temporal statistics (Welford online algorithm) |
| `ZarrSink` | Sink | Writes to Zarr store (per-variable groups, time-append) |
| `NetCDF4Sink` | Sink | Writes to NetCDF4 with optional year-based splitting |

## Getting Started

### Requirements

- **Python** >= 3.11
- **OS**: Linux x86_64
- **Rust** toolchain (for building the native extension from source)

### Installation

#### Option 1: PhysicsNeMo Docker container (recommended)

```bash
docker pull nvcr.io/nvidia/physicsnemo/physicsnemo:25.08

git clone git@github.com:NVIDIA/physicsnemo-curator.git
cd physicsnemo-curator
pip install -e ".[dev]"
```

#### Option 2: From source with uv

```bash
git clone git@github.com:NVIDIA/physicsnemo-curator.git
cd physicsnemo-curator

# Install all dev dependencies and build the Rust extension
make install
make develop

# (Optional) Install pre-commit hooks
pre-commit install
```

### Quick Start

```python
from physicsnemo_curator.mesh.sources.drivaerml import DrivAerMLSource
from physicsnemo_curator.mesh.filters.stats import StatsFilter
from physicsnemo_curator.mesh.filters.precision import PrecisionFilter
from physicsnemo_curator.mesh.sinks.mesh_writer import MeshSink
from physicsnemo_curator.run import run_pipeline

# Build a pipeline: Source → Filters → Sink
pipeline = (
    DrivAerMLSource(mesh_type="boundary")
    .filter(StatsFilter(output="stats.parquet"))
    .filter(PrecisionFilter(target_dtype="float32"))
    .write(MeshSink(output_dir="output/meshes/"))
)

# Execute in parallel (4 workers, first 10 runs)
results = run_pipeline(
    pipeline,
    n_jobs=4,
    backend="process_pool",
    indices=range(10),
    progress=True,
)
```

### Optional Dependencies

Install domain-specific extras as needed:

```bash
# LS-DYNA crash simulation support
pip install physicsnemo-curator[lsdyna]

# Ansys RST file support
pip install physicsnemo-curator[ansys]

# Interactive CLI
pip install physicsnemo-curator[cli]

# Parallel backends
pip install physicsnemo-curator[loky]    # Loky backend
pip install physicsnemo-curator[dask]    # Dask backend
pip install physicsnemo-curator[prefect] # Prefect backend
```

## Examples

Runnable Sphinx Gallery scripts in `examples/`. Each can be executed
directly or exported as a Jupyter notebook.

| Example | Domain | Description |
|---------|--------|-------------|
| [DrivAerML ETL](examples/mesh/mesh_drivaerml_etl.py) | Mesh | Boundary mesh curation with spatial means |
| [External Aerodynamics](examples/mesh/mesh_external_aerodynamics.py) | Mesh | Multi-pipeline surface + volume ETL |
| [Crash Simulation](examples/mesh/mesh_crash_simulation.py) | Mesh | LS-DYNA d3plot with wall-node filtering |
| [Ansys Thermal](examples/mesh/mesh_ansys_thermal.py) | Mesh | Ansys .rst thermal analysis pipeline |
| [ERA5 Reanalysis](examples/da/da_era5_etl.py) | DataArray | ERA5 climate data with temporal statistics |

## Development

### Toolchain

| Tool | Purpose | Command |
|------|---------|---------|
| [uv](https://github.com/astral-sh/uv) | Package & env management | `uv sync --group dev` |
| [maturin](https://github.com/PyO3/maturin) | Rust/Python build bridge | `uv run maturin develop` |
| [ruff](https://github.com/astral-sh/ruff) | Linting + formatting | `make format`, `make lint` |
| [ty](https://github.com/astral-sh/ty) | Type checking | `make typecheck` |
| [pytest](https://docs.pytest.org/) | Testing + coverage | `make test` |
| [interrogate](https://interrogate.readthedocs.io/) | Docstring coverage (99%) | `make interrogate` |
| [cargo](https://doc.rust-lang.org/cargo/) | Rust build + test | `make test-rust` |
| [clippy](https://github.com/rust-lang/rust-clippy) | Rust linting | `make lint` |

### Common Commands

```bash
make check          # Format + lint + typecheck + interrogate + cargo deny
make test           # Python tests with coverage
make test-rust      # Rust tests with nextest
make bench          # Python + Rust benchmarks
make docs           # Build Sphinx documentation
```

### Code Conventions

- **Python**: ruff defaults, line length 120, NumPy-style docstrings
- **Rust**: rustfmt defaults, all clippy warnings are errors (`-D warnings`)
- **Docstrings**: 99% coverage enforced by interrogate
- **Type checking**: all Python code must pass `ty check`
- **License**: Apache-2.0 with SPDX headers on all source files
- **Commits**: [Conventional Commits](https://www.conventionalcommits.org/)
  format (`feat`, `fix`, `refactor`, `test`, `docs`, etc.)

## Contributing to PhysicsNeMo Curator

PhysicsNeMo Curator is an open source project and its success is rooted in
community contributions. Thank you for contributing so others can build on
your work.

For guidance, please refer to the [contributing guidelines](CONTRIBUTING.md).

## Cite PhysicsNeMo

If PhysicsNeMo Curator helped your research, please refer to the
[citation guidelines](https://github.com/NVIDIA/physicsnemo/blob/main/CITATION.cff).

## Communication

- **GitHub Discussions** — new data formats, transformations, Physics-ML research
- **GitHub Issues** — bug reports, feature requests, installation issues
- **[PhysicsNeMo Forum](https://forums.developer.nvidia.com/t/welcome-to-the-physicsnemo-ml-model-framework-forum/178556)** —
  general chat, online discussions, collaboration

## Feedback

Suggestions for improvements? Use our
[feedback form](https://docs.google.com/forms/d/e/1FAIpQLSfX4zZ0Lp7MMxzi3xqvzX4IQDdWbkNh5H_a_clzIhclE2oSBQ/viewform?usp=sf_link).

## License

PhysicsNeMo Curator is provided under the Apache License 2.0. See
[LICENSE.txt](./LICENSE.txt) for the full license text.
