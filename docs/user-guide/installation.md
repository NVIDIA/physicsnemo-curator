# Installation

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## From Source (Development)

```bash
git clone https://github.com/NVIDIA/physicsnemo-curator.git
cd physicsnemo-curator

# Install core + dev tools
make install        # runs: uv sync --group dev

# (Optional) Build the Rust native extension
make develop        # runs: uv run maturin develop
```

## Dependency Groups

PhysicsNeMo Curator uses optional dependency groups so you only install what
you need.  Each submodule has its own group:

| Group | Contents | Install |
|-------|----------|---------|
| **mesh** | physicsnemo, pyvista, pyarrow, torch | `pip install .[mesh]` or `uv sync --group mesh` |
| **da** | xarray, earth2studio, zarr, gcsfs | `uv sync --group da` |
| **atm** | nvalchemi, ase, torch | `uv sync --group atm` |
| **cli** | click, questionary | `pip install .[cli]` |
| **dev** | ruff, ty, pytest, maturin, interrogate, pre-commit | `uv sync --group dev` |
| **docs** | sphinx, nvidia-sphinx-theme, myst-parser, etc. | `uv sync --group docs` |

### Execution Backend Extras

The `run_pipeline()` function supports multiple execution backends. The basic
backends (sequential, thread_pool, process_pool) work out of the box. For
advanced backends, install the corresponding extra:

| Extra | Backend | Install |
|-------|---------|---------|
| **loky** | joblib/loky (robust process pool) | `pip install 'physicsnemo-curator[loky]'` |
| **dask** | Dask bags (distributed execution) | `pip install 'physicsnemo-curator[dask]'` |
| **prefect** | Prefect (workflow orchestration) | `pip install 'physicsnemo-curator[prefect]'` |

```bash
# Install multiple backend extras
pip install 'physicsnemo-curator[loky,dask]'
```

### Installing Multiple Groups

```bash
# Development with mesh support
uv sync --group dev --group mesh

# Everything
uv sync --group dev --group mesh --group docs
```

## Verifying the Installation

```python
import physicsnemo_curator
print(physicsnemo_curator.__version__)  # "0.1.0"

# Check mesh dependencies
from physicsnemo_curator.domains.mesh.sources.vtk import VTKSource  # requires physicsnemo_curator[mesh]
```

## Building Documentation

```bash
uv sync --group docs
make docs
# Output in docs/_build/html/
```
