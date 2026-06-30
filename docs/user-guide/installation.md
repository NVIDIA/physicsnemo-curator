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
| **mesh** | physicsnemo, pyvista, pyarrow, torch | `pip install .[mesh]` or `uv sync --extra mesh` |
| **da** | xarray, earth2studio, zarr, gcsfs | `pip install .[da]` or `uv sync --extra da` |
| **atm** | nvalchemi, ase, torch | `pip install .[atm]` or `uv sync --extra atm` |
| **dev** | ruff, ty, pytest, maturin, interrogate, pre-commit | `uv sync --group dev` |
| **docs** | sphinx, nvidia-sphinx-theme, myst-parser, etc. | `uv sync --group docs` |

### Execution Backend Extras

The `run_pipeline()` function supports multiple execution backends. The basic
backends (sequential, process_pool) work out of the box. For
advanced backends, install the corresponding extra:

| Extra | Backend | Install |
|-------|---------|---------|
| **loky** | joblib/loky (robust process pool) | `uv sync --extra loky` |
| **dask** | Dask bags (distributed execution, experimental) | `uv sync --extra dask` |

### Installing Multiple Groups

```bash
# Development with mesh support
uv sync --group dev --extra mesh

# Everything
uv sync --group dev --extra mesh --group docs
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
