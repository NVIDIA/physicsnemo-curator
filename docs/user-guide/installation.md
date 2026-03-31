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
| **xr** | *(planned)* xarray, zarr | — |
| **mdt** | *(planned)* torch | — |
| **cli** | click, questionary | `pip install .[cli]` |
| **dev** | ruff, ty, pytest, maturin, interrogate, pre-commit | `uv sync --group dev` |
| **docs** | sphinx, nvidia-sphinx-theme, myst-parser, etc. | `uv sync --group docs` |

### Installing Multiple Groups

```bash
# Development with mesh support
uv sync --group dev --group mesh

# Everything
uv sync --group dev --group mesh --group docs
```

## Verifying the Installation

```python
import curator
print(curator.__version__)  # "0.1.0"

# Check mesh dependencies
from curator.mesh.sources.vtk import VTKSource  # requires curator[mesh]
```

## Building Documentation

```bash
uv sync --group docs
make docs
# Output in docs/_build/html/
```
