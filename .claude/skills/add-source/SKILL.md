---
name: add-source
description: >
  Guide for adding a new data source to PhysicsNeMo Curator. Use when adding
  a remote dataset (HuggingFace, S3, etc.) or local file-based source. Covers
  discovery questions, file format handling, Mesh/DataArray construction,
  testing, and registration.
---

# Adding a New Data Source

This skill walks through adding a new `Source` to PhysicsNeMo Curator,
from initial research through implementation, testing, and registration.

## Step 0: Gather Requirements

Before writing any code, answer these questions. Ask the user if anything
is unclear.

### Domain

Which submodule does this source belong to?

| Domain | Type parameter | Submodule | Dependency group |
|--------|---------------|-----------|-----------------|
| **mesh** | `Source[Mesh]` | `src/physicsnemo_curator/domains/mesh/` | `mesh` (physicsnemo, pyvista, pyarrow, torch) |
| **da** | `Source[xr.DataArray]` | `src/physicsnemo_curator/domains/da/` | `da` (xarray, earth2studio, zarr) |

### Dataset Information

Gather all of the following before proceeding:

1. **Name and URL** — What is the dataset called? Where is it hosted?
   (HuggingFace `hf://datasets/org/repo`, S3, HTTP, local path)
2. **File format** — What format are the files? (VTK/VTP/VTU, Parquet,
   HDF5, NetCDF, NumPy, Zarr, CSV, custom)
3. **Schema / fields** — What fields does the data contain? (coordinates,
   velocity, pressure, temperature, connectivity, etc.)
4. **Spatial dimensions** — Is the data 1D, 2D, 3D? Structured grid or
   unstructured mesh?
5. **Dataset size** — How large is the full dataset? How many samples?
6. **Organization** — How are files organized? Single file per sample?
   Run directories (`run_0/`, `run_1/`)? Splits (train/val/test)?
7. **License** — What license? Is it permissive enough for inclusion?
8. **Documentation** — Is there a paper, README, or data card?
9. **Parameters** — What varies between samples? (geometry, physics
   parameters, initial conditions, time steps)
10. **Authentication** — Does accessing the data require tokens or credentials?

### Output Type Construction

For **mesh** sources, determine how to construct `physicsnemo.mesh.Mesh`:

- **VTK-based**: Use `pyvista.read()` then `from_pyvista()` (see existing
  VTK sources)
- **Non-VTK** (Parquet, HDF5, NumPy): Construct `Mesh` directly:

```python
from physicsnemo.mesh import Mesh

mesh = Mesh(
    points=points_tensor,      # torch.Tensor, shape (n_points, 3)
    cells=cells_tensor,        # torch.Tensor, shape (n_cells, nodes_per_cell), dtype=long
    point_data=point_data_td,  # TensorDict with batch_size=[n_points], or None
    cell_data=cell_data_td,    # TensorDict with batch_size=[n_cells], or None
    global_data=global_data_td,# TensorDict with batch_size=[], or None
)
```

For **da** sources, determine how to construct `xarray.DataArray` with
appropriate dimensions, coordinates, and attributes.

### File Discovery Strategy

Each source handles its own file discovery and caching internally.
Choose the approach based on dataset organization:

| Pattern | Approach | When to use |
|---------|----------|------------|
| Flat directory of files | `pathlib.Path.glob()` | Local-only sources |
| Remote flat files | `fsspec` + `fs.glob()` | Generic remote file glob |
| Run-indexed dirs (`run_0/`, `run_1/`) | `fsspec` + `fs.ls()` + regex | Benchmark datasets with numbered runs |
| Single-file datasets | Direct `fsspec`/`pyarrow` | Parquet tables, Zarr stores |

## Step 1: Create the Source Module

Create the source file at the appropriate location:

- **mesh**: `src/physicsnemo_curator/domains/mesh/sources/<name>.py`
- **da**: `src/physicsnemo_curator/domains/da/sources/<name>.py`

### Required SPDX Header

Every source file MUST start with:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

### Source Class Template

```python
"""<Dataset name> data source for PhysicsNeMo Curator.

Reads <format> data from <location> and yields :class:`~physicsnemo.mesh.Mesh`
objects for use in curator pipelines.

Examples
--------
>>> source = <ClassName>()  # doctest: +SKIP
>>> len(source)  # doctest: +SKIP
<expected_count>
>>> mesh = next(source[0])  # doctest: +SKIP
"""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, ClassVar

from physicsnemo_curator.core.base import Param, Source

if TYPE_CHECKING:
    from collections.abc import Generator

    from physicsnemo.mesh import Mesh


_DEFAULT_URL = "<dataset_url>"


class <ClassName>(Source[Mesh]):
    """Read meshes from <Dataset Name>.

    <2-3 sentence description of the dataset, what it contains,
    and where it comes from. Include a link to the paper or dataset page.>

    Parameters
    ----------
    url : str
        Base URL for the dataset.
    storage_options : dict[str, object] | None
        Extra keyword arguments for the fsspec filesystem (e.g. HF token).
    cache_storage : str
        Local directory for caching downloaded files.
    """

    name: ClassVar[str] = "<Display Name>"
    description: ClassVar[str] = "<Short description for CLI>"

    @classmethod
    def params(cls) -> list[Param]:
        """Return configurable parameters for this source."""
        return [
            Param(
                name="url",
                description="Base dataset URL",
                type=str,
                default=_DEFAULT_URL,
            ),
            Param(
                name="cache_storage",
                description="Local cache directory for downloaded files",
                type=str,
                default="",
            ),
            # Add dataset-specific params (mesh_type, split, etc.)
        ]

    def __init__(
        self,
        url: str = _DEFAULT_URL,
        storage_options: dict[str, object] | None = None,
        cache_storage: str = "",
    ) -> None:
        self._url = url
        self._storage_options = storage_options or {}
        self._cache_storage = cache_storage or tempfile.mkdtemp(
            prefix="curator_<name>_"
        )
        # Initialize file discovery and data access here
        # Load lightweight metadata eagerly (e.g. parameter tables)
        # Defer heavy data loading (geometry, fields) to __getitem__

    def __len__(self) -> int:
        """Return the number of items in this source."""
        return self._count

    def __getitem__(self, index: int) -> Generator[Mesh]:
        """Yield mesh(es) for the given index.

        Parameters
        ----------
        index : int
            Zero-based index. Negative indices are supported.

        Yields
        ------
        Mesh
            One or more mesh objects for this index.

        Raises
        ------
        IndexError
            If *index* is out of range.
        """
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            msg = f"Index {index} out of range for source with {len(self)} items."
            raise IndexError(msg)

        # Load data for this index and construct Mesh
        yield mesh
```

### Key Implementation Patterns

**VTK-based sources** (use pyvista):

```python
import pyvista as pv
from physicsnemo.mesh.io import from_pyvista

def _read_vtk(self, path: str) -> Mesh:
    pv_mesh = pv.read(path)
    return from_pyvista(
        pv_mesh,
        manifold_dim=self._manifold_dim,
        point_source=self._point_source,
        warn_on_lost_data=self._warn_on_lost_data,
    )
```

**Parquet-based sources** (direct Mesh construction):

```python
import fsspec
import numpy as np
import pyarrow.parquet as pq
import torch
from tensordict import TensorDict

def _read_parquet(self, path: str) -> pa.Table:
    fs = fsspec.filesystem("hf", **self._storage_options)
    with fs.open(path, "rb") as f:
        return pq.read_table(f)

# In __getitem__:
points = torch.stack([
    torch.from_numpy(np.array(x_coords)),
    torch.from_numpy(np.array(y_coords)),
    torch.zeros(n_points),  # z=0 for 2D
], dim=1).float()

cells = torch.from_numpy(np.array(connectivity, dtype=np.int64))

point_data = TensorDict({
    "field_name": torch.from_numpy(np.array(values)).float(),
}, batch_size=[n_points])

mesh = Mesh(points=points, cells=cells, point_data=point_data)
```

**HDF5-based sources**:

```python
import h5py

with h5py.File(path, "r") as f:
    points = torch.from_numpy(f["coordinates"][:]).float()
    velocity = torch.from_numpy(f["velocity"][:]).float()
```

### Lazy Loading

Follow this pattern for efficiency:

- **Eager** (in `__init__`): lightweight metadata, parameter tables, file counts
- **Lazy** (in `__getitem__`): geometry, field data, large arrays
- **Cached**: data that is shared across indices (e.g. a single geometry
  used by all snapshots)

```python
def __init__(self, ...):
    self._geometry_loaded = False
    self._points: torch.Tensor | None = None
    self._cells: torch.Tensor | None = None

def _load_geometry(self) -> None:
    if self._geometry_loaded:
        return
    # ... load geometry ...
    self._geometry_loaded = True

def __getitem__(self, index):
    self._load_geometry()  # loads once, cached after
    # ... use self._points, self._cells ...
```

## Step 2: Write Tests

Create the test file at `test/<domain>/test_<name>.py`.

### Test File Structure

```python
"""Tests for <ClassName>."""

# SPDX header (same as above)

from __future__ import annotations

import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# ---------------------------------------------------------------------------
# Mock data helper
# ---------------------------------------------------------------------------

def _write_mock_dataset(root: pathlib.Path, ...) -> None:
    """Create a minimal mock dataset for unit tests."""
    # Write mock files that mirror the real dataset structure
    # Use small sizes (10-50 points, 3-5 samples)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

@pytest.mark.requires("<domain>")
class Test<ClassName>Unit:
    """Metadata and parameter tests (no data access)."""

    def test_params_list(self) -> None:
        from <module> import <ClassName>
        params = <ClassName>.params()
        assert len(params) > 0
        names = [p.name for p in params]
        assert "url" in names

    def test_name_and_description(self) -> None:
        from <module> import <ClassName>
        assert isinstance(<ClassName>.name, str)
        assert len(<ClassName>.name) > 0
        assert isinstance(<ClassName>.description, str)
        assert len(<ClassName>.description) > 0


@pytest.mark.requires("<domain>")
class Test<ClassName>Local:
    """Tests against local mock data."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: pathlib.Path) -> None:
        self.mock_root = tmp_path / "mock_dataset"
        _write_mock_dataset(self.mock_root, ...)

    def test_len(self) -> None:
        source = <ClassName>(url=str(self.mock_root))
        assert len(source) == <expected_count>

    def test_getitem_returns_mesh(self) -> None:
        from physicsnemo.mesh import Mesh
        source = <ClassName>(url=str(self.mock_root))
        mesh = next(source[0])
        assert isinstance(mesh, Mesh)
        assert mesh.n_points > 0

    def test_negative_index(self) -> None:
        source = <ClassName>(url=str(self.mock_root))
        mesh_neg = next(source[-1])
        mesh_pos = next(source[len(source) - 1])
        assert mesh_neg.n_points == mesh_pos.n_points

    def test_index_out_of_bounds(self) -> None:
        source = <ClassName>(url=str(self.mock_root))
        with pytest.raises(IndexError):
            next(source[len(source)])

    # Add tests for:
    # - Correct point_data fields
    # - Correct spatial/manifold dimensions
    # - global_data if applicable
    # - Different indices yield different data


@pytest.mark.requires("<domain>")
class Test<ClassName>Registry:
    """Test that the source is registered."""

    def test_source_registered(self) -> None:
        from physicsnemo_curator.core.registry import registry
        names = [s.name for s in registry.list_sources("<domain>")]
        assert "<Display Name>" in names


# ---------------------------------------------------------------------------
# E2E tests (hit real remote)
# ---------------------------------------------------------------------------

@pytest.mark.requires("<domain>")
@pytest.mark.e2e
@pytest.mark.slow
class Test<ClassName>E2E:
    """End-to-end tests against live dataset."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: pathlib.Path) -> None:
        from <module> import <ClassName>
        self.source = <ClassName>(cache_storage=str(tmp_path / "cache"))

    def test_discovers_items(self) -> None:
        assert len(self.source) == <expected_count>

    def test_reads_first_item(self) -> None:
        from physicsnemo.mesh import Mesh
        mesh = next(self.source[0])
        assert isinstance(mesh, Mesh)
        assert mesh.n_points == <expected_points>

    # Add tests for:
    # - Expected number of cells
    # - Expected spatial dimensions (n_spatial_dims)
    # - Expected manifold dimensions (n_manifold_dims)
    # - Point data field names and shapes
    # - Global data values
    # - Different samples have different field values
```

### Test Markers Reference

| Marker | Purpose |
|--------|---------|
| `@pytest.mark.requires("mesh")` | Skip if mesh dependencies not installed |
| `@pytest.mark.requires("da")` | Skip if da dependencies not installed |
| `@pytest.mark.unit` | Fast isolated test (auto-applied if no category marker) |
| `@pytest.mark.e2e` | End-to-end test |
| `@pytest.mark.slow` | Slow test, excluded from quick CI runs |

### Running Tests

```bash
# Unit tests only (fast, no network)
uv run pytest test/<domain>/test_<name>.py -v -k "not E2E"

# All tests including E2E (requires network)
uv run pytest test/<domain>/test_<name>.py -v

# Just the E2E tests
uv run pytest test/<domain>/test_<name>.py -v -k "E2E"
```

## Step 3: Register the Source

### Edit the domain `__init__.py`

For **mesh** sources, edit `src/physicsnemo_curator/domains/mesh/__init__.py`:

```python
# Add import (alphabetical order among sources)
from physicsnemo_curator.domains.mesh.sources.<module> import <ClassName>

# Add registration (after existing register_source calls)
registry.register_source("mesh", <ClassName>)

# Add to __all__ (alphabetical order)
__all__ = [
    ...,
    "<ClassName>",
    ...,
]
```

For **da** sources, edit `src/physicsnemo_curator/domains/da/__init__.py` with the
same pattern using `"da"` as the submodule name.

## Step 4: Quality Checks

Run all checks before committing:

```bash
# Format
uv run ruff format src/physicsnemo_curator/<domain>/sources/<name>.py test/<domain>/test_<name>.py

# Lint
uv run ruff check --fix src/physicsnemo_curator/<domain>/sources/<name>.py test/<domain>/test_<name>.py

# Docstring coverage (must be >= 99%)
uv run interrogate

# Type checking
uv run ty check

# Run unit tests
uv run pytest test/<domain>/test_<name>.py -v -k "not E2E"

# Run full test suite to check for regressions
uv run pytest test/<domain>/ -v -k "not slow"
```

## Step 5: Commit

Use the `commit` tool (or follow Conventional Commits format):

```text
feat(<domain>): add <ClassName> for <dataset> datasets

<Optional body describing the dataset, format, and what it provides.>
```

## Checklist

Before considering the source complete, verify:

- [ ] SPDX license headers on all new files
- [ ] Source inherits from `Source[Mesh]` or `Source[xr.DataArray]`
- [ ] `name` and `description` ClassVars are set
- [ ] `params()` returns all configurable parameters
- [ ] `__len__()` returns correct count
- [ ] `__getitem__()` yields correct type, supports negative indexing, raises IndexError for OOB
- [ ] NumPy-style docstrings on all public methods
- [ ] Lazy loading for heavy data (geometry, fields)
- [ ] Caching for shared data across indices
- [ ] `cache_storage` parameter wired to fsspec caching for remote URLs
- [ ] Registered in domain `__init__.py` with `registry.register_source()`
- [ ] Added to `__all__` in domain `__init__.py`
- [ ] Unit tests with mock data (no network required)
- [ ] Registry test verifying source is discoverable
- [ ] E2E tests marked `@pytest.mark.e2e` and `@pytest.mark.slow`
- [ ] `ruff format` clean
- [ ] `ruff check` clean
- [ ] `interrogate` >= 99%
- [ ] `ty check` clean
- [ ] All unit tests pass
- [ ] No regressions in existing tests

## Reference: Existing Sources

Study these for patterns:

| Source | File | Format | Complexity | Good example for |
|--------|------|--------|-----------|-----------------|
| `VTKSource` | `mesh/sources/vtk.py` | VTK | Medium | Local pathlib file discovery, from_pyvista |
| `DrivAerMLSource` | `mesh/sources/drivaerml.py` | VTP/VTU | High | Remote fsspec discovery, multiple mesh types |
| `WindTunnelSource` | `mesh/sources/windtunnel.py` | VTK | Low | Simplest HF source, splits |
| `NavierStokesCylinderSource` | `mesh/sources/ns_cylinder.py` | Parquet | Medium | Non-VTK, direct Mesh construction |
| `ERA5Source` | `da/sources/era5.py` | API | High | DataArray source, multi-backend |
