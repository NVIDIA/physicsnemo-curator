---
name: add-source
description: >
  Guide for adding a new data source to PhysicsNeMo Curator. Use when adding
  a remote dataset (HuggingFace, S3, etc.) or local file-based source. Covers
  discovery questions, file format handling, Mesh/DataArray/AtomicData
  construction, parallel partitioning, testing, and registration.
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
| **atm** | `Source[AtomicData]` | `src/physicsnemo_curator/domains/atm/` | `mesh` (nvalchemi, torch, lmdb) |

### Dataset Information

Gather all of the following before proceeding:

1. **Name and URL** — What is the dataset called? Where is it hosted?
   (HuggingFace `hf://datasets/org/repo`, S3, HTTP, local path)
2. **File format** — What format are the files? (VTK/VTP/VTU, Parquet,
   HDF5, NetCDF, NumPy, Zarr, LMDB, CSV, custom)
3. **Schema / fields** — What fields does the data contain? (coordinates,
   velocity, pressure, temperature, connectivity, forces, energies, etc.)
4. **Spatial dimensions** — Is the data 1D, 2D, 3D? Structured grid or
   unstructured mesh? Molecular graph?
5. **Dataset size** — How large is the full dataset? How many samples?
6. **Organization** — How are files organized? Single file per sample?
   Run directories (`run_0/`, `run_1/`)? Splits (train/val/test)?
   Multi-file databases?
7. **License** — What license? Is it permissive enough for inclusion?
8. **Documentation** — Is there a paper, README, or data card?
9. **Parameters** — What varies between samples? (geometry, physics
   parameters, initial conditions, time steps)
10. **Authentication** — Does accessing the data require tokens or credentials?
11. **Parallel constraints** — Does the data backend have concurrency
    limitations? (e.g. LMDB single-env-per-process, HDF5 locking)
12. **Backend options** — Is there a fast native reader (Rust) available
    as well as a Python fallback?

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

For **atm** sources, determine how to construct
`nvalchemi.data.AtomicData`:

```python
from nvalchemi.data import AtomicData

data = AtomicData(
    positions=positions,        # torch.Tensor, shape (n_atoms, 3)
    atomic_numbers=z,           # torch.Tensor, shape (n_atoms,)
    cell=cell,                  # torch.Tensor, shape (3, 3) or None
    pbc=pbc,                    # torch.Tensor, shape (3,) bool
    energy=energy,              # torch.Tensor, shape (1,) or None
    forces=forces,              # torch.Tensor, shape (n_atoms, 3) or None
)
```

### File Discovery Strategy

Each source handles its own file discovery and caching internally.
Choose the approach based on dataset organization:

| Pattern | Approach | When to use |
|---------|----------|------------|
| Flat directory of files | `pathlib.Path.glob()` | Local-only sources (VTKSource, ASELMDBSource) |
| Remote flat files | `fsspec` + `fs.glob()` | Generic remote file glob |
| Run-indexed dirs (`run_0/`, `run_1/`) | `fsspec` + `fs.ls()` + regex | Benchmark datasets with numbered runs (DrivAerML, AhmedML) |
| API/backend-driven | Domain library fetcher | Weather reanalysis (ERA5, HRRR via earth2studio) |
| Single-file datasets | Direct `fsspec`/`pyarrow` | Parquet tables, Zarr stores |

### Reference: Existing Sources

| Source | File | Format | Domain | Good example for |
|--------|------|--------|--------|-----------------|
| `VTKSource` | `mesh/sources/vtk.py` | VTK | mesh | Local file glob, pyvista/rust backend, relative_path |
| `DrivAerMLSource` | `mesh/sources/drivaerml.py` | VTP/VTU | mesh | Remote fsspec, multi-mesh, run_id, mesh_name |
| `AhmedMLSource` | `mesh/sources/ahmedml.py` | VTP | mesh | HF remote, CSV metadata, multi-mode, run_id |
| `NavierStokesCylinderSource` | `mesh/sources/ns_cylinder.py` | Parquet | mesh | Direct Mesh construction, non-VTK |
| `D3PlotSource` | `mesh/sources/d3plot.py` | d3plot | mesh | Binary format, python/rust dual backend |
| `AnsysRSTSource` | `mesh/sources/ansys_rst.py` | .rst | mesh | Auto-discovery of results, complex extraction |
| `OpenRadiossSource` | `mesh/sources/openradioss.py` | VTK | mesh | Per-timestep multi-file, run_id |
| `ERA5Source` | `da/sources/era5.py` | API | da | Multi-backend weather, DataArray, earth2studio |
| `HRRRSource` | `da/sources/hrrr.py` | API | da | HRRR 3km analysis, multiple providers |
| `ASELMDBSource` | `atm/sources/aselmdb.py` | LMDB | atm | partition_indices, rust fallback, relative_path |
| `RandomMeshSource` | `mesh/sources/random.py` | Synthetic | mesh | Testing, minimal source |
| `RandomDataArraySource` | `da/sources/random.py` | Synthetic | da | Testing, minimal DA source |
| `RandomAtomicSource` | `atm/sources/random.py` | Synthetic | atm | Testing, minimal atomic source |

## Step 1: Create the Source Module

Create the source file at the appropriate location:

- **mesh**: `src/physicsnemo_curator/domains/mesh/sources/<name>.py`
- **da**: `src/physicsnemo_curator/domains/da/sources/<name>.py`
- **atm**: `src/physicsnemo_curator/domains/atm/sources/<name>.py`

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

### Import Block Template

```python
"""<Dataset name> data source for PhysicsNeMo Curator.

Reads <format> data from <location> and yields :class:`~physicsnemo.mesh.Mesh`
objects for use in curator pipelines.

Examples
--------
>>> source = <ClassName>(...)  # doctest: +SKIP
>>> len(source)  # doctest: +SKIP
<expected_count>
>>> mesh = next(source[0])  # doctest: +SKIP
"""

from __future__ import annotations

import pathlib
import tempfile
from typing import TYPE_CHECKING, ClassVar, Literal

from physicsnemo_curator.core.base import Param, Source
from physicsnemo_curator.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Generator

    from physicsnemo.mesh import Mesh
```

### Source Class Template

```python
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

    Note
    ----
    - Dataset: `<org/repo> <<url>>`_
    - Paper: `arXiv:<id> <https://arxiv.org/abs/<id>>`_
    - License: `<license_name> <<license_url>>`_
    """

    name: ClassVar[str] = "<Display Name>"
    description: ClassVar[str] = "<Short description for CLI>"

    @classmethod
    def params(cls) -> list[Param]:
        """Return configurable parameters for this source.

        Returns
        -------
        list[Param]
            Parameter list for CLI configuration.
        """
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
            # Add dataset-specific params (mesh_type, split, backend, etc.)
        ]

    def __init__(
        self,
        url: str = _DEFAULT_URL,
        storage_options: dict[str, object] | None = None,
        cache_storage: str = "",
    ) -> None:
        """Initialize the source.

        Parameters
        ----------
        url : str
            Base dataset URL.
        storage_options : dict[str, object] | None
            Extra keyword arguments for the fsspec filesystem.
        cache_storage : str
            Local cache directory.
        """
        self._url = url
        self._storage_options = storage_options or {}
        self._cache_storage = cache_storage or tempfile.mkdtemp(
            prefix="curator_<name>_"
        )
        self._log = get_logger(self)

        # Initialize file discovery and data access here
        # Load lightweight metadata eagerly (e.g. parameter tables)
        # Defer heavy data loading (geometry, fields) to __getitem__
        self._discover_items()

    def _discover_items(self) -> None:
        """Eagerly discover available items (files, runs, etc.)."""
        ...

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

### Logging

Use the structured component logger:

```python
from physicsnemo_curator.core.logging import get_logger

class MySource(Source[Mesh]):
    def __init__(self, ...) -> None:
        self._log = get_logger(self)
        self._log.info("Discovered %d items in %s", count, url)

    def __getitem__(self, index: int) -> Generator[Mesh]:
        self._log.debug("Reading index %d", index)
        ...
```

For module-level logging (outside class scope):

```python
from physicsnemo_curator.core.logging import get_logger

logger = get_logger("MySourceModule")
```

**Do NOT use `logging.getLogger(__name__)`** — always use `get_logger`.

### Key Implementation Patterns

#### Pattern A: Local File-Based (VTKSource, ASELMDBSource)

Discovers files at construction time via `pathlib.Path.glob()`, then
reads them lazily in `__getitem__`:

```python
def __init__(self, input_path: str, file_pattern: str = "**/*") -> None:
    root = pathlib.Path(input_path)
    if not root.is_dir():
        msg = f"Path {root} is not a directory."
        raise FileNotFoundError(msg)
    self._root = root.resolve()

    # Discover and sort for deterministic order
    self._files: list[pathlib.Path] = sorted(
        p.resolve()
        for p in root.glob(file_pattern)
        if p.is_file() and p.suffix.lower() in self._VALID_EXTENSIONS
    )
    if not self._files:
        msg = f"No valid files found in {root} with pattern {file_pattern!r}"
        raise ValueError(msg)

    self._log.info("Discovered %d files in %s", len(self._files), root)

def __len__(self) -> int:
    return len(self._files)

def __getitem__(self, index: int) -> Generator[Mesh]:
    path = self._files[index]
    mesh = self._read_file(path)
    yield mesh
```

#### Pattern B: Remote HuggingFace Hub (DrivAerML, AhmedML)

Uses fsspec for file discovery and WholeFileCacheFileSystem for local
caching of downloaded files:

```python
import fsspec

def __init__(self, url: str, cache_storage: str = "", cache: bool = True, ...) -> None:
    self._url = url
    self._cache = cache
    self._protocol, self._path = fsspec.utils.split_protocol(url)

    # Build filesystem with caching layer
    target_fs = fsspec.filesystem(self._protocol, **storage_options)
    if cache:
        self._fs = fsspec.filesystem(
            "filecache",
            target_protocol=self._protocol,
            target_options=storage_options,
            cache_storage=cache_storage or tempfile.mkdtemp(),
        )
    else:
        self._fs = target_fs

    # Discover run directories
    self._run_indices = self._discover_runs()
    self._log.info("Discovered %d runs", len(self._run_indices))

def _discover_runs(self) -> list[int]:
    """Find run directories via regex pattern matching."""
    import re
    pattern = re.compile(r"run_(\d+)/?$")
    entries = self._fs.ls(self._path, detail=False)
    runs = sorted(int(m.group(1)) for e in entries if (m := pattern.search(e)))
    return runs

def __getitem__(self, index: int) -> Generator[Mesh]:
    run_id = self._run_indices[index]
    remote_path = f"{self._path}/run_{run_id}/boundary.vtp"
    local_path = self._fs.open(remote_path).name  # Cached locally
    mesh = self._read_vtk(local_path)
    yield mesh
```

#### Pattern C: API/Backend-Driven (ERA5Source, HRRRSource)

Uses a domain library (earth2studio) to fetch data, routing requests
to different backends:

```python
def __init__(
    self,
    variables: list[str],
    time_range: tuple[str, str],
    backend: str = "arco",
    ...
) -> None:
    self._variables = variables
    self._backend = backend
    self._log = get_logger(self)

    # Build time index eagerly
    self._times = self._build_time_index(time_range)

def __getitem__(self, index: int) -> Generator[xr.DataArray]:
    time = self._times[index]
    # Fetch from backend (earth2studio handles caching)
    da = self._fetch(time)
    yield da
```

#### Pattern D: Multi-File Database (ASELMDBSource)

Multiple files each containing many items, with cumulative indexing:

```python
def __init__(self, data_dir: str, file_pattern: str = "**/*.aselmdb") -> None:
    self._root = pathlib.Path(data_dir).resolve()
    self._db_files = sorted(
        p.resolve() for p in self._root.glob(file_pattern)
        if p.suffix == ".aselmdb"
    )

    # Count rows per file (lightweight metadata read)
    self._row_counts = [self._count_rows(f) for f in self._db_files]
    self._cumulative_counts = list(itertools.accumulate(self._row_counts))
    self._total = self._cumulative_counts[-1] if self._cumulative_counts else 0

def __len__(self) -> int:
    return self._total

def __getitem__(self, index: int) -> Generator[AtomicData]:
    file_idx = self._find_file_for_index(index)
    local_idx = index - (self._cumulative_counts[file_idx - 1] if file_idx > 0 else 0)
    data = self._read_single(self._db_files[file_idx], local_idx)
    yield data

def _find_file_for_index(self, index: int) -> int:
    """Map global index to file index using cumulative counts."""
    import bisect
    return bisect.bisect_right(self._cumulative_counts, index) - 1
```

#### Pattern E: Multi-Output Per Index (DrivAerML "multi" mode)

A single index can yield multiple items (e.g. multiple mesh parts):

```python
def __getitem__(self, index: int) -> Generator[Mesh | DomainMesh]:
    run_id = self._run_indices[index]

    for part in self._mesh_parts:
        remote_path = self._resolve_part_path(run_id, part)
        local_path = self._download(remote_path)
        mesh = self._read_vtk(local_path)
        yield mesh
```

### Dual Backend Pattern (Python + Rust)

Many sources support both a Python backend and a faster Rust backend:

```python
def __init__(self, ..., backend: Literal["python", "rust"] = "python") -> None:
    self._backend: Literal["python", "rust"] = backend

    # Validate rust availability at construction time
    if backend == "rust":
        try:
            from physicsnemo_curator._lib.some_module import reader as _  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            self._log.warning("Rust backend unavailable; falling back to 'python'.")
            self._backend = "python"

def __getitem__(self, index: int) -> Generator[Mesh]:
    if self._backend == "rust":
        mesh = self._read_with_rust(index)
    else:
        mesh = self._read_with_python(index)
    yield mesh
```

### Lazy Loading

Follow this pattern for efficiency:

- **Eager** (in `__init__`): lightweight metadata, parameter tables, file
  counts, directory listings
- **Lazy** (in `__getitem__`): geometry, field data, large arrays
- **Cached**: data that is shared across indices (e.g. a single geometry
  used by all snapshots)

```python
def __init__(self, ...):
    self._geometry_loaded = False
    self._points: torch.Tensor | None = None
    self._cells: torch.Tensor | None = None

def _load_geometry(self) -> None:
    """Load shared geometry (called once, result cached)."""
    if self._geometry_loaded:
        return
    # ... load geometry ...
    self._geometry_loaded = True

def __getitem__(self, index):
    self._load_geometry()  # loads once, cached after
    # ... use self._points, self._cells ...
```

## Step 1b: Sink Integration Protocol (Optional)

Sources can optionally provide methods that sinks use for output naming.
These are NOT part of the ABC but are checked via `hasattr()` by
`Pipeline.write()` and sinks.

### `relative_path()` Method

For file-based sources, enables directory mirroring in output:

```python
@property
def root(self) -> pathlib.Path:
    """Return the root directory of this source.

    Returns
    -------
    pathlib.Path
        The root directory containing the discovered files.
    """
    return self._root

def relative_path(self, index: int) -> str:
    """Return the path of the *index*-th file relative to the root.

    Used by sinks to resolve ``{relpath}`` and ``{stem}`` naming
    placeholders, enabling output layouts that mirror the input.

    Parameters
    ----------
    index : int
        Zero-based file index.

    Returns
    -------
    str
        POSIX-style relative path (e.g. ``"subdir/mesh.vtu"``).
    """
    return self._files[index].relative_to(self._root).as_posix()
```

### `run_id()` Method

For run-based datasets, provides the dataset run identifier:

```python
def run_id(self, index: int) -> int:
    """Return the dataset run ID for the given source index.

    Used by sinks to resolve the ``{run_id}`` naming placeholder.

    Parameters
    ----------
    index : int
        Zero-based index into the sorted run list.

    Returns
    -------
    int
        The dataset run ID (e.g. 1, 5, 12).
    """
    return self._run_indices[index]
```

### `mesh_name()` Method

For multi-output sources, provides canonical names for each part:

```python
def mesh_name(self, index: int, seq: int) -> str:
    """Return the canonical output name for mesh at (index, seq).

    Used by sinks to resolve the ``{mesh_name}`` naming placeholder.

    Parameters
    ----------
    index : int
        Source index (which run).
    seq : int
        Sequence number within this index (which part).

    Returns
    -------
    str
        Resolved name like ``"domain_1"`` or ``"drivaer_5.stl"``.
    """
    run_id = self._run_indices[index]
    part = self._mesh_parts[seq]
    return self._TEMPLATES[part].format(run_id=run_id)
```

## Step 1c: Parallel Partitioning (Optional)

### `partition_indices()` Method

Override when the source has concurrency constraints:

```python
def partition_indices(self, indices: list[int]) -> list[list[int]] | None:
    """Group indices so all from same file go to same worker.

    Improves data locality and avoids concurrent access issues
    (e.g. LMDB single-env-per-process constraint).

    Parameters
    ----------
    indices : list[int]
        The indices to partition.

    Returns
    -------
    list[list[int]] | None
        Groups of indices (one per file), or ``None`` if all from one file.
    """
    from collections import defaultdict

    file_groups: dict[int, list[int]] = defaultdict(list)
    for idx in indices:
        file_idx = self._find_file_for_index(idx)
        file_groups[file_idx].append(idx)

    if len(file_groups) <= 1:
        return None

    return [sorted(file_groups[k]) for k in sorted(file_groups)]
```

### Param Declaration Patterns

**Required parameter** (no default — user must provide):

```python
Param(name="input_path", description="Path to data directory", type=str)
```

**URL with default** (remote dataset):

```python
Param(name="url", description="Base HuggingFace Hub URL", type=str, default=_DEFAULT_URL)
```

**Backend choice**:

```python
Param(
    name="backend",
    description="Reading backend: pyvista (default) or rust (faster)",
    type=str,
    default="pyvista",
    choices=["pyvista", "rust"],
)
```

**Glob pattern**:

```python
Param(name="file_pattern", description="Glob pattern for file discovery", type=str, default="**/*")
```

**Cache toggle**:

```python
Param(
    name="cache",
    description="Persist downloaded files across sessions",
    type=bool,
    default=True,
)
```

**Mesh conversion options** (for VTK-based sources):

```python
Param(
    name="manifold_dim",
    description="Target manifold dimension (auto, 0, 1, 2, 3)",
    type=str,
    default="auto",
    choices=["auto", "0", "1", "2", "3"],
),
Param(
    name="point_source",
    description="Point source mode: vertices or cell_centroids",
    type=str,
    default="vertices",
    choices=["vertices", "cell_centroids"],
),
Param(
    name="warn_on_lost_data",
    description="Warn when data arrays are discarded during conversion",
    type=bool,
    default=True,
),
```

**Metadata path** (optional supplementary file):

```python
Param(
    name="metadata_path",
    description="Path to metadata file (empty = auto-detect)",
    type=str,
    default="",
)
```

## Step 2: Write Tests

Create the test file at `test/domains/<domain>/test_<name>.py`.

### Test File Structure

```python
"""Tests for <ClassName>."""

# SPDX header (same as above)

from __future__ import annotations

import pathlib

import numpy as np
import pytest

pytestmark = pytest.mark.requires("<domain>")


# ---------------------------------------------------------------------------
# Mock data helper
# ---------------------------------------------------------------------------

def _write_mock_dataset(root: pathlib.Path, n_samples: int = 3) -> None:
    """Create a minimal mock dataset for unit tests.

    Write mock files that mirror the real dataset structure.
    Use small sizes (10-50 points, 3-5 samples).
    """
    ...


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class Test<ClassName>Unit:
    """Metadata and parameter tests (no data access)."""

    def test_params_list(self) -> None:
        from <module> import <ClassName>

        params = <ClassName>.params()
        assert len(params) > 0
        names = [p.name for p in params]
        assert "url" in names  # or "input_path", "data_dir"

    def test_name_and_description(self) -> None:
        from <module> import <ClassName>

        assert isinstance(<ClassName>.name, str)
        assert len(<ClassName>.name) > 0
        assert isinstance(<ClassName>.description, str)
        assert len(<ClassName>.description) > 0
```

### Local Mock Tests

```python
class Test<ClassName>Local:
    """Tests against local mock data."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: pathlib.Path) -> None:
        self.mock_root = tmp_path / "mock_dataset"
        _write_mock_dataset(self.mock_root)

    def test_len(self) -> None:
        from <module> import <ClassName>

        source = <ClassName>(url=str(self.mock_root))
        assert len(source) == 3  # matches mock data count

    def test_getitem_returns_correct_type(self) -> None:
        from physicsnemo.mesh import Mesh
        from <module> import <ClassName>

        source = <ClassName>(url=str(self.mock_root))
        item = next(source[0])
        assert isinstance(item, Mesh)
        assert item.n_points > 0

    def test_negative_index(self) -> None:
        from <module> import <ClassName>

        source = <ClassName>(url=str(self.mock_root))
        item_neg = next(source[-1])
        item_pos = next(source[len(source) - 1])
        assert item_neg.n_points == item_pos.n_points

    def test_index_out_of_bounds(self) -> None:
        from <module> import <ClassName>

        source = <ClassName>(url=str(self.mock_root))
        with pytest.raises(IndexError):
            next(source[len(source)])

    def test_correct_fields(self) -> None:
        """Verify expected fields are present."""
        from <module> import <ClassName>

        source = <ClassName>(url=str(self.mock_root))
        item = next(source[0])
        # Assert expected point_data fields
        assert "temperature" in item.point_data.keys()

    def test_different_indices_different_data(self) -> None:
        """Different indices should yield different data."""
        from <module> import <ClassName>

        source = <ClassName>(url=str(self.mock_root))
        item_0 = next(source[0])
        item_1 = next(source[1])
        # Should have different data (or at least different metadata)
        ...

    def test_empty_directory_raises(self) -> None:
        """Source raises ValueError on empty directory."""
        from <module> import <ClassName>

        empty = self.mock_root / "empty"
        empty.mkdir()
        with pytest.raises((ValueError, FileNotFoundError)):
            <ClassName>(url=str(empty))
```

### Sink Integration Tests

```python
    def test_relative_path(self) -> None:
        """relative_path returns POSIX path relative to root."""
        from <module> import <ClassName>

        source = <ClassName>(input_path=str(self.mock_root))
        rel = source.relative_path(0)
        assert "/" not in rel or not rel.startswith("/")
        assert pathlib.PurePosixPath(rel).suffix != ""

    def test_run_id(self) -> None:
        """run_id returns integer run identifier."""
        from <module> import <ClassName>

        source = <ClassName>(url=str(self.mock_root))
        rid = source.run_id(0)
        assert isinstance(rid, (int, str))
```

### Partition Tests (if applicable)

```python
    def test_partition_indices_groups_by_file(self) -> None:
        """Items from same file go to same group."""
        from <module> import <ClassName>

        source = <ClassName>(data_dir=str(self.mock_root))
        groups = source.partition_indices(list(range(len(source))))
        if groups is not None:
            # All indices covered
            flat = [idx for g in groups for idx in g]
            assert sorted(flat) == list(range(len(source)))

    def test_partition_single_file_returns_none(self) -> None:
        """Single file = no partitioning needed."""
        from <module> import <ClassName>

        # Create mock with single file
        source = <ClassName>(data_dir=str(self.single_file_root))
        assert source.partition_indices(list(range(len(source)))) is None
```

### Backend Tests

```python
    def test_backend_fallback(self) -> None:
        """Rust backend falls back gracefully if unavailable."""
        from <module> import <ClassName>

        # Should not raise even if rust is unavailable
        source = <ClassName>(input_path=str(self.mock_root), backend="rust")
        # Verify it works (may use python fallback)
        item = next(source[0])
        assert item.n_points > 0
```

### Pipeline Integration Tests

```python
@pytest.mark.e2e
class Test<ClassName>Pipeline:
    """End-to-end pipeline tests."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: pathlib.Path) -> None:
        self.mock_root = tmp_path / "mock"
        _write_mock_dataset(self.mock_root)

    def test_source_in_pipeline(self, tmp_path: pathlib.Path) -> None:
        """Source works as pipeline head."""
        from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink
        from <module> import <ClassName>

        source = <ClassName>(url=str(self.mock_root))
        sink = MeshSink(output_dir=str(tmp_path / "output"))
        pipeline = source.write(sink)

        assert len(pipeline) == len(source)
        paths = pipeline[0]
        assert len(paths) >= 1
        assert pathlib.Path(paths[0]).exists()
```

### Registry Tests

```python
class Test<ClassName>Registry:
    """Test that the source is registered."""

    def test_source_registered(self) -> None:
        from physicsnemo_curator.core.registry import registry

        names = [s.name for s in registry.list_sources("<domain>")]
        assert "<Display Name>" in names
```

### E2E Tests (Remote)

```python
@pytest.mark.e2e
@pytest.mark.slow
class Test<ClassName>E2E:
    """End-to-end tests against live dataset (requires network)."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: pathlib.Path) -> None:
        from <module> import <ClassName>

        self.source = <ClassName>(cache_storage=str(tmp_path / "cache"))

    def test_discovers_items(self) -> None:
        assert len(self.source) == <expected_count>

    def test_reads_first_item(self) -> None:
        from physicsnemo.mesh import Mesh

        item = next(self.source[0])
        assert isinstance(item, Mesh)
        assert item.n_points == <expected_points>
```

### Test Markers Reference

| Marker | Purpose |
|--------|---------|
| `pytestmark = pytest.mark.requires("mesh")` | Module-level: skip all if mesh deps missing |
| `@pytest.mark.requires("da")` | Skip if da dependencies not installed |
| `@pytest.mark.requires("atm")` | Skip if atm dependencies not installed |
| `@pytest.mark.integration` | Tests that touch filesystem |
| `@pytest.mark.e2e` | End-to-end test |
| `@pytest.mark.slow` | Slow test, excluded from quick CI runs |

### Running Tests

```bash
# Unit tests only (fast, no network)
uv run pytest test/domains/<domain>/test_<name>.py -v -k "not E2E"

# All tests including E2E (requires network)
uv run pytest test/domains/<domain>/test_<name>.py -v

# Just the E2E tests
uv run pytest test/domains/<domain>/test_<name>.py -v -k "E2E"
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

For **da** sources, edit `src/physicsnemo_curator/domains/da/__init__.py` with
the same pattern using `"da"` as the submodule name.

For **atm** sources, edit `src/physicsnemo_curator/domains/atm/__init__.py`
with `"atm"` as the submodule name.

## Step 4: Quality Checks

Run all checks before committing:

```bash
# Format
uv run ruff format \
  src/physicsnemo_curator/domains/<domain>/sources/<name>.py \
  test/domains/<domain>/test_<name>.py

# Lint
uv run ruff check --fix \
  src/physicsnemo_curator/domains/<domain>/sources/<name>.py \
  test/domains/<domain>/test_<name>.py

# Docstring coverage (must be >= 99%)
uv run interrogate

# Type checking
uv run ty check

# Run unit tests
uv run pytest test/domains/<domain>/test_<name>.py -v -k "not E2E"

# Run full test suite to check for regressions
uv run pytest test/domains/<domain>/ -v -k "not slow"
```

## Step 5: Commit

Use the `commit` tool (or follow Conventional Commits format):

```text
feat(<domain>): add <ClassName> for <dataset> datasets

<Optional body describing the dataset, format, and what it provides.>
```

## Checklist

Before considering the source complete, verify:

### Core

- [ ] SPDX license headers on all new files
- [ ] Source inherits from `Source[Mesh]`, `Source[xr.DataArray]`, or `Source[AtomicData]`
- [ ] `name` and `description` ClassVars are set
- [ ] `params()` returns all configurable parameters
- [ ] `__len__()` returns correct count
- [ ] `__getitem__()` yields correct type
- [ ] `__getitem__()` supports negative indexing
- [ ] `__getitem__()` raises `IndexError` for out-of-bounds
- [ ] Logging via `get_logger(self)` (not `logging.getLogger(__name__)`)

### Data Access

- [ ] Lazy loading for heavy data (geometry, fields)
- [ ] Caching for shared data across indices (e.g. shared geometry)
- [ ] `cache_storage` parameter for remote sources
- [ ] Backend validation with graceful fallback (if dual backend)
- [ ] Empty directory / no-files raises clear error message

### Sink Integration (if applicable)

- [ ] `relative_path(index)` for file-based sources
- [ ] `root` property for file-based sources
- [ ] `run_id(index)` for run-based datasets
- [ ] `mesh_name(index, seq)` for multi-output sources

### Parallel Safety (if applicable)

- [ ] `partition_indices()` groups items with concurrency constraints
- [ ] Returns `None` when no partitioning needed

### Documentation

- [ ] NumPy-style docstrings on class, `__init__`, `__len__`, `__getitem__`
- [ ] Dataset reference links (paper, URL, license) in class docstring
- [ ] `from __future__ import annotations` at top
- [ ] `TYPE_CHECKING` block for `Generator` and domain type imports

### Registration

- [ ] Registered in domain `__init__.py` with `registry.register_source()`
- [ ] Added to `__all__` in domain `__init__.py`

### Testing

- [ ] Unit tests: params, name, description
- [ ] Local mock tests: len, getitem, negative index, OOB, fields, empty dir
- [ ] Sink integration tests: relative_path, run_id (if applicable)
- [ ] Partition tests (if applicable): correct grouping, None for single file
- [ ] Backend fallback test (if dual backend)
- [ ] Pipeline test: source works as pipeline head (marked `@pytest.mark.e2e`)
- [ ] Registry test: source is discoverable
- [ ] E2E tests: marked `@pytest.mark.e2e` and `@pytest.mark.slow`

### Quality Gates

- [ ] `ruff format` clean
- [ ] `ruff check` clean
- [ ] `interrogate` >= 99%
- [ ] `ty check` clean
- [ ] All unit tests pass
- [ ] No regressions in existing tests

## Reference: Source[T] ABC

From `src/physicsnemo_curator/core/base.py`:

```python
class Source[T](ABC):
    """Abstract data source that yields items of type *T*.

    A source represents a collection of data items (e.g. files on disk).
    Each item is accessed by integer index and may yield one or more *T*
    objects (generator semantics allow a single source item to expand into
    multiple outputs).

    Subclasses must set the class-level :attr:`name` and :attr:`description`
    attributes and implement :meth:`params`, :meth:`__len__`, and
    :meth:`__getitem__`.
    """

    name: ClassVar[str]
    """Human-readable display name for the interactive CLI."""
    description: ClassVar[str]
    """Short description shown in the interactive CLI."""

    @classmethod
    @abstractmethod
    def params(cls) -> list[Param]: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> Generator[T]: ...

    def partition_indices(self, indices: list[int]) -> list[list[int]] | None:
        """Group indices into partitions that MUST be processed by the same worker.

        Override when the source has constraints on concurrent access
        (e.g. LMDB allows only one environment open per file per process).

        Returns ``None`` by default (no partitioning required).
        """
        return None

    # -- Convenience builder methods --
    def filter(self, f: Filter[T]) -> Pipeline[T]: ...
    def write(self, s: Sink[T]) -> Pipeline[T]: ...
```

Key points:
- `__getitem__` returns `Generator[T]` (can yield multiple items per index)
- `partition_indices()` is optional (same as on Sink)
- `write()` automatically calls `s.set_source(self)` if sink has that method
- Sources do NOT have `dashboard_panel()`, `artifacts()`, or `flush()`

## Reference: Param Dataclass

```python
@dataclass(frozen=True)
class Param:
    """Descriptor for a configurable parameter."""

    name: str
    description: str
    type: type
    default: Any = REQUIRED  # Sentinel: no default = required param
    choices: list[Any] | None = None
```

- Use `REQUIRED` sentinel (or omit `default`) for mandatory parameters
- `choices` constrains values to a finite set
- `type` is used for CLI parsing and validation
