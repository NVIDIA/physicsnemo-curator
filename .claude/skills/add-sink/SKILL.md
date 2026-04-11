---
name: add-sink
description: >
  Guide for adding a new sink to PhysicsNeMo Curator. Covers discovery
  questions, implementation patterns (simple writer, append-based, split-based),
  output naming, testing, and registration.
---

# Adding a New Sink

This skill walks through adding a new `Sink` to PhysicsNeMo Curator,
from initial design through implementation, testing, and registration.

## Step 0: Gather Requirements

Before writing any code, answer these questions. Ask the user if anything
is unclear.

### Domain

Which submodule does this sink belong to?

| Domain | Type parameter | Submodule | Dependency group |
|--------|---------------|-----------|-----------------|
| **mesh** | `Sink["Mesh"]` | `src/physicsnemo_curator/domains/mesh/` | `mesh` (physicsnemo, pyvista, pyarrow, torch) |
| **da** | `Sink["xr.DataArray"]` | `src/physicsnemo_curator/domains/da/` | `da` (xarray, earth2studio, zarr) |

### Sink Design

1. **Output format** — What file format will the sink produce? (tensordict
   memmap, Zarr, NetCDF4, HDF5, Parquet, VTK, NumPy, custom)
2. **Naming strategy** — How are output files/directories named?
   - **Index-based**: Use pipeline `index` (e.g. `mesh_0001_0`). Simplest approach.
   - **Data-driven**: Use coordinate metadata from the data itself (e.g.
     variable names, time values). Used when data carries its own identity.
3. **Append or overwrite?** — When the same output path is hit twice, should
   the sink append to the existing file or overwrite it?
4. **Splitting?** — Should the output be split along a dimension? (e.g.
   one file per variable, one file per year, one file per run)
5. **Chunking/compression?** — Does the format support chunking or
   compression? Should these be configurable?
6. **Parameters** — What user-configurable parameters does the sink need?
   (output directory, chunk sizes, compression level, format options, etc.)
7. **Dependencies** — Does it require additional imports beyond the domain
   group? (e.g. `netCDF4`, `h5py`, `zarr`, specific I/O libraries)
8. **Streaming?** — Can items be written one-at-a-time as the iterator is
   consumed, or does the sink need to buffer all items first?

### Reference: Existing Sinks

| Sink | File | Type | Pattern | Good example for |
|------|------|------|---------|-----------------|
| `MeshSink` | `mesh/sinks/mesh_writer.py` | Mesh | Index-based naming, one dir per mesh | Simplest sink |
| `ZarrSink` | `da/sinks/zarr_writer.py` | DataArray | Data-driven, append | Zarr, chunking, sharding |
| `NetCDF4Sink` | `da/sinks/netcdf_writer.py` | DataArray | Data-driven, split | Splitting, compression |

## Step 1: Create the Sink Module

Create the sink file at the appropriate location:

- **mesh**: `src/physicsnemo_curator/domains/mesh/sinks/<name>.py`
- **da**: `src/physicsnemo_curator/domains/da/sinks/<name>.py`

### Required SPDX Header

Every file MUST start with:

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
"""Brief description of what this sink writes."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, ClassVar

from physicsnemo_curator.core.base import Param, Sink

if TYPE_CHECKING:
    from collections.abc import Iterator

    from physicsnemo.mesh import Mesh  # or: import xarray as xr
```

### Sink Class Template

```python
class <ClassName>(Sink["Mesh"]):
    """Write :class:`~physicsnemo.mesh.Mesh` objects to <format>.

    <2-3 sentence description of what the sink writes, how it names
    outputs, and any append/split behaviour.>

    Parameters
    ----------
    output_dir : str
        Directory where outputs will be written.

    Examples
    --------
    >>> sink = <ClassName>(output_dir="./output/")  # doctest: +SKIP
    >>> paths = sink(mesh_iterator, index=0)  # doctest: +SKIP
    """

    name: ClassVar[str] = "<Display Name>"
    description: ClassVar[str] = "<Short one-line description for CLI>"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for this sink.

        Returns
        -------
        list[Param]
            Parameter descriptors.
        """
        return [
            Param(
                name="output_dir",
                description="Output directory for files",
                type=str,
            ),
            # Add format-specific params (chunks, compression, etc.)
        ]

    def __init__(self, output_dir: str) -> None:
        self._output_dir = pathlib.Path(output_dir)

    def __call__(self, items: Iterator[Mesh], index: int) -> list[str]:
        """Consume items and write each to storage.

        Parameters
        ----------
        items : Iterator[Mesh]
            Stream of data items to persist.
        index : int
            Source index (used for naming output files).

        Returns
        -------
        list[str]
            Paths of the files written.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []

        for seq, item in enumerate(items):
            out_path = self._output_dir / f"item_{index:04d}_{seq}"
            self._write_item(item, out_path)
            paths.append(str(out_path))

        return paths

    def _write_item(self, item: Mesh, path: pathlib.Path) -> None:
        """Write a single item to disk.

        Parameters
        ----------
        item : Mesh
            Data item to write.
        path : pathlib.Path
            Output file/directory path.
        """
        # Implementation here
        ...
```

### Key Design Decisions

**`__call__` receives `Iterator[T]` and `index: int`, returns `list[str]`.**

This is the `Sink[T]` ABC contract. Key differences from filters:

- Sinks **consume** the iterator (no yielding back)
- Sinks **return paths** of written files as `list[str]`
- The `index` parameter comes from the source (pipeline position)
- Sinks create output directories as needed (`mkdir(parents=True, exist_ok=True)`)

### Implementation Patterns

#### Pattern A: Index-Based Naming (MeshSink)

The simplest pattern. Uses the pipeline `index` and a sequence number
for output naming. Good when each source item maps to one output.

```python
def __call__(self, items: Iterator[Mesh], index: int) -> list[str]:
    self._output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    for seq, mesh in enumerate(items):
        subdir = self._output_dir / f"mesh_{index:04d}_{seq}"
        mesh.save(str(subdir))
        paths.append(str(subdir))

    return paths
```

Best for: mesh formats, one-item-per-file outputs, simple datasets.

#### Pattern B: Data-Driven Naming (ZarrSink)

Uses coordinate metadata from the data itself (e.g. variable names)
instead of the pipeline index. Good when data carries identity info.

```python
def __call__(self, items: Iterator[xr.DataArray], index: int) -> list[str]:
    paths: list[str] = []

    for da in items:
        written = self._write_dataarray(da)
        paths.extend(written)

    return paths

def _write_dataarray(self, da: xr.DataArray) -> list[str]:
    paths: list[str] = []

    if "variable" not in da.dims:
        group_path = self._output_path / "data"
        self._append_to_store(da, group_path)
        paths.append(str(group_path))
        return paths

    # Split along variable dimension
    for var_name in da.coords["variable"].values:
        var_da = da.sel(variable=var_name).drop_vars("variable")
        group_path = self._output_path / str(var_name)
        self._append_to_store(var_da, group_path)
        paths.append(str(group_path))

    return paths
```

Best for: Zarr stores, multi-variable datasets, append-oriented formats.

#### Pattern C: Data-Driven with Coordinate Splitting (NetCDF4Sink)

Extends Pattern B with an additional split along a coordinate dimension
(e.g. one file per year). Most complex pattern.

```python
def _write_variable(self, da: xr.DataArray, var_name: str) -> list[str]:
    if self._split_dim is None or self._split_dim not in da.dims:
        nc_path = self._output_dir / var_name / "data.nc"
        self._append_to_file(da, nc_path)
        return [str(nc_path)]

    # Group by split key (e.g. year)
    groups = self._group_by_split(da)
    paths: list[str] = []
    for split_key, group_da in groups.items():
        nc_path = self._output_dir / var_name / f"{split_key}.nc"
        self._append_to_file(group_da, nc_path)
        paths.append(str(nc_path))

    return paths
```

Best for: NetCDF4, time-series data, formats that benefit from splitting.

### Append Logic

If the sink supports appending to existing files, implement a two-branch
write method:

```python
def _append_to_store(self, da: xr.DataArray, path: pathlib.Path) -> None:
    ds = da.to_dataset(name="data")

    if path.exists():
        # Append along the appropriate dimension
        ds.to_zarr(store=str(path), mode="a", append_dim="time")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_zarr(store=str(path), mode="w", encoding=self._encoding)
```

### Chunking and Compression

For formats that support chunking (Zarr, NetCDF4, HDF5), build encoding
dicts from user parameters:

```python
_DEFAULT_CHUNKS: ClassVar[dict[str, int]] = {"time": 1, "lat": 721, "lon": 1440}

def _build_encoding(self, da: xr.DataArray) -> dict[str, dict[str, Any]]:
    chunk_tuple = tuple(
        self._chunks.get(str(d), da.sizes[d]) for d in da.dims
    )
    enc: dict[str, Any] = {"chunksizes": chunk_tuple}
    if self._compression_level > 0:
        enc["zlib"] = True
        enc["complevel"] = self._compression_level
    return {"data": enc}
```

### Param Declaration Patterns

**Required parameter** (no default — user must provide):

```python
Param(name="output_dir", description="Output directory for files", type=str)
```

**Chunking as CLI-friendly string** (parsed in `__init__`):

```python
Param(
    name="chunks",
    description="Chunk sizes as dim:size pairs (e.g. time:1,lat:721,lon:1440)",
    type=str,
    default="time:1,lat:721,lon:1440",
)
```

**Compression level with range**:

```python
Param(
    name="compression_level",
    description="Zlib compression level (0=off, 9=max)",
    type=int,
    default=4,
)
```

**Optional split dimension** (None disables):

```python
Param(
    name="split_dim",
    description="Dimension to split files on (default: time, None=no split)",
    type=str,
    default="time",
)
```

### Property Accessors

Expose key configuration as read-only properties for testing and
introspection:

```python
@property
def output_dir(self) -> pathlib.Path:
    """Return the output directory path."""
    return self._output_dir

@property
def compression_level(self) -> int:
    """Return the configured compression level."""
    return self._compression_level
```

## Step 2: Write Tests

Create the test file at `test/<domain>/test_<name>.py`.

### Test File Structure

```python
"""Tests for <ClassName>."""

# SPDX header (same as source files)

from __future__ import annotations

import pathlib

import pytest

pytestmark = pytest.mark.requires("<domain>")


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _create_test_data(...) -> ...:
    """Create minimal test data for unit tests.

    Use small sizes (10-50 points, 3-5 time steps).
    """
    ...
```

### Test Classes

#### Unit Tests (Metadata)

```python
class Test<ClassName>Unit:
    """Metadata tests."""

    def test_params_list(self) -> None:
        from <module> import <ClassName>

        params = <ClassName>.params()
        assert len(params) > 0
        names = [p.name for p in params]
        assert "output_dir" in names  # or "output_path"

    def test_name_and_description(self) -> None:
        from <module> import <ClassName>

        assert isinstance(<ClassName>.name, str)
        assert len(<ClassName>.name) > 0
        assert len(<ClassName>.description) > 0
```

#### Write Tests

```python
class Test<ClassName>Write:
    """Tests for writing data."""

    def test_writes_single_item(self, tmp_path: pathlib.Path) -> None:
        """Sink writes one item and returns correct path."""
        sink = <ClassName>(output_dir=str(tmp_path / "out"))
        items = iter([_create_test_data()])
        paths = sink(items, index=0)

        assert len(paths) == 1
        assert pathlib.Path(paths[0]).exists()

    def test_writes_multiple_items(self, tmp_path: pathlib.Path) -> None:
        """Sink writes multiple items from one index."""
        sink = <ClassName>(output_dir=str(tmp_path / "out"))
        items = iter([_create_test_data(), _create_test_data()])
        paths = sink(items, index=0)

        assert len(paths) >= 1
        for p in paths:
            assert pathlib.Path(p).exists()

    def test_output_naming_uses_index(self, tmp_path: pathlib.Path) -> None:
        """Output paths incorporate the source index."""
        sink = <ClassName>(output_dir=str(tmp_path / "out"))
        paths_0 = sink(iter([_create_test_data()]), index=0)
        paths_1 = sink(iter([_create_test_data()]), index=1)

        assert paths_0[0] != paths_1[0]

    def test_creates_output_directory(self, tmp_path: pathlib.Path) -> None:
        """Sink creates output dir if it doesn't exist."""
        out = tmp_path / "nested" / "dir"
        sink = <ClassName>(output_dir=str(out))
        sink(iter([_create_test_data()]), index=0)

        assert out.exists()

    def test_empty_iterator(self, tmp_path: pathlib.Path) -> None:
        """Empty iterator produces no output paths."""
        sink = <ClassName>(output_dir=str(tmp_path / "out"))
        paths = sink(iter([]), index=0)

        assert paths == []
```

#### Roundtrip Tests (Read Back)

```python
    def test_roundtrip(self, tmp_path: pathlib.Path) -> None:
        """Written data can be read back and matches original."""
        original = _create_test_data()
        sink = <ClassName>(output_dir=str(tmp_path / "out"))
        paths = sink(iter([original]), index=0)

        # Read back and verify
        loaded = _read_output(paths[0])
        # Assert data matches original
```

#### Append Tests (if applicable)

```python
    def test_append_to_existing(self, tmp_path: pathlib.Path) -> None:
        """Sink appends to existing file rather than overwriting."""
        sink = <ClassName>(output_dir=str(tmp_path / "out"))

        # Write first batch
        paths_1 = sink(iter([_create_test_data(time_start=0)]), index=0)
        # Write second batch (same output path)
        paths_2 = sink(iter([_create_test_data(time_start=1)]), index=1)

        # Verify the file contains both batches
        ...
```

#### Split Tests (if applicable)

```python
    def test_splits_by_variable(self, tmp_path: pathlib.Path) -> None:
        """Data with variable dimension is split into separate outputs."""
        da = _create_test_data(variables=["temperature", "pressure"])
        sink = <ClassName>(output_dir=str(tmp_path / "out"))
        paths = sink(iter([da]), index=0)

        assert len(paths) == 2
        # Verify separate outputs for each variable
```

#### Pipeline Integration Tests

```python
@pytest.mark.e2e
class Test<ClassName>Pipeline:
    """End-to-end pipeline tests."""

    def test_in_pipeline(self, tmp_path: pathlib.Path) -> None:
        """Sink works correctly at the end of a pipeline."""
        from physicsnemo_curator.domains.mesh.sources.vtk import VTKSource
        # or: from physicsnemo_curator.domains.da.sources.era5 import ERA5Source

        # Create source data
        ...

        sink = <ClassName>(output_dir=str(tmp_path / "output"))

        pipeline = source.write(sink)

        for i in range(len(pipeline)):
            paths = pipeline[i]
            assert len(paths) >= 1
            for p in paths:
                assert pathlib.Path(p).exists()
```

#### Registry Tests

```python
class Test<ClassName>Registry:
    """Test that the sink is registered."""

    def test_sink_registered(self) -> None:
        from physicsnemo_curator.core.registry import registry

        names = [s.name for s in registry.list_sinks("<domain>")]
        assert "<Display Name>" in names
```

### Test Markers Reference

| Marker | Purpose |
|--------|---------|
| `pytestmark = pytest.mark.requires("mesh")` | Module-level: skip all if mesh deps missing |
| `@pytest.mark.requires("da")` | Skip if da dependencies not installed |
| `@pytest.mark.integration` | Tests that touch filesystem |
| `@pytest.mark.e2e` | End-to-end pipeline tests |
| `@pytest.mark.slow` | Slow tests, excluded from quick CI |

### Running Tests

```bash
# Unit and write tests (fast, no network)
uv run pytest test/<domain>/test_<name>.py -v

# Pipeline integration test
uv run pytest test/<domain>/test_<name>.py -v -k "Pipeline"

# Full domain test suite for regressions
uv run pytest test/<domain>/ -v -k "not slow"
```

## Step 3: Register the Sink

### Edit the domain `__init__.py`

For **mesh** sinks, edit `src/physicsnemo_curator/domains/mesh/__init__.py`:

```python
# Add import (alphabetical order among sinks)
from physicsnemo_curator.domains.mesh.sinks.<module> import <ClassName>

# Add registration (after existing register_sink calls)
registry.register_sink("mesh", <ClassName>)

# Add to __all__ (alphabetical order)
__all__ = [
    ...,
    "<ClassName>",
    ...,
]
```

For **da** sinks, edit `src/physicsnemo_curator/domains/da/__init__.py` with
the same pattern using `"da"` as the submodule name.

## Step 4: Quality Checks

Run all checks before committing:

```bash
# Format
uv run ruff format \
  src/physicsnemo_curator/<domain>/sinks/<name>.py \
  test/<domain>/test_<name>.py

# Lint
uv run ruff check --fix \
  src/physicsnemo_curator/<domain>/sinks/<name>.py \
  test/<domain>/test_<name>.py

# Docstring coverage (must be >= 99%)
uv run interrogate

# Type checking
uv run ty check

# Run sink tests
uv run pytest test/<domain>/test_<name>.py -v

# Run full domain test suite for regressions
uv run pytest test/<domain>/ -v -k "not slow"
```

## Step 5: Commit

Use the `commit` tool (or follow Conventional Commits format):

```text
feat(<domain>): add <ClassName> for <format> output

<Optional body describing the output format, naming, and features.>
```

## Checklist

Before considering the sink complete, verify:

- [ ] SPDX license headers on all new files
- [ ] Sink inherits from `Sink["Mesh"]` or `Sink["xr.DataArray"]`
- [ ] `name` and `description` ClassVars are set
- [ ] `params()` returns all configurable parameters
- [ ] `__call__()` receives `Iterator[T]` and `index: int`, returns `list[str]`
- [ ] Iterator is consumed lazily (items written as they arrive)
- [ ] Output directory is created automatically (`mkdir(parents=True, exist_ok=True)`)
- [ ] Empty iterator returns `[]` (no crash, no empty files)
- [ ] NumPy-style docstrings on class, `__init__`, `__call__`, and helpers
- [ ] `from __future__ import annotations` at top
- [ ] `TYPE_CHECKING` block for `Iterator` and domain type imports
- [ ] Read-only `@property` accessors for key config values
- [ ] Registered in domain `__init__.py` with `registry.register_sink()`
- [ ] Added to `__all__` in domain `__init__.py`
- [ ] Unit tests: params, name, description
- [ ] Write tests: single item, multiple items, naming, directory creation, empty iterator
- [ ] Roundtrip test: written data can be read back correctly
- [ ] Append tests (if applicable): data accumulates rather than overwrites
- [ ] Split tests (if applicable): data is split by variable/coordinate
- [ ] Pipeline test: sink works at end of a pipeline (marked `@pytest.mark.e2e`)
- [ ] Registry test: sink is discoverable
- [ ] `ruff format` clean
- [ ] `ruff check` clean
- [ ] `interrogate` >= 99%
- [ ] `ty check` clean
- [ ] All tests pass
- [ ] No regressions in existing tests

## Reference: Sink[T] ABC

From `src/physicsnemo_curator/core/base.py`:

```python
class Sink[T](ABC):
    """Abstract sink that persists items and returns output file paths."""

    name: ClassVar[str]
    description: ClassVar[str]

    @classmethod
    @abstractmethod
    def params(cls) -> list[Param]: ...

    @abstractmethod
    def __call__(self, items: Iterator[T], index: int) -> list[str]: ...
```

Key differences from `Source[T]` and `Filter[T]`:

- Sources have `__len__` and `__getitem__` — sinks do not
- Filters receive and return `Generator[T]` — sinks consume `Iterator[T]`
  and return `list[str]` (file paths)
- Sinks are the terminal stage: they **consume** items, they do not yield
- The `index` parameter is the source-level index from the pipeline
- Return value is a list of all file paths written (can be empty)
