---
name: add-sink
description: >
  Guide for adding a new sink to PhysicsNeMo Curator. Covers discovery
  questions, implementation patterns (simple writer, append-based, split-based),
  output naming, parallel partitioning, testing, and registration.
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
| **atm** | `Sink["AtomicData"]` | `src/physicsnemo_curator/domains/atm/` | `mesh` (nvalchemi, torch) |

### Sink Design

1. **Output format** — What file format will the sink produce? (tensordict
   memmap, Zarr, NetCDF4, HDF5, Parquet, VTK, NumPy, custom)
2. **Naming strategy** — How are output files/directories named?
   - **Index-based**: Use pipeline `index` (e.g. `mesh_0001_0`). Simplest approach.
   - **Template-based**: Use `naming_template` with placeholders like
     `{index}`, `{seq}`, `{relpath}`, `{stem}`, `{run_id}`, `{mesh_name}`.
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
9. **Parallel safety** — Does the sink need to coordinate concurrent writes?
   (e.g. multiple workers writing to the same Zarr chunk). If so, implement
   `partition_indices()`.
10. **Source integration** — Does the sink need metadata from the source for
    output naming? (directory mirroring, run IDs, etc.) If so, implement
    `set_source()`.

### Reference: Existing Sinks

| Sink | File | Type | Pattern | Good example for |
|------|------|------|---------|-----------------|
| `MeshSink` | `mesh/sinks/mesh_writer.py` | Mesh | Index/template naming, tensordict | Naming templates, set_source |
| `MeshVTUSink` | `mesh/sinks/mesh_vtu.py` | Mesh | Index/template naming, VTK | Atomic writes, complex serialization |
| `MeshZarrSink` | `mesh/sinks/mesh_zarr.py` | Mesh | Index naming, Zarr | Compression, chunking, reconstruction |
| `ZarrSink` | `da/sinks/zarr_writer.py` | DataArray | Data-driven, append | Chunk-aligned partitioning |
| `NetCDF4Sink` | `da/sinks/netcdf_writer.py` | DataArray | Data-driven, split | Splitting, compression, atomic append |
| `AtomicDataZarrSink` | `atm/sinks/zarr_writer.py` | AtomicData | Dual-mode (seq/parallel) | Pre-allocated stores, batch writes |

## Step 1: Create the Sink Module

Create the sink file at the appropriate location:

- **mesh**: `src/physicsnemo_curator/domains/mesh/sinks/<name>.py`
- **da**: `src/physicsnemo_curator/domains/da/sinks/<name>.py`
- **atm**: `src/physicsnemo_curator/domains/atm/sinks/<name>.py`

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
import time
from typing import TYPE_CHECKING, ClassVar

from physicsnemo_curator.core.base import Param, Sink
from physicsnemo_curator.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

    from physicsnemo.mesh import Mesh  # or: import xarray as xr
    # or: from physicsnemo_curator.domains.atm.types import AtomicData
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
    naming_template : str or None, optional
        Format string for output names. Placeholders: ``{index}``,
        ``{seq}``, ``{relpath}``, ``{stem}``. Default: ``mesh_{index:04d}_{seq}``.

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
            Param(
                name="naming_template",
                description=(
                    "Format string for output names. "
                    "Placeholders: {index}, {seq}, {relpath}, {stem}. "
                    "Default: mesh_{index:04d}_{seq}"
                ),
                type=str,
                default=None,
            ),
        ]

    def __init__(
        self,
        output_dir: str,
        naming_template: str | None = None,
    ) -> None:
        """Initialize the sink.

        Parameters
        ----------
        output_dir : str
            Output directory for files.
        naming_template : str or None, optional
            Format string for output names.
        """
        self._output_dir = pathlib.Path(output_dir)
        self._naming_template = naming_template
        self._source: Source[Mesh] | None = None
        self._log = get_logger(self)

    def set_source(self, source: Source[Mesh]) -> None:
        """Inject source reference for placeholder resolution.

        Called automatically by :meth:`Pipeline.write`.

        Parameters
        ----------
        source : Source[Mesh]
            The upstream source providing items.
        """
        self._source = source

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
        self._log.info("idx_%d: Starting write", index)
        t0 = time.perf_counter()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []

        for seq, item in enumerate(items):
            out_path = self._resolve_path(index, seq)
            self._write_item(item, out_path)
            paths.append(str(out_path))

        elapsed = time.perf_counter() - t0
        self._log.info("idx_%d: Wrote %d items (%.2fs)", index, len(paths), elapsed)
        return paths

    def _resolve_path(self, index: int, seq: int) -> pathlib.Path:
        """Resolve output path from naming template.

        Parameters
        ----------
        index : int
            Pipeline source index.
        seq : int
            Sequence number within the index.

        Returns
        -------
        pathlib.Path
            Resolved output path.
        """
        if self._naming_template is None:
            name = f"mesh_{index:04d}_{seq}"
        else:
            kwargs: dict[str, object] = {"index": index, "seq": seq}
            if self._source is not None:
                if hasattr(self._source, "relative_path"):
                    rel = self._source.relative_path(index)
                    kwargs["relpath"] = str(pathlib.PurePosixPath(rel).parent)
                    kwargs["stem"] = pathlib.PurePosixPath(rel).stem
            name = self._naming_template.format(**kwargs)
        return self._output_dir / name

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

    @property
    def output_dir(self) -> pathlib.Path:
        """Return the output directory path."""
        return self._output_dir
```

### Key Design Decisions

**`__call__` receives `Iterator[T]` and `index: int`, returns `list[str]`.**

This is the `Sink[T]` ABC contract. Key differences from filters:

- Sinks **consume** the iterator (no yielding back)
- Sinks **return paths** of written files as `list[str]`
- The `index` parameter comes from the source (pipeline position)
- Sinks create output directories as needed (`mkdir(parents=True, exist_ok=True)`)

### Logging

Use the structured component logger from `physicsnemo_curator.core.logging`:

```python
from physicsnemo_curator.core.logging import get_logger

class MySink(Sink["Mesh"]):
    def __init__(self, ...) -> None:
        self._log = get_logger(self)

    def __call__(self, items: Iterator[Mesh], index: int) -> list[str]:
        self._log.info("idx_%d: Starting write", index)
        t0 = time.perf_counter()
        # ... write items ...
        elapsed = time.perf_counter() - t0
        self._log.info("idx_%d: Wrote %d items (%.2fs)", index, n, elapsed)
```

The `get_logger(self)` wrapper provides:
- Automatic component name prefixing
- Process-aware output format: `[ProcessName:PID] ComponentName: message`
- Standard Python logging levels (debug, info, warning, error)

**Do NOT use `logging.getLogger(__name__)`** — always use `get_logger(self)`.

### Implementation Patterns

#### Pattern A: Index-Based with Naming Template (MeshSink)

The simplest pattern. Uses `naming_template` with `{index}` and `{seq}`
placeholders. Supports source-backed placeholders via `set_source()`.

```python
def __call__(self, items: Iterator[Mesh], index: int) -> list[str]:
    self._log.info("idx_%d: Starting write", index)
    t0 = time.perf_counter()
    self._output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    for seq, mesh in enumerate(items):
        name = self._resolve_name(index, seq)
        subdir = self._output_dir / name
        mesh.save(str(subdir))
        paths.append(str(subdir))

    elapsed = time.perf_counter() - t0
    self._log.info("idx_%d: Wrote %d items (%.2fs)", index, len(paths), elapsed)
    return paths
```

Best for: mesh formats, one-item-per-file outputs, simple datasets.

**Naming template placeholders** (MeshSink / MeshVTUSink):

| Placeholder | Source | Description |
|-------------|--------|-------------|
| `{index}` | Pipeline | Zero-based source index |
| `{seq}` | Sink | Sequence number within index |
| `{relpath}` | `source.relative_path()` | Parent directory of source file |
| `{stem}` | `source.relative_path()` | Filename without extension |
| `{run_id}` | `source.run_id()` | Run identifier from source |
| `{mesh_name}` | `source.mesh_name(index, seq)` | Mesh identifier |

#### Pattern B: Data-Driven Naming with Append (ZarrSink)

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

#### Pattern D: Dual-Mode Sequential/Pre-allocated (AtomicDataZarrSink)

The most sophisticated pattern. Supports two execution modes:

- **Sequential mode** (default): uses a third-party writer library with
  batched append semantics. One store per index.
- **Pre-allocated parallel mode**: allocates the full Zarr store upfront
  at construction time, then workers write to non-overlapping offsets
  with zero locking.

```python
def __init__(
    self,
    output_path: str,
    batch_size: int = 1000,
    natoms: np.ndarray | None = None,
    schema: AtomicData | None = None,
    chunk_size: int = 1024,
) -> None:
    self._output_path = pathlib.Path(output_path)
    self._batch_size = batch_size
    self._chunk_size = chunk_size

    # Pre-allocated mode if natoms and schema provided
    if natoms is not None and schema is not None:
        self._parallel = True
        self._preallocate(natoms, schema)
    else:
        self._parallel = False

def __call__(self, items: Iterator[AtomicData], index: int) -> list[str]:
    if self._parallel:
        return self._write_parallel(items, index)
    return self._write_sequential(items, index)

def _write_sequential(self, items: Iterator[AtomicData], index: int) -> list[str]:
    """Batch items and flush via writer.write()/append()."""
    batch: list[AtomicData] = []
    for item in items:
        batch.append(item)
        if len(batch) >= self._batch_size:
            self._flush(batch, store_key)
            batch = []
    if batch:
        self._flush(batch, store_key)
    return [str(store_path)]

def _write_parallel(self, items: Iterator[AtomicData], index: int) -> list[str]:
    """Write directly to pre-computed offset in Zarr store."""
    for item in items:
        offset = self._atom_offsets[index]
        store[field_path][offset:offset + n_atoms] = arr
    return [str(self._output_path)]
```

Best for: Molecular/atomic datasets with known sizes, maximum parallel throughput.

### Atomic Writes

For formats that don't support native append, use temp-file + rename to
prevent partial/corrupt outputs:

```python
def _write_item(self, item: Mesh, path: pathlib.Path) -> None:
    """Write item atomically using temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f".{path.stem}_temp{path.suffix}"
    try:
        self._serialize(item, temp_path)
        temp_path.rename(path)  # Atomic on same filesystem
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
```

Used by: `MeshVTUSink`

### Append Logic

If the sink supports appending to existing files, implement a two-branch
write method:

```python
def _append_to_store(self, da: xr.DataArray, path: pathlib.Path) -> None:
    """Append data to existing store, or create new one."""
    ds = da.to_dataset(name="data")

    if path.exists():
        # Append along the appropriate dimension
        ds.to_zarr(store=str(path), mode="a", append_dim="time")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_zarr(store=str(path), mode="w", encoding=self._encoding)
```

For NetCDF4 (which doesn't support native append), use load + concat + rewrite:

```python
def _append_existing(self, ds: xr.Dataset, nc_path: pathlib.Path) -> None:
    """Atomic append: load existing → concat → write temp → rename."""
    existing = xr.open_dataset(nc_path)
    combined = xr.concat([existing, ds], dim="time")
    existing.close()

    temp_path = nc_path.parent / f".{nc_path.stem}_temp.nc"
    combined.to_netcdf(temp_path, encoding=self._encoding)
    temp_path.rename(nc_path)
```

### Chunking and Compression

For formats that support chunking (Zarr, NetCDF4, HDF5), build encoding
dicts from user parameters:

```python
_DEFAULT_CHUNKS: ClassVar[dict[str, int]] = {"time": 1, "lat": 721, "lon": 1440}

def _build_encoding(self, da: xr.DataArray) -> dict[str, dict[str, Any]]:
    """Build encoding dict for xarray write operations."""
    chunk_tuple = tuple(
        self._chunks.get(str(d), da.sizes[d]) for d in da.dims
    )
    enc: dict[str, Any] = {"chunksizes": chunk_tuple}
    if self._compression_level > 0:
        enc["zlib"] = True
        enc["complevel"] = self._compression_level
    return {"data": enc}
```

For Zarr with Blosc compression:

```python
import zarr

def _build_zarr_compressor(self) -> zarr.codecs.BloscCodec:
    """Build Blosc codec for Zarr v3."""
    return zarr.codecs.BloscCodec(
        cname="zstd",
        clevel=self._compression_level,
    )
```

For adaptive chunk sizing based on target memory:

```python
def _compute_chunks(
    shape: tuple[int, ...], dtype: np.dtype, target_mb: float
) -> tuple[int, ...]:
    """Compute chunk sizes targeting a given memory footprint.

    Parameters
    ----------
    shape : tuple[int, ...]
        Array shape.
    dtype : np.dtype
        Array element type.
    target_mb : float
        Target chunk size in megabytes.

    Returns
    -------
    tuple[int, ...]
        Chunk dimensions.
    """
    target_bytes = target_mb * 1024 * 1024
    itemsize = dtype.itemsize
    # ... adaptive logic based on dimensionality ...
```

## Step 1b: Parallel Partitioning (Optional)

### `partition_indices()` Method

Override this method when the sink has constraints on concurrent writes.
The pipeline runner uses it to group indices that must be processed by
the same worker.

```python
def partition_indices(self, indices: list[int]) -> list[list[int]] | None:
    """Group indices into partitions for same-worker processing.

    Each returned group is a list of indices that must be handled
    sequentially by a single worker. The runner never splits a
    group across workers.

    Parameters
    ----------
    indices : list[int]
        The indices to partition.

    Returns
    -------
    list[list[int]] | None
        Partitioned groups, or ``None`` if no partitioning required.
    """
    return None  # Default: no constraints
```

#### Chunk-Aligned Partitioning (ZarrSink pattern)

When multiple indices map to the same Zarr chunk, they must be written
by the same worker to avoid corruption:

```python
from collections import defaultdict

def partition_indices(self, indices: list[int]) -> list[list[int]] | None:
    """Group indices by Zarr chunk alignment."""
    chunk_size = self._chunks.get(self._append_dim, 1)
    if chunk_size <= 1:
        return None  # One index per chunk = no constraints

    groups: dict[int, list[int]] = defaultdict(list)
    for idx in indices:
        chunk_id = idx // chunk_size
        groups[chunk_id].append(idx)

    return [sorted(group) for _, group in sorted(groups.items())]
```

#### Pre-Computed Partitioning (AtomicDataZarrSink pattern)

When the store is pre-allocated and the partition map is computed at
construction time:

```python
def partition_indices(self, indices: list[int] | None = None) -> list[list[int]] | None:
    """Return pre-computed chunk groups for parallel writes."""
    if not self._parallel:
        return None
    if indices is None:
        return self._chunk_groups  # Computed in _preallocate()
    # Filter to requested indices
    idx_set = set(indices)
    return [
        [i for i in group if i in idx_set]
        for group in self._chunk_groups
        if any(i in idx_set for i in group)
    ]
```

## Step 1c: Source Integration (Optional)

### `set_source()` Method

Implement this when the sink needs metadata from the source for output
path resolution (directory mirroring, run IDs, etc.). It is called
automatically by `Pipeline.write()`.

```python
def set_source(self, source: Source[Mesh]) -> None:
    """Inject source reference for placeholder resolution.

    Called automatically by :meth:`Pipeline.write` when the sink is
    attached to a pipeline. Use this to resolve naming template
    placeholders that depend on source metadata.

    Parameters
    ----------
    source : Source[Mesh]
        The upstream source providing items.
    """
    self._source = source
```

**Source methods available for naming**:

| Method | Returns | Use case |
|--------|---------|----------|
| `source.relative_path(index)` | `str` | Directory mirroring (relpath, stem) |
| `source.run_id()` | `str` | Group outputs by run identifier |
| `source.mesh_name(index, seq)` | `str` | Per-mesh naming from source metadata |

### Param Declaration Patterns

**Required parameter** (no default — user must provide):

```python
Param(name="output_dir", description="Output directory for files", type=str)
```

**Naming template** (optional, enables source-backed naming):

```python
Param(
    name="naming_template",
    description=(
        "Format string for output names. "
        "Placeholders: {index}, {seq}, {relpath}, {stem}, {run_id}. "
        "Default: mesh_{index:04d}_{seq}"
    ),
    type=str,
    default=None,
)
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
    description="Blosc/zstd compression level (0=off, 9=max)",
    type=int,
    default=3,
)
```

**Chunk size for adaptive chunking**:

```python
Param(
    name="chunk_size_mb",
    description="Target chunk size in megabytes",
    type=float,
    default=1.0,
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

**Batch size for buffered writes**:

```python
Param(
    name="batch_size",
    description="Items per write batch (sequential mode)",
    type=int,
    default=1000,
)
```

**Boolean flag**:

```python
Param(
    name="flip_triangle_normals",
    description="Reverse triangle vertex order for VTK normal convention",
    type=bool,
    default=True,
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

@property
def naming_template(self) -> str | None:
    """Return the configured naming template."""
    return self._naming_template
```

### Iterator Consumption Patterns

All sinks **fully consume** their input iterator before returning.

**Pattern 1: Sequential Enumeration** (most common):
```python
for seq, mesh in enumerate(items):
    # Process each item immediately
    paths.append(str(self._write(mesh, index, seq)))
```

**Pattern 2: Batch Collection** (buffered writes):
```python
batch: list[T] = []
for item in items:
    batch.append(item)
    if len(batch) >= self._batch_size:
        self._flush_batch(batch)
        batch = []
if batch:
    self._flush_batch(batch)  # Don't forget remainder
```

**Pattern 3: Stateful Accumulation** (data-driven):
```python
for da in items:
    written = self._write_dataarray(da)
    paths.extend(written)
return paths
```

## Step 2: Write Tests

Create the test file at `test/domains/<domain>/test_<name>.py`.

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

    def test_properties(self) -> None:
        from <module> import <ClassName>

        sink = <ClassName>(output_dir="/tmp/test")
        assert sink.output_dir == pathlib.Path("/tmp/test")
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

#### Naming Template Tests

```python
    def test_naming_template_default(self, tmp_path: pathlib.Path) -> None:
        """Default naming uses index and seq."""
        sink = <ClassName>(output_dir=str(tmp_path / "out"))
        paths = sink(iter([_create_test_data()]), index=5)
        assert "0005" in paths[0]

    def test_naming_template_custom(self, tmp_path: pathlib.Path) -> None:
        """Custom template resolves placeholders."""
        sink = <ClassName>(
            output_dir=str(tmp_path / "out"),
            naming_template="run_{index}_{seq}",
        )
        paths = sink(iter([_create_test_data()]), index=3)
        assert "run_3_0" in paths[0]

    def test_naming_template_with_source(self, tmp_path: pathlib.Path) -> None:
        """Source-backed placeholders resolve correctly."""
        from unittest.mock import MagicMock

        source = MagicMock()
        source.relative_path.return_value = "split_a/run_01.ext"

        sink = <ClassName>(
            output_dir=str(tmp_path / "out"),
            naming_template="{relpath}/{stem}",
        )
        sink.set_source(source)
        paths = sink(iter([_create_test_data()]), index=0)
        assert "split_a" in paths[0]
        assert "run_01" in paths[0]
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

#### Partition Tests (if applicable)

```python
    def test_partition_indices_chunk_aligned(self) -> None:
        """Indices are grouped by chunk alignment."""
        sink = <ClassName>(output_path="test.zarr", chunks={"time": 4})
        groups = sink.partition_indices(list(range(12)))

        assert len(groups) == 3
        assert groups[0] == [0, 1, 2, 3]
        assert groups[1] == [4, 5, 6, 7]
        assert groups[2] == [8, 9, 10, 11]

    def test_partition_indices_no_constraint(self) -> None:
        """Returns None when no partitioning needed."""
        sink = <ClassName>(output_path="test.zarr", chunks={"time": 1})
        assert sink.partition_indices(list(range(10))) is None
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
| `@pytest.mark.requires("atm")` | Skip if atm dependencies not installed |
| `@pytest.mark.integration` | Tests that touch filesystem |
| `@pytest.mark.e2e` | End-to-end pipeline tests |
| `@pytest.mark.slow` | Slow tests, excluded from quick CI |

### Running Tests

```bash
# Unit and write tests (fast, no network)
uv run pytest test/domains/<domain>/test_<name>.py -v

# Pipeline integration test
uv run pytest test/domains/<domain>/test_<name>.py -v -k "Pipeline"

# Full domain test suite for regressions
uv run pytest test/domains/<domain>/ -v -k "not slow"
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

For **atm** sinks, edit `src/physicsnemo_curator/domains/atm/__init__.py`
with `"atm"` as the submodule name.

### Lazy Loading Pattern (Optional Heavy Dependencies)

If the sink requires heavy optional dependencies (pyvista, zarr, etc.),
use lazy loading in the domain sinks `__init__.py`:

```python
__all__ = ["MeshSink", "MeshVTUSink", "MeshZarrSink"]

def __getattr__(name: str):
    """Lazy-load optional sinks to avoid import at module load."""
    if name == "MeshZarrSink":
        from physicsnemo_curator.domains.mesh.sinks.mesh_zarr import MeshZarrSink
        return MeshZarrSink
    if name == "MeshVTUSink":
        from physicsnemo_curator.domains.mesh.sinks.mesh_vtu import MeshVTUSink
        return MeshVTUSink
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

## Step 4: Quality Checks

Run all checks before committing:

```bash
# Format
uv run ruff format \
  src/physicsnemo_curator/domains/<domain>/sinks/<name>.py \
  test/domains/<domain>/test_<name>.py

# Lint
uv run ruff check --fix \
  src/physicsnemo_curator/domains/<domain>/sinks/<name>.py \
  test/domains/<domain>/test_<name>.py

# Docstring coverage (must be >= 99%)
uv run interrogate

# Type checking
uv run ty check

# Run sink tests
uv run pytest test/domains/<domain>/test_<name>.py -v

# Run full domain test suite for regressions
uv run pytest test/domains/<domain>/ -v -k "not slow"
```

## Step 5: Commit

Use the `commit` tool (or follow Conventional Commits format):

```text
feat(<domain>): add <ClassName> for <format> output

<Optional body describing the output format, naming, and features.>
```

## Checklist

Before considering the sink complete, verify:

### Core

- [ ] SPDX license headers on all new files
- [ ] Sink inherits from `Sink["Mesh"]`, `Sink["xr.DataArray"]`, or `Sink["AtomicData"]`
- [ ] `name` and `description` ClassVars are set
- [ ] `params()` returns all configurable parameters
- [ ] `__call__()` receives `Iterator[T]` and `index: int`, returns `list[str]`
- [ ] Iterator is fully consumed (no partial reads)
- [ ] Output directory is created automatically (`mkdir(parents=True, exist_ok=True)`)
- [ ] Empty iterator returns `[]` (no crash, no empty files)
- [ ] Logging via `get_logger(self)` (not `logging.getLogger(__name__)`)
- [ ] Timing logged for write operations

### Naming and Source Integration

- [ ] `naming_template` param if index-based naming is used
- [ ] `set_source()` implemented if template uses source-backed placeholders
- [ ] Placeholder validation in `__init__` (fail fast on invalid templates)

### Parallel Safety (if applicable)

- [ ] `partition_indices()` returns correct groups or `None`
- [ ] Concurrent writes to same file are impossible within a partition
- [ ] Atomic writes (temp + rename) where format doesn't support append

### Documentation

- [ ] NumPy-style docstrings on class, `__init__`, `__call__`, and helpers
- [ ] `from __future__ import annotations` at top
- [ ] `TYPE_CHECKING` block for `Iterator` and domain type imports
- [ ] Read-only `@property` accessors for key config values

### Registration

- [ ] Registered in domain `__init__.py` with `registry.register_sink()`
- [ ] Added to `__all__` in domain `__init__.py`
- [ ] Lazy loading if sink has heavy optional dependencies

### Testing

- [ ] Unit tests: params, name, description, properties
- [ ] Write tests: single item, multiple items, naming, directory creation, empty iterator
- [ ] Naming template tests: default, custom, source-backed placeholders
- [ ] Roundtrip test: written data can be read back correctly
- [ ] Append tests (if applicable): data accumulates rather than overwrites
- [ ] Split tests (if applicable): data is split by variable/coordinate
- [ ] Partition tests (if applicable): correct grouping, None when no constraint
- [ ] Pipeline test: sink works at end of a pipeline (marked `@pytest.mark.e2e`)
- [ ] Registry test: sink is discoverable

### Quality Gates

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
    """Abstract sink that persists items and returns output file paths.

    The sink consumes a generator of items and writes each one to storage,
    returning the file paths of the written outputs.

    Subclasses must set :attr:`name` and :attr:`description` and implement
    :meth:`params` and :meth:`__call__`.
    """

    name: ClassVar[str]
    """Human-readable display name for the interactive CLI."""
    description: ClassVar[str]
    """Short description shown in the interactive CLI."""

    @classmethod
    @abstractmethod
    def params(cls) -> list[Param]:
        """Declare the configurable parameters for this sink.

        Returns
        -------
        list[Param]
            Ordered list of parameter descriptors.
        """
        ...

    @abstractmethod
    def __call__(self, items: Iterator[T], index: int) -> list[str]:
        """Consume items and persist them to storage.

        Parameters
        ----------
        items : Iterator[T]
            Stream of data items to write.
        index : int
            Source index being processed (useful for naming output files).

        Returns
        -------
        list[str]
            Paths of the files written.
        """
        ...

    def partition_indices(self, indices: list[int]) -> list[list[int]] | None:
        """Group indices into partitions that MUST be processed by the same worker.

        Each returned group is a list of indices that must be handled
        sequentially by a single worker.  The runner will never split a
        group across workers.

        Override this method when the sink has constraints on concurrent
        writes (e.g., multiple indices writing to the same Zarr chunk must
        go through the same worker).

        Parameters
        ----------
        indices : list[int]
            The indices to partition.

        Returns
        -------
        list[list[int]] | None
            Partitioned groups, or ``None`` if no partitioning is required
            (the default).
        """
        return None
```

Key differences from `Source[T]` and `Filter[T]`:

- Sources have `__len__` and `__getitem__` — sinks do not
- Filters receive and return `Generator[T]` — sinks consume `Iterator[T]`
  and return `list[str]` (file paths)
- Sinks are the terminal stage: they **consume** items, they do not yield
- The `index` parameter is the source-level index from the pipeline
- Return value is a list of all file paths written (can be empty)
- Sinks do NOT have `dashboard_panel()`, `artifacts()`, or `merge()`
  (those are filter-only methods)
- `partition_indices()` is unique to sinks (not on sources or filters)
- `set_source()` is an optional protocol method (not in the ABC)

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
