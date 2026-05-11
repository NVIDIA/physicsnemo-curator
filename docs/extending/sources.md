<!---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
-->

# Writing Custom Sources

A **source** reads data from local or remote storage and yields domain objects
(e.g. meshes, xarray datasets).  Each source handles its own file discovery
and caching internally.  This page walks through the interface contract,
implementation patterns, and a worked example.

## Interface Contract

Subclass {class}`~physicsnemo_curator.core.base.Source` and implement three
methods:

| Method | Signature | Purpose |
|--------|-----------|---------|
| `__len__` | `() -> int` | Number of items the source contains |
| `__getitem__` | `(index: int) -> Generator[T]` | Yield one or more domain objects for the given index |
| `params` | `classmethod() -> list[Param]` | Declare constructor parameters for registry discovery |
| `partition_indices` | `(indices) -> list[list[int]] \| None` *(optional)* | Group indices for same-worker processing |

Key rules:

- Each source handles its own file discovery and caching internally (using
  `pathlib`, `fsspec`, or other appropriate libraries).
- `__getitem__` is a **generator** (uses `yield`).  It can yield multiple
  items per index if a single file contains several samples.
- `params()` documents the constructor interface and enables registry
  discovery.
- Override `partition_indices` when the source has constraints on concurrent
  access (e.g. LMDB allows only one environment open per file per process).
  Return a list of index groups — each group is guaranteed to be processed
  sequentially by the same worker.

## Minimal Example

```python
from __future__ import annotations
import pathlib
from typing import ClassVar, TYPE_CHECKING
from physicsnemo_curator.core.base import Source, Param

if TYPE_CHECKING:
    from collections.abc import Generator
    from physicsnemo.mesh import Mesh

class MySource(Source["Mesh"]):
    name: ClassVar[str] = "My Reader"
    description: ClassVar[str] = "Reads data from my custom format"

    @classmethod
    def params(cls) -> list[Param]:
        return [
            Param(name="input_path", description="Path to data directory", type=str),
            Param(name="option", description="Processing option", type=str, default="default"),
        ]

    def __init__(self, input_path: str, option: str = "default") -> None:
        self._option = option
        root = pathlib.Path(input_path)
        self._files = sorted(root.glob("**/*.vtk"))

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, index: int) -> Generator[Mesh]:
        path = str(self._files[index])
        mesh = self._load(path)
        yield mesh

    def _load(self, path: str) -> Mesh:
        """Load a single item from a local file path."""
        ...
```

## Implementation Patterns

### Single-item sources

The most common pattern — each file maps to exactly one domain object:

```python
def __getitem__(self, index: int) -> Generator[Mesh]:
    path = str(self._files[index])
    mesh = read_vtk(path)
    yield mesh
```

### Multi-item sources

When a single file contains multiple samples (e.g. time steps in an HDF5 file):

```python
def __getitem__(self, index: int) -> Generator[Mesh]:
    path = str(self._files[index])
    with h5py.File(path) as f:
        for timestep in f["timesteps"]:
            yield self._build_mesh(timestep)
```

### Eager metadata, lazy data

Load lightweight metadata up-front in `__init__` and defer heavy data loading
to `__getitem__`:

```python
def __init__(self, input_path: str) -> None:
    root = pathlib.Path(input_path)
    self._files = sorted(root.glob("**/*.vtk"))
    # Lightweight — just read headers
    self._metadata = [read_header(str(f)) for f in self._files]

def __getitem__(self, index: int) -> Generator[Mesh]:
    path = str(self._files[index])
    meta = self._metadata[index]
    mesh = read_full(path, meta)
    yield mesh
```

## Registration

Register your source in the submodule's `__init__.py` so the registry can
discover it:

```python
from physicsnemo_curator.core.registry import registry
from .sources.my_source import MySource

registry.register_source("mymodule", MySource)
```

## Gallery Example

For a complete worked example, see the
[custom_source example](https://github.com/NVIDIA/physicsnemo-curator/tree/main/examples/extending/custom_source).
