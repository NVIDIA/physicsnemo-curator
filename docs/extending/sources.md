<!---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
-->

# Writing Custom Sources

A **source** reads data from a {class}`~physicsnemo_curator.core.store.FileStore`
and yields domain objects (e.g. meshes, xarray datasets).  This page walks
through the interface contract, implementation patterns, and a worked example.

## Interface Contract

Subclass {class}`~physicsnemo_curator.core.base.Source` and implement three
methods:

| Method | Signature | Purpose |
|--------|-----------|---------|
| `__len__` | `() -> int` | Number of items the store contains |
| `__getitem__` | `(index: int) -> Generator[T]` | Yield one or more domain objects for the given index |
| `params` | `classmethod() -> list[Param]` | Declare constructor parameters for CLI discovery |

Key rules:

- The source receives a `FileStore` — it **never** handles file discovery or
  downloads directly.
- `__getitem__` is a **generator** (uses `yield`).  It can yield multiple
  items per index if a single file contains several samples.
- `params()` drives the CLI prompts.  Do **not** include a `store` parameter —
  the CLI constructs the store separately.

## Minimal Example

```python
from __future__ import annotations
from typing import ClassVar, TYPE_CHECKING
from physicsnemo_curator.core.base import Source, Param
from physicsnemo_curator.core.store import FileStore

if TYPE_CHECKING:
    from collections.abc import Generator
    from physicsnemo.mesh import Mesh

class MySource(Source["Mesh"]):
    name: ClassVar[str] = "My Reader"
    description: ClassVar[str] = "Reads data from my custom format"

    @classmethod
    def params(cls) -> list[Param]:
        return [
            Param(name="option", description="Processing option", type=str, default="default"),
        ]

    def __init__(self, store: FileStore, option: str = "default") -> None:
        self._store = store
        self._option = option

    def __len__(self) -> int:
        return len(self._store)

    def __getitem__(self, index: int) -> Generator[Mesh]:
        path = self._store[index]
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
    path = self._store[index]
    mesh = read_vtk(path)
    yield mesh
```

### Multi-item sources

When a single file contains multiple samples (e.g. time steps in an HDF5 file):

```python
def __getitem__(self, index: int) -> Generator[Mesh]:
    path = self._store[index]
    with h5py.File(path) as f:
        for timestep in f["timesteps"]:
            yield self._build_mesh(timestep)
```

### Eager metadata, lazy data

Load lightweight metadata up-front in `__init__` and defer heavy data loading
to `__getitem__`:

```python
def __init__(self, store: FileStore) -> None:
    self._store = store
    # Lightweight — just read headers
    self._metadata = [read_header(store[i]) for i in range(len(store))]

def __getitem__(self, index: int) -> Generator[Mesh]:
    path = self._store[index]
    meta = self._metadata[index]
    mesh = read_full(path, meta)
    yield mesh
```

## Registration

Register your source in the submodule's `__init__.py` so the CLI can discover it:

```python
from physicsnemo_curator.core.registry import registry
from .sources.my_source import MySource

registry.register_source("mymodule", MySource)
```

## Gallery Example

For a complete worked example, see
{doc}`/auto_examples/extending/extending_custom_source`.
