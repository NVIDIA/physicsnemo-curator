<!---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
-->

# Writing Custom Sinks

A **sink** persists domain objects to storage and returns the file paths it
wrote.  Sinks are the final stage of a pipeline.

## Interface Contract

Subclass {class}`~physicsnemo_curator.core.base.Sink` and implement:

| Method | Signature | Purpose |
|--------|-----------|---------|
| `__call__` | `(items: Generator[T], index: int) -> list[str]` | Consume items and write them; return paths |
| `params` | `classmethod() -> list[Param]` | Declare constructor parameters |

Key rules:

- `__call__` receives a generator of items and the pipeline **index** for that
  batch.  It must return a `list[str]` of every file path written.
- The sink is responsible for creating any necessary output directories.
- Naming conventions should be deterministic so that re-runs produce the same
  file names (enabling checkpointing and append logic).

## Minimal Example

```python
from __future__ import annotations
from typing import ClassVar, TYPE_CHECKING
from physicsnemo_curator.core.base import Sink, Param

if TYPE_CHECKING:
    from collections.abc import Generator
    from physicsnemo.mesh import Mesh

class VTKSink(Sink["Mesh"]):
    name: ClassVar[str] = "VTK Writer"
    description: ClassVar[str] = "Save meshes as VTK files"

    @classmethod
    def params(cls) -> list[Param]:
        return [
            Param(name="output_dir", description="Output directory", type=str),
        ]

    def __init__(self, output_dir: str) -> None:
        self._output_dir = output_dir

    def __call__(self, items: Generator[Mesh], index: int) -> list[str]:
        paths: list[str] = []
        for seq, mesh in enumerate(items):
            path = f"{self._output_dir}/mesh_{index:04d}_{seq}.vtu"
            self._save_vtk(mesh, path)
            paths.append(path)
        return paths
```

## Implementation Patterns

### Index-based naming

The simplest pattern — use the pipeline index to name output files:

```python
def __call__(self, items: Generator[Mesh], index: int) -> list[str]:
    paths: list[str] = []
    for seq, mesh in enumerate(items):
        path = f"{self._output_dir}/item_{index:04d}_{seq}.vtu"
        write(mesh, path)
        paths.append(path)
    return paths
```

### Data-driven naming

Use metadata from the domain object to name files (e.g. simulation ID,
timestamp):

```python
def __call__(self, items: Generator[Mesh], index: int) -> list[str]:
    paths: list[str] = []
    for mesh in items:
        sim_id = mesh.metadata["simulation_id"]
        path = f"{self._output_dir}/{sim_id}.vtu"
        write(mesh, path)
        paths.append(path)
    return paths
```

### Append logic

Support appending to existing files (useful for incremental pipelines):

```python
def __call__(self, items: Generator[Mesh], index: int) -> list[str]:
    path = f"{self._output_dir}/data_{index:04d}.h5"
    if os.path.exists(path):
        self._append(items, path)
    else:
        self._create(items, path)
    return [path]
```

## Registration

Register your sink in the submodule's `__init__.py`:

```python
from physicsnemo_curator.core.registry import registry
from .sinks.my_sink import MySink

registry.register_sink("mymodule", MySink)
```

## Gallery Example

For a complete worked example, see the
[custom_sink example](https://github.com/NVIDIA/physicsnemo-curator/tree/main/examples/extending/custom_sink).
