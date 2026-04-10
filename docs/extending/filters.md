<!---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
-->

# Writing Custom Filters

A **filter** transforms a stream of domain objects.  Filters are the most
versatile component — they can modify items in-place, expand one item into
many, skip items, or accumulate statistics across the entire stream.

## Interface Contract

Subclass {class}`~physicsnemo_curator.core.base.Filter` and implement:

| Method | Signature | Purpose |
|--------|-----------|---------|
| `__call__` | `(items: Generator[T]) -> Generator[T]` | Transform a stream of items |
| `params` | `classmethod() -> list[Param]` | Declare constructor parameters |
| `flush` | `() -> str \| None` *(optional)* | Write accumulated state after pipeline execution |

Key rules:

- `__call__` receives **and returns** a generator.  Processing is lazy — items
  flow through the pipeline one at a time.
- Always `yield` from inside a `for item in items:` loop to maintain the
  streaming contract.
- The return type annotation should be `Generator[T]`, not `Iterator[T]`.

## Minimal Example

```python
from __future__ import annotations
from typing import ClassVar, TYPE_CHECKING
from physicsnemo_curator.core.base import Filter, Param

if TYPE_CHECKING:
    from collections.abc import Generator
    from physicsnemo.mesh import Mesh

class ScaleFilter(Filter["Mesh"]):
    name: ClassVar[str] = "Scale"
    description: ClassVar[str] = "Scale mesh points by a constant factor"

    @classmethod
    def params(cls) -> list[Param]:
        return [
            Param(name="factor", description="Scale factor", type=float),
        ]

    def __init__(self, factor: float) -> None:
        self._factor = factor

    def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
        for mesh in items:
            mesh.points *= self._factor
            yield mesh
```

## Filter Patterns

### Pass-through with side effects

Yield items unchanged while accumulating information.  This is the pattern
used by {class}`~physicsnemo_curator.mesh.filters.mean.MeanFilter`:

```python
def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
    for mesh in items:
        self._accumulate(mesh)
        yield mesh  # unchanged
```

### In-place modification

Mutate each item and yield it:

```python
def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
    for mesh in items:
        mesh.points *= self._factor
        yield mesh
```

### Expand (one-to-many)

Yield multiple items per input — useful for data augmentation:

```python
def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
    for mesh in items:
        yield mesh                    # original
        yield self._augment(mesh)     # augmented copy
```

### Contract (filtering)

Skip items that don't meet criteria:

```python
def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
    for mesh in items:
        if self._passes_quality_check(mesh):
            yield mesh
```

## Stateful Filters

If your filter accumulates data across the entire stream, add a `flush()`
method.  The pipeline calls `flush()` automatically after all items have
been processed:

```python
class StatsFilter(Filter["Mesh"]):
    def __init__(self, output: str) -> None:
        self._output = output
        self._rows: list[dict] = []

    def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
        for mesh in items:
            self._rows.append(self._compute(mesh))
            yield mesh

    def flush(self) -> str | None:
        """Write accumulated data. Called after pipeline execution."""
        if not self._rows:
            return None
        self._write(self._rows, self._output)
        return self._output
```

The interactive CLI calls `flush()` automatically on any filter that has it.

## Registration

Register your filter in the submodule's `__init__.py`:

```python
from physicsnemo_curator.core.registry import registry
from .filters.my_filter import MyFilter

registry.register_filter("mymodule", MyFilter)
```

## Gallery Example

For a complete worked example, see
{doc}`/auto_examples/extending/extending_custom_filter`.
