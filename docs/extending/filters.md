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
| `artifacts` | `() -> list[str]` *(optional)* | Return paths of files produced since last call |
| `merge` | `staticmethod(parquet_paths: list[str], output: str) -> str` *(optional)* | Merge per-worker shard files after parallel execution |

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
used by {class}`~physicsnemo_curator.domains.mesh.filters.mean.MeanFilter`:

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
method.  The pipeline runner calls `flush()` automatically after all items
have been processed (via {func}`~physicsnemo_curator.run.run_pipeline`):

```python
class MeshStatsFilter(Filter["Mesh"]):
    def __init__(self, output: str) -> None:
        self._output_path = pathlib.Path(output)
        self._rows: list[dict] = []

    def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
        for mesh in items:
            self._rows.append(self._compute(mesh))
            yield mesh

    def flush(self) -> str | None:
        """Write accumulated data. Called after pipeline execution."""
        if not self._rows:
            return None
        self._write(self._rows, self._output_path)
        return str(self._output_path)

    def artifacts(self) -> list[str]:
        """Report produced files for pipeline store tracking."""
        if self._output_path.exists():
            return [str(self._output_path)]
        return []
```

### Parallel execution and `merge`

When running in parallel, each worker process gets its own copy of the
filter.  The framework rewrites the filter's `_output_path` to a
worker-specific shard (e.g. `stats_worker_0.parquet`).  After all workers
finish, call {func}`~physicsnemo_curator.run.gather_pipeline` to merge
the shards:

```python
from physicsnemo_curator import run_pipeline
from physicsnemo_curator.run import gather_pipeline

results = run_pipeline(pipeline, n_jobs=4, backend="process_pool")
merged = gather_pipeline(pipeline)  # merges per-worker shards
```

To support this, stateful filters should also implement a `merge` static
method:

```python
    @staticmethod
    def merge(parquet_paths: list[str], output: str) -> str:
        """Merge per-worker Parquet shards into a single output file."""
        import pyarrow.parquet as pq

        tables = [pq.read_table(p) for p in parquet_paths]
        merged = pa.concat_tables(tables)
        pq.write_table(merged, output)
        return output
```

The framework detects stateful filters by checking for the presence of
`flush`, `_output_path`, and `merge` attributes via `hasattr()`.  No
base-class registration is needed — just implement the methods.

## Registration

Register your filter in the submodule's `__init__.py`:

```python
from physicsnemo_curator.core.registry import registry
from .filters.my_filter import MyFilter

registry.register_filter("mymodule", MyFilter)
```

## Gallery Example

For a complete worked example, see the
[custom_filter example](https://github.com/NVIDIA/physicsnemo-curator/tree/main/examples/extending/custom_filter).
