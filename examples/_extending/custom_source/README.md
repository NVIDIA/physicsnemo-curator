# Creating a Custom Source

This example shows how to implement and register a custom Source. We create a `CylinderFlowSource`
that reads the Navier-Stokes Cylinder dataset from HuggingFace Hub using Parquet files. This
demonstrates the core source contract: indexed access with generator semantics, lazy loading, and
shared geometry caching.

## Prerequisites

```bash
uv sync --group mesh
uv run maturin develop
```

## Usage

```bash
uv run python main.py
```

## How It Works

A source inherits from `Source` and implements four things:

1. `name` / `description` class variables
2. `params()` class method (parameter descriptors)
3. `__len__()` — number of items
4. `__getitem__(index)` — yield data for a given index

### Step 1 — Define the Source

```python
from physicsnemo_curator.core.base import Param, Source

class CylinderFlowSource(Source["Mesh"]):
    name: ClassVar[str] = "Cylinder Flow (Custom)"
    description: ClassVar[str] = "Read NS cylinder flow from HF Parquet files"

    @classmethod
    def params(cls) -> list[Param]:
        return [
            Param(name="url", description="HuggingFace dataset URL", type=str, default=_DEFAULT_URL),
            Param(name="cache_storage", description="Local cache directory", type=str, default=""),
        ]

    def __init__(self, url: str = _DEFAULT_URL, cache_storage: str = "") -> None:
        # Eagerly load lightweight metadata
        ...

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int) -> Generator[Mesh]:
        # Lazily load geometry, read snapshot, yield Mesh
        ...
        yield Mesh(points=self._points, cells=self._cells, point_data=point_data, global_data=global_data)
```

Key design patterns:

- **Eager metadata** — `__init__` reads the lightweight parameter table to determine `__len__`
- **Lazy heavy data** — geometry is loaded on first `__getitem__` call
- **Shared caching** — geometry is cached across indices (shared mesh topology)
- **Generator semantics** — `__getitem__` must `yield` (not `return`)
- **Negative indexing** — support `source[-1]` by converting to positive index
- **IndexError** — raise for out-of-bounds access

### Step 2 — Register the Source (Optional)

Registration makes the source discoverable via the global registry and the interactive CLI:

```python
from physicsnemo_curator.core.registry import registry

registry.register_source("mesh", CylinderFlowSource)
```

### Step 3 — Use in a Pipeline

The custom source works with any compatible filter and sink:

```python
source = CylinderFlowSource()

pipeline = source.filter(
    MeanFilter(output="output/extending/cylinder_stats.parquet")
).write(
    MeshSink(output_dir="output/extending/cylinder_meshes/")
)

results = run_pipeline(pipeline, n_jobs=1, backend="sequential", indices=range(3), progress=True)
```

### Step 4 — Verify Output

Load a saved mesh and inspect its contents:

```python
mesh = Mesh.load(results[0][0])
print(f"Points: {mesh.n_points}")
print(f"Cells: {mesh.n_cells}")
print(f"Point fields: {list(mesh.point_data.keys())}")
print(f"Global fields: {list(mesh.global_data.keys())}")
```

## Extended API: `partition_indices()`

If your source has concurrency constraints (e.g. indices from the same LMDB file must be processed
by the same worker), override `partition_indices()`.

The method receives the list of indices to be processed and returns groups of indices that MUST stay
on the same worker. Return `None` if no grouping is needed.

Example: a multi-file LMDB source where each file can only have one reader open at a time per
process:

```python
def partition_indices(self, indices: list[int]) -> list[list[int]] | None:
    file_groups = defaultdict(list)
    for idx in indices:
        for file_idx in range(len(self._file_boundaries) - 1):
            if self._file_boundaries[file_idx] <= idx < self._file_boundaries[file_idx + 1]:
                file_groups[file_idx].append(idx)
                break
    if len(file_groups) <= 1:
        return None
    return [sorted(group) for group in file_groups.values()]
```

When `run_pipeline` receives a source with `partition_indices`, it ensures that indices in the same
group are never split across workers. The framework calls `intersect_partitions()` internally to
merge source and sink constraints, then `batch_groups()` to assign groups to workers.

## Summary

To create a custom source:

1. Subclass `Source` with a type parameter (`Source["Mesh"]`, `Source["xr.DataArray"]`, etc.)
2. Set `name` and `description` class variables
3. Implement `params()`, `__len__()`, and `__getitem__(index)`
4. Use generator semantics — `__getitem__` must `yield`
5. Support negative indexing and raise `IndexError` for out-of-bounds
6. Eagerly load lightweight metadata in `__init__`
7. Lazily load heavy data (geometry, fields) in `__getitem__`
8. Cache shared data (like geometry) across indices
9. Optionally register with `registry.register_source()`

Extended API:

- Override `partition_indices(indices)` to group indices that must share a worker (e.g. same LMDB
  file, same S3 prefix for locality)
