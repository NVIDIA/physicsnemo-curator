# Creating a Custom Source

This example shows how to implement and register a custom Source. We create a `SineFlowSource`
that generates synthetic 2D flow fields with sinusoidal velocity patterns on a triangular mesh.
This demonstrates the core source contract: indexed access with generator semantics, mesh
construction, and registration.

## Prerequisites

```bash
uv sync --extra mesh
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

class SineFlowSource(Source["Mesh"]):
    name: ClassVar[str] = "Sine Flow (Custom)"
    description: ClassVar[str] = "Generate synthetic sinusoidal flow on a 2D mesh"

    @classmethod
    def params(cls) -> list[Param]:
        return [
            Param(name="n_samples", description="Number of flow snapshots", type=int, default=10),
            Param(name="n_points", description="Number of mesh vertices", type=int, default=100),
            Param(name="seed", description="Random seed for geometry", type=int, default=42),
        ]

    def __init__(self, n_samples: int = 10, n_points: int = 100, seed: int = 42) -> None:
        # Build a 2D triangular grid with jitter
        ...

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, index: int) -> Generator[Mesh]:
        # Generate sinusoidal flow with phase offset per index
        ...
        yield Mesh(points=self._points, cells=self._cells, point_data=point_data, global_data=global_data)
```

Key design patterns:

- **Eager geometry** — `__init__` builds the mesh grid upfront (lightweight for synthetic data)
- **Generator semantics** — `__getitem__` must `yield` (not `return`)
- **Negative indexing** — support `source[-1]` by converting to positive index
- **IndexError** — raise for out-of-bounds access

### Step 2 — Register the Source (Optional)

Registration makes the source discoverable via the global registry and the interactive CLI:

```python
from physicsnemo_curator.core.registry import registry

registry.register_source("mesh", SineFlowSource)
```

### Step 3 — Use in a Pipeline

The custom source works with any compatible filter and sink:

```python
source = SineFlowSource(n_samples=5, n_points=64)

pipeline = source.filter(
    MeanFilter(output="output/extending/sine_stats.parquet")
).write(
    MeshSink(output_dir="output/extending/sine_meshes/")
)

results = run_pipeline(pipeline, n_jobs=1, backend="sequential", indices=range(3), use_tui=True)
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
