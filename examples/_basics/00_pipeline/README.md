# Creating a Pipeline

This example shows how to build a data curation pipeline using the **Source -> Filter -> Sink**
pattern. We generate random meshes, apply statistics and precision filters, and write outputs to
disk. A pipeline is lazy — nothing is executed until you index into it with `pipeline[i]`.

## Prerequisites

```bash
uv sync --extra mesh
uv run maturin develop
```

## Usage

```bash
uv run python main.py
```

## Step-by-Step Walkthrough

### 1. Create a Source

A **Source** is an indexed collection of data items. `RandomMeshSource` generates synthetic
tetrahedral meshes with configurable point/cell counts and random scalar fields.

```python
from physicsnemo_curator.domains.mesh.sources.random import RandomMeshSource

source = RandomMeshSource(n_samples=10, n_points=100, n_cells=50)
print(f"Items available: {len(source)}")
```

### 2. Add Filters

Filters transform or inspect items as they flow through the pipeline. The fluent `.filter()`
method chains multiple filters together:

- `MeanFilter` — computes spatial means and writes a Parquet summary file.
- `PrecisionFilter` — converts floating-point fields to a target dtype (e.g., `float32`).

```python
from physicsnemo_curator.domains.mesh.filters.mean import MeanFilter
from physicsnemo_curator.domains.mesh.filters.precision import PrecisionFilter

pipeline = (
    source
    .filter(MeanFilter(output="output/getting_started/stats.parquet"))
    .filter(PrecisionFilter(target_dtype="float32"))
)
```

### 3. Attach a Sink

A **Sink** persists items to storage. The `.write()` method attaches a sink and returns a
complete pipeline.

```python
from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink

pipeline = pipeline.write(MeshSink(output_dir="output/getting_started/meshes/"))
```

### 4. Execute One Index

Indexing into a pipeline runs the full **Source -> Filters -> Sink** chain for a single source
item and returns the file paths written by the sink.

```python
paths = pipeline[0]
print(f"Index 0 wrote: {paths}")
```

### Fluent One-Liner

The entire pipeline can also be built in a single expression:

```python
pipeline = (
    RandomMeshSource(n_samples=10, n_points=100, n_cells=50)
    .filter(MeanFilter(output="stats.parquet"))
    .filter(PrecisionFilter(target_dtype="float32"))
    .write(MeshSink(output_dir="meshes/"))
)
```

Each call returns a new immutable `Pipeline` — the original source, filters, and sink are never
modified.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Source** | Indexed data collection; `len(source)` gives the number of items |
| **Filter** | Transform applied to each item; chained via `.filter()` |
| **Sink** | Writes processed items to disk; attached via `.write()` |
| **Lazy execution** | Pipeline only runs when indexed (`pipeline[i]`) or via `run_pipeline` |
| **Immutable** | Each `.filter()` / `.write()` returns a new pipeline object |
