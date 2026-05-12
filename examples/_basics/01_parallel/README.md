# Running a Pipeline in Parallel

This example demonstrates `run_pipeline` to execute a pipeline across multiple source indices
using parallel workers. Building on the [Creating a Pipeline](../pipeline/) example, we process
multiple DrivAerML CFD meshes concurrently with a `process_pool` backend and then merge
per-worker statistics using `gather_pipeline`.

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

### 1. Build the Pipeline

`DrivAerMLSource` provides 500 DrivAerML automotive CFD meshes from HuggingFace Hub. We attach
a `MeanFilter` for spatial statistics and a `MeshSink` for output.

```python
from physicsnemo_curator.domains.mesh.sources.drivaerml import DrivAerMLSource
from physicsnemo_curator.domains.mesh.filters.mean import MeanFilter
from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink

pipeline = (
    DrivAerMLSource(mesh_type="boundary")
    .filter(MeanFilter(output="output/parallel/stats.parquet"))
    .write(MeshSink(output_dir="output/parallel/meshes/"))
)
```

### 2. Run in Parallel

`run_pipeline` dispatches indices to parallel workers. Key parameters:

- `n_jobs` — number of workers (`-1` = all CPUs)
- `backend` — `"sequential"`, `"process_pool"`, `"loky"`, or `"dask"`
- `indices` — which source indices to process (default: all)
- `progress` — show a progress bar

Each worker receives an independent copy of the pipeline, so data is read, filtered, and written
concurrently.

```python
from physicsnemo_curator.run import run_pipeline

results = run_pipeline(
    pipeline,
    n_jobs=4,
    backend="process_pool",
    indices=range(3),
    use_tui=True,
)
```

> **Note:** If the TUI progress display is not useful for your workflow, set `use_tui=False` to
> get a traditional console log instead.

### 3. Inspect Results

`results` is a list of lists — one entry per processed index, each containing the file paths
returned by the sink.

```python
for i, paths in enumerate(results):
    print(f"  Run {i}: {paths}")
```

### 4. Gather Statistics

When running in parallel, stateful filters (like `MeanFilter`) produce per-index shard files.
`gather_pipeline` discovers those shards, calls the filter's `merge()` method to combine them
into a single output file, and cleans up the temporaries.

```python
from physicsnemo_curator.run import gather_pipeline

merged = gather_pipeline(pipeline)
for path in merged:
    print(f"Merged statistics: {path}")
```

## Available Backends

| Backend | Install extra | Best for |
|---------|--------------|----------|
| `sequential` | (built-in) | Debugging, small datasets (default) |
| `process_pool` | (built-in) | CPU-bound tasks |
| `loky` | `uv sync --extra loky` | Robust multi-process |
| `dask` | `uv sync --extra dask` | Distributed clusters |
