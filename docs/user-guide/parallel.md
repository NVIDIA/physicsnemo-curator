# Parallel Execution

`run_pipeline()` processes every index of a pipeline — optionally in parallel —
and returns the collected sink outputs.  It replaces the manual
`for i in range(len(pipeline))` loop.

## Quick Start

```python
from curator import run_pipeline

# Sequential (default)
results = run_pipeline(pipeline)

# Parallel with 4 worker processes
results = run_pipeline(pipeline, n_jobs=4, backend="processes")

# Use all CPUs
results = run_pipeline(pipeline, n_jobs=-1)

# Process a subset of indices
results = run_pipeline(pipeline, indices=[0, 5, 10])
```

## Backends

`run_pipeline` supports multiple execution backends, selected via the
`backend` parameter:

| Backend | Dependency | Description |
|---------|-----------|-------------|
| `"sequential"` | None | Simple for-loop.  Always used when `n_jobs=1`. |
| `"processes"` | None | `concurrent.futures.ProcessPoolExecutor`.  Zero extra deps. |
| `"loky"` | `joblib` | joblib's robust process pool.  Better memory handling for large arrays. |
| `"dask"` | `dask` | `dask.bag` for distributed execution.  Scales to clusters. |
| `"auto"` | Varies | Picks the best available: dask → loky → processes. |

### Installing optional backends

```bash
# Both joblib and dask
pip install 'physicsnemo-curator[parallel]'

# Or individually
pip install joblib
pip install dask
```

## Parameters

```text
run_pipeline(
    pipeline,          # Pipeline with source + filters + sink
    *,
    n_jobs=1,          # Workers.  -1 = all CPUs.
    backend="auto",    # "auto", "sequential", "processes", "loky", "dask"
    indices=None,      # Subset of source indices, or None for all
    progress=True,     # Show progress bar (tqdm / dask diagnostics)
    **backend_kwargs,  # Extra args forwarded to the backend executor
)
```

**Returns:** `list[list[str]]` — outer list ordered by input indices, inner
list contains file paths returned by the sink for that index.

## Process Isolation

All multiprocess backends (`"processes"`, `"loky"`, `"dask"`) execute each
index in a **separate process**.  This means:

- Each worker gets an independent copy of the pipeline, source, filters, and sink.
- **Stateful filters are not merged back.**  For example, `MeanFilter` accumulates
  rows in `self._rows` — those rows exist only in the child process and are
  discarded when the worker exits.

If you need filter side-effects (like `MeanFilter.flush()`), use sequential
execution:

```python
# Sequential — filter state is preserved
results = run_pipeline(pipeline, n_jobs=1)
pipeline.filters[0].flush()
```

Or implement a post-hoc merge step that combines per-worker outputs.

## Examples

### Basic parallel ETL

```python
from curator.core.store import LocalFileStore
from curator.mesh.sources.vtk import VTKSource
from curator.mesh.sinks.mesh_writer import MeshSink
from curator import run_pipeline

pipeline = (
    VTKSource.from_path("./cfd_results/")
    .write(MeshSink(output_dir="./output/"))
)

# Process all items with 8 workers
results = run_pipeline(pipeline, n_jobs=8, backend="processes")
print(f"Wrote {sum(len(r) for r in results)} files")
```

### With HuggingFace dataset sources

```python
from curator.mesh.sources.drivaerml import DrivAerMLSource
from curator.mesh.sinks.mesh_writer import MeshSink
from curator import run_pipeline

pipeline = (
    DrivAerMLSource(mesh_type="boundary")
    .write(MeshSink(output_dir="./drivaer_output/"))
)

# Process first 10 runs in parallel
results = run_pipeline(pipeline, n_jobs=4, indices=list(range(10)))
```

### Choosing a backend at runtime

```python
import os
from curator import run_pipeline

# CI uses sequential; production uses all CPUs
n = 1 if os.getenv("CI") else -1
results = run_pipeline(pipeline, n_jobs=n)
```
