# Parallel Execution

`run_pipeline()` processes every index of a pipeline — optionally in parallel —
and returns the collected sink outputs.  It replaces the manual
`for i in range(len(pipeline))` loop.

## Quick Start

```python
from physicsnemo_curator import run_pipeline

# Sequential (default)
results = run_pipeline(pipeline)

# Parallel with 4 worker processes
results = run_pipeline(pipeline, n_jobs=4, backend="process_pool")

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
| `"thread_pool"` | None | `ThreadPoolExecutor`.  Good for I/O-bound tasks. |
| `"process_pool"` | None | `ProcessPoolExecutor`.  True parallelism for CPU-bound tasks. |
| `"loky"` | `joblib` | joblib's robust process pool.  Better memory handling for large arrays. |
| `"dask"` | `dask` | `dask.bag` for distributed execution.  Scales to clusters. |
| `"auto"` | Varies | Picks the best available: dask → loky → process_pool. |

### Installing optional backends

```bash
# Install individual backend extras
pip install 'physicsnemo-curator[loky]'
pip install 'physicsnemo-curator[dask]'

# Or install multiple
pip install 'physicsnemo-curator[loky,dask]'
```

### Custom backends

You can register your own backends:

```python
from physicsnemo_curator.run import register_backend, RunBackend, RunConfig

class MyBackend(RunBackend):
    name = "my_backend"
    description = "My custom execution backend"
    requires = ("my_package",)  # Optional dependencies

    def run(self, pipeline, config):
        # Your execution logic here
        results = []
        for idx in config.indices or range(len(pipeline)):
            results.append(pipeline[idx])
        return results

register_backend(MyBackend)

# Now use it:
results = run_pipeline(pipeline, n_jobs=4, backend="my_backend")
```

## Parameters

```text
run_pipeline(
    pipeline,          # Pipeline with source + filters + sink
    *,
    n_jobs=1,          # Workers.  -1 = all CPUs.
    backend="auto",    # "auto", "sequential", "thread_pool", "process_pool", "loky", "dask"
    indices=None,      # Subset of source indices, or None for all
    progress=True,     # Show progress (Textual TUI for sequential, tqdm for parallel)
    **backend_kwargs,  # Extra args forwarded to the backend executor
)
```

**Returns:** `list[list[str]]` — outer list ordered by input indices, inner
list contains file paths returned by the sink for that index.

## Process Isolation

All multiprocess backends (`"process_pool"`, `"loky"`, `"dask"`) execute each
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

## Profiling

To measure wall-clock time, memory, and GPU usage across parallel backends,
ensure `track_metrics=True` (the default). See [Profiling](profiling.md)
for details.

## Examples

### Basic parallel ETL

```python
from physicsnemo_curator.domains.mesh.sources.vtk import VTKSource
from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink
from physicsnemo_curator import run_pipeline

pipeline = (
    VTKSource("./cfd_results/")
    .write(MeshSink(output_dir="./output/"))
)

# Process all items with 8 workers
results = run_pipeline(pipeline, n_jobs=8, backend="process_pool")
print(f"Wrote {sum(len(r) for r in results)} files")
```

### With HuggingFace dataset sources

```python
from physicsnemo_curator.domains.mesh.sources.drivaerml import DrivAerMLSource
from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink
from physicsnemo_curator import run_pipeline

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
from physicsnemo_curator import run_pipeline

# CI uses sequential; production uses all CPUs
n = 1 if os.getenv("CI") else -1
results = run_pipeline(pipeline, n_jobs=n)
```

### Listing available backends

```python
from physicsnemo_curator.run import list_backends

backends = list_backends()
for name, info in backends.items():
    status = "available" if info["available"] else f"requires {info['requires']}"
    print(f"{name}: {info['description']} ({status})")
```
