# Profiling Pipelines

`Pipeline` includes built-in profiling that collects wall-clock time,
memory, and (optionally) GPU metrics at whole-pipeline, per-index, and
per-stage granularity — without requiring any wrappers or separate
configuration.

Profiling is **enabled by default** via the `track_metrics` field
(which also enables checkpointing).

## Quick Start

```python
from physicsnemo_curator import Pipeline, run_pipeline

# Pipeline with default profiling (track_metrics=True)
pipeline = (
    MySource(data_dir="/data/")
    .filter(MyFilter())
    .write(MySink(output_dir="/output/"))
)

# Run exactly as before — works with all backends
results = run_pipeline(pipeline, n_jobs=4, backend="process_pool")

# Inspect metrics
metrics = pipeline.metrics
metrics.to_console()
```

## Metrics Granularity

Profiling collects data at three levels:

| Level | What's measured |
|-------|----------------|
| **Whole-pipeline** | Total wall time, peak memory across all indices |
| **Per-index** | Wall time, peak memory, GPU memory for each source index |
| **Per-stage** | Wall time for source, each filter, and sink |

### Per-Stage Timing

The pipeline chain `source → filter₁ → filter₂ → … → sink` uses lazy
generators.  Profiling wraps each stage's generator with an internal
timer to attribute time accurately using chain subtraction:

- **Source time** = time spent yielding items from the source
- **Filter N time** = time spent in filter N's own logic (excluding upstream)
- **Sink time** = time spent in the sink (excluding all upstream generators)

### Memory Tracking

Peak Python memory per index is tracked via `tracemalloc`. This captures
Python-level allocations accurately but does not cover C-extension or
Rust-extension memory. Per-stage memory is not tracked because chained
lazy generators make per-stage attribution unreliable.

Disable memory tracking (to avoid `tracemalloc` overhead):

```python
pipeline = Pipeline(
    source=MySource(),
    sink=MySink(),
    track_memory=False,  # disable tracemalloc
)
```

### GPU Memory Tracking

To track GPU memory, set `track_gpu=True`:

```python
pipeline = Pipeline(
    source=MySource(),
    sink=MySink(),
    track_gpu=True,
)
```

This uses `torch.cuda.max_memory_allocated()` to capture peak GPU memory
per index. Requires PyTorch with CUDA support.  If `torch` is not installed
or CUDA is unavailable, GPU fields will be `None`.

## Output Formats

### Console Table

```python
metrics.to_console()
```

Prints a human-readable summary with per-index breakdown and stage averages.

### JSON File

```python
metrics.to_json("profile.json")
```

Writes full metrics (including per-stage breakdowns) as structured JSON.

### CSV File

```python
metrics.to_csv("profile.csv")
```

Writes one row per index with columns for wall time, memory, GPU memory,
and per-stage timing.

### Programmatic Access

```python
info = metrics.summary()
info["total_wall_time_ns"]    # int
info["mean_index_time_ns"]    # float
info["indices"][0]["stages"]  # list of stage dicts
```

## Using with Parallel Backends

Profiling works with **all** backends — sequential, thread_pool,
process_pool, loky, dask, and prefect — without any backend modifications.

Metrics are stored in a SQLite database using WAL mode, which supports
safe concurrent writes from multiple threads and processes.  Each worker
records its metrics independently and the `pipeline.metrics` property
aggregates them on demand.

```python
results = run_pipeline(pipeline, n_jobs=8, backend="process_pool")

# Aggregates metrics from all workers
metrics = pipeline.metrics
metrics.to_console()
```

## Disabling Profiling

Set `track_metrics=False` to disable all profiling and checkpointing:

```python
pipeline = Pipeline(
    source=MySource(),
    sink=MySink(),
    track_metrics=False,
)
```

## Full Example

```python
from physicsnemo_curator import Pipeline, run_pipeline

# Build a pipeline (profiling is on by default)
pipeline = (
    MySource(path="/data/cfd/")
    .filter(NormalizeFilter())
    .filter(ResampleFilter(target_resolution=0.01))
    .write(MeshSink(output_dir="/output/"))
)

# Enable GPU tracking
pipeline.track_gpu = True

# Run
results = run_pipeline(pipeline, n_jobs=4, backend="process_pool")

# Analyze
metrics = pipeline.metrics
metrics.to_console()             # Quick visual summary
metrics.to_json("profile.json")  # Detailed JSON for analysis
metrics.to_csv("profile.csv")    # CSV for spreadsheet

# Programmatic access
summary = metrics.summary()
print(f"Processed {summary['num_indices']} indices")
print(f"Total time: {summary['total_wall_time_ns'] / 1e9:.2f}s")
print(f"Peak memory: {summary['total_peak_memory_bytes'] / 1e6:.1f} MB")
```

## Pipeline Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `track_metrics` | `bool` | `True` | Enable timing, checkpointing, and metrics |
| `track_memory` | `bool` | `True` | Enable `tracemalloc` memory tracking |
| `track_gpu` | `bool` | `False` | Enable GPU memory tracking via PyTorch |
| `db_dir` | `Path \| None` | `None` | Override database directory (default: `.pnc/`) |

## API Reference

### `PipelineMetrics`

```python
class PipelineMetrics:
    indices: list[IndexMetrics]

    # Properties
    total_wall_time_ns: int
    mean_index_time_ns: float
    total_peak_memory_bytes: int

    # Output
    def to_console(self) -> None: ...
    def to_json(self, path: str | Path) -> None: ...
    def to_csv(self, path: str | Path) -> None: ...
    def summary(self) -> dict: ...
```

### `IndexMetrics`

```python
class IndexMetrics:
    index: int
    stages: list[StageMetrics]
    wall_time_ns: int
    peak_memory_bytes: int
    gpu_memory_bytes: int | None
```

### `StageMetrics`

```python
class StageMetrics:
    name: str
    wall_time_ns: int
```
