# Profiling Pipelines

`ProfiledPipeline` is a transparent wrapper around `Pipeline` that collects
wall-clock time, memory, and (optionally) GPU metrics at whole-pipeline,
per-index, and per-stage granularity — without requiring any changes to
your pipeline or backend configuration.

## Quick Start

```python
from physicsnemo_curator import Pipeline, ProfiledPipeline, run_pipeline

# Wrap any existing pipeline
profiled = ProfiledPipeline(pipeline)

# Run exactly as before — works with all backends
results = run_pipeline(profiled, n_jobs=4, backend="process_pool")

# Inspect metrics
metrics = profiled.metrics
metrics.to_console()

# Clean up temp files when done
profiled.cleanup()
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
generators.  `ProfiledPipeline` wraps each stage's generator with an
internal timer to attribute time accurately using chain subtraction:

- **Source time** = time spent yielding items from the source
- **Filter N time** = time spent in filter N's own logic (excluding upstream)
- **Sink time** = time spent in the sink (excluding all upstream generators)

### Memory Tracking

Peak Python memory per index is tracked via `tracemalloc`. This captures
Python-level allocations accurately but does not cover C-extension or
Rust-extension memory. Per-stage memory is not tracked because chained
lazy generators make per-stage attribution unreliable.

### GPU Memory Tracking

To track GPU memory, pass `track_gpu=True`:

```python
profiled = ProfiledPipeline(pipeline, track_gpu=True)
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
info = metrics.summary()      # dict
info["total_wall_time_ns"]    # int
info["mean_index_time_ns"]    # float
info["indices"][0]["stages"]  # list of stage dicts
```

## Using with Parallel Backends

`ProfiledPipeline` works with **all** backends — sequential, thread_pool,
process_pool, loky, dask, and prefect — without any backend modifications.

For multiprocess backends (`process_pool`, `loky`, `dask`, `prefect`),
metrics are serialized to a temporary directory as JSON files. Each worker
process writes its own index metrics to a uniquely-named file. After
`run_pipeline()` returns, call `profiled.metrics` (or
`profiled.collect_metrics()`) to read and aggregate all results.

```python
profiled = ProfiledPipeline(pipeline)
results = run_pipeline(profiled, n_jobs=8, backend="process_pool")

# Reads all temp files and aggregates
metrics = profiled.metrics
metrics.to_console()

# Clean up temp directory
profiled.cleanup()
```

## Full Example

```python
from physicsnemo_curator import Pipeline, ProfiledPipeline, run_pipeline

# Build a pipeline
pipeline = (
    MySource(path="/data/cfd/")
    .filter(NormalizeFilter())
    .filter(ResampleFilter(target_resolution=0.01))
    .write(MeshSink(output_dir="/output/"))
)

# Profile it
profiled = ProfiledPipeline(pipeline, track_gpu=True)
results = run_pipeline(profiled, n_jobs=4, backend="process_pool")

# Analyze
metrics = profiled.metrics
metrics.to_console()             # Quick visual summary
metrics.to_json("profile.json")  # Detailed JSON for analysis
metrics.to_csv("profile.csv")    # CSV for spreadsheet

# Programmatic access
summary = metrics.summary()
print(f"Processed {summary['num_indices']} indices")
print(f"Total time: {summary['total_wall_time_ns'] / 1e9:.2f}s")
print(f"Peak memory: {summary['total_peak_memory_bytes'] / 1e6:.1f} MB")

# Clean up
profiled.cleanup()
```

## API Reference

### `ProfiledPipeline`

```python
class ProfiledPipeline(Generic[T]):
    def __init__(self, pipeline: Pipeline[T], *, track_gpu: bool = False) -> None: ...

    # Duck-type compatibility
    source: Source[T]          # property
    filters: list[Filter[T]]  # property
    sink: Sink[T] | None      # property
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> list[str]: ...

    # Metrics
    def collect_metrics(self) -> PipelineMetrics: ...
    metrics: PipelineMetrics   # property (shortcut for collect_metrics)
    def cleanup(self) -> None: ...
```

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
