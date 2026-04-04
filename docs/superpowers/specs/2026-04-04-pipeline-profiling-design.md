---
orphan: true
---

# Pipeline Profiling Utility — Design Spec

**Date:** 2026-04-04
**Status:** Approved
**Approach:** Transparent Proxy Wrapper (Approach A)

## Problem

PhysicsNeMo Curator has zero built-in profiling infrastructure. Users running
ETL pipelines through `run_pipeline()` have no way to measure wall-clock time,
memory consumption, or GPU usage at whole-pipeline, per-index, or per-stage
granularity without writing custom instrumentation every time.

## Goals

1. Provide a `ProfiledPipeline` wrapper that is duck-type compatible with
   `Pipeline[T]` and can be passed directly to `run_pipeline()` without any
   backend changes.
2. Collect metrics at three granularity levels: whole-pipeline, per-index,
   per-stage (source / each filter / sink).
3. Support multiple output formats: console table, JSON file, CSV file, and
   programmatic dict access.
4. Work with all six backends (sequential, thread_pool, process_pool, loky,
   dask, prefect).
5. Add comprehensive tests with good coverage.
6. Integrate with ASV benchmarks to track profiling overhead across commits.
7. Update user and developer documentation.

## Non-Goals

- Adding new required dependencies (stdlib only for core; `psutil` and
  `torch.cuda` are optional).
- Modifying the `RunBackend` protocol or any existing backend implementation.
- Continuous / streaming metric export (batch collection only).

## Architecture

### Module Layout

```
src/physicsnemo_curator/core/profiling.py   # NEW — all profiling code
src/physicsnemo_curator/__init__.py          # EDIT — add exports
test/core/test_profiling.py                  # NEW — unit tests
test/run/test_profiling_backends.py          # NEW — integration tests
benchmarks/bench_profiling.py                # NEW — ASV benchmarks
docs/user-guide/profiling.md                 # NEW — user guide page
docs/user-guide/parallel.md                  # EDIT — cross-reference
docs/developer-guide/benchmarking.md         # EDIT — mention profiling benchmarks
```

### Data Classes

```python
@dataclasses.dataclass
class StageMetrics:
    """Metrics for a single stage (source, one filter, or sink)."""
    name: str                    # e.g. "source", "DoubleFilter", "sink"
    wall_time_ns: int            # time.perf_counter_ns delta
    # Note: memory is tracked at index level only (not per-stage),
    # because chained lazy generators make per-stage attribution unreliable.

@dataclasses.dataclass
class IndexMetrics:
    """Metrics for one __getitem__ call (one index)."""
    index: int
    stages: list[StageMetrics]
    wall_time_ns: int            # total wall time for this index
    peak_memory_bytes: int       # peak memory for this index
    gpu_memory_bytes: int | None

@dataclasses.dataclass
class PipelineMetrics:
    """Aggregated metrics across all indices."""
    indices: list[IndexMetrics]

    # Computed properties
    @property
    def total_wall_time_ns(self) -> int: ...
    @property
    def mean_index_time_ns(self) -> float: ...
    @property
    def total_peak_memory_bytes(self) -> int: ...

    # Output methods
    def to_console(self) -> None: ...
    def to_json(self, path: str | pathlib.Path) -> None: ...
    def to_csv(self, path: str | pathlib.Path) -> None: ...
    def summary(self) -> dict: ...
```

### ProfiledPipeline Class

```python
class ProfiledPipeline(Generic[T]):
    """Transparent profiling wrapper around Pipeline[T].

    Duck-type compatible with Pipeline — exposes source, filters, sink,
    __len__, and __getitem__. Can be passed directly to run_pipeline().
    """

    def __init__(self, pipeline: Pipeline[T], *, track_gpu: bool = False) -> None:
        self._pipeline = pipeline
        self._track_gpu = track_gpu
        self._session_id = uuid.uuid4().hex[:12]
        self._metrics_dir = pathlib.Path(tempfile.gettempdir()) / f"pnc_profile_{self._session_id}"
        self._metrics_dir.mkdir(exist_ok=True)

    # Duck-type compatibility
    @property
    def source(self) -> Source[T]: return self._pipeline.source
    @property
    def filters(self) -> list[Filter[T]]: return self._pipeline.filters
    @property
    def sink(self) -> Sink[T]: return self._pipeline.sink
    def __len__(self) -> int: return len(self._pipeline)

    def __getitem__(self, index: int) -> list[str]:
        # 1. Start tracemalloc
        # 2. Wrap source with timed generator proxy
        # 3. Wrap each filter with timed generator proxy
        # 4. Time sink call (forces evaluation of full chain)
        # 5. Record GPU memory if track_gpu
        # 6. Build IndexMetrics, serialize to temp file
        # 7. Return original list[str] result (contract preserved)
        ...

    def collect_metrics(self) -> PipelineMetrics:
        """Gather metrics from temp files after run_pipeline completes."""
        ...

    def cleanup(self) -> None:
        """Remove temp metric files."""
        ...

    @property
    def metrics(self) -> PipelineMetrics:
        """Convenience: calls collect_metrics()."""
        return self.collect_metrics()
```

### Per-Stage Instrumentation Detail

Generators in the pipeline are lazy. Calling `source[index]` returns a
generator object instantly; actual work happens when the sink iterates it.
Similarly, `filter(gen)` returns a new generator wrapping the input.

To attribute time to each stage accurately:

1. **Source timing**: Wrap the source's generator with a `_TimedGenerator`
   class (implements `Iterator[T]` protocol: `__iter__`, `__next__`) that
   accumulates `time.perf_counter_ns()` across all `__next__` calls.
2. **Filter timing**: Each filter's output generator is similarly wrapped
   with a `_TimedGenerator` instance. Time attributed to a filter is the
   delta between its output generator's accumulated time and the input
   generator's accumulated time (i.e., the time the filter itself spends,
   not the upstream source/filter).
3. **Sink timing**: The sink call is timed directly. Since the sink consumes
   the iterator, all upstream generators execute during this call. The
   sink's own time is `total_wall_time - last_wrapper.accumulated_time`.

In practice, the simplest correct approach:
- Wrap source generator: each `__next__` records time → accumulates into
  `source_time`.
- Wrap filter N generator: each `__next__` records time, but this includes
  pulling from filter N-1. Filter N's own time = its total `__next__` time
  minus filter N-1's total `__next__` time.
- Sink time = total `__getitem__` time minus last wrapper's accumulated time.

This chain-subtraction approach gives accurate per-stage attribution.

### Memory Tracking

- `tracemalloc.start()` / `tracemalloc.get_traced_memory()` for peak Python
  memory per index.
- Per-stage memory is harder to attribute reliably; we track per-index peak
  only. Stage-level memory is not included (not reliably separable with
  `tracemalloc` for chained generators).

### GPU Memory Tracking

When `track_gpu=True`:
- Check `torch.cuda.is_available()` at init time.
- Before `__getitem__`: `torch.cuda.reset_peak_memory_stats()`,
  record `torch.cuda.memory_allocated()`.
- After `__getitem__`: `torch.cuda.max_memory_allocated()` minus baseline.
- GPU memory is per-index only (same reason as RAM: generators interleave).

### Pickle Compatibility for Parallel Backends

For `process_pool`, `loky`, `dask`, and `prefect`, each worker receives a
pickled copy of the pipeline. Instance attributes like `self._local_metrics`
cannot flow back to the parent process.

Solution: **temp-file side channel**.

- `__getitem__` serializes each `IndexMetrics` to a JSON file in
  `self._metrics_dir` (named `{index}.json`).
- The temp directory path is a plain string, survives pickling.
- After `run_pipeline()` returns, the user calls `profiled.metrics` (or
  `profiled.collect_metrics()`), which reads all JSON files from the temp
  directory, deserializes them, and returns a `PipelineMetrics` object.
- `profiled.cleanup()` removes the temp directory.

This works because:
- `tempfile.gettempdir()` is the same path in parent and child processes
  (same filesystem).
- JSON files are written atomically (write to temp + rename).
- No changes to any backend code.

### Public API

Add to `src/physicsnemo_curator/__init__.py`:

```python
from physicsnemo_curator.core.profiling import ProfiledPipeline, PipelineMetrics
```

### Usage Example

```python
from physicsnemo_curator import Pipeline, ProfiledPipeline, run_pipeline

pipeline = Pipeline(source=my_source, filters=[f1, f2], sink=my_sink)
profiled = ProfiledPipeline(pipeline, track_gpu=True)

results = run_pipeline(profiled, n_jobs=4, backend="process_pool")
metrics = profiled.metrics

metrics.to_console()           # Pretty table to stdout
metrics.to_json("profile.json") # Save as JSON
metrics.to_csv("profile.csv")  # Save as CSV
info = metrics.summary()       # Dict for programmatic use

profiled.cleanup()             # Remove temp files
```

## Testing Strategy

### Unit Tests — `test/core/test_profiling.py`

Mark: `pytestmark = pytest.mark.unit`

Test implementations defined at module level for pickle compatibility:
- `_TimedSource(Source[int])` — yields integers with a small `time.sleep`
- `_SlowFilter(Filter[int])` — adds delay per item
- `_CollectSink(Sink[int])` — collects to temp files

Tests:
1. **Duck-type compatibility**: `ProfiledPipeline` has `source`, `filters`,
   `sink`, `__len__`, `__getitem__` matching `Pipeline` interface.
2. **Basic metrics collection**: Run one index, verify `IndexMetrics` has
   correct number of stages, wall times are positive.
3. **Per-stage timing accuracy**: Source time > 0, filter time > 0, sink
   time > 0. Sum of stage times approximates total time.
4. **Memory tracking**: Allocate a known-size list inside a source; verify
   `peak_memory_bytes` is in the right ballpark.
5. **GPU tracking (mocked)**: Mock `torch.cuda` functions, verify
   `gpu_memory_bytes` is populated when `track_gpu=True`.
6. **GPU tracking disabled**: When `track_gpu=False`, GPU fields are `None`.
7. **Output formats**: `.to_json()` produces valid JSON with expected keys.
   `.to_csv()` produces valid CSV. `.to_console()` writes to stdout.
   `.summary()` returns dict with expected structure.
8. **Pickle round-trip**: `pickle.dumps(profiled)` / `pickle.loads(...)` works
   and preserves configuration.
9. **Temp-file metrics**: Simulate parallel execution — call `__getitem__`
   from multiple threads, then `collect_metrics()`, verify all indices present.
10. **Empty pipeline**: Pipeline with zero-length source produces empty metrics.
11. **No-filter pipeline**: Pipeline with empty filter list still profiles
    source and sink correctly.
12. **Error propagation**: If a stage raises, the exception propagates
    unchanged (profiling doesn't swallow errors).

### Integration Tests — `test/run/test_profiling_backends.py`

Mark: `pytestmark = pytest.mark.integration`

Test implementations at module level (pickle-safe):
- `_NumberSource`, `_DoubleFilter`, `_ListSink` (reuse patterns from
  `test/run/test_backends.py`)

Tests:
1. **Sequential backend**: `ProfiledPipeline` + `run_pipeline(backend="sequential")`.
   Verify results match non-profiled run. Verify metrics collected for all indices.
2. **Thread pool backend**: Same verification with `backend="thread_pool"`.
3. **Process pool backend**: Same verification with `backend="process_pool"`.
   Verify temp-file metric collection works across processes.

## ASV Benchmark Integration

### New File — `benchmarks/bench_profiling.py`

Follows existing ASV conventions (classes with `time_` / `peakmem_` / `track_`
prefixed methods).

```python
class TimeProfilingOverhead:
    """Benchmark wall-clock overhead of ProfiledPipeline vs raw Pipeline."""
    params = [[10, 100, 1000]]
    param_names = ["n_indices"]

    def setup(self, n_indices):
        self.pipeline = Pipeline(source=_NumberSource(n_indices), ...)
        self.profiled = ProfiledPipeline(self.pipeline)

    def time_raw_pipeline(self, n_indices):
        for i in range(n_indices):
            self.pipeline[i]

    def time_profiled_pipeline(self, n_indices):
        for i in range(n_indices):
            self.profiled[i]

    def track_overhead_percent(self, n_indices):
        # (profiled_time - raw_time) / raw_time * 100
        ...

class MemProfilingOverhead:
    """Benchmark memory overhead of ProfiledPipeline."""
    params = [[10, 100]]
    param_names = ["n_indices"]

    def setup(self, n_indices): ...
    def peakmem_raw_pipeline(self, n_indices): ...
    def peakmem_profiled_pipeline(self, n_indices): ...
```

## Documentation Updates

### New Page — `docs/user-guide/profiling.md`

Contents:
- Overview: what ProfiledPipeline does
- Quick start example
- Metrics granularity (whole-pipeline, per-index, per-stage)
- Output formats (console, JSON, CSV, dict)
- GPU tracking
- Using with parallel backends
- Cleanup

### Edit — `docs/user-guide/parallel.md`

Add a "Profiling" subsection cross-referencing the profiling guide:
> "To profile pipeline execution across parallel backends, see
> the Profiling page in the user guide."

### Edit — `docs/developer-guide/benchmarking.md`

Add mention of `bench_profiling.py` in the ASV benchmarks section.

## Dependencies

**Required (stdlib only):**
- `time`, `tracemalloc`, `json`, `csv`, `dataclasses`, `pathlib`, `tempfile`,
  `uuid`, `typing`, `shutil`

**Optional:**
- `torch.cuda` — GPU memory tracking (auto-detected, no import error if absent)

No new entries in `pyproject.toml`.

## Constraints

- Return type of `ProfiledPipeline.__getitem__` must remain `list[str]` —
  the backend contract cannot change.
- All test helper classes must be defined at module level for pickle
  compatibility.
- Profiling overhead target: <5% wall-clock for typical workloads.
- SPDX license headers on all new files.
- NumPy-style docstrings, 99% interrogate coverage.
