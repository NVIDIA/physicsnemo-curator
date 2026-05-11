<!---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
-->

# Parallel Safety

This page covers the considerations you must address when building a pipeline
that will be executed in parallel.  While `run_pipeline` handles most
orchestration automatically, the design choices you make in sources, filters,
and sinks determine whether parallel execution is correct.

```{important}
The golden rule: **each source index must be processable independently.**
If two indices share mutable state or write to overlapping storage locations,
you must either constrain the parallelism (via `partition_indices`) or
restructure your pipeline.
```

## Process Isolation Model

All parallel backends (`"process_pool"`, `"loky"`, `"dask"`) execute each
index (or index group) in a **separate OS process**.  This means:

- Each worker gets a deep copy of the pipeline (serialized via pickle).
- Filter instance variables (`self._rows`, `self._count`, etc.) are private
  to each worker — they are never shared or merged automatically.
- Sink writes from different workers target the filesystem concurrently.
- The GIL is bypassed — true parallelism for CPU-bound work.

On Linux, workers are spawned with the `forkserver` start method (avoids
fork+thread deadlocks).  On macOS/Windows, `spawn` is used.

```text
Main Process                    Worker Processes
┌────────────┐                  ┌────────────┐
│            │  pickle pipeline  │ Worker 0   │
│ run_pipeline├─────────────────▶│  index 0   │
│            │                  │  index 4   │
│            │                  └────────────┘
│            │                  ┌────────────┐
│            │                  │ Worker 1   │
│            ├─────────────────▶│  index 1   │
│            │                  │  index 5   │
│            │                  └────────────┘
│            │                  ┌────────────┐
│            │                  │ Worker 2   │
│            ├─────────────────▶│  index 2   │
│            │                  │  index 6   │
│            │                  └────────────┘
│            │                  ┌────────────┐
│            │                  │ Worker 3   │
│            ├─────────────────▶│  index 3   │
│            │                  │  index 7   │
└────────────┘                  └────────────┘
```

## Stateful Filters in Parallel

### The Problem

Stateful filters accumulate data across indices (e.g. running statistics,
row collections for Parquet output).  In sequential mode, a single filter
instance processes all indices and calls `flush()` once at the end.  In
parallel mode, each worker has its own copy of the filter with its own state.

### The Solution: Flush + Merge

The framework provides a three-phase protocol:

1. **Per-index flush** — After each index, the runner calls `flush()` on
   every filter that has both a `flush` method and an `_output_path` attribute.
   Output paths are automatically rewritten to be worker-unique:

   ```text
   Original:  output/stats.parquet
   Worker 0:  output/stats_worker_12345.parquet
   Worker 1:  output/stats_worker_12346.parquet
   Worker 2:  output/stats_worker_12347.parquet
   ```

   If the path contains `{worker_id}`, it is used as a template instead:

   ```text
   Template:  output/stats_{worker_id}.parquet
   Worker 0:  output/stats_12345.parquet
   ```

2. **Gather** — After all workers complete, call `gather_pipeline(pipeline)`
   in the main process.  This discovers shard files via glob and calls each
   filter's `merge()` method.

3. **Merge** — The filter's `@staticmethod merge(shard_paths, output_path)`
   combines per-worker results into a single file and returns the merged path.

### Implementation Checklist for Stateful Filters

```python
class MyStatsFilter(Filter["Mesh"]):
    def __init__(self, output: str) -> None:
        self._output_path = pathlib.Path(output)  # Required for auto-flush
        self._rows: list[dict] = []

    def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
        for mesh in items:
            self._rows.append(self._compute(mesh))
            yield mesh

    def flush(self) -> str | None:
        """Write accumulated rows. Called per-index in parallel."""
        if not self._rows:
            return None
        # APPEND mode — multiple flushes per worker
        mode = "a" if self._output_path.exists() else "w"
        table = pa.Table.from_pylist(self._rows)
        # ... write with append semantics ...
        self._rows.clear()
        path = str(self._output_path)
        self._last_artifacts = [path]
        return path

    def artifacts(self) -> list[str]:
        """Report files produced since last call."""
        paths = self._last_artifacts
        self._last_artifacts = []
        return paths

    @staticmethod
    def merge(parquet_paths: list[str], output: str) -> str:
        """Concatenate per-worker shards into single output."""
        tables = [pq.read_table(p) for p in parquet_paths]
        merged = pa.concat_tables(tables)
        pq.write_table(merged, output)
        return output
```

### Key Design Principles

- **Flush appends, never overwrites** — A single worker may process multiple
  indices.  Each `flush()` call must append to the shard file rather than
  overwrite it.
- **Clear state after flush** — Reset `self._rows = []` (or equivalent) so
  the next index starts fresh.
- **Merge is a static method** — It runs in the main process after all
  workers exit.  It has no access to instance state.
- **Artifacts track what was written** — Return paths from `artifacts()` for
  the dashboard and checkpoint database to track.

### Full Parallel Workflow

```python
from physicsnemo_curator import run_pipeline, gather_pipeline

# 1. Build pipeline with stateful filter
pipeline = (
    VTKSource("./meshes/")
    .filter(MeshStatsFilter(output="stats.parquet"))
    .write(MeshSink(output_dir="./output/"))
)

# 2. Run in parallel — each worker produces its own shard
results = run_pipeline(pipeline, n_jobs=8, backend="process_pool")

# 3. Merge per-worker shards into single output
merged = gather_pipeline(pipeline)
# merged = ["stats.parquet"]  (shard files are cleaned up)
```

## Partition Constraints

### When Indices Must Be Processed Together

Some sources and sinks require that certain indices be processed by the
**same worker** to avoid data corruption.  The `partition_indices` method
declares these constraints.

### Source Partitioning

A source returns `partition_indices` when its data store has concurrency
limitations.  For example, LMDB databases allow only one reader environment
per process — all indices from the same `.lmdb` file must go to the same
worker.

```python
class ASELMDBSource(Source["AtomicData"]):
    def partition_indices(self, indices: list[int]) -> list[list[int]] | None:
        """Group indices by their backing LMDB file."""
        groups: dict[int, list[int]] = defaultdict(list)
        for idx in indices:
            file_id = self._index_to_file_id(idx)
            groups[file_id].append(idx)
        return [sorted(g) for g in groups.values()]
```

### Sink Partitioning

A sink returns `partition_indices` when concurrent writes to the same
storage region would corrupt data.  For example, a Zarr store with
`chunks={"time": 10}` requires that indices 0-9 (which map to the same
chunk along the time axis) are written by the same worker.

```python
class ZarrSink(Sink["xr.DataArray"]):
    def partition_indices(self, indices: Iterable[int]) -> list[list[int]] | None:
        """Group indices by their target Zarr chunk."""
        chunk_size = self._chunks.get(self._append_dim, 1)
        if chunk_size <= 1:
            return None  # Each index is its own chunk — no constraint

        groups: dict[int, list[int]] = defaultdict(list)
        for idx in indices:
            chunk_id = idx // chunk_size
            groups[chunk_id].append(idx)
        return [sorted(group) for _, group in sorted(groups.items())]
```

### Constraint Intersection

When both source and sink declare partition constraints, the runner computes
their **intersection** — the finest partition that satisfies both:

```text
Source constraint:   [[0, 1, 2], [3, 4, 5]]   (by LMDB file)
Sink constraint:    [[0, 1], [2, 3], [4, 5]]   (by Zarr chunk)
Intersection:       [[0, 1], [2], [3], [4, 5]] (satisfies both)
```

If the constraints are incompatible (one requires indices together that the
other requires apart), a `ValueError` is raised at runtime with a message
explaining the conflict.  The typical fix is adjusting the sink's
`chunk_size` to align with source file boundaries.

### Batch Packing

When there are more partition groups than workers, the runner uses greedy
bin-packing to distribute groups across workers while respecting constraints:

```text
8 groups, 4 workers → largest groups assigned to lightest worker
```

Groups are never split across workers — only combined.

## Chunked Output Safety

### The Race Condition

When multiple workers write to the same file or Zarr chunk concurrently,
data corruption occurs.  Three patterns prevent this:

### Pattern 1: Index-Based Naming (No Conflicts)

Each index writes to a unique path.  No coordination needed.

```python
class MeshSink(Sink["Mesh"]):
    def __call__(self, items, index):
        path = self._output_dir / f"mesh_{index:04d}_0"
        mesh.save(str(path))
        return [str(path)]
```

This is the simplest and safest approach for parallel execution.

### Pattern 2: Partition Constraints (Coordinated Access)

When multiple indices must write to the same output (e.g. Zarr chunks),
use `partition_indices` to ensure they go to the same worker:

```python
# ZarrSink: chunk_size=10 means indices 0-9 are one chunk
# partition_indices groups them → same worker handles all 10
# Worker writes sequentially within its group → no race
```

### Pattern 3: Atomic Writes (Concurrent-Safe)

When writing to independent files, use temp-file + rename for atomicity:

```python
def _write_mesh(self, mesh, output_path):
    temp = output_path.parent / f".{output_path.stem}_temp.vtu"
    grid.save(str(temp))
    temp.rename(output_path)  # Atomic on POSIX filesystems
```

This prevents readers from seeing partially-written files.

## Filter Artifacts and the Dashboard

### How Artifacts Flow in Parallel

When `track_metrics=True` (the default), the framework records which files
each filter produces for each index.  This enables the dashboard to
visualize results.

```text
Pipeline execution:
  Worker 0 processes index 0 → flush → artifacts=["stats_worker_123.parquet"]
  Worker 1 processes index 1 → flush → artifacts=["stats_worker_456.parquet"]

gather_pipeline():
  merge(["stats_worker_123.parquet", "stats_worker_456.parquet"], "stats.parquet")
  DB updated: replace shard paths with "stats.parquet"
  Shard files deleted
```

### Dashboard Widget Integration

Filters can provide `dashboard_panel()` to visualize their artifacts.  The
dashboard reads artifact paths from the checkpoint database — after
`gather_pipeline()`, these point to the merged file:

```python
@classmethod
def dashboard_panel(cls, artifact_paths, selected_index=None):
    """Build a Panel widget from merged artifact files."""
    # artifact_paths = ["stats.parquet"] (post-merge)
    # or ["stats_worker_123.parquet", ...] (pre-merge)
    ...
```

Design your dashboard widgets to handle both pre-merge (multiple shards)
and post-merge (single file) states gracefully.

## Common Pitfalls

### 1. Shared Mutable State

```python
# BAD: Class variable shared across instances
class BadFilter(Filter["Mesh"]):
    _global_counter = 0  # Shared across forks — each worker starts at 0!

    def __call__(self, items):
        for mesh in items:
            BadFilter._global_counter += 1  # Isolated per process
            yield mesh
```

**Fix:** Use instance variables (`self._counter`) and accept that each
worker has its own counter.  If you need a global count, compute it
post-hoc from per-worker results.

### 2. File-Based State Without Locking

```python
# BAD: Multiple workers writing to same file
class BadSink(Sink["Mesh"]):
    def __call__(self, items, index):
        with open("results.jsonl", "a") as f:  # Race condition!
            for mesh in items:
                f.write(json.dumps(info) + "\n")
```

**Fix:** Use index-based naming (`results_{index}.jsonl`) or
`partition_indices` to serialize access.

### 3. Missing flush() Append Semantics

```python
# BAD: Overwrites previous data on each flush
def flush(self):
    table = pa.Table.from_pylist(self._rows)
    pq.write_table(table, str(self._output_path))  # Overwrites!
```

**Fix:** Check if the file exists and append:

```python
def flush(self):
    table = pa.Table.from_pylist(self._rows)
    if self._output_path.exists():
        existing = pq.read_table(str(self._output_path))
        table = pa.concat_tables([existing, table])
    pq.write_table(table, str(self._output_path))
    self._rows.clear()
```

### 4. Forgetting gather_pipeline()

```python
# BAD: Stateful filter outputs are never merged
results = run_pipeline(pipeline, n_jobs=8, backend="process_pool")
# output/stats_worker_12345.parquet exists
# output/stats_worker_12346.parquet exists
# output/stats.parquet does NOT exist

# GOOD: Merge after parallel execution
results = run_pipeline(pipeline, n_jobs=8, backend="process_pool")
merged = gather_pipeline(pipeline)  # Produces output/stats.parquet
```

### 5. Incompatible Partition Constraints

```python
# Source groups: [[0,1,2,3,4], [5,6,7,8,9]]  (by LMDB file)
# Sink groups:   [[0,1,2], [3,4,5], [6,7,8,9]]  (by Zarr chunk)
#
# Index 3 must be with 0-4 (source) but also with 3-5 (sink)
# Index 5 must be with 5-9 (source) but also with 3-5 (sink)
# → ValueError: constraints conflict
```

**Fix:** Adjust sink chunk boundaries to align with source boundaries,
or restructure the pipeline to avoid the conflict.

## Decision Guide

Use this table to determine the right parallel strategy:

| Pipeline Design | Safe for Parallel? | Action Required |
|----------------|--------------------|-----------------|
| Source → Sink (no filters) | Yes | None (index-based naming is sufficient) |
| Source → Stateless filter → Sink | Yes | None |
| Source → Stateful filter → Sink | Partial | Implement `flush()` + `merge()`, call `gather_pipeline()` |
| Sink writes to shared store | No (without constraints) | Implement `partition_indices` on sink |
| Source has concurrency limits | No (without constraints) | Implement `partition_indices` on source |
| Multiple stateful filters | Partial | Each needs `flush()` + `merge()` |

## Sequential Fallback

When parallel safety is too complex or not worth the engineering effort,
sequential execution preserves all invariants:

```python
# Simple and correct — filter state accumulates naturally
results = run_pipeline(pipeline, n_jobs=1)  # or backend="sequential"
pipeline.filters[0].flush()  # Single output file, no merge needed
```

This is the recommended approach for:

- Prototyping and debugging
- Small datasets where parallelism overhead exceeds benefit
- Filters with complex inter-index dependencies
- Output formats that don't support concurrent writes

## Summary

| Concept | Sequential | Parallel |
|---------|-----------|----------|
| Filter state | Single instance, all indices | Per-worker copy, isolated |
| flush() | Called once at end by user | Called per-index automatically |
| Output naming | User controls | Framework adds `_worker_{pid}` suffix |
| merge() | Not needed | Required for stateful filters |
| gather_pipeline() | Not needed | Required after run_pipeline() |
| partition_indices | Ignored | Enforced by runner |
| Checkpoint DB | Single process writes | WAL mode for concurrent writes |
