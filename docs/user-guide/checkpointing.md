<!---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
-->

# Checkpointing Pipelines

`Pipeline` includes built-in checkpointing that records completed indices
in a SQLite database.  On restart, indices that already finished are
skipped — their cached output paths are returned immediately without
re-executing the source, filters, or sink.

Checkpointing is **enabled by default** via the `track_metrics` field
(which also enables timing and memory profiling).

## Quick Start

```python
from physicsnemo_curator import Pipeline, run_pipeline

# Pipeline with default checkpointing (track_metrics=True)
pipeline = (
    MySource(data_dir="/data/")
    .filter(MyFilter())
    .write(MySink(output_dir="/output/"))
)

# Run — works with all backends
results = run_pipeline(pipeline, n_jobs=4, backend="process_pool")

# Interrupt and restart — completed indices are skipped
results = run_pipeline(pipeline, n_jobs=4, backend="process_pool")

# Inspect progress
print(pipeline.summary())
```

## How It Works

When `track_metrics=True` (the default), each `pipeline[index]` call:

1. **Checks the database** for a prior completion record for this index.
2. If found, **returns the cached paths** immediately (no computation).
3. If not found, **runs the pipeline** and records timing, memory, and
   output paths to SQLite.
4. If the pipeline **raises an exception**, records the error and re-raises.

## Controlling the Database Location

By default, the database is stored at `.pnc/{config_hash[:16]}.db`
relative to the current working directory.  Each unique pipeline
configuration gets its own database file (based on its SHA-256 hash).

Override the directory with `db_dir`:

```python
from pathlib import Path

pipeline = Pipeline(
    source=MySource(data_dir="/data/"),
    filters=[MyFilter()],
    sink=MySink(output_dir="/output/"),
    db_dir=Path("/output/checkpoints"),
)
```

## Disabling Checkpointing

Set `track_metrics=False` to disable all checkpointing and metrics:

```python
pipeline = Pipeline(
    source=MySource(data_dir="/data/"),
    filters=[MyFilter()],
    sink=MySink(output_dir="/output/"),
    track_metrics=False,
)
```

## Provenance Tracking

The checkpoint stores full pipeline provenance — source class, filter
parameters, sink configuration — as a JSON document with a SHA-256 hash.
Each unique configuration gets its own database file, so different
pipelines never collide.

## Error Handling

Failed indices are recorded with their error message but are **not**
marked as completed.  On the next run they will be retried automatically:

```python
# First run — index 42 fails
results = run_pipeline(pipeline, n_jobs=4)

# Check what failed
print(pipeline.failed_indices)
# {42: "RuntimeError: corrupt file at /data/sample_42.lmdb"}

# Fix the underlying issue, then retry — only index 42 runs
results = run_pipeline(pipeline, n_jobs=4)
```

## Query API

| Property / Method | Returns | Description |
|---|---|---|
| `pipeline.completed_indices` | `set[int]` | Successfully completed indices |
| `pipeline.failed_indices` | `dict[int, str]` | Failed indices with error messages |
| `pipeline.remaining_indices()` | `list[int]` | Indices not yet completed (sorted) |
| `pipeline.summary()` | `dict` | Total, completed, failed, remaining counts + elapsed time + worker count |
| `pipeline.metrics` | `PipelineMetrics` | Full timing and memory metrics |
| `pipeline.active_workers` | `list[dict]` | Workers registered for this run |
| `pipeline.reset()` | `None` | Clear all records and start fresh |
| `pipeline.reset_index(i)` | `None` | Re-run a single index |

### Summary Example

```python
>>> pipeline.summary()
{'total': 80, 'completed': 65, 'failed': 2, 'remaining': 13,
 'errors': 2, 'avg_time_ms': 48.2}
```

## Combining with Profiling

Checkpointing and profiling are unified — both are controlled by
`track_metrics`.  When enabled, you get both checkpoint/resume and
per-index timing and memory metrics automatically.  See
[Profiling](profiling.md) for details on accessing metrics.

## SQLite Database

The checkpoint uses a SQLite database in WAL (Write-Ahead Logging)
mode for safe concurrent writes from multiple threads or processes.  The
database contains four tables:

**`pipeline_runs`** — one row per unique pipeline configuration:

| Column | Type | Description |
|---|---|---|
| `run_id` | INTEGER | Auto-incrementing primary key |
| `config_hash` | TEXT | SHA-256 of the pipeline config JSON |
| `config_json` | TEXT | Full pipeline configuration |
| `started_at` | TEXT | ISO-8601 timestamp |

**`index_results`** — one row per processed index:

| Column | Type | Description |
|---|---|---|
| `idx` | INTEGER | Source index |
| `run_id` | INTEGER | Foreign key to `pipeline_runs` |
| `status` | TEXT | `completed` or `error` |
| `output_paths` | TEXT | JSON array of output file paths |
| `completed_at` | TEXT | ISO-8601 timestamp |
| `wall_time_ns` | INTEGER | Wall-clock time in nanoseconds |
| `peak_memory_bytes` | INTEGER | Peak Python memory (bytes) |
| `gpu_memory_bytes` | INTEGER | Peak GPU memory (bytes, or NULL) |
| `error` | TEXT | Error message (NULL for success) |

**`stage_metrics`** — per-stage timing for each index:

| Column | Type | Description |
|---|---|---|
| `idx` | INTEGER | Source index |
| `run_id` | INTEGER | Foreign key |
| `stage_order` | INTEGER | 0 = source, 1..N = filters, N+1 = sink |
| `stage_name` | TEXT | Stage class name |
| `wall_time_ns` | INTEGER | Wall-clock time in nanoseconds |

**`workers`** — one row per worker thread/process:

| Column | Type | Description |
|---|---|---|
| `worker_id` | TEXT | UUID4 hex string (primary key) |
| `run_id` | INTEGER | Foreign key to `pipeline_runs` |
| `pid` | INTEGER | OS process ID |
| `hostname` | TEXT | Machine hostname |
| `started_at` | TEXT | ISO-8601 timestamp |
| `last_heartbeat` | TEXT | ISO-8601 timestamp (updated on index start/finish) |
| `current_index` | INTEGER | Index currently being processed (NULL if idle) |

## Worker Progress Tracking

Pipeline execution automatically registers workers in the database.
Each thread (or process) gets a stable UUID identifier.  The worker
record is updated when an index starts and when it finishes.

```python
# Run a multi-worker pipeline
results = run_pipeline(pipeline, n_jobs=4, backend="process_pool")

# Check which workers participated
for w in pipeline.active_workers:
    print(f"Worker {w['worker_id'][:8]} (PID {w['pid']}) on {w['hostname']}")
    if w["current_index"] is not None:
        print(f"  Currently processing index {w['current_index']}")
```

Worker tracking works with all backends — the instrumentation is in
`Pipeline.__getitem__`, not in the backends themselves.

## Full Example

```python
from pathlib import Path
from physicsnemo_curator import Pipeline, run_pipeline
from physicsnemo_curator.atm import ASELMDBSource, AtomicDataZarrSink

# Build pipeline — checkpointing is on by default
pipeline = (
    ASELMDBSource(data_dir="/data/val/")
    .write(AtomicDataZarrSink(output_path="/output/dataset.zarr"))
)

# Optionally control DB location
pipeline.db_dir = Path("/output/checkpoints")

# Run — can be interrupted and resumed
results = run_pipeline(pipeline, n_jobs=8, backend="process_pool")

# Check progress
s = pipeline.summary()
print(f"Completed: {s['completed']}/{s['total']}")

if s['failed'] > 0:
    print(f"Failed indices: {list(pipeline.failed_indices.keys())}")

# Start fresh if needed
# pipeline.reset()
```
