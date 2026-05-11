<!---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
-->

# Checkpointing Pipelines

`Pipeline` includes built-in metrics tracking that records completed indices,
timing, memory, and output paths in a SQLite database.

By default, **each pipeline run creates a fresh database** — this captures
metrics and provenance for every execution without automatically resuming
from prior runs.  To enable checkpoint resumption (skipping previously
completed indices), set `resume=True`.

## Quick Start

```python
from physicsnemo_curator import Pipeline, run_pipeline

# Default behavior — fresh database each run (metrics only, no resume)
pipeline = (
    MySource(data_dir="/data/")
    .filter(MyFilter())
    .write(MySink(output_dir="/output/"))
)
results = run_pipeline(pipeline, n_jobs=4, backend="process_pool")

# Enable checkpoint resumption with resume=True
pipeline = Pipeline(
    source=MySource(data_dir="/data/"),
    filters=[MyFilter()],
    sink=MySink(output_dir="/output/"),
    resume=True,
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

The `resume` flag controls whether prior results can be found:

### Default Behavior (``resume=False``)

A fresh database is created with a unique timestamp in its filename
(``{hash}_{timestamp}.db``).  Because the database is new, the checkpoint
table is always empty — all indices are processed.  The database still
provides metrics, provenance, and error tracking for the current run.

### Resume Mode (``resume=True``)

The database uses a stable filename (``{hash}.db``).  If a database for
the same pipeline configuration already exists, the run picks up where it
left off — completed indices return their cached output paths immediately
without re-executing any stages.

## Controlling the Database Location

By default, the database is stored in `~/.cache/psnc/`.  The filename
depends on the `resume` setting:

- **Default** (`resume=False`): `{config_hash[:16]}_{timestamp}.db` — unique
  per instantiation.
- **Resume mode** (`resume=True`): `{config_hash[:16]}.db` — stable across
  runs with the same pipeline configuration.

Each unique pipeline configuration gets its own hash (SHA-256 of the
pipeline config).  The cache directory follows XDG conventions and can be
controlled with environment variables:

| Priority | Method | Example |
|----------|--------|---------|
| 1 (highest) | `db_dir` field on Pipeline | `Pipeline(..., db_dir="/output/checkpoints")` |
| 2 | `PSNC_CACHE_DIR` environment variable | `export PSNC_CACHE_DIR=/data/pipeline-cache` |
| 3 (lowest) | XDG default | `~/.cache/psnc/` (or `$XDG_CACHE_HOME/psnc/`) |

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

## Disabling Metrics Tracking

Set `track_metrics=False` to disable all database creation, checkpointing,
and metrics recording:

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
marked as completed.  When using `resume=True`, they will be retried
automatically on the next run:

```python
# Pipeline with resume=True to enable retry on restart
pipeline = Pipeline(
    source=MySource(data_dir="/data/"),
    filters=[MyFilter()],
    sink=MySink(output_dir="/output/"),
    resume=True,
)

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
[Metrics & Dashboard](dashboard.md) for details on accessing metrics.

## SQLite Database

The checkpoint uses a SQLite database in WAL (Write-Ahead Logging)
mode for safe concurrent writes from multiple threads or processes.  The
database contains seven tables:

**`pipeline_runs`** — one row per unique pipeline configuration:

| Column | Type | Description |
|---|---|---|
| `run_id` | INTEGER | Auto-incrementing primary key |
| `config_hash` | TEXT | SHA-256 of the pipeline config JSON |
| `config_json` | TEXT | Full pipeline configuration |
| `started_at` | TEXT | ISO-8601 timestamp |
| `run_dir` | TEXT | Working directory for the run |
| `total_indices` | INTEGER | Total number of source indices |

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
| `completed_count` | INTEGER | Number of indices completed by this worker |
| `invocation_id` | TEXT | Groups workers from the same `run_pipeline` call |

**`output_files`** — normalized reverse-lookup table for output paths:

| Column | Type | Description |
|---|---|---|
| `path` | TEXT | Output file path |
| `idx` | INTEGER | Source index that produced this file |
| `run_id` | INTEGER | Foreign key |
| `seq` | INTEGER | Ordering within the index's output list |

**`filter_artifacts`** — files produced by filters (e.g. intermediate outputs):

| Column | Type | Description |
|---|---|---|
| `path` | TEXT | Artifact file path |
| `idx` | INTEGER | Source index |
| `run_id` | INTEGER | Foreign key |
| `filter_name` | TEXT | Name of the filter that produced the artifact |
| `filter_order` | INTEGER | Position of the filter in the pipeline |

**`logs`** — structured log entries captured during execution:

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Auto-incrementing primary key |
| `run_id` | INTEGER | Foreign key to `pipeline_runs` |
| `timestamp` | TEXT | ISO-8601 timestamp |
| `level` | INTEGER | Numeric log level |
| `level_name` | TEXT | Log level name (DEBUG, INFO, etc.) |
| `logger_name` | TEXT | Logger name |
| `message` | TEXT | Log message |
| `worker_id` | TEXT | Worker that emitted the log (nullable) |
| `idx` | INTEGER | Index being processed (nullable) |

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
from physicsnemo_curator.domains.atm import ASELMDBSource, AtomicDataZarrSink

# Build pipeline with resume=True so it can be interrupted and restarted
pipeline = Pipeline(
    source=ASELMDBSource(data_dir="/data/val/"),
    sink=AtomicDataZarrSink(output_path="/output/dataset.zarr"),
    resume=True,
    db_dir=Path("/output/checkpoints"),
)

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
