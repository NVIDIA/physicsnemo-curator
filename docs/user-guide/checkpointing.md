<!---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
-->

# Checkpointing Pipelines

`CheckpointedPipeline` is a transparent wrapper around `Pipeline` that
records completed indices in a SQLite database.  On restart, indices that
already finished are skipped — their cached output paths are returned
immediately without re-executing the source, filters, or sink.

## Quick Start

```python
from physicsnemo.curator import Pipeline, CheckpointedPipeline, run_pipeline

# Wrap any existing pipeline
cp = CheckpointedPipeline(pipeline, db_path="run.checkpoint.db")

# Run exactly as before — works with all backends
results = run_pipeline(cp, n_jobs=4, backend="process_pool")

# Interrupt and restart — completed indices are skipped
results = run_pipeline(cp, n_jobs=4, backend="process_pool")

# Inspect progress
print(cp.summary())
```

## How It Works

`CheckpointedPipeline` is duck-type compatible with `Pipeline` — it
exposes `source`, `filters`, `sink`, `__len__`, and `__getitem__`.  This
means it works with **all** backends (sequential, thread\_pool,
process\_pool, loky, dask, prefect) without any modifications.

On each `__getitem__` call:

1. **Check the database** for a prior completion record for this index.
2. If found, **return the cached paths** immediately (no computation).
3. If not found, **run the inner pipeline** and record the result.
4. If the inner pipeline **raises an exception**, record the error and
   re-raise.

## Provenance Tracking

The checkpoint stores full pipeline provenance — source class, filter
parameters, sink configuration — as a JSON document with a SHA-256 hash.
When you resume from a checkpoint with a different pipeline configuration,
a warning is logged but processing continues:

```text
WARNING:physicsnemo.curator.core.checkpoint:Pipeline config has changed
since the original checkpoint (stored hash a1b2c3d4e5f6…, current hash
9f8e7d6c5b4a…). Resuming anyway — completed indices from prior config
will be kept.
```

This **warn-only** strategy avoids blocking restarts for minor config
changes (e.g. adjusting filter parameters for new indices) while still
alerting you to potential inconsistencies.

## Error Handling

Failed indices are recorded with their error message but are **not**
marked as completed.  On the next run they will be retried automatically:

```python
# First run — index 42 fails
results = run_pipeline(cp, n_jobs=4)

# Check what failed
print(cp.failed_indices)
# {42: "RuntimeError: corrupt file at /data/sample_42.lmdb"}

# Fix the underlying issue, then retry — only index 42 runs
results = run_pipeline(cp, n_jobs=4)
```

## Query API

| Property / Method | Returns | Description |
|---|---|---|
| `cp.completed_indices` | `set[int]` | Successfully completed indices |
| `cp.failed_indices` | `dict[int, str]` | Failed indices with error messages |
| `cp.remaining_indices` | `list[int]` | Indices not yet completed (sorted) |
| `cp.summary()` | `dict` | Total, completed, failed, remaining counts + elapsed time |
| `cp.db_path` | `pathlib.Path` | Path to the SQLite database file |
| `cp.config_hash` | `str` | SHA-256 hash of the current pipeline config |
| `cp.reset()` | `None` | Clear all records and start fresh |

### Summary Example

```python
>>> cp.summary()
{'total': 80, 'completed': 65, 'failed': 2, 'remaining': 13,
 'config_hash': 'a1b2c3...', 'db_path': 'run.checkpoint.db',
 'total_elapsed_s': 3847.5}
```

## Composing with ProfiledPipeline

`CheckpointedPipeline` and `ProfiledPipeline` can be composed.  Wrap the
profiled pipeline with the checkpoint to get both profiling and
resumability:

```python
from physicsnemo.curator import ProfiledPipeline, CheckpointedPipeline

profiled = ProfiledPipeline(pipeline, track_gpu=True)
cp = CheckpointedPipeline(profiled, db_path="run.checkpoint.db")

results = run_pipeline(cp, n_jobs=4)

# Profiling metrics (only for indices that actually ran)
profiled.metrics.to_console()

# Checkpoint progress
print(cp.summary())
```

## SQLite Database

The checkpoint uses a single SQLite database in WAL (Write-Ahead Logging)
mode for safe concurrent writes from multiple threads or processes.  The
database contains two tables:

**`pipeline_runs`** — one row per unique pipeline configuration:

| Column | Type | Description |
|---|---|---|
| `run_id` | INTEGER | Auto-incrementing primary key |
| `config_hash` | TEXT | SHA-256 of the pipeline config JSON |
| `config_json` | TEXT | Full pipeline configuration |
| `started_at` | TEXT | ISO-8601 timestamp |

**`completed_indices`** — one row per processed index:

| Column | Type | Description |
|---|---|---|
| `idx` | INTEGER | Source index (primary key) |
| `run_id` | INTEGER | Foreign key to `pipeline_runs` |
| `output_paths` | TEXT | JSON array of output file paths |
| `completed_at` | TEXT | ISO-8601 timestamp |
| `elapsed_ns` | INTEGER | Wall-clock time in nanoseconds |
| `error` | TEXT | Error message (NULL for success) |

## Full Example

```python
from physicsnemo.curator import CheckpointedPipeline, run_pipeline
from physicsnemo.curator.atm import ASELMDBSource, AtomicDataZarrSink

# Build pipeline
pipeline = (
    ASELMDBSource(data_dir="/data/val/")
    .write(AtomicDataZarrSink(output_path="/output/dataset.zarr"))
)

# Wrap with checkpointing
cp = CheckpointedPipeline(pipeline, db_path="/output/etl.checkpoint.db")

# Run — can be interrupted and resumed
results = run_pipeline(cp, n_jobs=8, backend="process_pool")

# Check progress
s = cp.summary()
print(f"Completed: {s['completed']}/{s['total']} "
      f"({s['total_elapsed_s']:.1f}s elapsed)")

if s['failed'] > 0:
    print(f"Failed indices: {list(cp.failed_indices.keys())}")

# Start fresh if needed
# cp.reset()
```

## API Reference

### `CheckpointedPipeline`

```python
class CheckpointedPipeline(Generic[T]):
    def __init__(
        self,
        pipeline: Pipeline[T] | ProfiledPipeline[T] | Any,
        db_path: str | pathlib.Path,
    ) -> None: ...

    # Duck-type compatibility
    source: Source[T]          # property
    filters: list[Filter[T]]  # property
    sink: Sink[T] | None      # property
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> list[str]: ...

    # Query API
    completed_indices: set[int]       # property
    failed_indices: dict[int, str]    # property
    remaining_indices: list[int]      # property
    db_path: pathlib.Path             # property
    config_hash: str                  # property
    def summary(self) -> dict: ...
    def reset(self) -> None: ...
```
