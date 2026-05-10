# Checkpointing a Pipeline

This example demonstrates how to checkpoint pipeline execution using the built-in `resume=True`
flag. When enabled, the Pipeline uses a stable SQLite database keyed by its configuration hash.
Completed indices are skipped and their cached output paths are returned immediately on subsequent
runs. This is especially useful for long-running pipelines over large datasets where you want
crash resilience without re-processing.

## Prerequisites

```bash
uv sync --group mesh
uv run maturin develop
```

## Usage

```bash
uv run python main.py
```

## Step-by-Step Walkthrough

### 1. Build a Resumable Pipeline

Two fields control checkpointing:

- `resume=True` — reuse a stable database (keyed by config hash) rather than creating a fresh
  timestamped one.
- `db_dir` — optional directory for the SQLite file (defaults to `~/.cache/psnc/` or
  `$PSNC_CACHE_DIR`).

When `resume=False` (default), each pipeline run gets a unique database and there is no automatic
skipping of completed indices.

```python
from physicsnemo_curator import Pipeline
from physicsnemo_curator.domains.mesh.filters.precision import PrecisionFilter
from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink
from physicsnemo_curator.domains.mesh.sources.ns_cylinder import NavierStokesCylinderSource

resumable = Pipeline(
    source=NavierStokesCylinderSource(),
    filters=[PrecisionFilter(target_dtype="float32")],
    sink=MeshSink(output_dir="output/checkpoint/meshes/"),
    resume=True,
    db_dir="output/checkpoint/",
)
```

### 2. First Run — Process Indices

On the first run, all indices are new and will be fully executed. Results are recorded in the
SQLite database.

```python
from physicsnemo_curator.run import run_pipeline

results = run_pipeline(
    resumable,
    n_jobs=1,
    backend="sequential",
    indices=range(5),
    use_tui=True,
)
```

### 3. Resume from Checkpoint

If you run the same pipeline again with overlapping indices, completed indices are skipped. Their
cached output paths are returned from the database without re-executing the pipeline.

```python
results_resumed = run_pipeline(
    resumable,
    n_jobs=1,
    backend="sequential",
    indices=range(8),  # 0-4 cached, 5-7 new
    use_tui=True,
)
```

### 4. Query Checkpoint State

The pipeline exposes its checkpoint state through properties and methods:

```python
resumable.completed_indices   # Set of indices that finished successfully
resumable.failed_indices      # Set of indices that raised an error
resumable.remaining_indices() # Indices not yet completed
resumable.summary()           # Human-readable status string
```

### 5. Individual Index Lookup

Query which output paths were produced for a given index, or find which index produced a given
output path:

```python
paths_for_0 = resumable.output_paths_for_index(0)
idx = resumable.index_for_path(paths_for_0[0])  # Reverse lookup
```

### 6. Reset Checkpoint

To re-process all indices from scratch, call `reset()`:

```python
resumable.reset()           # Reset everything
resumable.reset_index(3)    # Only re-processes index 3 next time
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Config hash** | Deterministic hash of pipeline configuration; identifies the SQLite DB |
| **resume=True** | Enables stable DB reuse across runs |
| **db_dir** | Custom location for the checkpoint database |
| **Completed indices** | Skipped on subsequent runs; cached output paths returned instantly |
| **Failed indices** | Recorded but not skipped; will be retried on next run |
| **reset()** | Clears all checkpoint state so indices are re-processed |

## Combining with Metrics

Both features are always active together. When `resume=True`, checkpointed pipelines still
collect full timing/memory metrics for newly processed indices. Cached (skipped) indices bypass
instrumentation entirely. Access metrics via:

```python
metrics = resumable.metrics
metrics.to_console()
```
