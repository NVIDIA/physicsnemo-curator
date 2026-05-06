<!---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
-->

# Metrics & Dashboard

PhysicsNeMo Curator includes built-in pipeline profiling and an interactive
web dashboard for inspecting run metrics.  Profiling collects wall-clock
time, memory, and (optionally) GPU metrics at whole-pipeline, per-index,
and per-stage granularity — without requiring any wrappers or separate
configuration.

The dashboard reads the SQLite database produced by the profiling system
and presents timing, memory, stage-level breakdowns, and filter artifact
previews in a browser-based interface using
[Panel](https://panel.holoviz.org/) with
[Material UI](https://panel-material-ui.holoviz.org/) theming and
[HoloViews](https://holoviews.org/) / [Bokeh](https://bokeh.org/) for
interactive plotting.

## Metrics Collection

Profiling is **enabled by default** via the `track_metrics` pipeline field
(which also enables checkpointing).

### Quick Start

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

# Inspect metrics programmatically
metrics = pipeline.metrics
metrics.to_console()
```

### Metrics Granularity

Profiling collects data at three levels:

| Level | What's measured |
|-------|----------------|
| **Whole-pipeline** | Total wall time, peak memory across all indices |
| **Per-index** | Wall time, peak memory, GPU memory for each source index |
| **Per-stage** | Wall time for source, each filter, and sink |

#### Per-Stage Timing

The pipeline chain `source → filter₁ → filter₂ → … → sink` uses lazy
generators.  Profiling wraps each stage's generator with an internal
timer to attribute time accurately using chain subtraction:

- **Source time** = time spent yielding items from the source
- **Filter N time** = time spent in filter N's own logic (excluding upstream)
- **Sink time** = time spent in the sink (excluding all upstream generators)

#### Memory Tracking

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

#### GPU Memory Tracking

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

### Pipeline Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `track_metrics` | `bool` | `True` | Enable timing, checkpointing, and metrics |
| `track_memory` | `bool` | `True` | Enable `tracemalloc` memory tracking |
| `track_gpu` | `bool` | `False` | Enable GPU memory tracking via PyTorch |
| `db_dir` | `Path \| None` | `None` | Override database directory (default: `~/.cache/psnc/`) |

### Disabling Metrics

Set `track_metrics=False` to disable all profiling and checkpointing:

```python
pipeline = Pipeline(
    source=MySource(),
    sink=MySink(),
    track_metrics=False,
)
```

### Using with Parallel Backends

Profiling works with **all** backends — sequential, process_pool,
loky, and dask — without any backend modifications.

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

## Dashboard Installation

```bash
pip install "physicsnemo-curator[dashboard]"
```

This installs the required dependencies:

| Package | Version | Purpose |
|---------|---------|---------|
| `panel` | >=1.3 | Reactive web application framework |
| `panel-material-ui` | >=0.9 | Material Design theming |
| `holoviews` | >=1.18 | Declarative data visualization |
| `bokeh` | >=3.3 | Interactive plotting backend |
| `pandas` | >=2.0 | DataFrame manipulation |
| `pyarrow` | >=14.0 | Parquet file reading |

## Launching the Dashboard

### From the TUI (recommended)

The easiest way to launch the dashboard is through the interactive wizard:

```bash
psnc
```

This opens the full-screen Textual TUI.  On the welcome screen, click
**"Open dashboard"** to reach the dashboard launcher.  The launcher
provides:

- A **dropdown** of all cached pipeline databases (auto-discovered from
  `~/.cache/psnc/`)
- A **port** input (default 5006)
- **Launch / Stop** controls

The dashboard runs as a subprocess so the TUI remains responsive.  Press
**Stop** or navigate back to terminate it.

```text
┌──────────────────────────────────────────────┐
│  Open Dashboard                              │
│  Select a pipeline database to visualize     │
│                                              │
│  [2025-05-01 14:30] VTKSource → MeshSink ▼  │
│                                              │
│  Port: [5006]                                │
│                                              │
│  Dashboard running at http://localhost:5006   │
│                                              │
│  [ Stop Dashboard ]                          │
│  [ ← Back ]                                  │
└──────────────────────────────────────────────┘
```

### From Python

```python
from physicsnemo_curator.dashboard import launch

launch("pipeline.db", port=5006)
```

For more control, instantiate the app directly:

```python
from physicsnemo_curator.dashboard import DashboardApp

app = DashboardApp("pipeline.db")

# Serve as a standalone web application
app.serve(port=5006, open_browser=True)
```

Or embed in a Jupyter notebook:

```python
from physicsnemo_curator.dashboard import DashboardApp

app = DashboardApp("pipeline.db")
page = app.servable()  # returns a Panel Page object
page  # display inline in notebook
```

### Resolving database paths programmatically

The `resolve_db_path` utility supports flexible path resolution:

```python
from physicsnemo_curator.dashboard._cli import resolve_db_path, launch_dashboard

# 1. Direct .db file path
path = resolve_db_path("./runs/abc123.db")

# 2. Pipeline config file — computes config hash and locates matching DB
path = resolve_db_path("my_pipeline.yaml")

# 3. Hash prefix — glob-matched against ~/.cache/psnc/*.db
path = resolve_db_path("a1b2")

# Full launch with resolution
launch_dashboard("a1b2", port=5007)
```

Resolution order:

1. **Existing `.db` file** — returned as-is
2. **Pipeline file** (`.yaml`, `.yml`, `.json`) — config hash is computed
   and the matching database is located in the cache directory
3. **Hash prefix** — glob-matched against `*.db` in `~/.cache/psnc/`

## Dashboard Tabs

The dashboard provides three tabs in a Material UI layout with NVIDIA green
theming, a dark/light theme toggle, and a **Refresh** button in the toolbar
to re-query the database.

### Overview

Summary of the pipeline run:

- **Progress cards** — completed, failed, remaining counts with
  elapsed time (formatted as seconds, minutes, or hours)
- **Workers** — table of registered workers with heartbeat status
  (useful when monitoring a running pipeline)
- **Pipeline structure** — source → filters → sink chain with
  stage timing summary table (mean and total per stage)
- **Recent output files** — last 20 files produced by the sink
- **Error log** — indices that failed with error messages (up to 10 shown)

### Pipeline

Inspect the pipeline structure and drill into individual indices:

- **Structure flow** — visual cards for each pipeline component
  (source, filters, sink) with color-coded backgrounds
- **Index query** — filter by index range (e.g. `10-20`, `1,5,10`)
  or status (completed, error).  Supports `all` to show everything
- **Pagination** — configurable page size (20, 50, 100) with
  prev/next navigation for large datasets
- **Artifact inspection** — select an index to view its output files
  and filter artifacts.  If the filter class overrides
  {meth}`~physicsnemo_curator.core.base.Filter.dashboard_panel`,
  a rich visualization is shown inline (e.g. a scatter plot for
  AtomicStatsFilter or a bar chart for MeanFilter)
- **Aggregate view** — when no index is selected, browse all
  artifacts grouped by filter name with Parquet file previews
  (first 20 rows displayed inline)

### Performance

Timing and resource analysis:

- **Timeline scatter** — wall time per index, colored by status
  (green = completed, red = error).  Click a point to select it in
  the Pipeline tab.  Toggle a memory overlay to show peak memory
  alongside timing
- **Stage breakdown** — stacked bar chart of per-stage time for each
  index.  Filter by stage name via dropdown.  Summary statistics table
  (mean, median, p95, max)
- **Resource summary** — CPU memory distribution histogram, GPU memory
  histogram (if tracked), and a table of the 10 slowest indices

## Filter Dashboard Widgets

Filters that produce artifacts (Parquet, Zarr, etc.) can provide custom
visualizations in the Pipeline tab.  Built-in widgets are registered
automatically for:

- {class}`~physicsnemo_curator.domains.mesh.filters.stats.MeshStatsFilter`
- {class}`~physicsnemo_curator.domains.mesh.filters.mean.MeanFilter`
- {class}`~physicsnemo_curator.domains.atm.filters.stats.AtomicStatsFilter`
- {class}`~physicsnemo_curator.domains.da.filters.stats.DataArrayStatsFilter`

### Writing a custom widget

Override two classmethods on your
{class}`~physicsnemo_curator.core.base.Filter` subclass:

```python
from __future__ import annotations

from physicsnemo_curator.core.base import Filter


class MyFilter(Filter[MyDataType]):
    """Filter with a custom dashboard visualization."""

    name = "My Filter"
    description = "Computes custom statistics"

    @classmethod
    def dashboard_panel(
        cls,
        artifact_paths: list[str],
        selected_index: int | None = None,
    ) -> Any:
        """Return a Panel viewable for the dashboard Pipeline tab."""
        import panel as pn

        # Read artifacts, build visualization
        ...
        return pn.Column(...)

    @classmethod
    def dashboard_layout_hints(cls) -> dict[str, int]:
        """Return GridStack layout hints for the tile."""
        return {"sizing_mode": "stretch_width", "height": 400}
```

### Automatic discovery

The `WidgetRegistry` automatically discovers filters that override
`dashboard_panel()` at import time.  No manual registration is needed —
simply ensure your filter is importable and the dashboard will pick it up.

To add auto-discovery for a new filter, add its import to
`physicsnemo_curator/dashboard/widgets/__init__.py` in the
`_auto_discover()` function following the existing pattern:

```python
try:
    from physicsnemo_curator.domains.my_domain.filters.my_filter import MyFilter

    self.register(MyFilter)
except Exception:
    logger.debug("MyFilter not available", exc_info=True)
```

## Live Monitoring

The dashboard can be launched while a pipeline is still running.  The
Overview tab shows worker heartbeats and progress updates.  Click the
**Refresh** button in the toolbar to poll the database for new results.

```python
from physicsnemo_curator.dashboard import DashboardApp

app = DashboardApp("pipeline.db")
app.serve()
```

The `PipelineStore` uses WAL-mode SQLite, so concurrent reads from the
dashboard and writes from the pipeline are safe.

## Programmatic Metrics Access

If you need the underlying metrics data without the web UI, access the
`pipeline.metrics` property or use
{class}`~physicsnemo_curator.core.pipeline_store.PipelineStore` directly.

### Via Pipeline object

```python
from physicsnemo_curator import Pipeline, run_pipeline

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

### Via PipelineStore

```python
from physicsnemo_curator.core.pipeline_store import PipelineStore

store = PipelineStore.from_db("pipeline.db")

# Summary
print(store.summary(total=100))

# Per-index metrics
metrics = store.metrics()
for im in metrics.indices:
    print(f"Index {im.index}: {im.wall_time_ns / 1e9:.2f}s")

# Artifacts
artifacts = store.all_filter_artifacts()
for filter_name, paths in artifacts.items():
    print(f"{filter_name}: {len(paths)} files")
```

### Output Formats

| Method | Description |
|--------|-------------|
| `metrics.to_console()` | Human-readable summary table to stdout |
| `metrics.to_json(path)` | Full metrics with per-stage breakdowns as JSON |
| `metrics.to_csv(path)` | One row per index, stage timings as columns |
| `metrics.summary()` | Dictionary for programmatic use |

### API Reference

#### `PipelineMetrics`

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

#### `IndexMetrics`

```python
class IndexMetrics:
    index: int
    stages: list[StageMetrics]
    wall_time_ns: int
    peak_memory_bytes: int
    gpu_memory_bytes: int | None
```

#### `StageMetrics`

```python
class StageMetrics:
    name: str
    wall_time_ns: int
```
