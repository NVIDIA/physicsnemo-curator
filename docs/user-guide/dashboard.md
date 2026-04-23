<!---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
-->

# Metrics Dashboard

PhysicsNeMo Curator includes an interactive web dashboard for inspecting
pipeline run metrics.  It reads the SQLite database produced by the
pipeline's profiling system and presents timing, memory, stage-level
breakdowns, and filter artifact previews in a browser-based interface.

## Installation

```bash
pip install physicsnemo-curator[dashboard]
```

This installs the required dependencies:

- `panel` — reactive web application framework
- `holoviews` — declarative data visualization
- `bokeh` — interactive plotting backend
- `pandas` — DataFrame manipulation
- `pyarrow` — Parquet file reading

## Launch

### From the command line

```bash
psnc dashboard pipeline.db
```

You can also pass a serialized pipeline file (`.yaml` or `.json`) — the
dashboard computes the config hash and locates the matching database
automatically:

```bash
psnc dashboard my_pipeline.yaml
```

Or pass a hash prefix instead of a full path.  The dashboard will look
up the matching database in the cache directory:

```bash
psnc dashboard a1b2
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 5006 | Server port |
| `--no-browser` | off | Don't open a browser window on launch |

### From Python

```python
from physicsnemo_curator.dashboard import launch

launch("pipeline.db", port=5006)
```

Or for more control:

```python
from physicsnemo_curator.dashboard import DashboardApp

app = DashboardApp("pipeline.db")
tabs = app.servable()  # for embedding in a notebook
```

## Prerequisites

The dashboard reads the SQLite database that the pipeline creates when
`track_metrics=True` (the default).  Make sure your pipeline was
configured to collect metrics:

```python
from physicsnemo_curator.core.base import Pipeline

pipeline = Pipeline(
    source=source,
    filters=[...],
    sink=sink,
    track_metrics=True,   # default
    track_memory=True,    # default
    db_dir="./runs/",     # directory for the .db file
)
```

After `run_pipeline()` completes, the database is at
`<db_dir>/<config_hash>.db`.

## Tabs

### Overview

Summary of the pipeline run:

- **Progress cards** — completed, failed, remaining counts with
  elapsed time
- **Workers** — table of registered workers with heartbeat status
  (useful when monitoring a running pipeline)
- **Pipeline structure** — source → filters → sink chain
- **Recent output files** — last 20 files produced by the sink
- **Error log** — indices that failed with error messages

### Pipeline

Inspect the pipeline structure and drill into individual indices:

- **Structure flow** — visual cards for each pipeline component
  with parameters
- **Index query** — filter by index range (e.g. `10-20`, `1,5,10`)
  or status (completed, error)
- **Artifact inspection** — click an index to see its output files
  and filter artifacts.  If the filter class overrides
  {meth}`~physicsnemo_curator.core.base.Filter.dashboard_panel`,
  a rich visualization is shown inline (e.g. a scatter plot for
  AtomicStatsFilter or a bar chart for MeanFilter Parquet files)
- **Aggregate view** — when no index is selected, browse all
  artifacts grouped by filter name with Parquet previews

### Performance

Timing and resource analysis:

- **Timeline scatter** — wall time per index, colored by status.
  Click a point to select it in the Pipeline tab.  Toggle a memory
  overlay
- **Stage breakdown** — stacked bar chart of per-stage time for each
  index.  Filter by stage name.  Summary statistics table (mean,
  median, p95, max)
- **Resource summary** — memory distribution histogram, GPU memory
  histogram (if tracked), and a table of the 10 slowest indices

## Filter Dashboard Widgets

Filters that produce artifacts (Parquet, Zarr, etc.) can provide custom
visualizations in the Pipeline tab.  Built-in widgets are provided for
{class}`~physicsnemo_curator.domains.mesh.filters.mean.MeanFilter` and
{class}`~physicsnemo_curator.domains.atm.filters.stats.AtomicStatsFilter`.

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
`_auto_discover()` function following the existing pattern.

## Live Monitoring

The dashboard can be launched while a pipeline is still running.  The
Overview tab shows worker heartbeats and progress updates.  Use the
refresh mechanism to poll the database for new results:

```python
from physicsnemo_curator.dashboard import DashboardApp

app = DashboardApp("pipeline.db")
# The store auto-refreshes on parameter events
app.serve()
```

The PipelineStore uses WAL-mode SQLite, so concurrent reads from the
dashboard and writes from the pipeline are safe.

## Programmatic Access

If you need the data without the web UI, use
{class}`~physicsnemo_curator.core.pipeline_store.PipelineStore`
directly:

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

See {doc}`profiling` for details on the metrics system.
