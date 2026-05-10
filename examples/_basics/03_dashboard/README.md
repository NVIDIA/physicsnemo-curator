# Launching the Metrics Dashboard

This example shows how to run a pipeline, collect execution metrics, and launch the interactive
web dashboard for visualization and analysis. Every pipeline automatically stores timing and
memory metrics in a SQLite database — the dashboard reads this database and provides charts for
progress, per-index timing, stage breakdowns, and artifact inspection.

## Prerequisites

```bash
uv sync --group mesh --extra dashboard
uv run maturin develop
```

## Usage

```bash
uv run python main.py
```

## Step-by-Step Walkthrough

### 1. Run a Pipeline with Metrics

All pipelines collect metrics by default. Set `db_dir` to choose where the SQLite database is
stored (defaults to `~/.cache/psnc/`). Setting `resume=True` reuses the same database across
runs so metrics accumulate.

```python
from pathlib import Path
from physicsnemo_curator import Pipeline
from physicsnemo_curator.domains.mesh.filters.mean import MeanFilter
from physicsnemo_curator.domains.mesh.filters.precision import PrecisionFilter
from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink
from physicsnemo_curator.domains.mesh.sources.random import RandomMeshSource
from physicsnemo_curator.run import run_pipeline

pipeline = Pipeline(
    source=RandomMeshSource(n_samples=10, n_points=100, n_cells=50),
    filters=[
        MeanFilter(output="output/dashboard/stats.parquet"),
        PrecisionFilter(target_dtype="float32"),
    ],
    sink=MeshSink(output_dir="output/dashboard/meshes/"),
    resume=True,
    db_dir=Path("output/dashboard/"),
)

results = run_pipeline(pipeline, n_jobs=1, backend="sequential", indices=range(10))
```

### 2. Inspect Metrics Programmatically

Before opening the dashboard you can access metrics directly from the pipeline object:

```python
metrics = pipeline.metrics
metrics.to_console()              # Pretty-printed summary
metrics.to_json("metrics.json")   # Full detail as JSON
metrics.to_csv("metrics.csv")     # Tabular export
```

### 3. Launch the Dashboard

The dashboard is a web application built with Panel. Pass the database path to `launch()` and it
opens in your default browser:

```python
from physicsnemo_curator.dashboard import launch

launch(str(pipeline.db_path), port=5006, open_browser=True)
```

The dashboard has three tabs:

- **Overview** — Progress cards, worker status, pipeline structure, recent outputs, error log.
- **Pipeline** — Browse indices, inspect artifacts, view custom filter visualizations.
- **Performance** — Timeline scatter, per-stage breakdown, memory histograms, slowest indices.

### 4. Alternative Launch Methods

You can also launch the dashboard from the command-line TUI:

```bash
psnc  # then navigate to "Open dashboard"
```

Or resolve a database from a pipeline config file or hash prefix:

```python
from physicsnemo_curator.dashboard._cli import launch_dashboard

launch_dashboard("output/dashboard/")   # Resolves .db in the directory
launch_dashboard("a1b2c3")              # Hash prefix lookup in ~/.cache/psnc/
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Metrics database** | SQLite file storing timing, memory, status for every index and stage |
| **db_dir** | Directory where the database file is stored |
| **db_path** | Full path to the generated `.db` file (available after first run) |
| **metrics.to_console()** | Quick programmatic summary without launching the dashboard |
| **launch()** | Starts the Panel web server and opens the dashboard in a browser |
| **Dashboard tabs** | Overview (progress), Pipeline (artifacts), Performance (charts) |
