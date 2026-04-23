# Quickstart

This guide walks through building ETL pipelines with PhysicsNeMo Curator
for two common domains: **CAE mesh processing** and **weather/climate
reanalysis**.

Both examples follow the same pattern: **Source → Filter → Sink →
`run_pipeline`**.

## CAE: DrivAerML Surface Meshes

Process automotive CFD boundary meshes from the
[DrivAerML](https://huggingface.co/datasets/neashton/drivaerml) dataset on
HuggingFace Hub.  DrivAerML contains 500 parametrically morphed variants of
the DrivAer notchback vehicle with high-fidelity scale-resolving CFD.

```bash
pip install physicsnemo-curator[mesh]
```

```python
from physicsnemo_curator.domains.mesh.filters.mean import MeanFilter
from physicsnemo_curator.domains.mesh.filters.precision import PrecisionFilter
from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink
from physicsnemo_curator.domains.mesh.sources.drivaerml import DrivAerMLSource
from physicsnemo_curator.run import gather_pipeline, run_pipeline

# 1. Source — reads boundary VTP files from HuggingFace Hub
source = DrivAerMLSource(mesh_type="boundary")
print(f"Runs available: {len(source)}")

# 2. Build the pipeline: Source → MeanFilter → PrecisionFilter → Sink
pipeline = (
    source
    .filter(MeanFilter(output="outputs/drivaerml/mean_stats.parquet"))
    .filter(PrecisionFilter(target_dtype="float32"))
    .write(MeshSink(output_dir="outputs/drivaerml/meshes/"))
)

# 3. Run in parallel (first 3 runs)
results = run_pipeline(
    pipeline,
    n_jobs=4,
    backend="process_pool",
    indices=range(3),
)

# 4. Merge per-worker statistics
merged = gather_pipeline(pipeline)

print(f"Processed {len(results)} runs")
for path in merged:
    print(f"Merged statistics: {path}")
```

This produces:

```text
outputs/drivaerml/
├── mean_stats.parquet      # Per-field spatial means
└── meshes/
    ├── mesh_0000_0/        # Run 0 (tensordict format)
    ├── mesh_0001_0/        # Run 1
    └── mesh_0002_0/        # Run 2
```

## Weather/Climate: ERA5 Reanalysis

Download ERA5 reanalysis fields and compute temporal statistics over one
month.  ERA5 data is accessed via
[earth2studio](https://github.com/NVIDIA/earth2studio) backends — no API
keys required for the default ARCO backend.

```bash
pip install physicsnemo-curator[da]
```

```python
from datetime import datetime, timedelta

from physicsnemo_curator.domains.da.filters.moments import MomentsFilter
from physicsnemo_curator.domains.da.sinks.zarr_writer import ZarrSink
from physicsnemo_curator.domains.da.sources.era5 import ERA5Source
from physicsnemo_curator.run import gather_pipeline, run_pipeline

# 1. Source — one month of 6-hourly ERA5 snapshots
start = datetime(2020, 1, 1)
times = [start + timedelta(hours=6 * i) for i in range(4 * 31)]  # ~124 steps
variables = ["t2m", "u10m", "v10m"]

source = ERA5Source(times=times, variables=variables, backend="arco")
print(f"Timesteps: {len(source)}")

# 2. Build the pipeline: Source → MomentsFilter → ZarrSink
pipeline = (
    source
    .filter(MomentsFilter(output="outputs/era5/moments.zarr", dims=("time",)))
    .write(ZarrSink(output_path="outputs/era5/data.zarr"))
)

# 3. Run in parallel
results = run_pipeline(
    pipeline,
    n_jobs=4,
    backend="process_pool",
    indices=range(len(source)),
)

# 4. Merge per-worker moment statistics
merged = gather_pipeline(pipeline)

print(f"Processed {len(results)} timesteps")
for path in merged:
    print(f"Merged moments: {path}")
```

This produces:

```text
outputs/era5/
├── moments.zarr/           # Temporal statistics (mean, variance, skewness, min, max)
│   ├── t2m/
│   ├── u10m/
│   └── v10m/
└── data.zarr/              # Raw fields in Zarr format
    ├── t2m/
    ├── u10m/
    └── v10m/
```

## Next Steps

- See the full [Examples gallery](https://nvidia.github.io/physicsnemo-curator/auto_examples/index.html)
  for crash simulation, external aerodynamics, and more.
- Read the {doc}`parallel` guide for details on execution backends and
  stateful filter handling.
- Use the interactive wizard (`psnc`) to build pipelines without writing
  code — see {doc}`cli`.
