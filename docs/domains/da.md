# DataArray Processing (``da``)

The ``da`` submodule provides pipeline components for working with
{class}`xarray.DataArray` objects — the standard data structure for
labelled multi-dimensional arrays in the scientific Python ecosystem.

## Installation

```bash
# Install with the da dependency group
uv sync --group da
```

Required packages: ``xarray``, ``earth2studio``, ``zarr>=3.0``, ``gcsfs``.

## Components

### ERA5Source

Fetches ERA5 reanalysis data from Google's Analysis-Ready, Cloud-Optimized
(ARCO) Zarr store via {class}`earth2studio.data.ARCO`.  Each pipeline index
maps to a single timestamp.

```python
from datetime import datetime
from physicsnemo_curator.da.sources.era5 import ERA5Source

source = ERA5Source(
    times=[datetime(2020, 6, 1, 0), datetime(2020, 6, 1, 6)],
    variables=["t2m", "u10m", "v10m"],
    cache=True,  # cache downloaded chunks locally
)
print(f"{len(source)} timestamps")  # 2
```

Each ``source[i]`` yields a single {class}`xarray.DataArray` with dimensions
``(time, variable, lat, lon)`` — ``time`` is length 1 (the requested
timestamp), ``variable`` spans the requested fields, and the spatial grid is
ERA5's native 0.25° resolution (721 lat × 1440 lon).

**Variable naming** follows the earth2studio lexicon:

| Category | Examples |
|----------|----------|
| Surface  | ``t2m``, ``u10m``, ``v10m``, ``sp``, ``msl``, ``sst``, ``tp`` |
| Pressure-level | ``t500``, ``z500``, ``u850``, ``q925`` (37 levels) |

**Time range**: 1940-01-01 through ~2023-11-11, hourly resolution.

### MomentsFilter

Computes running statistical moments (mean, variance, skewness, min, max)
along specified dimensions using Welford's online algorithm.  The DataArray
is yielded unchanged (pass-through).

```python
from physicsnemo_curator.da.filters.moments import MomentsFilter

filt = MomentsFilter(
    output="stats.zarr",
    dims=("time",),  # reduce over time → per-spatial-point statistics
)
```

Call {meth}`~physicsnemo_curator.da.filters.moments.MomentsFilter.flush` after processing
to write accumulated statistics to the output Zarr store.  Each variable gets
its own group with arrays: ``mean``, ``variance``, ``skewness``, ``min``,
``max``, plus a ``count`` attribute.

### ZarrSink

Writes incoming DataArrays to a Zarr v3 store.  Each variable is written to
its own Zarr group (e.g. ``output.zarr/t2m/``) with dimensions
``(time, lat, lon)``.  Subsequent calls append along the ``time`` dimension.

```python
from physicsnemo_curator.da.sinks.zarr_writer import ZarrSink

sink = ZarrSink(
    output_path="output.zarr",
    chunks={"time": 1, "lat": 721, "lon": 1440},
    shards={"time": 24, "lat": 721, "lon": 1440},  # optional, Zarr v3
)
```

**Chunking** controls how data is split into individual Zarr chunks.
**Sharding** (Zarr v3) groups multiple chunks into larger shard files,
reducing the number of objects in cloud storage.

### NetCDF4Sink

Writes incoming DataArrays to NetCDF4 files.  Each variable gets its own
subdirectory, and files are **split** along a configurable coordinate
dimension (default: ``time``, grouped by year).  The output layout is:

``<output_dir>/<variable>/<split_key>.nc``

For example, with the default settings, data spanning 2020–2021 produces:

```text
output_nc/
    t2m/
        2020.nc       # all 2020 timestamps
        2021.nc       # all 2021 timestamps
```

Subsequent calls with the same split key **append** along the time
dimension using the unlimited dimension.

```python
from physicsnemo_curator.da.sinks.netcdf_writer import NetCDF4Sink

# Default: split by year
sink = NetCDF4Sink(
    output_dir="output_nc",
    chunks={"time": 1, "lat": 721, "lon": 1440},
    compression_level=4,       # zlib 0-9, default 4
)

# Split by month instead
sink = NetCDF4Sink(
    output_dir="output_nc",
    split_func=lambda t: str(np.datetime64(t, "M")),
)

# No splitting — one file per variable
sink = NetCDF4Sink(output_dir="output_nc", split_dim=None)
```

**Chunking** controls the HDF5 chunk layout inside the NetCDF4 file.
**Compression** uses zlib (level 0 disables it, 9 is maximum compression).
The ``time`` dimension is unlimited by default, allowing efficient appends.

## Full Pipeline Example

```python
from datetime import datetime
from physicsnemo_curator import run_pipeline
from physicsnemo_curator.da.sources.era5 import ERA5Source
from physicsnemo_curator.da.filters.moments import MomentsFilter
from physicsnemo_curator.da.sinks.zarr_writer import ZarrSink

# Fetch 24 hours of surface weather data
times = [datetime(2020, 6, 1, h) for h in range(24)]

source = ERA5Source(
    times=times,
    variables=["t2m", "u10m", "v10m", "sp"],
)

filt = MomentsFilter(output="era5_stats.zarr", dims=("time",))
sink = ZarrSink(
    output_path="era5_output.zarr",
    chunks={"time": 1, "lat": 721, "lon": 1440},
)

pipeline = source.filter(filt).write(sink)

# Process all 24 timestamps
results = run_pipeline(pipeline)

# Write accumulated statistics
filt.flush()
```

After running, the output directory contains:

```text
era5_output.zarr/
    t2m/          # Zarr group: (time=24, lat=721, lon=1440)
    u10m/         # Zarr group: (time=24, lat=721, lon=1440)
    v10m/         # ...
    sp/           # ...

era5_stats.zarr/
    t2m/          # mean, variance, skewness, min, max arrays
    u10m/         # with shape (lat=721, lon=1440)
    v10m/
    sp/
```

### Using NetCDF4 output

Replace the sink to write NetCDF4 files instead of Zarr:

```python
from physicsnemo_curator.da.sinks.netcdf_writer import NetCDF4Sink

sink = NetCDF4Sink(
    output_dir="era5_output_nc",
    chunks={"time": 1, "lat": 721, "lon": 1440},
    compression_level=4,
)

pipeline = source.filter(filt).write(sink)
results = run_pipeline(pipeline)
filt.flush()
```

This produces one subdirectory per variable, with files split by year:

```text
era5_output_nc/
    t2m/
        2020.nc       # NetCDF4: (time=24, lat=721, lon=1440)
    u10m/
        2020.nc
    v10m/
        2020.nc
    sp/
        2020.nc
```

## Caching

ERA5Source uses earth2studio's built-in caching (``cache=True`` by default).
Downloaded Zarr chunks are stored in ``~/.cache/earth2studio/arco/``.  Set
the ``EARTH2STUDIO_CACHE`` environment variable to override the cache
location:

```bash
export EARTH2STUDIO_CACHE=/fast-scratch/era5-cache
```

## Process Isolation

When using ``run_pipeline`` with ``n_jobs > 1``, each worker process gets
its own copy of the pipeline.  Stateful filters like ``MomentsFilter``
accumulate statistics **independently** in each process — their results are
**not merged** automatically.  For accurate statistics across all indices,
use ``n_jobs=1`` (sequential) or post-process the per-worker outputs.

See the {doc}`parallel execution guide </user-guide/parallel>` for details.
