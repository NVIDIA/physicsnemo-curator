# DataArray Submodule

The `curator.da` submodule provides pipeline components for working with
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
from physicsnemo_curator.domains.da.sources.era5 import ERA5Source

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

### DataArrayStatsFilter

Computes running statistical moments (mean, variance, skewness, min, max)
along specified dimensions using Welford's online algorithm.  The DataArray
is yielded unchanged (pass-through).

```python
from physicsnemo_curator.domains.da.filters.stats import DataArrayStatsFilter

filt = DataArrayStatsFilter(
    output="stats.zarr",
    dims=("time",),  # reduce over time → per-spatial-point statistics
)
```

Statistics are automatically flushed to disk after each pipeline index.
Each variable gets its own group with arrays: ``mean``, ``variance``,
``skewness``, ``min``, ``max``, plus a ``count`` attribute.  When
multiple workers write to the same path, results are merged using
Chan's parallel Welford algorithm.

### ZarrSink

Writes incoming DataArrays to a Zarr v3 store.  Each variable is written to
its own Zarr group (e.g. ``output.zarr/t2m/``) with dimensions
``(time, lat, lon)``.  Subsequent calls append along the ``time`` dimension.

```python
from physicsnemo_curator.domains.da.sinks.zarr_writer import ZarrSink

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
from physicsnemo_curator.domains.da.sinks.netcdf_writer import NetCDF4Sink

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

## Dependencies

The `da` domain depends on:

| Package | Purpose |
|---------|---------|
| [xarray](https://docs.xarray.dev/) | Labelled multi-dimensional arrays |
| [earth2studio](https://nvidia.github.io/earth2studio/) | Weather/climate data backends (ERA5, HRRR, GFS) |
| [zarr](https://zarr.readthedocs.io/) | Zarr v3 store I/O |
| [gcsfs](https://gcsfs.readthedocs.io/) | Google Cloud Storage filesystem for ARCO data |
