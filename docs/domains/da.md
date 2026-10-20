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

### OPERASource

Fetches [EUMETNET OPERA](https://eumetnet.github.io/openradardata-documentation/)
pan-European radar composites from the CloudFerro S3 archive via
{class}`earth2studio.data.OPERA`.  Each pipeline index maps to a single
timestamp.

```python
from datetime import datetime
from physicsnemo_curator.domains.da.sources.opera import OPERASource

source = OPERASource(
    times=[datetime(2024, 9, 1, 0, 0), datetime(2024, 9, 1, 0, 5)],
    variables=["refc"],
)
```

Each ``source[i]`` yields a {class}`xarray.DataArray` with dimensions
``(time, variable, y, x)`` on OPERA's native Lambert Equal-Area grid, with
2-D geographic coordinates ``_lat`` / ``_lon``.

| Variable | Description |
|----------|-------------|
| ``refc`` | Maximum reflectivity (dBZ) |
| ``tprate`` | Instantaneous rain rate |
| ``tp01`` | 1-hour accumulated precipitation |

**Time range**: composites are published every 15 minutes before
2024-07-01 and every 5 minutes from 2024-07-01 onward; requested
timestamps must align to the interval for their era.  Variables with
differing pixel resolutions cannot be mixed in a single source.

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

Non-finite values (NaN/±Inf) are skipped **per element**, so gridpoints
that are missing in some samples — for example pixels outside a radar
domain — do not poison the statistics of their neighbors.  The
per-element tally of finite observations is stored alongside the
accumulator state as ``welford_n``, which is what makes the parallel
merge exact when a gridpoint is observed by some workers but not others.
Gridpoints with no finite observations at all report NaN for ``mean``,
``variance``, ``skewness``, ``min``, and ``max``.  The scalar ``count``
attribute still counts every sample seen, including all-NaN ones.

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

**Tracking which indices were written.** Pre-allocated stores are sized
up front, so an unwritten slot is indistinguishable from one legitimately
containing fill values.  Pass ``track_valid=True`` to record a per-index
boolean ``valid`` array, flipped only once *every* pre-allocated variable
has been written at that index:

```python
sink = ZarrSink(
    output_path="output.zarr",
    n_indices=72,
    variables=["refc"],
    track_valid=True,
)
# ... run the pipeline ...
sink.finalize()  # only once no workers are writing
```

The array is chunked ``(1,)`` while writing so concurrent workers never
contend for the same chunk.  ``finalize()`` rechunks it into a single
chunk for fast reads and must be called after all workers have finished.
This is useful for resuming a partial ingest, or for skipping gaps when a
remote archive is missing timestamps.

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
| [earth2studio](https://nvidia.github.io/earth2studio/) | Weather/climate data backends (ERA5, HRRR, GFS, OPERA) |
| [zarr](https://zarr.readthedocs.io/) | Zarr v3 store I/O |
| [gcsfs](https://gcsfs.readthedocs.io/) | Google Cloud Storage filesystem for ARCO data |
