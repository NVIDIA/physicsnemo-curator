# NOAA Global Forecast System (GFS) Analysis Data ETL

Curate [GFS](https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast)
0.25-degree global weather analysis data through a Source -> Filter -> Sink
pipeline, fetching data from AWS cloud storage and writing the curated
dataset to a Zarr store (including remote Zarr backends such as S3).

The pipeline:

1. **DataArrayStatsFilter** -- computes running statistics (mean, variance,
   skewness, min, max) per spatial grid point using Welford's online
   algorithm.
2. **ZarrSink** -- writes each timestep to a Zarr v3 store with one
   array per variable and dimensions `(time, lat, lon)`.

## Prerequisites

```bash
uv sync --extra da
uv pip install python-dotenv matplotlib

# or with pip
pip install physicsnemo-curator[da] python-dotenv matplotlib
```

Required packages: `xarray`, `earth2studio`, `zarr>=3.0`, `python-dotenv`.

## Data Access

GFS data is fetched on-the-fly from cloud object stores
([AWS](https://registry.opendata.aws/noaa-gfs-bdp-pds/)) via earth2studio.
**No manual download is required** -- the source streams data directly from
cloud storage.

> **Note:** The AWS backend requires no authentication for GFS archive
> data. By default caching is disabled and temporary files are cleaned up
> after each fetch.

### Variables

The example fetches 73 variables:

| Category | Variables |
|----------|-----------|
| Pressure levels (13) | 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa |
| Level variables (5) | t, u, v, z, q (temperature, wind, geopotential, humidity) |
| Surface variables (8) | u10m, v10m, u100m, v100m, t2m, sp, msl, tcwv |

### Remote Output (S3)

This example shows how an ETL pipeline can interface with remote storage systems.
The script supports configuring a remote S3 location for the output Zarr store.
S3 credentials can be configured via a `.env` file in this directory:

```bash
ZARR_S3_ACCESS_KEY_ID=your-access-key
ZARR_S3_SECRET_ACCESS_KEY=your-secret-key
ZARR_S3_REGION=us-east-1
ZARR_S3_ENDPOINT_URL=https://s3.amazonaws.com
```

> **Note:** These use `ZARR_S3_*` prefixes to avoid conflicting with the GFS source,
> which uses anonymous S3 access to the public `noaa-gfs-bdp-pds` bucket.

## Usage

The full time range is fixed (2021-01-01 to 2027-01-01, ~8760 6-hourly timestamps).
You specify which indices to process via `--start-index` and `--end-index`.
This pipeline is configured with checkpoint resume enabled, so reruns continue
from previously completed work.

```bash
# Process first 10 indices (0-9)
python main.py --zarr-path s3://<bucket>/path --start-index 0 --end-index 10

# Process next 20 indices (10-29)
python main.py --zarr-path s3://<bucket>/path --start-index 10 --end-index 30 --workers 8
```

## Monitor Progress and Errors

Use the `psnc` CLI to open the dashboard for this run's checkpoint database:

```bash
# Show the checkpoint database created by this example
ls outputs/checkpoint/*.db --port 8080

# Open the dashboard (replace with the actual DB filename)
uv run psnc dashboard outputs/checkpoint/<database-name>.db --port 8090
```

In the dashboard, use the **Overview** tab to monitor runner progress
(completed/remaining/failed indices) and inspect the error log for failures.

## Output Structure

```text
output/
├── stats.zarr/          # Per-variable statistics (mean, var, skew, min, max)
│   ├── t850/
│   ├── u500/
│   ├── t2m/
│   └── ...
├── checkpoint/          # Pipeline checkpoint database
│   └── *.db
└── data.zarr/           # Full dataset (time, lat, lon) per variable
    ├── t850
    ├── u500
    ├── t2m
    └── ...
```

Each variable in `data.zarr` is a 3-D array with dimensions
`(time=n, lat=721, lon=1440)` at 0.25-degree resolution on a regular
latitude-longitude grid.

## Plotting

After running the pipeline, visualize the analysis fields at a specific
timestep:

```bash
# Plot all variables at index 0
python plot_index.py --index 0

# Plot from remote S3 store
python plot_index.py --store s3://my-bucket/gfs/data.zarr --index 100
```

This produces a grid showing all variables at the specified timestep.

## References

- [NOAA GFS](https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast)
- [GFS Datasource](https://nvidia.github.io/earth2studio/modules/generated/data/earth2studio.data.GFS.html)
- [NVIDIA Earth2Studio](https://nvidia.github.io/earth2studio/)
