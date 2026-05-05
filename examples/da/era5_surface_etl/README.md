# ERA5 Surface Data ETL

Curate [ERA5][era5] surface reanalysis data through a Source -> Filter -> Sink
pipeline, fetching one month of hourly data from Google's Analysis-Ready,
Cloud-Optimized (ARCO) store.

The pipeline:

1. **MomentsFilter** -- computes running statistics (mean, variance,
   skewness, min, max) per spatial grid point using Welford's online
   algorithm.
2. **ZarrSink** -- writes each timestep to a Zarr v3 store with one
   group per variable and dimensions `(time, lat, lon)`.

## Prerequisites

```bash
uv sync --group da

# or with pip
pip install physicsnemo-curator[da]
```

Required packages: `xarray`, `earth2studio`, `zarr>=3.0`, `gcsfs`.

## Data Access

ERA5 data is fetched on-the-fly from [Google's ARCO ERA5 Zarr store][arco]
via earth2studio. **No manual download is required** -- the source
streams data directly from Google Cloud Storage.

> **Note:** The ARCO backend requires no authentication. Downloaded
> chunks are cached locally at `~/.cache/earth2studio/arco/` by default.
> Set the `EARTH2STUDIO_CACHE` environment variable to override the
> cache location.

Available surface variables (earth2studio lexicon):

| Variable | Description                |
|----------|----------------------------|
| `t2m`   | 2-metre temperature         |
| `u10m`  | 10-metre U wind component   |
| `v10m`  | 10-metre V wind component   |
| `sp`    | Surface pressure            |
| `msl`   | Mean sea level pressure     |

## Usage

```bash
# Default: January 2020, 8 workers, ARCO backend
python main.py

# Custom month and output directory
python main.py --year 2020 --month 6 --output /path/to/output

# Adjust worker count
python main.py --workers 4

# Use alternative backend (wb2, ncar, or cds)
python main.py --backend ncar
```

## Output Structure

```text
output/era5_surface/
├── stats.zarr/          # Per-variable statistics (mean, var, skew, min, max)
│   ├── t2m/
│   ├── u10m/
│   ├── v10m/
│   ├── sp/
│   └── msl/
└── dataset.zarr/        # Full dataset (time, lat, lon) per variable
    ├── t2m/
    ├── u10m/
    ├── v10m/
    ├── sp/
    └── msl/
```

Each variable group in `dataset.zarr` contains a 3-D array with
dimensions `(time=n, lat=721, lon=1440)` at 0.25 degree resolution.

[era5]: https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5
[arco]: https://cloud.google.com/storage/docs/public-datasets/era5
