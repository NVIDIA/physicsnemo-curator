# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""ASV benchmarks for the DA (Data Assimilation) domain.

Covers end-to-end pipeline performance, MomentsFilter throughput, and sink
write comparisons (ZarrSink vs NetCDF4Sink).

Because ERA5Source requires remote network access, benchmarks use a lightweight
synthetic source that produces xarray DataArrays matching the ERA5 schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from physicsnemo_curator.core.base import Param, Source

from ._helpers import cleanup_temp_dir, create_temp_dir, make_synthetic_dataarray

if TYPE_CHECKING:
    from collections.abc import Generator

    import xarray  # noqa: F401 — used in Source["xarray.DataArray"]


# ---------------------------------------------------------------------------
# Synthetic DA source (ERA5 requires network access)
# ---------------------------------------------------------------------------
class _SyntheticDASource(Source["xarray.DataArray"]):
    """Minimal Source-compatible object yielding synthetic ERA5-like DataArrays.

    Parameters
    ----------
    n_timesteps : int
        Number of pipeline indices (one DataArray per index).
    n_lat : int
        Latitude grid size.
    n_lon : int
        Longitude grid size.
    """

    name = "synthetic_da"
    description = "Synthetic ERA5-like source for benchmarking."

    def __init__(
        self,
        n_timesteps: int,
        n_lat: int = 36,
        n_lon: int = 72,
    ) -> None:
        self._n_timesteps = n_timesteps
        self._n_lat = n_lat
        self._n_lon = n_lon

    @classmethod
    def params(cls) -> list[Param]:
        """Return empty params list."""
        return []

    def __len__(self) -> int:
        return self._n_timesteps

    def __getitem__(self, index: int) -> Generator:
        """Yield a single DataArray for the given index."""
        da = make_synthetic_dataarray(
            n_timesteps=1,
            n_lat=self._n_lat,
            n_lon=self._n_lon,
            seed=42 + index,
        )
        yield da


# ---------------------------------------------------------------------------
# E2E: Full DA pipeline
# ---------------------------------------------------------------------------
class TimeDAE2E:
    """End-to-end DA pipeline: SyntheticSource -> MomentsFilter -> ZarrSink."""

    params = [4, 12]
    param_names = ["n_timesteps"]

    def setup(self, n_timesteps):
        """Build the DA pipeline with synthetic source."""
        from physicsnemo_curator.core.base import Pipeline
        from physicsnemo_curator.domains.da.filters.moments import MomentsFilter
        from physicsnemo_curator.domains.da.sinks.zarr_writer import ZarrSink

        self._output_dir = create_temp_dir()
        self._moments_dir = create_temp_dir()

        source = _SyntheticDASource(n_timesteps=n_timesteps, n_lat=36, n_lon=72)
        moments = MomentsFilter(
            output=str(Path(self._moments_dir) / "moments.zarr"),
            dims=("time",),
        )
        sink = ZarrSink(output_path=str(Path(self._output_dir) / "out.zarr"))

        self.pipeline = Pipeline(
            source=source,
            filters=[moments],  # ty: ignore[invalid-argument-type]
            sink=sink,
            track_metrics=False,
            track_memory=False,
        )
        self.n_timesteps = n_timesteps

    def time_e2e(self, n_timesteps):
        """Run the full pipeline for all indices."""
        for i in range(len(self.pipeline)):
            self.pipeline[i]

    def teardown(self, n_timesteps):
        """Remove temporary directories."""
        cleanup_temp_dir(self._output_dir)
        cleanup_temp_dir(self._moments_dir)


class MemDAE2E:
    """Peak memory for full DA pipeline."""

    params = [4, 12]
    param_names = ["n_timesteps"]

    def setup(self, n_timesteps):
        """Build the DA pipeline with synthetic source."""
        from physicsnemo_curator.core.base import Pipeline
        from physicsnemo_curator.domains.da.filters.moments import MomentsFilter
        from physicsnemo_curator.domains.da.sinks.zarr_writer import ZarrSink

        self._output_dir = create_temp_dir()
        self._moments_dir = create_temp_dir()

        source = _SyntheticDASource(n_timesteps=n_timesteps, n_lat=36, n_lon=72)
        moments = MomentsFilter(
            output=str(Path(self._moments_dir) / "moments.zarr"),
            dims=("time",),
        )
        sink = ZarrSink(output_path=str(Path(self._output_dir) / "out.zarr"))

        self.pipeline = Pipeline(
            source=source,
            filters=[moments],  # ty: ignore[invalid-argument-type]
            sink=sink,
            track_metrics=False,
            track_memory=False,
        )
        self.n_timesteps = n_timesteps

    def peakmem_e2e(self, n_timesteps):
        """Run full pipeline, tracking peak RSS."""
        for i in range(len(self.pipeline)):
            self.pipeline[i]

    def teardown(self, n_timesteps):
        """Remove temporary directories."""
        cleanup_temp_dir(self._output_dir)
        cleanup_temp_dir(self._moments_dir)


# ---------------------------------------------------------------------------
# Component: MomentsFilter throughput
# ---------------------------------------------------------------------------
class TimeMomentsFilter:
    """Time MomentsFilter throughput for different grid sizes."""

    params = [36, 72]
    param_names = ["n_lat"]

    def setup(self, n_lat):
        """Create synthetic DataArrays and filter."""
        from physicsnemo_curator.domains.da.filters.moments import MomentsFilter

        self._output_dir = create_temp_dir()
        self.filt = MomentsFilter(
            output=str(Path(self._output_dir) / "moments.zarr"),
            dims=("time",),
        )
        self.n_lat = n_lat
        # Pre-generate a DataArray for filtering
        self.da = make_synthetic_dataarray(n_timesteps=1, n_lat=n_lat, n_lon=n_lat * 2)

    def time_filter(self, n_lat):
        """Apply MomentsFilter to a single DataArray."""

        def _gen():
            yield self.da

        for _ in self.filt(_gen()):
            pass

    def teardown(self, n_lat):
        """Remove temporary directory."""
        cleanup_temp_dir(self._output_dir)


# ---------------------------------------------------------------------------
# Component: ZarrSink write
# ---------------------------------------------------------------------------
class TimeZarrSinkWrite:
    """Time ZarrSink write for different grid sizes."""

    params = [36, 72]
    param_names = ["n_lat"]

    def setup(self, n_lat):
        """Create synthetic DataArray and sink."""
        from physicsnemo_curator.domains.da.sinks.zarr_writer import ZarrSink

        self._output_dir = create_temp_dir()
        self.sink = ZarrSink(output_path=str(Path(self._output_dir) / "out.zarr"))
        self.da = make_synthetic_dataarray(n_timesteps=1, n_lat=n_lat, n_lon=n_lat * 2)

    def time_write(self, n_lat):
        """Write a single DataArray via ZarrSink."""
        self.sink(iter([self.da]), 0)  # ty: ignore[invalid-argument-type]

    def teardown(self, n_lat):
        """Remove temporary directory."""
        cleanup_temp_dir(self._output_dir)


# ---------------------------------------------------------------------------
# Component: NetCDF4Sink write
# ---------------------------------------------------------------------------
class TimeNetCDF4SinkWrite:
    """Time NetCDF4Sink write for different grid sizes."""

    params = [36, 72]
    param_names = ["n_lat"]

    def setup(self, n_lat):
        """Create synthetic DataArray and sink."""
        from physicsnemo_curator.domains.da.sinks.netcdf_writer import NetCDF4Sink

        self._output_dir = create_temp_dir()
        self.sink = NetCDF4Sink(output_dir=self._output_dir)
        self.da = make_synthetic_dataarray(n_timesteps=1, n_lat=n_lat, n_lon=n_lat * 2)

    def time_write(self, n_lat):
        """Write a single DataArray via NetCDF4Sink."""
        self.sink(iter([self.da]), 0)  # ty: ignore[invalid-argument-type]

    def teardown(self, n_lat):
        """Remove temporary directory."""
        cleanup_temp_dir(self._output_dir)


# ---------------------------------------------------------------------------
# Component: Sink comparison (Zarr vs NetCDF4)
# ---------------------------------------------------------------------------
class TimeSinkComparison:
    """Side-by-side Zarr vs NetCDF4 sink comparison."""

    params = [36, 72]
    param_names = ["n_lat"]

    def setup(self, n_lat):
        """Create synthetic DataArray and both sinks."""
        from physicsnemo_curator.domains.da.sinks.netcdf_writer import NetCDF4Sink
        from physicsnemo_curator.domains.da.sinks.zarr_writer import ZarrSink

        self._zarr_dir = create_temp_dir()
        self._nc_dir = create_temp_dir()
        self.zarr_sink = ZarrSink(output_path=str(Path(self._zarr_dir) / "out.zarr"))
        self.nc_sink = NetCDF4Sink(output_dir=self._nc_dir)
        self.da = make_synthetic_dataarray(n_timesteps=1, n_lat=n_lat, n_lon=n_lat * 2)

    def time_zarr(self, n_lat):
        """Write via ZarrSink."""
        self.zarr_sink(iter([self.da]), 0)  # ty: ignore[invalid-argument-type]

    def time_netcdf4(self, n_lat):
        """Write via NetCDF4Sink."""
        self.nc_sink(iter([self.da]), 0)  # ty: ignore[invalid-argument-type]

    def teardown(self, n_lat):
        """Remove temporary directories."""
        cleanup_temp_dir(self._zarr_dir)
        cleanup_temp_dir(self._nc_dir)
