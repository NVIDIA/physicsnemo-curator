# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NetCDF4 writer sink for xarray DataArrays.

Writes incoming :class:`xarray.DataArray` objects to NetCDF4 files,
creating one file per variable.  Files are **split** by a configurable
coordinate dimension (default: ``time``, grouped by year) so that each
output file covers a self-contained slice of the data.

The directory layout is::

    <output_dir>/
        <variable>/
            <split_key>.nc      # e.g. 2020.nc, 2021.nc
        <variable>/
            <split_key>.nc

When no ``variable`` dimension is present, the variable subdirectory
is called ``data``.

Supports user-specified chunking (HDF5 chunk sizes) and zlib
compression.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any, ClassVar

from physicsnemo_curator.core.base import Param, Sink

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import xarray as xr


def _year_from_time(coord_value: Any) -> str:
    """Extract the year string from a numpy datetime64 or datetime value.

    Parameters
    ----------
    coord_value : Any
        A scalar coordinate value (``numpy.datetime64``, ``datetime``,
        or anything with a ``year`` attribute or convertible via
        ``astype("datetime64[Y]")``).

    Returns
    -------
    str
        Four-digit year string, e.g. ``"2020"``.
    """
    import numpy as np

    dt = np.datetime64(coord_value, "Y")
    return str(dt.astype("datetime64[Y]").astype(int) + 1970)


class NetCDF4Sink(Sink["xr.DataArray"]):
    """Write :class:`xarray.DataArray` fields to NetCDF4 files.

    Each incoming DataArray is expected to carry coordinate metadata
    (e.g. ``time``, ``variable``, ``lat``, ``lon``).  The sink uses
    these coordinates — not the pipeline index — to organise the
    output.

    DataArrays with a ``variable`` dimension are first split along it
    so that each variable gets its own subdirectory.  Then the data is
    further split along *split_dim* (default ``"time"``) using
    *split_func* (default: extract year) so that each distinct split
    key becomes a separate ``.nc`` file:

    ``<output_dir>/<variable>/<split_key>.nc``

    Within each file, data is **appended** along the *split_dim*
    (which is marked as an unlimited dimension), so multiple pipeline
    indices that share the same split key accumulate into the same
    file.

    Parameters
    ----------
    output_dir : str
        Path to the output directory where ``.nc`` files are created.
    chunks : dict[str, int] | None
        Chunk sizes per dimension for NetCDF4 internal chunking.
        Defaults to ``{"time": 1, "lat": 721, "lon": 1440}`` (one
        time-step per chunk, full spatial extent).  These control the
        HDF5 chunk layout used for on-disk storage and compression.
    compression_level : int
        Zlib compression level (0–9).  0 disables compression, 9 is
        maximum compression.  Defaults to 4 (a good speed/size
        trade-off).
    unlimited_dims : list[str] | None
        Dimensions to mark as *unlimited* in the NetCDF4 file.
        Defaults to ``["time"]`` so new timesteps can be appended.
    split_dim : str | None
        Coordinate dimension along which to split the data into
        separate files.  Defaults to ``"time"``.  Set to ``None`` to
        disable splitting (all data goes into a single file per
        variable).
    split_func : Callable[[Any], str] | None
        A function that takes a single coordinate value from
        *split_dim* and returns a string used as the file name
        (without the ``.nc`` extension).  Defaults to year extraction
        for ``"time"`` (e.g. ``datetime(2020, 6, 1)`` → ``"2020"``).

    Examples
    --------
    >>> # Default: split by year
    >>> sink = NetCDF4Sink(output_dir="output_nc")
    >>>
    >>> # Split by month
    >>> sink = NetCDF4Sink(
    ...     output_dir="output_nc",
    ...     split_func=lambda t: f"{t.astype('datetime64[M]')}",
    ... )
    >>>
    >>> # No splitting — one file per variable
    >>> sink = NetCDF4Sink(output_dir="output_nc", split_dim=None)
    """

    name: ClassVar[str] = "NetCDF4 Writer"
    description: ClassVar[str] = "Write DataArrays to NetCDF4 files with chunking and compression"

    _DEFAULT_CHUNKS: ClassVar[dict[str, int]] = {"time": 1, "lat": 721, "lon": 1440}

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the NetCDF4 sink.

        Returns
        -------
        list[Param]
            Descriptors for *output_dir*, *chunks*, *compression_level*,
            and *split_dim*.
        """
        return [
            Param(name="output_dir", description="Path to output directory for .nc files", type=str),
            Param(
                name="chunks",
                description="Chunk sizes as dim:size pairs (e.g. time:1,lat:721,lon:1440)",
                type=str,
                default="time:1,lat:721,lon:1440",
            ),
            Param(
                name="compression_level",
                description="Zlib compression level (0=off, 9=max)",
                type=int,
                default=4,
            ),
            Param(
                name="split_dim",
                description="Dimension to split files on (default: time, None=no split)",
                type=str,
                default="time",
            ),
        ]

    def __init__(
        self,
        output_dir: str,
        chunks: dict[str, int] | None = None,
        compression_level: int = 4,
        unlimited_dims: list[str] | None = None,
        split_dim: str | None = "time",
        split_func: Callable[[Any], str] | None = None,
    ) -> None:
        self._output_dir = pathlib.Path(output_dir)
        self._chunks = chunks if chunks is not None else dict(self._DEFAULT_CHUNKS)
        self._compression_level = compression_level
        self._unlimited_dims = unlimited_dims if unlimited_dims is not None else ["time"]
        self._split_dim = split_dim
        self._split_func = split_func if split_func is not None else _year_from_time

    def __call__(self, items: Iterator[xr.DataArray], index: int) -> list[str]:
        """Consume DataArrays and write each variable to NetCDF4 files.

        Output paths are derived from the coordinates of the incoming
        data (one subdirectory per variable, one file per split key),
        not from *index*.

        Parameters
        ----------
        items : Iterator[xr.DataArray]
            Stream of DataArrays with dims ``(time, variable, lat, lon)``.
        index : int
            Pipeline source index (not used for path naming — the data's
            own coordinates determine the output layout).

        Returns
        -------
        list[str]
            Paths of the NetCDF4 files written or appended to.
        """
        paths: list[str] = []

        for da in items:
            written = self._write_dataarray(da)
            paths.extend(written)

        return paths

    def _write_dataarray(self, da: xr.DataArray) -> list[str]:
        """Split a DataArray by variable (and optionally by split_dim) and write.

        Parameters
        ----------
        da : xr.DataArray
            Input array, typically with dims ``(time, variable, lat, lon)``.

        Returns
        -------
        list[str]
            Paths of the written NetCDF4 files.
        """
        paths: list[str] = []

        if "variable" not in da.dims:
            written = self._write_variable(da, "data")
            paths.extend(written)
            return paths

        # Split along the variable dimension and write each separately.
        variables = da.coords["variable"].values
        for var_name in variables:
            var_da = da.sel(variable=var_name).drop_vars("variable")
            written = self._write_variable(var_da, str(var_name))
            paths.extend(written)

        return paths

    def _write_variable(self, da: xr.DataArray, var_name: str) -> list[str]:
        """Write a single-variable DataArray, optionally splitting by coordinate.

        Parameters
        ----------
        da : xr.DataArray
            Data without the ``variable`` dimension.
        var_name : str
            Variable name used for the subdirectory.

        Returns
        -------
        list[str]
            Paths of the written files.
        """
        if self._split_dim is None or self._split_dim not in da.dims:
            # No splitting — single file per variable
            nc_path = self._output_dir / var_name / "data.nc"
            self._append_to_netcdf(da, nc_path)
            return [str(nc_path)]

        # Group by split key
        groups = self._group_by_split(da)
        paths: list[str] = []
        for split_key, group_da in groups.items():
            nc_path = self._output_dir / var_name / f"{split_key}.nc"
            self._append_to_netcdf(group_da, nc_path)
            paths.append(str(nc_path))

        return paths

    def _group_by_split(self, da: xr.DataArray) -> dict[str, xr.DataArray]:
        """Group a DataArray along split_dim into keyed sub-arrays.

        Parameters
        ----------
        da : xr.DataArray
            Data to split.

        Returns
        -------
        dict[str, xr.DataArray]
            Mapping from split key (e.g. ``"2020"``) to sub-array.
        """
        import numpy as np

        coord_values = da.coords[self._split_dim].values
        groups: dict[str, list[int]] = {}

        for i, val in enumerate(np.atleast_1d(coord_values)):
            key = self._split_func(val)
            groups.setdefault(key, []).append(i)

        result: dict[str, xr.DataArray] = {}
        for key, indices in groups.items():
            result[key] = da.isel({self._split_dim: indices})

        return result

    def _append_to_netcdf(self, da: xr.DataArray, nc_path: pathlib.Path) -> None:
        """Append a DataArray to a NetCDF4 file, creating it if needed.

        The coordinate from the DataArray determines where in the
        output file the data lands.

        Parameters
        ----------
        da : xr.DataArray
            Data to write.
        nc_path : pathlib.Path
            Path to the NetCDF4 file.
        """
        # Convert to Dataset for xarray's NetCDF writer
        ds = da.to_dataset(name="data")

        # Build encoding with chunk sizes and compression
        encoding = self._build_encoding(da)

        if nc_path.exists():
            self._append_existing(ds, nc_path)
        else:
            nc_path.parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(
                path=str(nc_path),
                mode="w",
                format="NETCDF4",
                encoding=encoding,
                unlimited_dims=self._unlimited_dims,
            )

    def _build_encoding(self, da: xr.DataArray) -> dict[str, dict[str, Any]]:
        """Build NetCDF4 encoding dict with chunking and compression.

        Parameters
        ----------
        da : xr.DataArray
            Data whose dimensions determine chunk shape.

        Returns
        -------
        dict[str, dict[str, Any]]
            Encoding dict for ``xr.Dataset.to_netcdf()``.
        """
        chunk_tuple = tuple(min(self._chunks.get(str(d), da.sizes[d]), da.sizes[d]) for d in da.dims)

        enc: dict[str, Any] = {"chunksizes": chunk_tuple}
        if self._compression_level > 0:
            enc["zlib"] = True
            enc["complevel"] = self._compression_level
        else:
            enc["zlib"] = False

        return {"data": enc}

    def _append_existing(self, ds: xr.Dataset, nc_path: pathlib.Path) -> None:
        """Append a Dataset to an existing NetCDF4 file along time.

        Reads the existing file with xarray, concatenates the new data
        along the ``time`` dimension, and rewrites.  This avoids
        incompatible HDF5 library issues that can arise when mixing
        ``xarray.to_netcdf`` with ``netCDF4.Dataset(mode='a')``.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset to append (must contain ``data`` variable).
        nc_path : pathlib.Path
            Path to the existing NetCDF4 file.
        """
        import xarray

        existing = xarray.open_dataset(str(nc_path))
        try:
            merged = xarray.concat([existing, ds], dim="time")
        finally:
            existing.close()

        encoding = self._build_encoding(merged["data"])
        merged.to_netcdf(
            path=str(nc_path),
            mode="w",
            format="NETCDF4",
            encoding=encoding,
            unlimited_dims=self._unlimited_dims,
        )

    @property
    def output_dir(self) -> pathlib.Path:
        """Return the output directory path."""
        return self._output_dir

    @property
    def compression_level(self) -> int:
        """Return the configured zlib compression level."""
        return self._compression_level

    @property
    def unlimited_dims(self) -> list[str]:
        """Return the list of unlimited dimensions."""
        return list(self._unlimited_dims)

    @property
    def split_dim(self) -> str | None:
        """Return the dimension used for file splitting."""
        return self._split_dim

    @property
    def split_func(self) -> Callable[[Any], str]:
        """Return the function used to compute split keys."""
        return self._split_func
