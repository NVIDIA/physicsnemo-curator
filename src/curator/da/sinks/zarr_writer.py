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

"""Zarr writer sink for xarray DataArrays.

Writes incoming :class:`xarray.DataArray` objects to a Zarr store,
creating one Zarr group per variable with dimensions
``(time, lat, lon)``.  Supports user-specified chunking and Zarr v3
sharding.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any, ClassVar

import zarr

from curator.core.base import Param, Sink

if TYPE_CHECKING:
    from collections.abc import Generator

    import xarray as xr


class ZarrSink(Sink["xr.DataArray"]):
    """Write :class:`xarray.DataArray` fields to a Zarr store.

    Each incoming DataArray is expected to have dimensions
    ``(time, variable, lat, lon)``.  The sink splits along the
    ``variable`` dimension and writes each variable to its own Zarr
    group: ``<output_path>/<variable_name>/``, with dimensions
    ``(time, lat, lon)``.

    Subsequent calls **append** along the ``time`` dimension, so the
    sink accumulates data across multiple pipeline indices.

    Parameters
    ----------
    output_path : str
        Path to the output Zarr store directory.
    chunks : dict[str, int] | None
        Chunk sizes per dimension for the Zarr arrays.  Defaults to
        ``{"time": 1, "lat": 721, "lon": 1440}`` (one time-step per
        chunk, full spatial extent).
    shards : dict[str, int] | None
        Shard sizes per dimension (Zarr v3 only).  When provided, each
        shard is a container for multiple chunks.  Requires ``zarr>=3.0``.
        If *None*, sharding is not used.

    Examples
    --------
    >>> sink = ZarrSink(
    ...     output_path="output.zarr",
    ...     chunks={"time": 1, "lat": 721, "lon": 1440},
    ... )
    """

    name: ClassVar[str] = "Zarr Writer"
    description: ClassVar[str] = "Write DataArrays to a Zarr store with configurable chunking and sharding"

    _DEFAULT_CHUNKS: ClassVar[dict[str, int]] = {"time": 1, "lat": 721, "lon": 1440}

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the Zarr sink.

        Returns
        -------
        list[Param]
            Descriptors for *output_path* and *chunks*.
        """
        return [
            Param(name="output_path", description="Path to output Zarr store", type=str),
            Param(
                name="chunks",
                description="Chunk sizes as dim:size pairs (e.g. time:1,lat:721,lon:1440)",
                type=str,
                default="time:1,lat:721,lon:1440",
            ),
        ]

    def __init__(
        self,
        output_path: str,
        chunks: dict[str, int] | None = None,
        shards: dict[str, int] | None = None,
    ) -> None:
        self._output_path = pathlib.Path(output_path)
        self._chunks = chunks if chunks is not None else dict(self._DEFAULT_CHUNKS)
        self._shards = shards

    def __call__(self, items: Generator[xr.DataArray], index: int) -> list[str]:
        """Consume DataArrays and write each variable to the Zarr store.

        Parameters
        ----------
        items : Generator[xr.DataArray]
            Stream of DataArrays with dims ``(time, variable, lat, lon)``.
        index : int
            Source index (used for logging; data is appended by time).

        Returns
        -------
        list[str]
            Paths of the Zarr groups written (one per variable).
        """
        paths: list[str] = []

        for da in items:
            written = self._write_dataarray(da)
            paths.extend(written)

        return paths

    def _write_dataarray(self, da: xr.DataArray) -> list[str]:
        """Split a DataArray by variable and write each to its Zarr group.

        Parameters
        ----------
        da : xr.DataArray
            Input with dims ``(time, variable, lat, lon)``.

        Returns
        -------
        list[str]
            Paths of the written Zarr groups.
        """
        paths: list[str] = []

        # If no 'variable' dim, write directly
        if "variable" not in da.dims:
            group_path = self._output_path / "data"
            self._append_to_zarr(da, group_path)
            paths.append(str(group_path))
            return paths

        # Split along the variable dimension and write each separately.
        variables = da.coords["variable"].values
        for var_name in variables:
            var_da = da.sel(variable=var_name).drop_vars("variable")
            var_str = str(var_name)
            group_path = self._output_path / var_str
            self._append_to_zarr(var_da, group_path)
            paths.append(str(group_path))

        return paths

    def _append_to_zarr(self, da: xr.DataArray, group_path: pathlib.Path) -> None:
        """Append a DataArray to a Zarr group, creating it if needed.

        Parameters
        ----------
        da : xr.DataArray
            Data to write (should have ``time`` as the first dim).
        group_path : pathlib.Path
            Path to the Zarr group directory.
        """
        # Convert to Dataset for xarray's Zarr writer
        ds = da.to_dataset(name="data")

        # Build encoding with chunk sizes
        encoding: dict[str, dict[str, Any]] = {}
        var_chunks: dict[str, int] = {}
        for dim in da.dims:
            dim_str = str(dim)
            if dim_str in self._chunks:
                var_chunks[dim_str] = self._chunks[dim_str]
            else:
                var_chunks[dim_str] = da.sizes[dim]

        chunk_tuple = tuple(var_chunks.get(str(d), da.sizes[d]) for d in da.dims)
        encoding["data"] = {"chunks": chunk_tuple}

        # Add sharding config if specified (Zarr v3)
        if self._shards is not None:
            shard_tuple = tuple(self._shards.get(str(d), chunk_tuple[i]) for i, d in enumerate(da.dims))
            encoding["data"]["shards"] = shard_tuple

        if group_path.exists():
            # Append along the time dimension
            ds.to_zarr(
                store=str(group_path),
                mode="a",
                append_dim="time",
                zarr_format=3,
            )
        else:
            group_path.parent.mkdir(parents=True, exist_ok=True)
            ds.to_zarr(
                store=str(group_path),
                mode="w",
                encoding=encoding,
                zarr_format=3,
            )

    @property
    def output_path(self) -> pathlib.Path:
        """Return the output Zarr store path."""
        return self._output_path

    @property
    def zarr_version(self) -> int:
        """Return the Zarr format version in use."""
        return int(zarr.__version__[0])
