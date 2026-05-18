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

When *n_indices* and *variables* are provided, the sink pre-allocates
the full Zarr store at construction time and uses region-based writes
(``mode="r+"``) which are safe for fully concurrent workers — each
write touches only independent chunk data without modifying shared
array metadata.

When *n_indices* and *variables* are omitted, the sink falls back to
append-based writes (``append_dim``) which are only safe for
sequential execution.

When executed with a parallel backend (``process_pool``), the sink
partitions pipeline indices into chunk-aligned groups so that no two
workers write to the same Zarr chunk concurrently.  The partitioning
dimension defaults to the ``append_dim`` (``"time"``).

Remote Storage Support
----------------------
The sink supports writing to remote storage (S3, GCS, Azure) via fsspec.
Pass an fsspec-compatible URL as the *output_path* (e.g.,
``s3://bucket/path/dataset.zarr``) and provide authentication via
*storage_options*::

    sink = ZarrSink(
        output_path="s3://my-bucket/weather/gfs.zarr",
        storage_options={"key": "...", "secret": "..."},
        ...
    )

For AWS S3, you can also rely on ambient credentials from environment
variables or ``~/.aws/credentials``.
"""

from __future__ import annotations

import pathlib
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, ClassVar

import fsspec
import numpy as np
import xarray as xr
import zarr
from fsspec.core import split_protocol

from physicsnemo_curator.core.base import Param, Sink
from physicsnemo_curator.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


class ZarrSink(Sink["xr.DataArray"]):
    """Write :class:`xarray.DataArray` fields to a Zarr store.

    Each incoming DataArray is expected to carry coordinate metadata
    (e.g. ``time``, ``variable``, ``lat``, ``lon``).  The sink uses
    these coordinates — not the pipeline index — to organise the
    output.

    DataArrays with a ``variable`` dimension are split along it so
    that each variable gets its own Zarr group:
    ``<output_path>/<variable_name>/``, with dimensions
    ``(time, lat, lon)``.  Subsequent calls **append** along the
    append dimension, so the sink accumulates data across pipeline
    indices based on the coordinate in the incoming data.

    For parallel execution, the sink partitions pipeline indices into
    chunk-aligned groups via :meth:`partition_indices` so that no two
    workers write to the same Zarr chunk concurrently.

    Parameters
    ----------
    output_path : str
        Path to the output Zarr store directory.  Supports local paths
        and remote fsspec URLs (e.g., ``s3://bucket/path/store.zarr``).
    chunks : dict[str, int] | None
        Chunk sizes per dimension for the Zarr arrays.  Defaults to
        ``{"time": 1, "lat": 721, "lon": 1440}`` (one time-step per
        chunk, full spatial extent).
    shards : dict[str, int] | None
        Shard sizes per dimension (Zarr v3 only).  When provided, each
        shard is a container for multiple chunks.  Requires ``zarr>=3.0``.
        If *None*, sharding is not used.
    append_dim : str
        The dimension along which new data is appended on subsequent
        writes.  This is also the dimension used for chunk-aligned
        partitioning.  Defaults to ``"time"``.
    n_indices : int | None
        Total number of pipeline indices (time steps) that will be
        written.  When provided together with *variables*, the store
        is pre-allocated at construction time enabling safe concurrent
        region writes.
    variables : list[str] | None
        Variable names for pre-allocation.  Each variable becomes a
        separate Zarr group.  Required when *n_indices* is set.
    overwrite : bool
        If *True*, an existing store at *output_path* will be
        overwritten during pre-allocation.  If *False* (default) and
        the store already exists, a :class:`FileExistsError` is raised
        to prevent accidental data loss.
    storage_options : dict[str, Any] | None
        Extra keyword arguments for the fsspec filesystem.  Use this
        to provide credentials for remote stores (e.g., S3 keys).
        If *None*, uses ambient credentials from environment or config.

    Examples
    --------
    Sequential (append-based, backward compatible):

    >>> sink = ZarrSink(
    ...     output_path="output.zarr",
    ...     chunks={"time": 1, "lat": 721, "lon": 1440},
    ... )

    Parallel-safe (pre-allocated with region writes):

    >>> sink = ZarrSink(
    ...     output_path="output.zarr",
    ...     chunks={"time": 1, "hrrr_y": 1059, "hrrr_x": 1799},
    ...     n_indices=72,
    ...     variables=["t2m", "q2m", "tcwv"],
    ...     overwrite=True,
    ... )

    Remote S3 storage:

    >>> sink = ZarrSink(
    ...     output_path="s3://my-bucket/weather/gfs.zarr",
    ...     chunks={"time": 1, "lat": 721, "lon": 1440},
    ...     n_indices=28,
    ...     variables=["t2m", "u10m"],
    ...     storage_options={"anon": False},  # Use ambient AWS credentials
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
            Descriptors for *output_path*, *chunks*, and *append_dim*.
        """
        return [
            Param(name="output_path", description="Path to output Zarr store", type=str),
            Param(
                name="chunks",
                description="Chunk sizes as dim:size pairs (e.g. time:1,lat:721,lon:1440)",
                type=str,
                default="time:1,lat:721,lon:1440",
            ),
            Param(
                name="append_dim",
                description="Dimension along which data is appended (used for partitioning)",
                type=str,
                default="time",
            ),
        ]

    def __init__(
        self,
        output_path: str,
        chunks: dict[str, int] | None = None,
        shards: dict[str, int] | None = None,
        append_dim: str = "time",
        n_indices: int | None = None,
        variables: list[str] | None = None,
        overwrite: bool = False,
        storage_options: dict[str, Any] | None = None,
    ) -> None:
        self._log = get_logger(self)
        self._output_path = output_path
        self._storage_options = storage_options or {}
        self._chunks = chunks if chunks is not None else dict(self._DEFAULT_CHUNKS)
        self._shards = shards
        self._append_dim = append_dim
        self._n_indices = n_indices
        self._variables = variables
        self._overwrite = overwrite
        self._preallocated = False

        # Determine if using remote storage based on URL scheme.
        self._is_remote = "://" in output_path

        # Build filesystem for path operations.
        if self._is_remote:
            protocol, _ = split_protocol(output_path)
            self._fs = fsspec.filesystem(protocol, **self._storage_options)
        else:
            self._fs = fsspec.filesystem("file")

        # Validate: n_indices and variables must both be set or both be None.
        if (n_indices is None) != (variables is None):
            msg = "n_indices and variables must both be provided or both be omitted."
            raise ValueError(msg)

        # Pre-allocate the store if schema is provided.
        if n_indices is not None and variables is not None:
            self._preallocate_store(n_indices, variables)

    def _preallocate_store(self, n_indices: int, variables: list[str]) -> None:
        """Create the Zarr store with pre-allocated arrays for region writes.

        Each variable gets its own Zarr group at
        ``<output_path>/<variable_name>/`` with a ``data`` array of
        shape ``(n_indices, *spatial_dims)``.  Spatial dimension sizes
        are taken from :attr:`_chunks` (all dimensions except the
        append dimension).

        Parameters
        ----------
        n_indices : int
            Number of time steps to allocate along the append dimension.
        variables : list[str]
            Variable names — one Zarr group per variable.
        """
        # Determine spatial dims (everything in chunks except append_dim).
        spatial_dims = {k: v for k, v in self._chunks.items() if k != self._append_dim}
        if not spatial_dims:
            msg = (
                f"chunks must contain at least one spatial dimension besides '{self._append_dim}'. Got: {self._chunks}"
            )
            raise ValueError(msg)

        # Full shape: (n_indices, *spatial_sizes)
        dim_names = [self._append_dim, *spatial_dims.keys()]
        shape = (n_indices, *spatial_dims.values())
        chunk_sizes = tuple(self._chunks.get(d, s) for d, s in zip(dim_names, shape, strict=True))

        # Build encoding
        encoding: dict[str, dict[str, Any]] = {"data": {"chunks": chunk_sizes}}
        if self._shards is not None:
            shard_tuple = tuple(self._shards.get(d, c) for d, c in zip(dim_names, chunk_sizes, strict=True))
            encoding["data"]["shards"] = shard_tuple

        # Create coords (integer indices for spatial, empty for time)
        coords: dict[str, Any] = {self._append_dim: np.arange(n_indices)}
        for dim_name, dim_size in spatial_dims.items():
            coords[dim_name] = np.arange(dim_size)

        # Check for existing store and respect overwrite flag.
        # Use fsspec for path existence checks to support remote stores.
        store_exists = self._fs.exists(self._output_path)
        if store_exists:
            var_paths_exist = any(self._fs.exists(f"{self._output_path}/{v}") for v in variables)
            if var_paths_exist and not self._overwrite:
                msg = (
                    f"Zarr store already exists at '{self._output_path}'. "
                    f"Set overwrite=True to replace it, or remove the directory manually."
                )
                raise FileExistsError(msg)
            if var_paths_exist:
                self._log.warning("Overwriting existing Zarr store at: %s", self._output_path)

        # Create parent directory (local only; remote stores handle this automatically).
        if not self._is_remote:
            pathlib.Path(self._output_path).mkdir(parents=True, exist_ok=True)

        for var_name in variables:
            group_path = f"{self._output_path}/{var_name}"

            # Create empty NaN-filled DataArray with correct shape
            data = np.full(shape, np.nan, dtype=np.float32)
            da = xr.DataArray(data, dims=dim_names, coords=coords)
            ds = da.to_dataset(name="data")

            ds.to_zarr(
                store=str(group_path),
                mode="w",
                encoding=encoding,
                zarr_format=3,
            )
            self._log.debug("Pre-allocated Zarr group: %s (shape=%s)", group_path, shape)

        self._preallocated = True
        self._log.info(
            "Pre-allocated Zarr store: %d variables, %d time steps, shape=%s",
            len(variables),
            n_indices,
            shape,
        )

    def __call__(self, items: Iterator[xr.DataArray], index: int) -> list[str]:
        """Consume DataArrays and write each variable to the Zarr store.

        Output paths are derived from the coordinates of the incoming
        data (one Zarr group per variable), not from *index*.

        When the store was pre-allocated (via *n_indices* + *variables*),
        writes use region indexing keyed by *index* for safe concurrent
        access.

        Parameters
        ----------
        items : Iterator[xr.DataArray]
            Stream of DataArrays with dims ``(time, variable, lat, lon)``.
        index : int
            Pipeline source index.  Used as the time-slot position for
            region writes when the store is pre-allocated.

        Returns
        -------
        list[str]
            Paths of the Zarr groups written (one per variable).
        """
        t0 = time.perf_counter()
        self._log.info("Writing index %d to Zarr", index)
        paths: list[str] = []

        for da in items:
            written = self._write_dataarray(da, index)
            paths.extend(written)
            self._log.debug("Wrote %d groups for DataArray", len(written))

        self._log.info("Write complete: %d paths (%.2fs)", len(paths), time.perf_counter() - t0)
        return paths

    def partition_indices(self, indices: Iterable[int]) -> list[list[int]] | None:
        """Group pipeline indices by chunk along the append dimension.

        Each returned group contains indices whose data lands within the
        same Zarr chunk along :attr:`append_dim`.  The runner dispatches
        each group to a single worker, preventing concurrent writes to
        the same chunk.

        If the store is pre-allocated with region writes, or if the chunk
        size along the append dimension is 1, every index is its own
        chunk and no partitioning is needed — returns *None* to signal
        that one-index-per-worker dispatch is fine.

        Parameters
        ----------
        indices : Iterable[int]
            Pipeline indices to partition.

        Returns
        -------
        list[list[int]] | None
            Chunk-aligned groups, or *None* if each index already maps
            to a unique chunk (chunk size == 1 or pre-allocated store).
        """
        # Pre-allocated stores use region writes — fully concurrent-safe.
        if self._preallocated:
            return None

        chunk_size = self._chunks.get(self._append_dim, 1)
        if chunk_size <= 1:
            return None

        # Group indices by which chunk they fall into.
        # Pipeline index i appends at position i along the append dim.
        groups: dict[int, list[int]] = defaultdict(list)
        for idx in indices:
            chunk_id = idx // chunk_size
            groups[chunk_id].append(idx)

        # Return groups sorted by chunk id, each internally sorted.
        return [sorted(group) for _, group in sorted(groups.items())]

    def _write_dataarray(self, da: xr.DataArray, index: int) -> list[str]:
        """Split a DataArray by variable and write each to its Zarr group.

        Parameters
        ----------
        da : xr.DataArray
            Input with dims ``(time, variable, lat, lon)``.
        index : int
            Pipeline index used for region writes.

        Returns
        -------
        list[str]
            Paths of the written Zarr groups.
        """
        paths: list[str] = []

        # If no 'variable' dim, write directly
        if "variable" not in da.dims:
            group_path = f"{self._output_path}/data"
            self._write_to_zarr(da, group_path, index)
            paths.append(group_path)
            return paths

        # Split along the variable dimension and write each separately.
        variables = da.coords["variable"].values
        for var_name in variables:
            var_da = da.sel(variable=var_name).drop_vars("variable")
            var_str = str(var_name)
            group_path = f"{self._output_path}/{var_str}"
            self._write_to_zarr(var_da, group_path, index)
            paths.append(group_path)

        return paths

    def _write_to_zarr(self, da: xr.DataArray, group_path: str, index: int) -> None:
        """Write a DataArray to a Zarr group using the appropriate strategy.

        When the store is pre-allocated, uses region writes
        (``mode="r+"``).  Otherwise falls back to append-based writes
        for backward compatibility with sequential pipelines.

        Parameters
        ----------
        da : xr.DataArray
            Data to write (should have the append dim as the first dim).
        group_path : str
            Path to the Zarr group directory (local or remote URL).
        index : int
            Pipeline index — used as the position along the append
            dimension for region writes.
        """
        if self._preallocated:
            self._region_write(da, group_path, index)
        else:
            self._append_to_zarr(da, group_path)

    def _region_write(self, da: xr.DataArray, group_path: str, index: int) -> None:
        """Write a DataArray to a pre-allocated Zarr group using region indexing.

        Each call writes to a specific slice along the append dimension,
        determined by *index*.  This is concurrent-safe because each
        worker writes to disjoint chunk regions without modifying shared
        array metadata.

        Parameters
        ----------
        da : xr.DataArray
            Data to write.  Must have exactly one element along the
            append dimension.
        group_path : str
            Path to the pre-allocated Zarr group (local or remote URL).
        index : int
            Position along the append dimension to write to.
        """
        ds = da.to_dataset(name="data")

        # Build region spec: slice the append dim at the index position.
        region: dict[str, slice] = {self._append_dim: slice(index, index + 1)}

        # Drop all coordinates/variables that don't share a dimension with
        # the region.  xarray's region write requires every variable in the
        # dataset to have at least one dim in common with the region dims.
        # Dimension coordinates (e.g. hrrr_x, hrrr_y, time) are 1-D and
        # only indexed on themselves — drop those that lack the append dim.
        region_dims = set(region.keys())
        drop_vars = [
            name
            for name in list(ds.coords) + list(ds.data_vars)
            if name != "data" and not region_dims.intersection(ds[name].dims)
        ]
        ds = ds.drop_vars(drop_vars)

        ds.to_zarr(
            store=group_path,
            mode="r+",
            region=region,
            storage_options=self._storage_options if self._is_remote else None,
        )

    def _append_to_zarr(self, da: xr.DataArray, group_path: str) -> None:
        """Append a DataArray to a Zarr group, creating it if needed.

        This is the legacy write path used when the store is NOT
        pre-allocated.  It is only safe for sequential execution.

        The coordinate from the DataArray along the append dimension
        determines where in the output store the data lands.

        Parameters
        ----------
        da : xr.DataArray
            Data to write (should have the append dim as the first dim).
        group_path : str
            Path to the Zarr group directory (local or remote URL).
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

        if self._fs.exists(group_path):
            # Append along the configured dimension
            ds.to_zarr(
                store=group_path,
                mode="a",
                append_dim=self._append_dim,
                zarr_format=3,
                storage_options=self._storage_options if self._is_remote else None,
            )
        else:
            # Create parent directory (local only; remote stores handle this automatically).
            if not self._is_remote:
                pathlib.Path(group_path).parent.mkdir(parents=True, exist_ok=True)
            ds.to_zarr(
                store=group_path,
                mode="w",
                encoding=encoding,
                zarr_format=3,
                storage_options=self._storage_options if self._is_remote else None,
            )

    @property
    def output_path(self) -> str:
        """Return the output Zarr store path."""
        return self._output_path

    @property
    def append_dim(self) -> str:
        """Return the dimension along which data is appended."""
        return self._append_dim

    @property
    def zarr_version(self) -> int:
        """Return the Zarr format version in use."""
        return int(zarr.__version__[0])
