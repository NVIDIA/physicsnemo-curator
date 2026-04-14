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

"""Running statistical moments filter for DataArray pipelines.

Computes online (streaming) statistics — mean, variance, skewness, min,
max, and count — along specified dimensions of each incoming
:class:`xarray.DataArray`.  The DataArray is yielded unchanged
(pass-through).  Call :meth:`flush` to write the accumulated statistics
to a Zarr store.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import xarray as xr

from physicsnemo_curator.core.base import Filter, Param

if TYPE_CHECKING:
    from collections.abc import Generator


class MomentsFilter(Filter["xr.DataArray"]):
    r"""Compute running statistical moments and accumulate to a Zarr store.

    For each incoming DataArray the filter updates online accumulators for
    the first three raw moments (used to derive mean, variance, and
    skewness), as well as element-wise min and max.  Reduction is
    performed over the specified *dims* (e.g. ``("time",)``), so the
    output statistics retain the remaining dimensions.

    The DataArray is yielded unchanged so downstream filters and sinks
    still receive the full data.

    Statistics are accumulated in memory using Welford's online algorithm
    for numerically stable computation.  Call :meth:`flush` after
    processing to write the results.

    Parameters
    ----------
    output : str
        Path for the output Zarr store containing the statistics.
    dims : tuple[str, ...]
        Dimension names along which to reduce (e.g. ``("time",)`` to
        compute per-spatial-point statistics across time).

    Examples
    --------
    >>> filt = MomentsFilter(output="stats.zarr", dims=("time",))
    >>> pipeline = source.filter(filt).write(sink)
    >>> for i in range(len(pipeline)):
    ...     pipeline[i]
    >>> filt.flush()  # write accumulated statistics
    """

    name: ClassVar[str] = "Statistical Moments"
    description: ClassVar[str] = "Compute running moments (mean, var, skew, min, max) along given dimensions"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the moments filter.

        Returns
        -------
        list[Param]
            Descriptors for *output* and *dims*.
        """
        return [
            Param(name="output", description="Output Zarr store path for statistics", type=str),
            Param(
                name="dims",
                description="Comma-separated dimension names to reduce (e.g. time)",
                type=str,
                default="time",
            ),
        ]

    def __init__(self, output: str, dims: tuple[str, ...] = ("time",)) -> None:
        self._output_path = pathlib.Path(output)
        self._dims = dims
        self._last_artifacts: list[str] = []

        # Welford accumulators keyed by variable name.
        # Each entry stores: count, mean (M1), M2, M3, min, max
        # with shapes equal to the remaining (non-reduced) dimensions.
        self._accumulators: dict[str, _MomentAccumulator] = {}

    def __call__(self, items: Generator[xr.DataArray]) -> Generator[xr.DataArray]:
        """Update running statistics for each DataArray and yield it unchanged.

        Parameters
        ----------
        items : Generator[xr.DataArray]
            Stream of incoming DataArrays.

        Yields
        ------
        xr.DataArray
            The same DataArray, unmodified.
        """
        for da in items:
            self._update(da)
            yield da

    def flush(self) -> str | None:
        """Write accumulated statistics to the output Zarr store.

        The output store contains one group per variable, each with
        arrays: ``mean``, ``variance``, ``skewness``, ``min``, ``max``,
        ``count``.

        If the output store already exists (e.g., from a previous flush in
        the same worker process), the statistics are merged using Chan's
        parallel Welford algorithm. This enables worker-level aggregation
        when running pipelines in parallel.

        Returns
        -------
        str or None
            The path of the written Zarr store, or ``None`` if no data
            was accumulated.
        """
        if not self._accumulators:
            return None

        self._output_path.mkdir(parents=True, exist_ok=True)

        # Build xarray Dataset for each variable and write/merge to Zarr
        for var_name, acc in self._accumulators.items():
            new_stats = acc.finalize()
            group_path = self._output_path / var_name

            # Merge with existing data if it exists (worker-level aggregation)
            if group_path.exists():
                existing_stats = xr.open_zarr(str(group_path))
                merged_stats = _merge_moment_datasets([existing_stats, new_stats])
                merged_stats.to_zarr(str(group_path), mode="w", zarr_format=3)
            else:
                new_stats.to_zarr(str(group_path), mode="w", zarr_format=3)

        path = str(self._output_path)
        self._accumulators.clear()
        self._last_artifacts = [path]
        return path

    def artifacts(self) -> list[str]:
        """Return paths written by the last :meth:`flush` call.

        Returns
        -------
        list[str]
            Paths of files written since the last call, or ``[]``.
        """
        paths = self._last_artifacts
        self._last_artifacts = []
        return paths

    def _update(self, da: xr.DataArray) -> None:
        """Update accumulators with data from a single DataArray.

        Parameters
        ----------
        da : xr.DataArray
            Input DataArray.  If it has a ``variable`` dimension, each
            variable is accumulated separately.
        """
        if "variable" in da.dims:
            for var_name in da.coords["variable"].values:
                var_da = da.sel(variable=var_name).drop_vars("variable")
                var_str = str(var_name)
                self._update_single(var_str, var_da)
        else:
            self._update_single("data", da)

    def _update_single(self, var_name: str, da: xr.DataArray) -> None:
        """Update the accumulator for a single variable.

        Parameters
        ----------
        var_name : str
            Variable name (used as a key).
        da : xr.DataArray
            Data for one variable, without the ``variable`` dimension.
        """
        if var_name not in self._accumulators:
            # Determine the shape of the remaining dimensions
            remaining_dims: list[str] = [str(d) for d in da.dims if d not in self._dims]
            remaining_coords: dict[str, object] = {
                str(d): da.coords[d] for d in da.dims if d not in self._dims and d in da.coords
            }
            self._accumulators[var_name] = _MomentAccumulator(
                remaining_dims=remaining_dims,
                remaining_coords=remaining_coords,
            )

        # Reduce over the specified dims to get per-sample aggregates
        # For each sample along the reduction dims, update the accumulator
        self._accumulators[var_name].update(da, self._dims)

    @property
    def output_path(self) -> pathlib.Path:
        """Return the output path."""
        return self._output_path

    @property
    def dims(self) -> tuple[str, ...]:
        """Return the reduction dimensions."""
        return self._dims

    @staticmethod
    def merge(zarr_paths: list[str], output: str) -> str:
        """Merge moment-statistics Zarr stores produced by parallel workers.

        Each worker writes a Zarr store with one group per variable,
        containing ``mean``, ``variance``, ``skewness``, ``min``, ``max``
        arrays and a ``count`` attribute.  This method recovers the Welford
        accumulator state from each store and merges them using Chan's
        parallel algorithm.

        Parameters
        ----------
        zarr_paths : list[str]
            Paths to per-worker Zarr stores.
        output : str
            Path for the merged output Zarr store.

        Returns
        -------
        str
            The path of the written merged Zarr store.

        Raises
        ------
        ValueError
            If *zarr_paths* is empty.

        Examples
        --------
        >>> paths = ["worker_0/stats.zarr", "worker_1/stats.zarr"]
        >>> MomentsFilter.merge(paths, output="merged.zarr")  # doctest: +SKIP
        'merged.zarr'
        """
        if not zarr_paths:
            msg = "zarr_paths must be a non-empty list."
            raise ValueError(msg)

        # Discover all variable names across all stores.
        var_groups: dict[str, list[xr.Dataset]] = {}
        for zpath in zarr_paths:
            store_path = pathlib.Path(zpath)
            if not store_path.exists():
                msg = f"Zarr store not found: {zpath}"
                raise FileNotFoundError(msg)

            for child in sorted(store_path.iterdir()):
                if child.is_dir():
                    var_name = child.name
                    ds = xr.open_zarr(str(child))
                    var_groups.setdefault(var_name, []).append(ds)

        if not var_groups:
            msg = "No variable groups found in any Zarr store."
            raise ValueError(msg)

        # Merge each variable group using Chan's parallel Welford algorithm.
        out_path = pathlib.Path(output)
        out_path.mkdir(parents=True, exist_ok=True)

        for var_name, datasets in var_groups.items():
            merged_ds = _merge_moment_datasets(datasets)
            group_path = out_path / var_name
            merged_ds.to_zarr(str(group_path), mode="w", zarr_format=3)

        return str(out_path)


class _MomentAccumulator:
    """Online accumulator for statistical moments using Welford's algorithm.

    Maintains running count, mean (M1), second central moment (M2),
    third central moment (M3), element-wise min and max.

    Parameters
    ----------
    remaining_dims : list[str]
        Names of the dimensions that are *not* reduced.
    remaining_coords : dict[str, Any]
        Coordinate arrays for the remaining dimensions.
    """

    def __init__(
        self,
        remaining_dims: list[str],
        remaining_coords: dict[str, object],
    ) -> None:
        self._remaining_dims = remaining_dims
        self._remaining_coords = remaining_coords
        self._count: int = 0
        self._mean: np.ndarray | None = None
        self._m2: np.ndarray | None = None
        self._m3: np.ndarray | None = None
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None

    def update(self, da: xr.DataArray, reduce_dims: tuple[str, ...]) -> None:
        """Incorporate new data into the running accumulators.

        Parameters
        ----------
        da : xr.DataArray
            New data.  Must have the same non-reduced shape.
        reduce_dims : tuple[str, ...]
            Dimensions to reduce over.
        """
        # Get the values along the reduction dims.
        # We iterate over each "slice" along the reduction dims and
        # update Welford's algorithm for each element.
        present_reduce_dims = tuple(d for d in reduce_dims if d in da.dims)

        if not present_reduce_dims:
            # No reduction dims present — treat the whole array as one sample
            self._update_sample(da.values)
            return

        # Stack all reduction dims into a single "sample" dim and iterate
        stacked = da.stack(_sample_=present_reduce_dims)
        n_samples = stacked.sizes["_sample_"]

        for i in range(n_samples):
            sample = stacked.isel(_sample_=i).values
            self._update_sample(sample)

    def _update_sample(self, x: np.ndarray) -> None:
        """Update accumulators with a single sample (Welford's online).

        Parameters
        ----------
        x : np.ndarray
            Sample array with shape matching the remaining dims.
        """
        x = np.asarray(x, dtype=np.float64)

        if self._mean is None:
            self._mean = np.zeros_like(x)
            self._m2 = np.zeros_like(x)
            self._m3 = np.zeros_like(x)
            self._min = x.copy()
            self._max = x.copy()

        self._count += 1
        n = self._count

        delta = x - self._mean
        delta_n = delta / n
        term1 = delta * delta_n * (n - 1)

        # Update M3 before M2 (needs old M2 value)
        self._m3 += term1 * delta_n * (n - 2) - 3 * delta_n * self._m2
        # Update M2
        self._m2 += term1
        # Update mean
        self._mean += delta_n

        # Min/max
        self._min = np.minimum(self._min, x)  # ty: ignore[no-matching-overload]  # numpy ufunc stub limitation
        self._max = np.maximum(self._max, x)  # ty: ignore[no-matching-overload]  # numpy ufunc stub limitation

    def finalize(self) -> xr.Dataset:
        """Compute final statistics and return as an xarray Dataset.

        Returns
        -------
        xr.Dataset
            Dataset with variables: ``mean``, ``variance``,
            ``skewness``, ``min``, ``max``, ``count``.
        """
        if self._mean is None or self._count == 0:
            return xr.Dataset()

        # After the None guard above, all accumulators are guaranteed to be set.
        assert self._m2 is not None  # noqa: S101
        assert self._m3 is not None  # noqa: S101
        assert self._min is not None  # noqa: S101
        assert self._max is not None  # noqa: S101

        mean = self._mean
        variance = self._m2 / self._count if self._count > 0 else np.zeros_like(self._mean)

        # Compute skewness from the third central moment.
        with np.errstate(divide="ignore", invalid="ignore"):
            m2_safe = np.where(self._m2 > 0, self._m2, np.nan)
            skewness = (np.sqrt(self._count) * self._m3) / np.power(m2_safe, 1.5)
            skewness = np.where(np.isfinite(skewness), skewness, 0.0)

        data_vars: dict[str, tuple[list[str], np.ndarray]] = {
            "mean": (self._remaining_dims, mean),
            "variance": (self._remaining_dims, variance),
            "skewness": (self._remaining_dims, skewness),
            "min": (self._remaining_dims, self._min),
            "max": (self._remaining_dims, self._max),
        }

        ds = xr.Dataset(
            data_vars=data_vars,
            coords=self._remaining_coords,
            attrs={"count": self._count},
        )
        return ds


def _merge_moment_datasets(datasets: list[xr.Dataset]) -> xr.Dataset:
    """Merge finalized moment datasets using Chan's parallel Welford algorithm.

    Each dataset must contain ``mean``, ``variance``, ``skewness``,
    ``min``, ``max`` data variables and a ``count`` attribute.

    The Welford accumulator state (count, mean, M2, M3) is recovered
    from the finalized statistics and merged pairwise.

    Parameters
    ----------
    datasets : list[xr.Dataset]
        Per-worker finalized moment datasets for a single variable.

    Returns
    -------
    xr.Dataset
        Merged dataset with the same structure.
    """
    if len(datasets) == 1:
        return datasets[0]

    # Recover Welford state from first dataset.
    ds_a = datasets[0]
    n_a = int(ds_a.attrs["count"])
    mean_a = ds_a["mean"].values.astype(np.float64)
    var_a = ds_a["variance"].values.astype(np.float64)
    m2_a = var_a * n_a
    skew_a = ds_a["skewness"].values.astype(np.float64)
    # Recover m3: skewness = sqrt(n) * m3 / m2^1.5
    # => m3 = skewness * m2^1.5 / sqrt(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        m2_safe = np.where(m2_a > 0, m2_a, np.nan)
        m3_a = np.where(m2_a > 0, skew_a * np.power(m2_safe, 1.5) / np.sqrt(n_a), 0.0)
    min_a = ds_a["min"].values.astype(np.float64)
    max_a = ds_a["max"].values.astype(np.float64)

    # Merge remaining datasets one at a time.
    for ds_b in datasets[1:]:
        n_b = int(ds_b.attrs["count"])
        mean_b = ds_b["mean"].values.astype(np.float64)
        var_b = ds_b["variance"].values.astype(np.float64)
        m2_b = var_b * n_b
        skew_b = ds_b["skewness"].values.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            m2_b_safe = np.where(m2_b > 0, m2_b, np.nan)
            m3_b = np.where(m2_b > 0, skew_b * np.power(m2_b_safe, 1.5) / np.sqrt(n_b), 0.0)
        min_b = ds_b["min"].values.astype(np.float64)
        max_b = ds_b["max"].values.astype(np.float64)

        n_ab = n_a + n_b
        delta = mean_b - mean_a
        delta2 = delta * delta

        # Chan's parallel formulas.
        mean_ab = (n_a * mean_a + n_b * mean_b) / n_ab
        m2_ab = m2_a + m2_b + delta2 * n_a * n_b / n_ab
        m3_ab = (
            m3_a
            + m3_b
            + delta * delta2 * n_a * n_b * (n_a - n_b) / (n_ab * n_ab)
            + 3 * delta * (n_a * m2_b - n_b * m2_a) / n_ab
        )

        n_a = n_ab
        mean_a = mean_ab
        m2_a = m2_ab
        m3_a = m3_ab
        min_a = np.minimum(min_a, min_b)
        max_a = np.maximum(max_a, max_b)

    # Finalize merged statistics.
    variance = m2_a / n_a if n_a > 0 else np.zeros_like(mean_a)
    with np.errstate(divide="ignore", invalid="ignore"):
        m2_safe = np.where(m2_a > 0, m2_a, np.nan)
        skewness = (np.sqrt(n_a) * m3_a) / np.power(m2_safe, 1.5)
        skewness = np.where(np.isfinite(skewness), skewness, 0.0)

    # Reconstruct the Dataset with the same structure as the inputs.
    ref = datasets[0]
    dims: list[str] = [str(d) for d in ref["mean"].dims]
    coords = dict(ref.coords.items())

    data_vars = {
        "mean": (dims, mean_a),
        "variance": (dims, variance),
        "skewness": (dims, skewness),
        "min": (dims, min_a),
        "max": (dims, max_a),
    }

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs={"count": n_a})
