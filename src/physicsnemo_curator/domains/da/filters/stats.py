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

The ``DataArrayStatsFilter`` name follows the same naming convention
as ``MeshStatsFilter`` in the CAE domain.
"""

from __future__ import annotations

import pathlib
import time
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import xarray as xr
import zarr
import zarr.storage

from physicsnemo_curator.core.base import Filter, Param
from physicsnemo_curator.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Generator


class DataArrayStatsFilter(Filter["xr.DataArray"]):
    r"""Compute running statistical moments and accumulate to a Zarr store.

    For each incoming DataArray the filter updates online accumulators for
    the first three raw moments (used to derive mean, variance, and
    skewness), as well as element-wise min and max.  Reduction is
    performed over the specified *dims* (e.g. ``("time",)``), so the
    output statistics retain the remaining dimensions.

    The DataArray is yielded unchanged so downstream filters and sinks
    still receive the full data.

    Statistics are accumulated in memory using Welford's online algorithm
    for numerically stable computation.  When run via :func:`run_pipeline`,
    the framework automatically flushes statistics after each index and
    records artifacts in the pipeline database.  If the output Zarr store
    already exists, new statistics are merged using Chan's parallel
    algorithm so multi-worker pipelines produce correct results.

    Parameters
    ----------
    output : str
        Path for the output Zarr store containing the statistics.
    dims : tuple[str, ...]
        Dimension names along which to reduce (e.g. ``("time",)`` to
        compute per-spatial-point statistics across time).
    keep_shards : bool
        If ``True``, per-worker shard Zarr stores are retained after
        merging via :func:`~physicsnemo_curator.run.gather_pipeline`.
        Useful for debugging or inspecting intermediate results.
        Default is ``False``.

    Examples
    --------
    >>> filt = DataArrayStatsFilter(output="stats.zarr", dims=("time",))
    >>> pipeline = source.filter(filt).write(sink)
    >>> run_pipeline(pipeline, n_jobs=4)  # flush is automatic
    """

    name: ClassVar[str] = "DataArray Statistics"
    description: ClassVar[str] = "Compute running statistics (mean, var, skew, min, max) along given dimensions"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the statistics filter.

        Returns
        -------
        list[Param]
            Descriptors for *output*, *dims*, and *keep_shards*.
        """
        return [
            Param(name="output", description="Output Zarr store path for statistics", type=str),
            Param(
                name="dims",
                description="Comma-separated dimension names to reduce (e.g. time)",
                type=str,
                default="time",
            ),
            Param(
                name="keep_shards",
                description="Keep per-worker shard Zarr stores after merging (default: False)",
                type=bool,
                default=False,
            ),
        ]

    def __init__(
        self,
        output: str,
        dims: tuple[str, ...] = ("time",),
        keep_shards: bool = False,
    ) -> None:
        self._log = get_logger(self)
        self._output_path = pathlib.Path(output)
        self._dims = dims
        self._keep_shards = keep_shards
        self._accumulators: dict[str, _MomentAccumulator] = {}
        self._last_artifacts: list[str] = []

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
        for counter, da in enumerate(items):
            t0 = time.perf_counter()
            self._update(da)
            self._log.info("Updated stats for item %d: %s (%.2fs)", counter, da.dims, time.perf_counter() - t0)
            yield da

    def flush(self) -> str | None:
        """Write accumulated statistics to the output Zarr store.

        Called automatically by the framework after each pipeline index
        via :func:`~physicsnemo_curator.run.base._flush_filters`.  Can
        also be called manually in custom loops.

        The output store contains one group per variable, each with
        arrays: ``mean``, ``variance``, ``skewness``, ``min``, ``max``,
        ``count``.

        If the output store already exists (e.g., from a previous flush in
        the same worker), the statistics are merged using Chan's parallel
        Welford algorithm.  This enables incremental aggregation across
        multiple indices processed by the same worker.

        Returns
        -------
        str or None
            The path of the written Zarr store, or ``None`` if no data
            was accumulated.
        """
        if not self._accumulators:
            return None

        t0 = time.perf_counter()
        self._log.info("Flushing stats for %d variables", len(self._accumulators))

        output_path = self._output_path
        output_path.mkdir(parents=True, exist_ok=True)

        # Build xarray Dataset for each variable and write/merge to Zarr
        for var_name, acc in self._accumulators.items():
            new_stats = acc.finalize()
            group_path = output_path / var_name

            # Merge with existing data if it exists (worker-level aggregation)
            if group_path.exists():
                existing_stats = _open_stats_zarr(group_path)
                merged_stats = _merge_moment_datasets([existing_stats, new_stats])
                merged_stats.to_zarr(str(group_path), mode="w", zarr_format=3)
                self._log.debug("Merged stats for %s", var_name)
            else:
                new_stats.to_zarr(str(group_path), mode="w", zarr_format=3)
                self._log.debug("Wrote stats for %s", var_name)

        path = str(output_path)
        self._accumulators.clear()
        self._last_artifacts = [path]
        self._log.info("Flush complete: %s (%.2fs)", path, time.perf_counter() - t0)
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

    @property
    def keep_shards(self) -> bool:
        """Return whether to keep per-worker shard files after merging."""
        return self._keep_shards

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
        >>> DataArrayStatsFilter.merge(paths, output="merged.zarr")  # doctest: +SKIP
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
                    ds = _open_stats_zarr(child)
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

    # -- Dashboard ------------------------------------------------------------

    @classmethod
    def dashboard_panel(
        cls,
        artifact_paths: list[str],
        selected_index: int | None = None,
    ) -> Any:
        """Return an interactive panel for viewing accumulated statistics.

        Displays either an ``imshow`` heatmap (for 2D spatial data per
        variable) or a scatter plot (for 1D data per variable).

        Parameters
        ----------
        artifact_paths : list[str]
            Paths to Zarr stores produced by this filter.
        selected_index : int or None
            Currently selected pipeline index, if any.

        Returns
        -------
        pn.viewable.Viewable or None
            A Panel layout with variable selector and appropriate plot,
            or a Markdown message if no data is available.
        """
        import holoviews as hv
        import panel as pn

        hv.extension("bokeh")

        if not artifact_paths:
            return pn.pane.Markdown("*No DataArray Statistics artifacts found.*")

        # Load the first valid Zarr store
        store_path: pathlib.Path | None = None
        for path in artifact_paths:
            candidate = pathlib.Path(path)
            if candidate.exists() and candidate.is_dir():
                store_path = candidate
                break
        else:
            paths_str = ", ".join(artifact_paths[:3])
            return pn.pane.Markdown(
                f"*Could not read any DataArray Statistics artifacts.*\n\nPaths checked: `{paths_str}`"
            )

        assert store_path is not None  # noqa: S101

        # Discover variable groups
        var_names = sorted(d.name for d in store_path.iterdir() if d.is_dir())
        if not var_names:
            return pn.pane.Markdown("*No variable groups found in statistics store.*")

        stat_names = ["mean", "variance", "skewness", "min", "max"]

        # Widgets
        var_select = pn.widgets.Select(name="Variable", options=var_names, value=var_names[0])
        stat_select = pn.widgets.Select(name="Statistic", options=stat_names, value="mean")

        @pn.depends(
            var_select.param.value,
            stat_select.param.value,
        )
        def update_plot(var_name: str, stat_name: str) -> object:
            """Update plot based on variable and statistic selection."""
            group_path = store_path / var_name  # type: ignore[operator]
            try:
                ds_var = xr.open_zarr(str(group_path))
            except Exception:  # noqa: BLE001
                return hv.Div("<p>Could not load variable data.</p>")

            if stat_name not in ds_var:
                return hv.Div(f"<p>Statistic '{stat_name}' not found.</p>")

            arr = ds_var[stat_name]
            ndim = len(arr.dims)

            if ndim >= 2:
                # 2D+ data: show imshow heatmap of first two spatial dims
                dim_y, dim_x = arr.dims[0], arr.dims[1]
                # If more than 2D, select first slice of remaining dims
                if ndim > 2:
                    sel = {str(d): 0 for d in arr.dims[2:]}
                    arr = arr.isel(sel)

                img = hv.Image(
                    (arr.coords[dim_x].values, arr.coords[dim_y].values, arr.values),
                    kdims=[str(dim_x), str(dim_y)],
                    vdims=[stat_name],
                )
                return img.opts(
                    colorbar=True,
                    cmap="viridis",
                    responsive=True,
                    height=450,
                    title=f"{var_name} — {stat_name}",
                    tools=["hover", "pan", "wheel_zoom", "reset"],
                )
            elif ndim == 1:
                # 1D data: scatter plot
                dim_x = str(arr.dims[0])
                coords = arr.coords[dim_x].values
                values = arr.values

                scatter = hv.Scatter(
                    (coords, values),
                    kdims=[dim_x],
                    vdims=[stat_name],
                )
                return scatter.opts(
                    color="#1f77b4",
                    size=6,
                    responsive=True,
                    height=450,
                    title=f"{var_name} — {stat_name}",
                    tools=["hover", "pan", "wheel_zoom", "reset"],
                )
            else:
                # 0D scalar — just display text
                val = float(arr.values)
                return hv.Div(f"<h3>{var_name} — {stat_name}: {val:.6g}</h3>")

        sidebar = pn.Column(
            "### Controls",
            var_select,
            stat_select,
            width=250,
            sizing_mode="fixed",
        )

        plot_pane = pn.pane.HoloViews(update_plot, sizing_mode="stretch_both", min_height=450)

        return pn.Row(
            sidebar,
            plot_pane,
            sizing_mode="stretch_width",
            min_height=500,
        )

    @classmethod
    def dashboard_layout_hints(cls) -> dict[str, int]:
        """Declare grid space for the statistics widget.

        Returns
        -------
        dict[str, int]
            Full width (12 columns), 3 rows tall.
        """
        return {"cols": 12, "rows": 3}


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
        self._n: np.ndarray | None = None  # per-element finite observation counts
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

        Non-finite values (NaN/±Inf) are skipped per element so missing
        pixels (e.g. outside a radar domain) do not poison the running
        mean.  ``self._count`` still increments once per sample so callers
        can track how many frames were processed.

        Parameters
        ----------
        x : np.ndarray
            Sample array with shape matching the remaining dims.
        """
        x = np.asarray(x, dtype=np.float64)
        finite = np.isfinite(x)

        if self._mean is None:
            self._mean = np.zeros_like(x)
            self._m2 = np.zeros_like(x)
            self._m3 = np.zeros_like(x)
            self._n = np.zeros(x.shape, dtype=np.int64)
            self._min = np.full_like(x, np.nan)
            self._max = np.full_like(x, np.nan)

        # After the None guard above, all accumulators are guaranteed to be set.
        assert self._m2 is not None  # noqa: S101
        assert self._m3 is not None  # noqa: S101
        assert self._n is not None  # noqa: S101
        assert self._min is not None  # noqa: S101
        assert self._max is not None  # noqa: S101

        self._count += 1
        if not finite.any():
            return

        # Dense arithmetic with the increments zeroed at non-finite elements.
        n_old = self._n
        n_new = n_old + finite
        delta = np.where(finite, x - self._mean, 0.0)
        delta_n = np.where(finite, delta / np.maximum(n_new, 1), 0.0)
        term1 = delta * delta_n * n_old

        # Update M3 before M2 (needs old M2 value)
        self._m3 += term1 * delta_n * (n_new - 2) - 3.0 * delta_n * self._m2
        # Update M2
        self._m2 += term1
        # Update mean
        self._mean += delta_n
        self._n = n_new

        # Min/max over finite observations only
        self._min = np.where(finite, np.fmin(self._min, x), self._min)
        self._max = np.where(finite, np.fmax(self._max, x), self._max)

    def finalize(self) -> xr.Dataset:
        """Compute final statistics and return as an xarray Dataset.

        The output includes both user-facing statistics and the raw
        Welford accumulator state (``welford_mean``, ``welford_m2``,
        ``welford_m3``, ``welford_n``) so that runs can be resumed exactly
        without recovering state from derived statistics.

        Returns
        -------
        xr.Dataset
            Dataset with variables: ``mean``, ``variance``,
            ``skewness``, ``min``, ``max``, ``welford_mean``,
            ``welford_m2``, ``welford_m3``, ``welford_n``.  The attribute
            ``count`` stores the total number of samples (elements along
            the reduced dimensions) seen, including all-NaN samples.
            Elements with no finite observations have NaN mean/min/max
            /variance/skewness.
        """
        if self._mean is None or self._count == 0:
            return xr.Dataset()

        # After the None guard above, all accumulators are guaranteed to be set.
        assert self._m2 is not None  # noqa: S101
        assert self._m3 is not None  # noqa: S101
        assert self._n is not None  # noqa: S101
        assert self._min is not None  # noqa: S101
        assert self._max is not None  # noqa: S101

        observed = self._n > 0
        mean = np.where(observed, self._mean, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            variance = np.where(observed, self._m2 / self._n, np.nan)
            # Compute skewness from the third central moment.
            m2_safe = np.where(self._m2 > 0, self._m2, np.nan)
            skewness = (np.sqrt(self._n) * self._m3) / np.power(m2_safe, 1.5)
            skewness = np.where(np.isfinite(skewness), skewness, 0.0)
            skewness = np.where(observed, skewness, np.nan)

        data_vars: dict[str, tuple[list[str], np.ndarray]] = {
            "mean": (self._remaining_dims, mean),
            "variance": (self._remaining_dims, variance),
            "skewness": (self._remaining_dims, skewness),
            "min": (self._remaining_dims, self._min.copy()),
            "max": (self._remaining_dims, self._max.copy()),
            # Raw Welford state for exact resumption
            "welford_mean": (self._remaining_dims, self._mean.copy()),
            "welford_m2": (self._remaining_dims, self._m2.copy()),
            "welford_m3": (self._remaining_dims, self._m3.copy()),
            "welford_n": (self._remaining_dims, self._n.astype(np.float64)),
        }

        ds = xr.Dataset(
            data_vars=data_vars,
            coords=self._remaining_coords,
            attrs={"count": int(self._count)},
        )
        return ds


def _open_stats_zarr(path: str | pathlib.Path) -> xr.Dataset:
    """Open a stats Zarr v3 store using zarr-python and return an xr.Dataset.

    This avoids ``xr.open_zarr`` which can fail on zarr v3 stores that
    lack the xarray-specific ``.zmetadata`` consolidation.  We read
    arrays directly from the zarr group and reconstruct the Dataset.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the zarr group directory.

    Returns
    -------
    xr.Dataset
        Reconstructed dataset with the same variables and attributes.
    """
    store = zarr.storage.LocalStore(str(path), read_only=True)
    group = zarr.open_group(store, mode="r")
    attrs = dict(group.attrs)

    # Known stat variable names written by _MomentAccumulator.finalize()
    stat_vars = {
        "mean",
        "variance",
        "skewness",
        "min",
        "max",
        "welford_mean",
        "welford_m2",
        "welford_m3",
        "welford_n",
    }

    data_vars: dict[str, tuple[list[str], np.ndarray]] = {}
    dims: list[str] = []

    for var_name in stat_vars:
        if var_name in group:
            arr = group[var_name]
            data = arr[:]
            # zarr v3 stores dimension names in array metadata
            dim_names = getattr(arr.metadata, "dimension_names", None)
            arr_dims: list[str] = [str(d) for d in dim_names] if dim_names else []
            if not dims and arr_dims:
                dims = arr_dims
            data_vars[var_name] = (arr_dims if arr_dims else dims, np.asarray(data))

    # Reconstruct coordinates from arrays that are NOT stat variables
    coords: dict[str, Any] = {}
    for name in group:
        if name not in stat_vars:
            arr = group[name]
            if not hasattr(arr, "ndim"):
                continue
            if arr.ndim == 0:
                coords[name] = np.asarray(arr[()])
            else:
                coords[name] = np.asarray(arr[:])

    if not data_vars:
        return xr.Dataset(attrs=attrs)

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


def _merge_moment_datasets(datasets: list[xr.Dataset]) -> xr.Dataset:
    """Merge finalized moment datasets using Chan's parallel Welford algorithm.

    Each dataset must contain ``mean``, ``variance``, ``skewness``,
    ``min``, ``max`` data variables and a ``count`` attribute.
    If ``welford_mean``, ``welford_m2``, ``welford_m3`` (and optionally
    ``welford_n``) are present, the Welford state is used directly for
    exact merging; otherwise it is recovered from the derived statistics
    (backward compatible).

    When ``welford_n`` is present, Chan's merge is applied **per element**
    so pixels that were missing (NaN) in one shard but observed in another
    combine correctly.  The scalar ``count`` attribute is still the sum of
    sample counts (e.g. frames processed) across inputs.

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

    def _extract_state(
        ds: xr.Dataset,
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract (sample_count, n, mean, m2, m3, min, max) from a dataset.

        ``sample_count`` is the scalar attrs count (frames/samples seen).
        ``n`` is the per-element finite observation count used for Chan merge.
        """
        sample_count = int(ds.attrs.get("count", 0))
        if sample_count == 0 and len(ds.data_vars) == 0:
            # Empty dataset - return zeros
            zeros = np.zeros((), dtype=np.float64)
            return 0, zeros, zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy()

        # Get a reference shape from any available variable
        ref_var = next(iter(ds.data_vars.values()))
        shape = ref_var.shape

        # Extract min/max with fallback to inf/-inf
        min_val = ds["min"].values.astype(np.float64) if "min" in ds else np.full(shape, np.inf, dtype=np.float64)
        max_val = ds["max"].values.astype(np.float64) if "max" in ds else np.full(shape, -np.inf, dtype=np.float64)

        # Prefer exact Welford state if all three variables available
        if "welford_mean" in ds and "welford_m2" in ds and "welford_m3" in ds:
            mean = ds["welford_mean"].values.astype(np.float64)
            m2 = ds["welford_m2"].values.astype(np.float64)
            m3 = ds["welford_m3"].values.astype(np.float64)
        elif "welford_mean" in ds and "welford_m2" in ds:
            # Partial Welford state (missing m3) - use what we have, set m3 to 0
            mean = ds["welford_mean"].values.astype(np.float64)
            m2 = ds["welford_m2"].values.astype(np.float64)
            m3 = np.zeros(shape, dtype=np.float64)
        elif "mean" in ds and "variance" in ds:
            # Recover from derived statistics (legacy stores)
            mean = ds["mean"].values.astype(np.float64)
            var = ds["variance"].values.astype(np.float64)
            m2 = var * sample_count
            if "skewness" in ds:
                skew = ds["skewness"].values.astype(np.float64)
                with np.errstate(divide="ignore", invalid="ignore"):
                    m2_safe = np.where(m2 > 0, m2, np.nan)
                    m3 = np.where(
                        m2 > 0,
                        skew * np.power(m2_safe, 1.5) / np.sqrt(max(sample_count, 1)),
                        0.0,
                    )
            else:
                m3 = np.zeros(shape, dtype=np.float64)
        else:
            # Minimal data - extract what we can
            if "welford_mean" in ds:
                mean = ds["welford_mean"].values.astype(np.float64)
            elif "mean" in ds:
                mean = ds["mean"].values.astype(np.float64)
            else:
                mean = np.zeros(shape, dtype=np.float64)
            m2 = np.zeros(shape, dtype=np.float64)
            m3 = np.zeros(shape, dtype=np.float64)

        if "welford_n" in ds:
            n = ds["welford_n"].values.astype(np.float64)
        else:
            # Legacy stores: treat every element as observed in all samples,
            # except positions that are already non-finite in the mean.
            n = np.full(shape, float(sample_count), dtype=np.float64)
            n[~np.isfinite(mean)] = 0.0

        # Zero out Welford state where nothing was observed.
        unobserved = n <= 0
        if np.any(unobserved):
            mean = mean.copy()
            m2 = m2.copy()
            m3 = m3.copy()
            mean[unobserved] = 0.0
            m2[unobserved] = 0.0
            m3[unobserved] = 0.0

        return sample_count, n, mean, m2, m3, min_val, max_val

    # Initialize from first dataset
    sample_count_a, n_a, mean_a, m2_a, m3_a, min_a, max_a = _extract_state(datasets[0])

    # Merge remaining datasets one at a time.
    for ds_b in datasets[1:]:
        sample_count_b, n_b, mean_b, m2_b, m3_b, min_b, max_b = _extract_state(ds_b)

        if sample_count_b == 0 and not np.any(n_b > 0):
            continue  # Skip empty datasets
        if sample_count_a == 0 and not np.any(n_a > 0):
            sample_count_a, n_a, mean_a, m2_a, m3_a, min_a, max_a = (
                sample_count_b,
                n_b,
                mean_b,
                m2_b,
                m3_b,
                min_b,
                max_b,
            )
            continue

        sample_count_a = sample_count_a + sample_count_b

        both = (n_a > 0) & (n_b > 0)
        only_b = (n_a <= 0) & (n_b > 0)
        # only_a: keep A's state (already in the copied buffers below)

        n_ab = n_a + n_b
        mean_ab = mean_a.copy()
        m2_ab = m2_a.copy()
        m3_ab = m3_a.copy()
        min_ab = min_a.copy()
        max_ab = max_a.copy()

        if np.any(both):
            n_a_b = n_a[both]
            n_b_b = n_b[both]
            n_ab_b = n_ab[both]
            mean_a_b = mean_a[both]
            mean_b_b = mean_b[both]
            m2_a_b = m2_a[both]
            m2_b_b = m2_b[both]
            m3_a_b = m3_a[both]
            m3_b_b = m3_b[both]

            delta = mean_b_b - mean_a_b
            delta2 = delta * delta

            # Chan's parallel formulas (per-element where both sides observed).
            mean_ab[both] = (n_a_b * mean_a_b + n_b_b * mean_b_b) / n_ab_b
            m2_ab[both] = m2_a_b + m2_b_b + delta2 * n_a_b * n_b_b / n_ab_b
            m3_ab[both] = (
                m3_a_b
                + m3_b_b
                + delta * delta2 * n_a_b * n_b_b * (n_a_b - n_b_b) / (n_ab_b * n_ab_b)
                + 3.0 * delta * (n_a_b * m2_b_b - n_b_b * m2_a_b) / n_ab_b
            )
            min_ab[both] = np.fmin(min_a[both], min_b[both])
            max_ab[both] = np.fmax(max_a[both], max_b[both])

        if np.any(only_b):
            mean_ab[only_b] = mean_b[only_b]
            m2_ab[only_b] = m2_b[only_b]
            m3_ab[only_b] = m3_b[only_b]
            min_ab[only_b] = min_b[only_b]
            max_ab[only_b] = max_b[only_b]

        n_a = n_ab
        mean_a = mean_ab
        m2_a = m2_ab
        m3_a = m3_ab
        min_a = min_ab
        max_a = max_ab

    # Finalize merged statistics.
    observed = n_a > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        variance = np.where(observed, m2_a / n_a, np.nan)
        m2_safe = np.where(m2_a > 0, m2_a, np.nan)
        skewness = (np.sqrt(n_a) * m3_a) / np.power(m2_safe, 1.5)
        skewness = np.where(np.isfinite(skewness), skewness, 0.0)
        skewness = np.where(observed, skewness, np.nan)
    mean_out = np.where(observed, mean_a, np.nan)

    # Reconstruct the Dataset with the same structure as the inputs.
    ref = datasets[0]
    dims: list[str] = [str(d) for d in next(iter(ref.data_vars.values())).dims]
    coords = dict(ref.coords.items())

    data_vars = {
        "mean": (dims, mean_out),
        "variance": (dims, variance),
        "skewness": (dims, skewness),
        "min": (dims, min_a),
        "max": (dims, max_a),
        "welford_mean": (dims, mean_a.copy()),
        "welford_m2": (dims, m2_a),
        "welford_m3": (dims, m3_a),
        "welford_n": (dims, n_a.astype(np.float64)),
    }

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs={"count": int(sample_count_a)})
