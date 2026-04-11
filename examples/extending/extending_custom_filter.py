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

"""
Creating a Custom Filter
=========================

This example shows how to implement and register a custom
:class:`~physicsnemo_curator.core.base.Filter`.

We create a ``LogTransformFilter`` that applies a ``log1p`` transform
to a chosen variable in an :class:`xarray.DataArray` — a common
preprocessing step for ERA5 total precipitation (``tp``), which has
a highly skewed distribution.

A filter is a callable that receives a **generator** of items and
yields transformed (or unchanged) items downstream.  Filters can be
**pass-through** (side-effect only), **in-place** (modify items),
or **stateful** (accumulate results and flush at the end).

.. note::

   Install the DataArray extras before running::

       pip install physicsnemo-curator[da]
"""

# %%
# Step 1 — Define the Filter
# ----------------------------
#
# A filter inherits from :class:`~physicsnemo_curator.core.base.Filter`
# and implements three things:
#
# 1. ``name`` / ``description`` class variables (for CLI discovery)
# 2. ``params()`` class method (parameter descriptors)
# 3. ``__call__(items)`` (the transform logic)

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from physicsnemo_curator.core.base import Filter, Param

if TYPE_CHECKING:
    from collections.abc import Generator

    import xarray as xr


class LogTransformFilter(Filter["xr.DataArray"]):
    """Apply log1p transform to a variable in the DataArray.

    Replaces values for the target variable with ``log1p(x)``
    (i.e. ``ln(1 + x)``), which is useful for right-skewed fields
    like precipitation.  All other variables pass through unchanged.

    Parameters
    ----------
    variable : str
        Variable name to transform (e.g. ``"tp"`` for total
        precipitation).
    """

    name: ClassVar[str] = "Log Transform"
    description: ClassVar[str] = "Apply log1p transform to a DataArray variable"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for this filter.

        Returns
        -------
        list[Param]
            Single parameter: variable name to transform.
        """
        return [
            Param(
                name="variable",
                description="Variable name to apply log1p to",
                type=str,
            ),
        ]

    def __init__(self, variable: str) -> None:
        self._variable = variable

    def __call__(self, items: Generator[xr.DataArray]) -> Generator[xr.DataArray]:
        """Apply log1p to the target variable in each DataArray.

        Parameters
        ----------
        items : Generator[xr.DataArray]
            Incoming stream of DataArrays.

        Yields
        ------
        xr.DataArray
            DataArray with the target variable log-transformed.
        """
        for da in items:
            if "variable" in da.dims and self._variable in da.coords["variable"].values:
                # Select the target variable, transform, and put back
                transformed = da.copy()
                var_idx = list(da.coords["variable"].values).index(self._variable)
                transformed.values[:, var_idx] = np.log1p(da.values[:, var_idx])
                yield transformed
            else:
                yield da


# %%
# Step 2 — Register the Filter (Optional)
# ----------------------------------------
#
# Registration makes the filter discoverable via the global registry
# and the interactive CLI.  This is optional — unregistered filters
# work fine in pipelines built with Python code.

from physicsnemo_curator.core.registry import registry

registry.register_filter("da", LogTransformFilter)

# Verify registration
registered = registry.filters("da")
print(f"Registered DA filters: {list(registered.keys())}")
assert "Log Transform" in registered

# %%
# Step 3 — Use in a Pipeline
# ---------------------------
#
# The custom filter plugs into the standard pipeline API just like
# any built-in filter.
#
# Here we fetch ERA5 data with total precipitation, apply the log
# transform, and write to a Zarr store.

from datetime import datetime

from physicsnemo_curator.domains.da.sinks.zarr_writer import ZarrSink
from physicsnemo_curator.domains.da.sources.era5 import ERA5Source
from physicsnemo_curator.run import run_pipeline

source = ERA5Source(
    times=[datetime(2020, 6, 1, 0), datetime(2020, 6, 1, 6)],
    variables=["tp", "t2m"],
    backend="arco",
)

pipeline = source.filter(LogTransformFilter(variable="tp")).write(ZarrSink(output_path="outputs/extending/log_tp.zarr"))

print(f"Source items: {len(pipeline)}")

results = run_pipeline(
    pipeline,
    n_jobs=1,
    backend="sequential",
    indices=range(len(pipeline)),
    progress=True,
)

print(f"\nProcessed {len(results)} items")
for i, paths in enumerate(results):
    print(f"  Index {i}: {paths}")

# %%
# Summary
# -------
#
# To create a custom filter:
#
# 1. Subclass :class:`~physicsnemo_curator.core.base.Filter` with a
#    type parameter (``Filter["xr.DataArray"]``, ``Filter["Mesh"]``,
#    etc.)
# 2. Set ``name`` and ``description`` class variables
# 3. Implement ``params()`` and ``__call__(items)``
# 4. Optionally register with ``registry.register_filter()``
#
# For **stateful** filters (like statistics accumulators), add a
# ``flush()`` method and an ``_output_path`` attribute.  See
# :class:`~physicsnemo_curator.domains.mesh.filters.stats.StatsFilter` for
# an example with Welford accumulators and cross-worker merging.
