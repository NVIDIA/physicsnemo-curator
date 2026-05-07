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

"""Creating a Custom Filter.

See README.md for a full walkthrough of this example.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from physicsnemo_curator.core.base import Filter, Param

if TYPE_CHECKING:
    from collections.abc import Generator

    import xarray as xr


# Step 1 — Define the Filter


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
                transformed = da.copy()
                var_idx = list(da.coords["variable"].values).index(self._variable)
                transformed.values[:, var_idx] = np.log1p(da.values[:, var_idx])
                yield transformed
            else:
                yield da


# Step 2 — Register the Filter (Optional)

from physicsnemo_curator.core.registry import registry

registry.register_filter("da", LogTransformFilter)

registered = registry.filters("da")
print(f"Registered DA filters: {list(registered.keys())}")
assert "Log Transform" in registered

# Step 3 — Use in a Pipeline

from datetime import datetime

from physicsnemo_curator.domains.da.sinks.zarr_writer import ZarrSink
from physicsnemo_curator.domains.da.sources.era5 import ERA5Source
from physicsnemo_curator.run import run_pipeline

source = ERA5Source(
    times=[datetime(2020, 6, 1, 0), datetime(2020, 6, 1, 6)],
    variables=["tp", "t2m"],
    backend="arco",
)

pipeline = source.filter(LogTransformFilter(variable="tp")).write(ZarrSink(output_path="output/extending/log_tp.zarr"))

print(f"Source items: {len(pipeline)}")

results = run_pipeline(
    pipeline,
    n_jobs=1,
    backend="sequential",
    indices=range(len(pipeline)),
    use_tui=True,
)

print(f"\nProcessed {len(results)} items")
for i, paths in enumerate(results):
    print(f"  Index {i}: {paths}")

# Extended API: Stateful Filters with flush() and artifacts()


class RunningVarianceFilter(Filter["xr.DataArray"]):
    """Accumulate running mean/variance across indices using Welford's algorithm.

    Demonstrates the stateful filter pattern: data is accumulated in
    __call__, written to disk in flush(), and reported via artifacts().

    Parameters
    ----------
    output : str
        Path for the output Parquet statistics file.
    variable : str
        Variable name to track.
    """

    name: ClassVar[str] = "Running Variance"
    description: ClassVar[str] = "Track running mean/variance via Welford's algorithm"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors.

        Returns
        -------
        list[Param]
            Parameters: output path and variable name.
        """
        return [
            Param(name="output", description="Output Parquet file path", type=str),
            Param(name="variable", description="Variable name to track", type=str),
        ]

    def __init__(self, output: str, variable: str) -> None:
        import pathlib

        self._output_path = pathlib.Path(output)
        self._variable = variable
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._rows: list[dict[str, object]] = []
        self._last_artifacts: list[str] = []

    def __call__(self, items: Generator[xr.DataArray]) -> Generator[xr.DataArray]:
        """Accumulate statistics and yield items unchanged (pass-through).

        Parameters
        ----------
        items : Generator[xr.DataArray]
            Incoming stream of DataArrays.

        Yields
        ------
        xr.DataArray
            The same DataArray, unmodified.
        """
        for da in items:
            if "variable" in da.dims and self._variable in da.coords["variable"].values:
                var_idx = list(da.coords["variable"].values).index(self._variable)
                values = da.values[:, var_idx].flatten()
                # Welford's online algorithm
                for x in values:
                    self._count += 1
                    delta = x - self._mean
                    self._mean += delta / self._count
                    delta2 = x - self._mean
                    self._m2 += delta * delta2

            self._rows.append(
                {
                    "count": self._count,
                    "mean": self._mean,
                    "variance": self._m2 / self._count if self._count > 0 else 0.0,
                }
            )
            yield da

    def flush(self) -> str | None:
        """Write accumulated statistics to the Parquet file.

        Returns
        -------
        str or None
            Path of the written Parquet file, or None.
        """
        if not self._rows:
            return None

        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table(
            {
                "count": [r["count"] for r in self._rows],
                "mean": [r["mean"] for r in self._rows],
                "variance": [r["variance"] for r in self._rows],
            }
        )

        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        if self._output_path.exists():
            existing = pq.read_table(str(self._output_path))
            table = pa.concat_tables([existing, table])

        pq.write_table(table, str(self._output_path))
        path = str(self._output_path)
        self._rows.clear()
        self._last_artifacts = [path]
        return path

    def artifacts(self) -> list[str]:
        """Return paths of files written since the last flush.

        Returns
        -------
        list[str]
            Artifact paths, or empty list.
        """
        paths = self._last_artifacts
        self._last_artifacts = []
        return paths

    @staticmethod
    def merge(parquet_paths: list[str], output: str) -> str:
        """Merge per-worker statistics files into one.

        Parameters
        ----------
        parquet_paths : list[str]
            Per-worker Parquet file paths.
        output : str
            Destination path for the merged file.

        Returns
        -------
        str
            Path of the merged output file.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        tables = [pq.read_table(p) for p in parquet_paths]
        merged = pa.concat_tables(tables, promote_options="default")

        import pathlib

        pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(merged, output)
        return output
