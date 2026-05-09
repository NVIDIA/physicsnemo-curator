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

"""Random DataArray source for testing and examples.

Generates synthetic :class:`xarray.DataArray` objects with random
gridded data on a regular latitude/longitude grid.  Useful for unit
tests, example pipelines, and quick prototyping without needing real
weather/climate data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from physicsnemo_curator.core.base import Param, Source

if TYPE_CHECKING:
    from collections.abc import Generator

    import xarray as xr


class RandomDataArraySource(Source["xr.DataArray"]):
    """Generate random DataArrays on a regular lat/lon grid.

    Each index yields a single :class:`xarray.DataArray` with dimensions
    ``(time, variable, lat, lon)`` containing random values.  The time
    coordinate advances by one hour per index.

    Parameters
    ----------
    n_samples : int
        Number of DataArrays this source provides (i.e. ``len(source)``).
    n_lat : int
        Number of latitude grid points.
    n_lon : int
        Number of longitude grid points.
    variables : str
        Comma-separated variable names (e.g. ``"u10m,v10m,t2m"``).
    seed : int
        Base random seed.  Each index uses ``seed + index`` for
        reproducibility.

    Examples
    --------
    >>> from physicsnemo_curator.domains.da.sources import RandomDataArraySource
    >>> source = RandomDataArraySource(n_samples=5, n_lat=90, n_lon=180)
    >>> len(source)
    5
    >>> da = next(source[0])
    >>> da.dims
    ('time', 'variable', 'lat', 'lon')
    """

    name: ClassVar[str] = "Random DataArray"
    description: ClassVar[str] = "Generate random DataArrays on a lat/lon grid for testing and prototyping"

    @classmethod
    def params(cls) -> list[Param]:
        """Declare configurable parameters."""
        return [
            Param(name="n_samples", description="Number of DataArrays to generate", type=int, default=10),
            Param(name="n_lat", description="Number of latitude points", type=int, default=181),
            Param(name="n_lon", description="Number of longitude points", type=int, default=360),
            Param(name="variables", description="Comma-separated variable names", type=str, default="u10m,v10m,t2m"),
            Param(name="seed", description="Base random seed for reproducibility", type=int, default=42),
        ]

    def __init__(
        self,
        n_samples: int = 10,
        n_lat: int = 181,
        n_lon: int = 360,
        variables: str = "u10m,v10m,t2m",
        seed: int = 42,
    ) -> None:
        """Initialize the random DataArray source."""
        self._n_samples = n_samples
        self._n_lat = n_lat
        self._n_lon = n_lon
        self._variables = [v.strip() for v in variables.split(",")]
        self._seed = seed

    def __len__(self) -> int:
        """Return the number of DataArrays available."""
        return self._n_samples

    def __getitem__(self, index: int) -> Generator[xr.DataArray]:
        """Yield a random DataArray for the given index.

        Parameters
        ----------
        index : int
            Zero-based index into the source.

        Yields
        ------
        xarray.DataArray
            A randomly generated DataArray with lat/lon coordinates.
        """
        import xarray as xr

        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self)})")

        rng = np.random.default_rng(self._seed + index)

        n_vars = len(self._variables)
        data = rng.standard_normal((1, n_vars, self._n_lat, self._n_lon)).astype(np.float32)

        # Regular lat/lon grid
        lat = np.linspace(90, -90, self._n_lat, dtype=np.float64)
        lon = np.linspace(0, 360, self._n_lon, endpoint=False, dtype=np.float64)

        # Time coordinate: one hour per index from a reference date
        time = np.array([np.datetime64("2020-01-01") + np.timedelta64(index, "h")])

        da = xr.DataArray(
            data=data,
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": time,
                "variable": self._variables,
                "lat": lat,
                "lon": lon,
            },
            name="random_field",
        )
        yield da
