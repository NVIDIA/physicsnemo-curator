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

"""ERA5 reanalysis source backed by earth2studio ARCO.

Fetches ERA5 data from Google's Analysis-Ready, Cloud-Optimized (ARCO)
Zarr store via the :class:`earth2studio.data.ARCO` data source.  Each
pipeline index corresponds to a single timestamp, and the returned
:class:`xarray.DataArray` has dimensions ``(time, variable, lat, lon)``
with a single time step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from earth2studio.data import ARCO

from curator.core.base import Param, Source

if TYPE_CHECKING:
    from collections.abc import Generator
    from datetime import datetime

    import xarray as xr


class ERA5Source(Source["xr.DataArray"]):
    """Fetch ERA5 reanalysis fields from Google ARCO via earth2studio.

    Each index maps to one timestamp from *times*.  The returned
    :class:`xarray.DataArray` has dimensions ``(time, variable, lat, lon)``
    where ``time`` is length-1 (the single requested timestamp) and
    ``variable`` spans the requested *variables*.

    Parameters
    ----------
    times : list[datetime]
        Timestamps to fetch.  Must be hourly-aligned and within the ARCO
        range (1940-01-01 through ~2023-11-11).
    variables : list[str]
        Earth2studio variable identifiers (e.g. ``"t2m"``, ``"z500"``).
    cache : bool
        Whether to cache downloaded chunks locally (default ``True``).

    Examples
    --------
    >>> from datetime import datetime
    >>> source = ERA5Source(
    ...     times=[datetime(2020, 6, 1, 0), datetime(2020, 6, 1, 6)],
    ...     variables=["t2m", "u10m", "v10m"],
    ... )
    >>> len(source)
    2
    """

    name: ClassVar[str] = "ERA5 (ARCO)"
    description: ClassVar[str] = "ERA5 reanalysis via earth2studio ARCO data source"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the ERA5 source.

        Returns
        -------
        list[Param]
            Descriptors for *times*, *variables*, and *cache*.
        """
        return [
            Param(
                name="times",
                description="Comma-separated ISO timestamps (e.g. 2020-06-01T00:00)",
                type=str,
            ),
            Param(
                name="variables",
                description="Comma-separated earth2studio variable IDs (e.g. t2m,u10m,v10m)",
                type=str,
            ),
            Param(
                name="cache",
                description="Cache downloaded chunks locally",
                type=bool,
                default=True,
            ),
        ]

    def __init__(
        self,
        times: list[datetime],
        variables: list[str],
        *,
        cache: bool = True,
    ) -> None:
        if not times:
            msg = "times must be a non-empty list of datetime objects."
            raise ValueError(msg)
        if not variables:
            msg = "variables must be a non-empty list of variable IDs."
            raise ValueError(msg)

        self._times = list(times)
        self._variables = list(variables)
        self._arco = ARCO(cache=cache, verbose=False)

    def __len__(self) -> int:
        """Return the number of timestamps in this source."""
        return len(self._times)

    def __getitem__(self, index: int) -> Generator[xr.DataArray]:
        """Fetch ERA5 data for the *index*-th timestamp.

        Parameters
        ----------
        index : int
            Positional index into *times*.

        Yields
        ------
        xr.DataArray
            A single DataArray with dims ``(time, variable, lat, lon)``
            where ``time`` is length-1.
        """
        n = len(self._times)
        if index < 0:
            index += n
        if index < 0 or index >= n:
            msg = f"Index {index} out of range for source with {n} timestamps."
            raise IndexError(msg)

        time = self._times[index]
        da = self._arco(time=[time], variable=self._variables)
        yield da

    @property
    def times(self) -> list[datetime]:
        """Return the list of timestamps in this source."""
        return list(self._times)

    @property
    def variables(self) -> list[str]:
        """Return the list of variable IDs in this source."""
        return list(self._variables)
