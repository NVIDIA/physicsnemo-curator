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

"""GFS analysis data source via earth2studio.

Fetches `Global Forecast System (GFS)
<https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast>`_
0.25-degree analysis data from cloud object stores (AWS or NCEP) using
:class:`earth2studio.data.GFS`.  GFS provides global weather analysis on an
equirectangular grid with spatial shape ``(721, 1440)`` (lat, lon).

Each pipeline index corresponds to a single timestamp, and the returned
:class:`xarray.DataArray` has dimensions
``(time, variable, lat, lon)`` with a single time step.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, ClassVar

from physicsnemo_curator.core.base import Param, Source

if TYPE_CHECKING:
    from collections.abc import Generator
    from datetime import datetime

    import xarray as xr

_VALID_SOURCES = ("aws", "ncep")


def _import_gfs(**kwargs: Any) -> Any:
    """Lazily import and instantiate the earth2studio GFS data source.

    Parameters
    ----------
    **kwargs : Any
        Keyword arguments forwarded to :class:`earth2studio.data.GFS`.

    Returns
    -------
    Any
        Instantiated ``GFS(...)`` data source.

    Raises
    ------
    ImportError
        If earth2studio is not installed.
    """
    mod = importlib.import_module("earth2studio.data")
    cls = mod.GFS
    return cls(**kwargs)


def _import_lexicon() -> Any:
    """Lazily import the GFS lexicon for variable validation.

    Returns
    -------
    Any
        ``GFSLexicon`` class.  Supports ``var in lexicon``.
    """
    mod = importlib.import_module("earth2studio.lexicon")
    return mod.GFSLexicon


class GFSSource(Source["xr.DataArray"]):
    """Fetch GFS analysis fields from cloud object stores.

    `GFS <https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast>`_
    is a NOAA global weather analysis and forecasting model providing data on a
    0.25-degree equirectangular grid (721 x 1440).  Data is accessed via
    :mod:`earth2studio` and can be served from AWS or NCEP.

    Parameters
    ----------
    times : list[datetime]
        Timestamps to fetch.  Must be at 6-hour intervals (00, 06, 12, 18 UTC).
        AWS source: available after 2021-01-01.
        NCEP source: last 10 days only.
    variables : list[str]
        Earth2studio variable identifiers (e.g. ``"t2m"``, ``"u10m"``).
        Must be present in :class:`earth2studio.lexicon.GFSLexicon`.
    source : str
        Cloud data source.  One of ``"aws"`` (default) or ``"ncep"``.
    cache : bool
        Whether to cache downloaded data locally.  Default ``False``.

    Examples
    --------
    >>> from datetime import datetime
    >>> source = GFSSource(
    ...     times=[datetime(2024, 1, 1, 0)],
    ...     variables=["t2m", "u10m"],
    ... )  # doctest: +SKIP
    >>> len(source)  # doctest: +SKIP
    1
    >>> da = next(source[0])  # doctest: +SKIP
    >>> da.dims  # doctest: +SKIP
    ('time', 'variable', 'lat', 'lon')

    Note
    ----
    - GFS documentation: `NOAA GFS <https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast>`_
    - AWS archive: `GFS on AWS <https://registry.opendata.aws/noaa-gfs-bdp-pds/>`_
    - earth2studio: `NVIDIA earth2studio <https://nvidia.github.io/earth2studio/>`_
    """

    name: ClassVar[str] = "GFS"
    description: ClassVar[str] = "GFS 0.25° analysis via earth2studio (AWS, NCEP)"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the GFS source.

        Returns
        -------
        list[Param]
            Descriptors for *times*, *variables*, *source*, and *cache*.
        """
        return [
            Param(
                name="times",
                description="Comma-separated ISO timestamps at 6h intervals (e.g. 2024-01-01T00:00)",
                type=str,
            ),
            Param(
                name="variables",
                description="Comma-separated earth2studio variable IDs (e.g. t2m,u10m,v10m)",
                type=str,
            ),
            Param(
                name="source",
                description="Cloud data source",
                type=str,
                default="aws",
                choices=list(_VALID_SOURCES),
            ),
            Param(
                name="cache",
                description="Cache downloaded data locally",
                type=bool,
                default=False,
            ),
        ]

    def __init__(
        self,
        times: list[datetime],
        variables: list[str],
        *,
        source: str = "aws",
        cache: bool = False,
    ) -> None:
        """Initialize the GFS source.

        Parameters
        ----------
        times : list[datetime]
            Timestamps to fetch.  Must be at 6-hour intervals.
        variables : list[str]
            Earth2studio variable identifiers.
        source : str
            Cloud data source (``"aws"`` or ``"ncep"``).
        cache : bool
            Whether to cache downloaded data locally.
        """
        if not times:
            msg = "times must be a non-empty list of datetime objects."
            raise ValueError(msg)
        if not variables:
            msg = "variables must be a non-empty list of variable IDs."
            raise ValueError(msg)
        if source not in _VALID_SOURCES:
            msg = f"Unknown source {source!r}. Valid: {sorted(_VALID_SOURCES)}"
            raise ValueError(msg)

        self._times = list(times)
        self._variables = list(variables)
        self._source = source
        self._cache = cache

        # Validate variables against the GFS lexicon.
        lexicon = _import_lexicon()
        unknown = [v for v in self._variables if v not in lexicon]
        if unknown:
            msg = f"Variables not found in GFSLexicon: {unknown}"
            raise ValueError(msg)

        # Instantiate the earth2studio backend.
        self._backend = _import_gfs(
            source=source,
            cache=cache,
            verbose=False,
        )

    def __getstate__(self) -> dict[str, Any]:
        """Return picklable state, excluding the unpicklable backend.

        The :class:`earth2studio.data.GFS` backend contains local
        functions and async filesystem handles that cannot be serialized
        with standard :mod:`pickle`.  We exclude it here and lazily
        re-create it on the worker side via :meth:`__setstate__`.
        """
        state = self.__dict__.copy()
        state.pop("_backend", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state and lazily re-create the backend.

        Called when a worker process unpickles this source.  The backend
        is reconstructed from the stored configuration parameters.
        """
        self.__dict__.update(state)
        self._backend = _import_gfs(
            source=self._source,
            cache=self._cache,
            verbose=False,
        )

    def __len__(self) -> int:
        """Return the number of timestamps in this source."""
        return len(self._times)

    def __getitem__(self, index: int) -> Generator[xr.DataArray]:
        """Fetch GFS data for the *index*-th timestamp.

        Parameters
        ----------
        index : int
            Positional index into *times*.  Negative indices are supported.

        Yields
        ------
        xr.DataArray
            A single DataArray with dims ``(time, variable, lat, lon)``
            where ``time`` is length-1.

        Raises
        ------
        IndexError
            If *index* is out of range.
        """
        n = len(self._times)
        if index < 0:
            index += n
        if index < 0 or index >= n:
            msg = f"Index {index} out of range for source with {n} timestamps."
            raise IndexError(msg)

        time = self._times[index]
        result = self._backend(time=[time], variable=self._variables)
        yield result

    @property
    def times(self) -> list[datetime]:
        """Return the list of timestamps in this source."""
        return list(self._times)

    @property
    def variables(self) -> list[str]:
        """Return the list of variable IDs in this source."""
        return list(self._variables)

    @property
    def source(self) -> str:
        """Return the cloud data source name."""
        return self._source

    @property
    def cache(self) -> bool:
        """Return whether local caching is enabled."""
        return self._cache
