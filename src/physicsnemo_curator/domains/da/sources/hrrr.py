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

"""HRRR analysis data source via earth2studio.

Fetches `High-Resolution Rapid Refresh (HRRR)
<https://www.nco.ncep.noaa.gov/pmb/products/hrrr/>`_ hourly analysis data
from cloud object stores (AWS, Google, or NOMADS) using
:class:`earth2studio.data.HRRR`.  HRRR provides 3 km North-American weather
analysis on a Lambert conformal grid with spatial shape ``(1799, 1059)``.

Each pipeline index corresponds to a single timestamp, and the returned
:class:`xarray.DataArray` has dimensions
``(time, variable, hrrr_x, hrrr_y)`` with a single time step.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, ClassVar

from physicsnemo_curator.core.base import Param, Source

if TYPE_CHECKING:
    from collections.abc import Generator
    from datetime import datetime

    import xarray as xr

_VALID_SOURCES = ("aws", "google", "nomads")


def _import_hrrr(**kwargs: Any) -> Any:
    """Lazily import and instantiate the earth2studio HRRR data source.

    Parameters
    ----------
    **kwargs : Any
        Keyword arguments forwarded to :class:`earth2studio.data.HRRR`.

    Returns
    -------
    Any
        Instantiated ``HRRR(...)`` data source.

    Raises
    ------
    ImportError
        If earth2studio is not installed.
    """
    mod = importlib.import_module("earth2studio.data")
    cls = mod.HRRR
    return cls(**kwargs)


def _import_lexicon() -> Any:
    """Lazily import the HRRR lexicon for variable validation.

    Returns
    -------
    Any
        ``HRRRLexicon`` class.  Supports ``var in lexicon``.
    """
    mod = importlib.import_module("earth2studio.lexicon")
    return mod.HRRRLexicon


class HRRRSource(Source["xr.DataArray"]):
    """Fetch HRRR analysis fields from cloud object stores.

    `HRRR <https://rapidrefresh.noaa.gov/hrrr/>`_ is a NOAA 3 km,
    hourly-updated weather analysis covering North America on a Lambert
    conformal grid.  Data is accessed via :mod:`earth2studio` and can be
    served from AWS, Google Cloud, or NOMADS.

    Parameters
    ----------
    times : list[datetime]
        Timestamps to fetch.  Must be within the range supported by the
        chosen *source* (AWS/Google: after 2018-07-12T13:00; NOMADS:
        last 2 days only).
    variables : list[str]
        Earth2studio variable identifiers (e.g. ``"t2m"``, ``"u10m"``).
        Must be present in :class:`earth2studio.lexicon.HRRRLexicon`.
    source : str
        Cloud data source.  One of ``"aws"`` (default), ``"google"``, or
        ``"nomads"``.
    cache : bool
        Whether to cache downloaded data locally.  Default ``False``.
    max_workers : int
        Maximum async I/O workers for downloads.  Default ``24``.

    Examples
    --------
    >>> from datetime import datetime
    >>> source = HRRRSource(
    ...     times=[datetime(2024, 1, 1, 0)],
    ...     variables=["t2m", "u10m"],
    ... )  # doctest: +SKIP
    >>> len(source)  # doctest: +SKIP
    1
    >>> da = next(source[0])  # doctest: +SKIP
    >>> da.dims  # doctest: +SKIP
    ('time', 'variable', 'hrrr_x', 'hrrr_y')

    Note
    ----
    - HRRR documentation: `NOAA HRRR <https://www.nco.ncep.noaa.gov/pmb/products/hrrr/>`_
    - AWS archive: `HRRR on AWS <https://aws.amazon.com/marketplace/pp/prodview-yd5ydptv3vuz2>`_
    - earth2studio: `NVIDIA earth2studio <https://nvidia.github.io/earth2studio/>`_
    """

    name: ClassVar[str] = "HRRR"
    description: ClassVar[str] = "HRRR 3 km analysis via earth2studio (AWS, Google, NOMADS)"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the HRRR source.

        Returns
        -------
        list[Param]
            Descriptors for *times*, *variables*, *source*, *cache*, and
            *max_workers*.
        """
        return [
            Param(
                name="times",
                description="Comma-separated ISO timestamps (e.g. 2024-01-01T00:00)",
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
            Param(
                name="max_workers",
                description="Maximum async I/O workers for downloads",
                type=int,
                default=24,
            ),
        ]

    def __init__(
        self,
        times: list[datetime],
        variables: list[str],
        *,
        source: str = "aws",
        cache: bool = False,
        max_workers: int = 24,
    ) -> None:
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
        self._max_workers = max_workers

        # Validate variables against the HRRR lexicon.
        lexicon = _import_lexicon()
        unknown = [v for v in self._variables if v not in lexicon]
        if unknown:
            msg = f"Variables not found in HRRRLexicon: {unknown}"
            raise ValueError(msg)

        # Instantiate the earth2studio backend.
        self._backend = _import_hrrr(
            source=source,
            cache=cache,
            verbose=False,
            max_workers=max_workers,
        )

    def __getstate__(self) -> dict[str, Any]:
        """Return picklable state, excluding the unpicklable backend.

        The :class:`earth2studio.data.HRRR` backend contains local
        functions that cannot be serialized with standard :mod:`pickle`.
        We exclude it here and lazily re-create it on the worker side
        via :meth:`__setstate__`.
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
        self._backend = _import_hrrr(
            source=self._source,
            cache=self._cache,
            verbose=False,
            max_workers=self._max_workers,
        )

    def __len__(self) -> int:
        """Return the number of timestamps in this source."""
        return len(self._times)

    def __getitem__(self, index: int) -> Generator[xr.DataArray]:
        """Fetch HRRR data for the *index*-th timestamp.

        Parameters
        ----------
        index : int
            Positional index into *times*.  Negative indices are supported.

        Yields
        ------
        xr.DataArray
            A single DataArray with dims ``(time, variable, hrrr_x, hrrr_y)``
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
