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

"""EUMETNET OPERA composite radar data source via earth2studio.

Fetches pan-European OPERA composite radar products from the EUMETNET Open
Radar Data archive on CloudFerro S3 using :class:`earth2studio.data.OPERA`.
Data is provided on a Lambert Equal-Area (LAEA) grid with native dimensions
``(y, x)`` and geographic coordinates ``_lat`` / ``_lon``.

Each pipeline index corresponds to a single timestamp, and the returned
:class:`xarray.DataArray` has dimensions ``(time, variable, y, x)`` with a
single time step.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, ClassVar

from physicsnemo_curator.core.base import Param, Source

if TYPE_CHECKING:
    from collections.abc import Generator
    from datetime import datetime

    import xarray as xr


def _import_opera(**kwargs: Any) -> Any:
    """Lazily import and instantiate the earth2studio OPERA data source."""
    mod = importlib.import_module("earth2studio.data")
    cls = mod.OPERA
    return cls(**kwargs)


def _import_lexicon() -> Any:
    """Lazily import the OPERA lexicon for variable validation."""
    mod = importlib.import_module("earth2studio.lexicon")
    return mod.OPERALexicon


class OPERASource(Source["xr.DataArray"]):
    """Fetch OPERA composite radar fields from the EUMETNET archive.

    `OPERA <https://eumetnet.github.io/openradardata-documentation/>`_ provides
    pan-European weather radar composites (reflectivity, rain rate, hourly
    accumulation) on a Lambert Equal-Area grid.  Data is accessed via
    :mod:`earth2studio` from the public CloudFerro S3 archive.

    Parameters
    ----------
    times : list[datetime]
        Timestamps to fetch (UTC).  Must align to the era-appropriate OPERA
        composite interval (15-minute before 2024-07-01, 5-minute from
        2024-07-01 onward).
    variables : list[str]
        Earth2studio variable identifiers (``"refc"``, ``"tprate"``, or
        ``"tp01"``).  Must be present in
        :class:`earth2studio.lexicon.OPERALexicon`.  Variables with
        different pixel resolutions cannot be mixed in one call.
    cache : bool
        Whether to cache downloaded ODIM HDF5 files locally.  Default ``False``.
    async_workers : int
        Maximum concurrent async fetch tasks.  Default ``16``.
    async_timeout : int
        Total timeout in seconds for a synchronous fetch call.  Default ``600``.
    retries : int
        Retry attempts per failed download.  Default ``3``.

    Examples
    --------
    >>> from datetime import datetime
    >>> source = OPERASource(
    ...     times=[datetime(2024, 9, 1, 0)],
    ...     variables=["refc"],
    ... )  # doctest: +SKIP
    >>> len(source)  # doctest: +SKIP
    1
    >>> da = next(source[0])  # doctest: +SKIP
    >>> da.dims  # doctest: +SKIP
    ('time', 'variable', 'y', 'x')

    Note
    ----
    - OPERA documentation: `EUMETNET Open Radar Data <https://eumetnet.github.io/openradardata-documentation/>`_
    - earth2studio: `NVIDIA earth2studio <https://nvidia.github.io/earth2studio/>`_
    """

    name: ClassVar[str] = "OPERA"
    description: ClassVar[str] = "EUMETNET OPERA composite radar via earth2studio"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the OPERA source."""
        return [
            Param(
                name="times",
                description="Comma-separated ISO timestamps (e.g. 2024-09-01T00:00)",
                type=str,
            ),
            Param(
                name="variables",
                description="Comma-separated earth2studio variable IDs (e.g. refc,tprate)",
                type=str,
            ),
            Param(
                name="cache",
                description="Cache downloaded ODIM HDF5 files locally",
                type=bool,
                default=False,
            ),
            Param(
                name="async_workers",
                description="Maximum concurrent async fetch tasks",
                type=int,
                default=16,
            ),
            Param(
                name="async_timeout",
                description="Total timeout in seconds for a fetch call",
                type=int,
                default=600,
            ),
            Param(
                name="retries",
                description="Retry attempts per failed download",
                type=int,
                default=3,
            ),
        ]

    def __init__(
        self,
        times: list[datetime],
        variables: list[str],
        *,
        cache: bool = False,
        async_workers: int = 16,
        async_timeout: int = 600,
        retries: int = 3,
    ) -> None:
        if not times:
            msg = "times must be a non-empty list of datetime objects."
            raise ValueError(msg)
        if not variables:
            msg = "variables must be a non-empty list of variable IDs."
            raise ValueError(msg)

        self._times = list(times)
        self._variables = list(variables)
        self._cache = cache
        self._async_workers = async_workers
        self._async_timeout = async_timeout
        self._retries = retries

        lexicon = _import_lexicon()
        unknown = [v for v in self._variables if v not in lexicon]
        if unknown:
            msg = f"Variables not found in OPERALexicon: {unknown}"
            raise ValueError(msg)

        self._backend = _import_opera(
            cache=cache,
            verbose=False,
            async_workers=async_workers,
            async_timeout=async_timeout,
            retries=retries,
        )

    def __getstate__(self) -> dict[str, Any]:
        """Return picklable state, excluding the unpicklable backend."""
        state = self.__dict__.copy()
        state.pop("_backend", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state and lazily re-create the backend in worker processes."""
        self.__dict__.update(state)
        self._backend = _import_opera(
            cache=self._cache,
            verbose=False,
            async_workers=self._async_workers,
            async_timeout=self._async_timeout,
            retries=self._retries,
        )

    def __len__(self) -> int:
        """Return the number of timestamps in this source."""
        return len(self._times)

    def __getitem__(self, index: int) -> Generator[xr.DataArray]:
        """Fetch OPERA data for the *index*-th timestamp.

        Parameters
        ----------
        index : int
            Positional index into *times*.  Negative indices are supported.

        Yields
        ------
        xr.DataArray
            A single DataArray with dims ``(time, variable, y, x)`` where
            ``time`` is length-1.

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
    def cache(self) -> bool:
        """Return whether local caching is enabled."""
        return self._cache
