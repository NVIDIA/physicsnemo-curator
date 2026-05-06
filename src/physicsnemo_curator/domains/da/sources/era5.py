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

"""ERA5 reanalysis source with multi-backend support.

Fetches ERA5 data from one or more earth2studio backends (ARCO, WB2,
NCAR, CDS).  Each requested variable is routed to the highest-priority
backend whose lexicon contains it.  When variables span multiple
backends, results are fetched separately and merged along the
``variable`` dimension.

Each pipeline index corresponds to a single timestamp, and the returned
:class:`xarray.DataArray` has dimensions ``(time, variable, lat, lon)``
with a single time step.
"""

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING, Any, ClassVar

from physicsnemo_curator.core.base import Param, Source

if TYPE_CHECKING:
    from collections.abc import Generator
    from datetime import datetime

    import xarray as xr

# ---------------------------------------------------------------------------
# Backend registry: name -> (data_module, data_class, lexicon_module, lexicon_class)
# ---------------------------------------------------------------------------
_BACKEND_REGISTRY: dict[str, tuple[str, str, str, str]] = {
    "arco": ("earth2studio.data", "ARCO", "earth2studio.lexicon", "ARCOLexicon"),
    "wb2": ("earth2studio.data", "WB2ERA5", "earth2studio.lexicon", "WB2Lexicon"),
    "ncar": ("earth2studio.data", "NCAR_ERA5", "earth2studio.lexicon", "NCAR_ERA5Lexicon"),
    "cds": ("earth2studio.data", "CDS", "earth2studio.lexicon", "CDSLexicon"),
}


def _import_backend(name: str, **kwargs: Any) -> Any:
    """Lazily import and instantiate an earth2studio data source.

    Parameters
    ----------
    name : str
        Backend name (key in ``_BACKEND_REGISTRY``).
    **kwargs : Any
        Additional keyword arguments forwarded to the backend constructor.

    Returns
    -------
    Any
        Instantiated data source (e.g. ``ARCO(...)``).

    Raises
    ------
    ImportError
        If the earth2studio module cannot be imported.
    Exception
        If the backend constructor fails (e.g. missing CDS credentials).
    """
    data_module, data_class, _, _ = _BACKEND_REGISTRY[name]
    mod = importlib.import_module(data_module)
    cls = getattr(mod, data_class)
    return cls(**kwargs)


def _import_lexicon(name: str) -> Any:
    """Lazily import an earth2studio lexicon class.

    Parameters
    ----------
    name : str
        Backend name (key in ``_BACKEND_REGISTRY``).

    Returns
    -------
    Any
        Lexicon class (e.g. ``ARCOLexicon``).  Supports ``var in lexicon``.
    """
    _, _, lex_module, lex_class = _BACKEND_REGISTRY[name]
    mod = importlib.import_module(lex_module)
    return getattr(mod, lex_class)


class ERA5Source(Source["xr.DataArray"]):
    """Fetch ERA5 reanalysis fields from earth2studio backends.

    Supports four backends — ARCO, WB2, NCAR, and CDS — with automatic
    per-variable routing.  Each variable is assigned to the highest-priority
    backend whose lexicon contains it.

    Parameters
    ----------
    times : list[datetime]
        Timestamps to fetch.  Must be within the range of the selected
        backend(s).
    variables : list[str]
        Earth2studio variable identifiers (e.g. ``"t2m"``, ``"z500"``).
    backend : str | list[str]
        Backend name or priority-ordered list.  Valid names: ``"arco"``,
        ``"wb2"``, ``"ncar"``, ``"cds"``.  Default ``"arco"`` preserves
        backward compatibility.
    backend_options : dict[str, dict[str, Any]] | None
        Per-backend keyword arguments forwarded to the constructor.
        Example: ``{"ncar": {"max_workers": 8}}``.  The ``cache`` and
        ``verbose`` parameters are set automatically.
    cache : bool
        Whether to cache downloaded chunks locally (default ``True``).

    Examples
    --------
    >>> from datetime import datetime
    >>> source = ERA5Source(
    ...     times=[datetime(2020, 6, 1, 0)],
    ...     variables=["t2m", "u10m"],
    ... )
    >>> len(source)
    1

    Multi-backend with fallback:

    >>> source = ERA5Source(
    ...     times=[datetime(2020, 6, 1, 0)],
    ...     variables=["t2m", "cp"],
    ...     backend=["arco", "ncar"],
    ... )
    >>> source.variable_routing  # doctest: +SKIP
    {'t2m': 'arco', 'cp': 'ncar'}

    Note
    ----
    - ERA5 documentation: `ECMWF ERA5 <https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5>`_
    - earth2studio: `NVIDIA earth2studio <https://nvidia.github.io/earth2studio/>`_
    - CDS API: `Climate Data Store <https://cds.climate.copernicus.eu/>`_
    """

    name: ClassVar[str] = "ERA5"
    description: ClassVar[str] = "ERA5 reanalysis via earth2studio (ARCO, WB2, NCAR, CDS)"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the ERA5 source.

        Returns
        -------
        list[Param]
            Descriptors for *times*, *variables*, *backend*, and *cache*.
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
                name="backend",
                description="Backend priority (comma-separated)",
                type=str,
                default="arco",
                choices=["arco", "wb2", "ncar", "cds"],
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
        backend: str | list[str] = "arco",
        backend_options: dict[str, dict[str, Any]] | None = None,
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

        # Normalize backend to a list.
        backend_names = [backend] if isinstance(backend, str) else list(backend)

        # Validate backend names.
        for bname in backend_names:
            if bname not in _BACKEND_REGISTRY:
                msg = f"Unknown backend {bname!r}. Valid: {sorted(_BACKEND_REGISTRY)}"
                raise ValueError(msg)

        # Import lexicons and resolve routing.
        self._routing = self._resolve_routing(variables, backend_names)

        # Instantiate backends eagerly.  Each forked process gets its own copy.
        self._backend_instances: dict[str, Any] = {}
        needed = set(self._routing.values())
        failed_backends: set[str] = set()

        for bname in backend_names:
            if bname not in needed:
                continue
            extra = (backend_options or {}).get(bname, {})
            kwargs = {"cache": cache, "verbose": False, **extra}
            try:
                self._backend_instances[bname] = _import_backend(bname, **kwargs)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"Backend {bname!r} failed to initialize: {exc}. Re-routing its variables to remaining backends.",
                    stacklevel=2,
                )
                failed_backends.add(bname)

        # Re-route variables from failed backends.
        if failed_backends:
            remaining = [b for b in backend_names if b not in failed_backends]
            vars_to_reroute = [v for v, b in self._routing.items() if b in failed_backends]
            if vars_to_reroute and not remaining:
                msg = f"All backends failed. Cannot serve variables: {vars_to_reroute}"
                raise RuntimeError(msg)
            rerouted = self._resolve_routing(vars_to_reroute, remaining)
            self._routing.update(rerouted)

            # Instantiate any newly needed backends.
            for bname in set(rerouted.values()) - set(self._backend_instances):
                extra = (backend_options or {}).get(bname, {})
                kwargs = {"cache": cache, "verbose": False, **extra}
                self._backend_instances[bname] = _import_backend(bname, **kwargs)

    def _resolve_routing(
        self,
        variables: list[str],
        backend_names: list[str],
    ) -> dict[str, str]:
        """Map each variable to its highest-priority available backend.

        Parameters
        ----------
        variables : list[str]
            Variable IDs to route.
        backend_names : list[str]
            Priority-ordered backend names.

        Returns
        -------
        dict[str, str]
            Mapping of variable name to backend name.

        Raises
        ------
        ValueError
            If any variable is not found in any backend's lexicon.
        """
        routing: dict[str, str] = {}
        unresolved: list[str] = []
        lexicons = {b: _import_lexicon(b) for b in backend_names}
        for var in variables:
            for bname in backend_names:
                if var in lexicons[bname]:
                    routing[var] = bname
                    break
            else:
                unresolved.append(var)
        if unresolved:
            msg = f"Variables not found in any backend ({', '.join(backend_names)}): {unresolved}"
            raise ValueError(msg)
        return routing

    def __len__(self) -> int:
        """Return the number of timestamps in this source."""
        return len(self._times)

    def __getitem__(self, index: int) -> Generator[xr.DataArray]:
        """Fetch ERA5 data for the *index*-th timestamp.

        Variables are grouped by backend, fetched separately, and merged
        along the ``variable`` dimension.  When all variables use a single
        backend, no concat occurs.

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
        import numpy as np
        import xarray as xr

        n = len(self._times)
        if index < 0:
            index += n
        if index < 0 or index >= n:
            msg = f"Index {index} out of range for source with {n} timestamps."
            raise IndexError(msg)

        time = self._times[index]

        # Group variables by backend, preserving input order.
        groups: dict[str, list[str]] = {}
        for var in self._variables:
            bname = self._routing[var]
            groups.setdefault(bname, []).append(var)

        # Fetch from each backend.
        parts: list[xr.DataArray] = []
        for bname, var_list in groups.items():
            backend_instance = self._backend_instances[bname]
            da = backend_instance(time=[time], variable=var_list)
            parts.append(da)

        # Merge.
        if len(parts) == 1:
            result = parts[0]
        else:
            # Verify grid alignment before concat.
            ref_lat = parts[0].coords["lat"].values
            ref_lon = parts[0].coords["lon"].values
            for i, part in enumerate(parts[1:], 1):
                if not np.allclose(part.coords["lat"].values, ref_lat):
                    msg = f"Latitude grid mismatch between backend 0 and {i}"
                    raise ValueError(msg)
                if not np.allclose(part.coords["lon"].values, ref_lon):
                    msg = f"Longitude grid mismatch between backend 0 and {i}"
                    raise ValueError(msg)
            result = xr.concat(parts, dim="variable")

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
    def variable_routing(self) -> dict[str, str]:
        """Return mapping of variable name to backend name."""
        return dict(self._routing)

    @property
    def backends_used(self) -> set[str]:
        """Return set of backend names that have variables routed to them."""
        return set(self._routing.values())

    @property
    def active_backend(self) -> str | None:
        """Return the single backend name if all variables use one backend.

        Returns ``None`` if variables are split across multiple backends.

        Returns
        -------
        str | None
            Backend name or ``None``.
        """
        backends = self.backends_used
        return next(iter(backends)) if len(backends) == 1 else None
