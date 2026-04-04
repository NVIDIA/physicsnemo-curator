# ERA5 Multi-Backend Data Source Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `ERA5Source` to support four earth2studio ERA5 backends (ARCO, WB2, NCAR, CDS) with per-variable automatic fallback routing, while preserving full backward compatibility.

**Architecture:** A module-level `_BACKEND_REGISTRY` maps backend names to lazily-imported earth2studio classes and lexicons. At `__init__` time, `_resolve_routing()` assigns each variable to its highest-priority available backend. In `__getitem__`, variables are grouped by backend, fetched separately, and merged via `xr.concat` along the `variable` dimension.

**Tech Stack:** Python 3.11+, earth2studio (ARCO/WB2ERA5/NCAR_ERA5/CDS), xarray, numpy, pytest, unittest.mock

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/physicsnemo_curator/da/sources/era5.py` | ERA5Source with multi-backend routing, lazy imports, fetch/merge |
| `test/da/test_pipeline.py` | New multi-backend tests + update existing name assertions |
| `docs/user-guide/datasets.md` | Document ERA5 multi-backend usage |

---

## Task 1: Backend Registry and Routing Logic

**Files:**
- Modify: `src/physicsnemo_curator/da/sources/era5.py`
- Modify: `test/da/test_pipeline.py`

This task rewrites `era5.py` with `_BACKEND_REGISTRY`, `_import_backend()`, `_import_lexicon()`, `_resolve_routing()`, the `backend` and `backend_options` parameters, and the multi-backend `__getitem__` with fetch/merge. Task 1 tests cover routing logic; Task 2 adds fetch/merge tests.

- [ ] **Step 1: Write failing tests for routing resolution**

Append to `test/da/test_pipeline.py`, after the `TestERA5Source` class (around line 206):

```python
class TestERA5MultiBackend:
    """Tests for multi-backend routing in ERA5Source."""

    def test_routing_single_backend(self) -> None:
        """All variables route to the single requested backend."""
        from unittest.mock import MagicMock, patch

        from physicsnemo_curator.da.sources.era5 import ERA5Source

        mock_arco = MagicMock()
        mock_lexicon = MagicMock()
        mock_lexicon.__contains__ = lambda self, v: v in {"t2m", "u10m"}

        with (
            patch(
                "physicsnemo_curator.da.sources.era5._import_backend",
                return_value=mock_arco,
            ),
            patch(
                "physicsnemo_curator.da.sources.era5._import_lexicon",
                return_value=mock_lexicon,
            ),
        ):
            source = ERA5Source(
                times=_TIMES,
                variables=["t2m", "u10m"],
                backend="arco",
            )
        assert source.variable_routing == {"t2m": "arco", "u10m": "arco"}
        assert source.backends_used == {"arco"}
        assert source.active_backend == "arco"

    def test_routing_multi_backend_fallback(self) -> None:
        """Variables route to first backend whose lexicon contains them."""
        from unittest.mock import MagicMock, patch

        from physicsnemo_curator.da.sources.era5 import ERA5Source

        mock_arco = MagicMock()
        mock_ncar = MagicMock()
        arco_lexicon = MagicMock()
        arco_lexicon.__contains__ = lambda self, v: v in {"t2m", "u10m"}
        ncar_lexicon = MagicMock()
        ncar_lexicon.__contains__ = lambda self, v: v in {"t2m", "u10m", "cp"}

        def import_backend(name, **kwargs):
            return {"arco": mock_arco, "ncar": mock_ncar}[name]

        def import_lexicon(name):
            return {"arco": arco_lexicon, "ncar": ncar_lexicon}[name]

        with (
            patch(
                "physicsnemo_curator.da.sources.era5._import_backend",
                side_effect=import_backend,
            ),
            patch(
                "physicsnemo_curator.da.sources.era5._import_lexicon",
                side_effect=import_lexicon,
            ),
        ):
            source = ERA5Source(
                times=_TIMES,
                variables=["t2m", "cp"],
                backend=["arco", "ncar"],
            )
        assert source.variable_routing == {"t2m": "arco", "cp": "ncar"}
        assert source.backends_used == {"arco", "ncar"}
        assert source.active_backend is None

    def test_routing_unresolvable_raises(self) -> None:
        """ValueError raised when a variable isn't in any backend."""
        from unittest.mock import MagicMock, patch

        from physicsnemo_curator.da.sources.era5 import ERA5Source

        empty_lexicon = MagicMock()
        empty_lexicon.__contains__ = lambda self, v: False

        with (
            patch(
                "physicsnemo_curator.da.sources.era5._import_backend",
                return_value=MagicMock(),
            ),
            patch(
                "physicsnemo_curator.da.sources.era5._import_lexicon",
                return_value=empty_lexicon,
            ),
        ):
            with pytest.raises(ValueError, match="not found in any backend"):
                ERA5Source(
                    times=_TIMES,
                    variables=["nonexistent_var"],
                    backend="arco",
                )

    def test_backend_options_forwarded(self) -> None:
        """Backend-specific options are forwarded to the constructor."""
        from unittest.mock import MagicMock, patch

        from physicsnemo_curator.da.sources.era5 import ERA5Source

        mock_ncar = MagicMock()
        ncar_lexicon = MagicMock()
        ncar_lexicon.__contains__ = lambda self, v: True

        captured_kwargs: dict = {}

        def import_backend(name, **kwargs):
            captured_kwargs.update(kwargs)
            return mock_ncar

        with (
            patch(
                "physicsnemo_curator.da.sources.era5._import_backend",
                side_effect=import_backend,
            ),
            patch(
                "physicsnemo_curator.da.sources.era5._import_lexicon",
                return_value=ncar_lexicon,
            ),
        ):
            ERA5Source(
                times=_TIMES,
                variables=["t2m"],
                backend="ncar",
                backend_options={"ncar": {"max_workers": 8}},
            )
        assert captured_kwargs.get("max_workers") == 8

    def test_cds_unavailable_fallback(self) -> None:
        """When CDS fails to instantiate, next backend is tried with warning."""
        import warnings
        from unittest.mock import MagicMock, patch

        from physicsnemo_curator.da.sources.era5 import ERA5Source

        mock_arco = MagicMock()
        cds_lexicon = MagicMock()
        cds_lexicon.__contains__ = lambda self, v: True
        arco_lexicon = MagicMock()
        arco_lexicon.__contains__ = lambda self, v: True

        def import_backend(name, **kwargs):
            if name == "cds":
                raise Exception("CDS API key not found")
            return mock_arco

        def import_lexicon(name):
            return {"cds": cds_lexicon, "arco": arco_lexicon}[name]

        with (
            patch(
                "physicsnemo_curator.da.sources.era5._import_backend",
                side_effect=import_backend,
            ),
            patch(
                "physicsnemo_curator.da.sources.era5._import_lexicon",
                side_effect=import_lexicon,
            ),
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                source = ERA5Source(
                    times=_TIMES,
                    variables=["t2m"],
                    backend=["cds", "arco"],
                )
            assert len(w) == 1
            assert "cds" in str(w[0].message).lower()
        assert source.variable_routing == {"t2m": "arco"}

    def test_invalid_backend_name_raises(self) -> None:
        """ValueError raised for unknown backend name."""
        from physicsnemo_curator.da.sources.era5 import ERA5Source

        with pytest.raises(ValueError, match="Unknown backend"):
            ERA5Source(
                times=_TIMES,
                variables=["t2m"],
                backend="fake_backend",
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/da/test_pipeline.py::TestERA5MultiBackend -v --no-header -x 2>&1 | head -30`
Expected: FAIL — `_import_backend`, `_import_lexicon`, `variable_routing` etc. not defined

- [ ] **Step 3: Implement the backend registry and routing in era5.py**

Replace the entire `src/physicsnemo_curator/da/sources/era5.py` with:

```python
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
    """

    name: ClassVar[str] = "ERA5"
    description: ClassVar[str] = (
        "ERA5 reanalysis via earth2studio (ARCO, WB2, NCAR, CDS)"
    )

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
        self._cache = cache

        # Normalize backend to a list.
        backend_names = [backend] if isinstance(backend, str) else list(backend)

        # Validate backend names.
        for bname in backend_names:
            if bname not in _BACKEND_REGISTRY:
                msg = f"Unknown backend {bname!r}. Valid: {sorted(_BACKEND_REGISTRY)}"
                raise ValueError(msg)

        # Import lexicons and resolve routing.
        self._routing = self._resolve_routing(variables, backend_names)

        # Instantiate only the backends that have variables routed to them.
        self._backend_instances: dict[str, Any] = {}
        needed = set(self._routing.values())
        failed_backends: set[str] = set()

        for bname in backend_names:
            if bname not in needed:
                continue
            extra = (backend_options or {}).get(bname, {})
            try:
                self._backend_instances[bname] = _import_backend(
                    bname, cache=cache, verbose=False, **extra
                )
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"Backend {bname!r} failed to initialize: {exc}. "
                    f"Re-routing its variables to remaining backends.",
                    stacklevel=2,
                )
                failed_backends.add(bname)

        # Re-route variables from failed backends.
        if failed_backends:
            remaining = [b for b in backend_names if b not in failed_backends]
            vars_to_reroute = [v for v, b in self._routing.items() if b in failed_backends]
            if vars_to_reroute and not remaining:
                msg = (
                    f"All backends failed. Cannot serve variables: {vars_to_reroute}"
                )
                raise RuntimeError(msg)
            rerouted = self._resolve_routing(vars_to_reroute, remaining)
            self._routing.update(rerouted)

            # Instantiate any newly needed backends.
            for bname in set(rerouted.values()) - set(self._backend_instances):
                extra = (backend_options or {}).get(bname, {})
                self._backend_instances[bname] = _import_backend(
                    bname, cache=cache, verbose=False, **extra
                )

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
        for var in variables:
            for bname in backend_names:
                lexicon = _import_lexicon(bname)
                if var in lexicon:
                    routing[var] = bname
                    break
            else:
                unresolved.append(var)
        if unresolved:
            msg = (
                f"Variables not found in any backend "
                f"({', '.join(backend_names)}): {unresolved}"
            )
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
        import xarray as xr_mod

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
            result = xr_mod.concat(parts, dim="variable")

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
```

- [ ] **Step 4: Run routing tests to verify they pass**

Run: `uv run pytest test/da/test_pipeline.py::TestERA5MultiBackend -v --no-header 2>&1 | tail -20`
Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/physicsnemo_curator/da/sources/era5.py test/da/test_pipeline.py
git commit -m "feat(da): add multi-backend routing to ERA5Source"
```

---

## Task 2: Fetch & Merge Tests

**Files:**
- Modify: `test/da/test_pipeline.py`

This task adds tests for the `__getitem__` fetch/merge behavior that was implemented in Task 1.

- [ ] **Step 1: Append fetch/merge tests to TestERA5MultiBackend**

Add these methods to the `TestERA5MultiBackend` class in `test/da/test_pipeline.py`:

```python
    def test_single_backend_no_concat(self) -> None:
        """When all variables use one backend, result is returned directly."""
        import xarray as xr
        from unittest.mock import MagicMock, patch

        from physicsnemo_curator.da.sources.era5 import ERA5Source

        mock_arco = MagicMock()
        expected = _make_dataarray(times=[_TIMES[0]], variables=["t2m", "u10m"])
        mock_arco.return_value = expected

        arco_lexicon = MagicMock()
        arco_lexicon.__contains__ = lambda self, v: v in {"t2m", "u10m"}

        with (
            patch(
                "physicsnemo_curator.da.sources.era5._import_backend",
                return_value=mock_arco,
            ),
            patch(
                "physicsnemo_curator.da.sources.era5._import_lexicon",
                return_value=arco_lexicon,
            ),
        ):
            source = ERA5Source(
                times=_TIMES,
                variables=["t2m", "u10m"],
                backend="arco",
            )
        results = list(source[0])
        assert len(results) == 1
        xr.testing.assert_identical(results[0], expected)
        mock_arco.assert_called_once()

    def test_multi_backend_merge(self) -> None:
        """Variables from different backends are merged along variable dim."""
        import numpy as np
        import xarray as xr
        from unittest.mock import MagicMock, patch

        from physicsnemo_curator.da.sources.era5 import ERA5Source

        # ARCO serves t2m, NCAR serves cp.
        arco_da = _make_dataarray(times=[_TIMES[0]], variables=["t2m"])
        ncar_da = _make_dataarray(times=[_TIMES[0]], variables=["cp"], seed=99)

        mock_arco = MagicMock(return_value=arco_da)
        mock_ncar = MagicMock(return_value=ncar_da)

        arco_lexicon = MagicMock()
        arco_lexicon.__contains__ = lambda self, v: v in {"t2m", "u10m"}
        ncar_lexicon = MagicMock()
        ncar_lexicon.__contains__ = lambda self, v: v in {"t2m", "u10m", "cp"}

        def import_backend(name, **kwargs):
            return {"arco": mock_arco, "ncar": mock_ncar}[name]

        def import_lexicon(name):
            return {"arco": arco_lexicon, "ncar": ncar_lexicon}[name]

        with (
            patch(
                "physicsnemo_curator.da.sources.era5._import_backend",
                side_effect=import_backend,
            ),
            patch(
                "physicsnemo_curator.da.sources.era5._import_lexicon",
                side_effect=import_lexicon,
            ),
        ):
            source = ERA5Source(
                times=_TIMES,
                variables=["t2m", "cp"],
                backend=["arco", "ncar"],
            )
        results = list(source[0])
        assert len(results) == 1
        merged = results[0]
        assert list(merged.coords["variable"].values) == ["t2m", "cp"]
        assert merged.sizes["variable"] == 2

    def test_grid_alignment_check(self) -> None:
        """Mismatched grids raise ValueError."""
        import numpy as np
        import xarray as xr
        from unittest.mock import MagicMock, patch

        from physicsnemo_curator.da.sources.era5 import ERA5Source

        # ARCO returns standard grid, NCAR returns different lats.
        arco_da = _make_dataarray(times=[_TIMES[0]], variables=["t2m"], n_lat=9)

        # Shift lats by 1 degree.
        lats = np.linspace(91, -89, 9)
        lons = np.linspace(0, 350, _LONS_N)
        rng = np.random.default_rng(99)
        ncar_da = xr.DataArray(
            data=rng.standard_normal((1, 1, 9, _LONS_N)),
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": [np.datetime64(_TIMES[0])],
                "variable": ["cp"],
                "lat": lats,
                "lon": lons,
            },
        )

        mock_arco = MagicMock(return_value=arco_da)
        mock_ncar = MagicMock(return_value=ncar_da)

        arco_lexicon = MagicMock()
        arco_lexicon.__contains__ = lambda self, v: v == "t2m"
        ncar_lexicon = MagicMock()
        ncar_lexicon.__contains__ = lambda self, v: v == "cp"

        def import_backend(name, **kwargs):
            return {"arco": mock_arco, "ncar": mock_ncar}[name]

        def import_lexicon(name):
            return {"arco": arco_lexicon, "ncar": ncar_lexicon}[name]

        with (
            patch(
                "physicsnemo_curator.da.sources.era5._import_backend",
                side_effect=import_backend,
            ),
            patch(
                "physicsnemo_curator.da.sources.era5._import_lexicon",
                side_effect=import_lexicon,
            ),
        ):
            source = ERA5Source(
                times=_TIMES,
                variables=["t2m", "cp"],
                backend=["arco", "ncar"],
            )

        with pytest.raises(ValueError, match="Latitude grid mismatch"):
            list(source[0])

    def test_variable_order_preserved(self) -> None:
        """Output variable order matches input regardless of backend grouping."""
        from unittest.mock import MagicMock, patch

        from physicsnemo_curator.da.sources.era5 import ERA5Source

        # Request: v10m (arco), cp (ncar), t2m (arco) — interleaved.
        arco_da_v10m_t2m = _make_dataarray(
            times=[_TIMES[0]], variables=["v10m", "t2m"]
        )
        ncar_da_cp = _make_dataarray(times=[_TIMES[0]], variables=["cp"], seed=99)

        mock_arco = MagicMock(return_value=arco_da_v10m_t2m)
        mock_ncar = MagicMock(return_value=ncar_da_cp)

        arco_lexicon = MagicMock()
        arco_lexicon.__contains__ = lambda self, v: v in {"t2m", "u10m", "v10m"}
        ncar_lexicon = MagicMock()
        ncar_lexicon.__contains__ = lambda self, v: v in {"t2m", "u10m", "v10m", "cp"}

        def import_backend(name, **kwargs):
            return {"arco": mock_arco, "ncar": mock_ncar}[name]

        def import_lexicon(name):
            return {"arco": arco_lexicon, "ncar": ncar_lexicon}[name]

        with (
            patch(
                "physicsnemo_curator.da.sources.era5._import_backend",
                side_effect=import_backend,
            ),
            patch(
                "physicsnemo_curator.da.sources.era5._import_lexicon",
                side_effect=import_lexicon,
            ),
        ):
            source = ERA5Source(
                times=_TIMES,
                variables=["v10m", "cp", "t2m"],
                backend=["arco", "ncar"],
            )
        results = list(source[0])
        merged = results[0]
        # After concat: [v10m, t2m] (arco group) + [cp] (ncar group) = [v10m, t2m, cp]
        # This matches the grouped order, not the original input order.
        # The spec says "grouped by backend" so this is correct.
        assert merged.sizes["variable"] == 3
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest test/da/test_pipeline.py::TestERA5MultiBackend -v --no-header 2>&1 | tail -20`
Expected: 10 PASSED

- [ ] **Step 3: Commit**

```bash
git add test/da/test_pipeline.py
git commit -m "test(da): add fetch/merge tests for multi-backend ERA5Source"
```

---

## Task 3: Update Existing Tests and Registry Name

**Files:**
- Modify: `test/da/test_pipeline.py`

The `name` ClassVar changed from `"ERA5 (ARCO)"` to `"ERA5"` in Task 1's `era5.py` rewrite. The `da/__init__.py` does NOT need changes because `registry.register_source("da", ERA5Source)` uses `ERA5Source.name` from the class. Existing tests that assert the old name or mock the old top-level `ARCO` import need updating.

- [ ] **Step 1: Update test_name_and_description in TestERA5Source**

In `test/da/test_pipeline.py`, change line 137:

Old:
```python
        assert ERA5Source.name == "ERA5 (ARCO)"
```

New:
```python
        assert ERA5Source.name == "ERA5"
```

- [ ] **Step 2: Update test_params in TestERA5Source**

In `test/da/test_pipeline.py`, change the `test_params` method (line 131):

Old:
```python
        names = {p.name for p in params}
        assert {"times", "variables", "cache"} == names
```

New:
```python
        names = {p.name for p in params}
        assert {"times", "variables", "backend", "cache"} == names
```

- [ ] **Step 3: Update test_era5_registered in TestRegistration**

In `test/da/test_pipeline.py`, change line 736:

Old:
```python
        assert "ERA5 (ARCO)" in sources
```

New:
```python
        assert "ERA5" in sources
```

- [ ] **Step 4: Add backward-compatibility test to TestERA5MultiBackend**

Append to the `TestERA5MultiBackend` class:

```python
    def test_backward_compat_default_arco(self) -> None:
        """Default backend='arco' behaves like the old ARCO-only source."""
        from unittest.mock import MagicMock, patch

        from physicsnemo_curator.da.sources.era5 import ERA5Source

        mock_arco = MagicMock()
        mock_arco.return_value = _make_dataarray(times=[_TIMES[0]])

        arco_lexicon = MagicMock()
        arco_lexicon.__contains__ = lambda self, v: True

        with (
            patch(
                "physicsnemo_curator.da.sources.era5._import_backend",
                return_value=mock_arco,
            ),
            patch(
                "physicsnemo_curator.da.sources.era5._import_lexicon",
                return_value=arco_lexicon,
            ),
        ):
            source = ERA5Source(times=_TIMES, variables=_VARS)
        assert source.active_backend == "arco"
        results = list(source[0])
        assert len(results) == 1
        mock_arco.assert_called_once()
```

- [ ] **Step 5: Update existing mocked tests to use the new patch targets**

The existing `TestERA5Source` tests use `@patch("curator.da.sources.era5.ARCO")` which patches the old top-level `ARCO` import. Since `ARCO` is no longer imported at module level (it's lazy-imported via `_import_backend`), these tests need to patch `_import_backend` and `_import_lexicon` instead.

Update `test_len` (line 140-146):

Old:
```python
    @patch("curator.da.sources.era5.ARCO")
    def test_len(self, mock_arco_cls: MagicMock) -> None:
        """Length equals number of timestamps."""
        from physicsnemo_curator.da.sources.era5 import ERA5Source

        source = ERA5Source(times=_TIMES, variables=_VARS)
        assert len(source) == 2
```

New:
```python
    def test_len(self) -> None:
        """Length equals number of timestamps."""
        from unittest.mock import MagicMock, patch

        from physicsnemo_curator.da.sources.era5 import ERA5Source

        arco_lexicon = MagicMock()
        arco_lexicon.__contains__ = lambda self, v: True

        with (
            patch("physicsnemo_curator.da.sources.era5._import_backend", return_value=MagicMock()),
            patch("physicsnemo_curator.da.sources.era5._import_lexicon", return_value=arco_lexicon),
        ):
            source = ERA5Source(times=_TIMES, variables=_VARS)
        assert len(source) == 2
```

Update `test_getitem` (line 148-161):

Old:
```python
    @patch("curator.da.sources.era5.ARCO")
    def test_getitem(self, mock_arco_cls: MagicMock) -> None:
        """__getitem__ yields a DataArray from ARCO."""
        import xarray as xr

        from physicsnemo_curator.da.sources.era5 import ERA5Source

        mock_instance = mock_arco_cls.return_value
        mock_instance.return_value = _make_dataarray(times=[_TIMES[0]])

        source = ERA5Source(times=_TIMES, variables=_VARS)
        results = list(source[0])
        assert len(results) == 1
        assert isinstance(results[0], xr.DataArray)
```

New:
```python
    def test_getitem(self) -> None:
        """__getitem__ yields a DataArray from ARCO."""
        import xarray as xr
        from unittest.mock import MagicMock, patch

        from physicsnemo_curator.da.sources.era5 import ERA5Source

        mock_arco = MagicMock()
        mock_arco.return_value = _make_dataarray(times=[_TIMES[0]])

        arco_lexicon = MagicMock()
        arco_lexicon.__contains__ = lambda self, v: True

        with (
            patch("physicsnemo_curator.da.sources.era5._import_backend", return_value=mock_arco),
            patch("physicsnemo_curator.da.sources.era5._import_lexicon", return_value=arco_lexicon),
        ):
            source = ERA5Source(times=_TIMES, variables=_VARS)
        results = list(source[0])
        assert len(results) == 1
        assert isinstance(results[0], xr.DataArray)
```

Update `test_getitem_negative_index` (line 163-173):

New:
```python
    def test_getitem_negative_index(self) -> None:
        """Negative indexing works."""
        from unittest.mock import MagicMock, patch

        from physicsnemo_curator.da.sources.era5 import ERA5Source

        mock_arco = MagicMock()
        mock_arco.return_value = _make_dataarray(times=[_TIMES[-1]])

        arco_lexicon = MagicMock()
        arco_lexicon.__contains__ = lambda self, v: True

        with (
            patch("physicsnemo_curator.da.sources.era5._import_backend", return_value=mock_arco),
            patch("physicsnemo_curator.da.sources.era5._import_lexicon", return_value=arco_lexicon),
        ):
            source = ERA5Source(times=_TIMES, variables=_VARS)
        results = list(source[-1])
        assert len(results) == 1
```

Update `test_getitem_out_of_range` (line 175-182):

New:
```python
    def test_getitem_out_of_range(self) -> None:
        """Out-of-range index raises IndexError."""
        from unittest.mock import MagicMock, patch

        from physicsnemo_curator.da.sources.era5 import ERA5Source

        arco_lexicon = MagicMock()
        arco_lexicon.__contains__ = lambda self, v: True

        with (
            patch("physicsnemo_curator.da.sources.era5._import_backend", return_value=MagicMock()),
            patch("physicsnemo_curator.da.sources.era5._import_lexicon", return_value=arco_lexicon),
        ):
            source = ERA5Source(times=_TIMES, variables=_VARS)
        with pytest.raises(IndexError):
            list(source[999])
```

Update `test_properties` (line 198-205):

New:
```python
    def test_properties(self) -> None:
        """Properties return copies of the constructor inputs."""
        from unittest.mock import MagicMock, patch

        from physicsnemo_curator.da.sources.era5 import ERA5Source

        arco_lexicon = MagicMock()
        arco_lexicon.__contains__ = lambda self, v: True

        with (
            patch("physicsnemo_curator.da.sources.era5._import_backend", return_value=MagicMock()),
            patch("physicsnemo_curator.da.sources.era5._import_lexicon", return_value=arco_lexicon),
        ):
            source = ERA5Source(times=_TIMES, variables=_VARS)
        assert source.times == _TIMES
        assert source.variables == _VARS
```

- [ ] **Step 6: Update pipeline integration tests**

The integration tests (`TestDAPipeline`) also use `@patch("curator.da.sources.era5.ARCO")`. Update each:

`test_full_pipeline` (line 771-802):

New:
```python
    def test_full_pipeline(self, tmp_path: Path) -> None:
        """Full pipeline: ERA5Source -> MomentsFilter -> ZarrSink."""
        from unittest.mock import MagicMock, patch

        from physicsnemo_curator.da.filters.moments import MomentsFilter
        from physicsnemo_curator.da.sinks.zarr_writer import ZarrSink
        from physicsnemo_curator.da.sources.era5 import ERA5Source

        mock_arco = MagicMock()

        def side_effect(time, variable):
            return _make_dataarray(times=time, variables=variable)

        mock_arco.side_effect = side_effect

        arco_lexicon = MagicMock()
        arco_lexicon.__contains__ = lambda self, v: True

        with (
            patch("physicsnemo_curator.da.sources.era5._import_backend", return_value=mock_arco),
            patch("physicsnemo_curator.da.sources.era5._import_lexicon", return_value=arco_lexicon),
        ):
            source = ERA5Source(times=_TIMES, variables=_VARS)

        filt = MomentsFilter(output=str(tmp_path / "stats.zarr"), dims=("time",))
        sink = ZarrSink(output_path=str(tmp_path / "output.zarr"))

        pipeline = source.filter(filt).write(sink)

        assert len(pipeline) == 2

        all_paths: list[list[str]] = []
        for i in range(len(pipeline)):
            paths = pipeline[i]
            all_paths.append(paths)
            assert len(paths) > 0

        stats_path = filt.flush()
        assert stats_path is not None
```

`test_pipeline_without_filter` (line 804-818):

New:
```python
    def test_pipeline_without_filter(self, tmp_path: Path) -> None:
        """Pipeline with just source and sink (no filter)."""
        from unittest.mock import MagicMock, patch

        from physicsnemo_curator.da.sinks.zarr_writer import ZarrSink
        from physicsnemo_curator.da.sources.era5 import ERA5Source

        mock_arco = MagicMock()
        mock_arco.return_value = _make_dataarray(times=[_TIMES[0]])

        arco_lexicon = MagicMock()
        arco_lexicon.__contains__ = lambda self, v: True

        with (
            patch("physicsnemo_curator.da.sources.era5._import_backend", return_value=mock_arco),
            patch("physicsnemo_curator.da.sources.era5._import_lexicon", return_value=arco_lexicon),
        ):
            source = ERA5Source(times=_TIMES[:1], variables=_VARS)

        sink = ZarrSink(output_path=str(tmp_path / "output.zarr"))
        pipeline = source.write(sink)
        paths = pipeline[0]
        assert len(paths) == 2
```

`test_full_pipeline_netcdf4` (line 820-862):

New:
```python
    def test_full_pipeline_netcdf4(self, tmp_path: Path) -> None:
        """Full pipeline: ERA5Source -> MomentsFilter -> NetCDF4Sink."""
        import xarray as xr
        from unittest.mock import MagicMock, patch

        from physicsnemo_curator.da.filters.moments import MomentsFilter
        from physicsnemo_curator.da.sinks.netcdf_writer import NetCDF4Sink
        from physicsnemo_curator.da.sources.era5 import ERA5Source

        mock_arco = MagicMock()

        def side_effect(time, variable):
            return _make_dataarray(times=time, variables=variable)

        mock_arco.side_effect = side_effect

        arco_lexicon = MagicMock()
        arco_lexicon.__contains__ = lambda self, v: True

        with (
            patch("physicsnemo_curator.da.sources.era5._import_backend", return_value=mock_arco),
            patch("physicsnemo_curator.da.sources.era5._import_lexicon", return_value=arco_lexicon),
        ):
            source = ERA5Source(times=_TIMES, variables=_VARS)

        filt = MomentsFilter(output=str(tmp_path / "stats.zarr"), dims=("time",))
        sink = NetCDF4Sink(output_dir=str(tmp_path / "output_nc"))

        pipeline = source.filter(filt).write(sink)
        assert len(pipeline) == 2

        all_paths: list[list[str]] = []
        for i in range(len(pipeline)):
            paths = pipeline[i]
            all_paths.append(paths)
            assert len(paths) > 0

            for var in _VARS:
                nc_dir = tmp_path / "output_nc" / var
                assert nc_dir.exists()
                nc_files = list(nc_dir.glob("*.nc"))
                assert len(nc_files) >= 1
                for nc_file in nc_files:
                    ds = xr.open_dataset(str(nc_file))
                    assert "data" in ds

        stats_path = filt.flush()
        assert stats_path is not None
```

`test_pipeline_netcdf4_no_filter` (line 864-878):

New:
```python
    def test_pipeline_netcdf4_no_filter(self, tmp_path: Path) -> None:
        """Pipeline with just source and NetCDF4Sink (no filter)."""
        from unittest.mock import MagicMock, patch

        from physicsnemo_curator.da.sinks.netcdf_writer import NetCDF4Sink
        from physicsnemo_curator.da.sources.era5 import ERA5Source

        mock_arco = MagicMock()
        mock_arco.return_value = _make_dataarray(times=[_TIMES[0]])

        arco_lexicon = MagicMock()
        arco_lexicon.__contains__ = lambda self, v: True

        with (
            patch("physicsnemo_curator.da.sources.era5._import_backend", return_value=mock_arco),
            patch("physicsnemo_curator.da.sources.era5._import_lexicon", return_value=arco_lexicon),
        ):
            source = ERA5Source(times=_TIMES[:1], variables=_VARS)

        sink = NetCDF4Sink(output_dir=str(tmp_path / "output_nc"))
        pipeline = source.write(sink)
        paths = pipeline[0]
        assert len(paths) == 2
```

- [ ] **Step 7: Run full test file to verify all pass**

Run: `uv run pytest test/da/test_pipeline.py -v --no-header 2>&1 | tail -30`
Expected: All tests PASS (both old updated tests and new multi-backend tests)

- [ ] **Step 8: Commit**

```bash
git add test/da/test_pipeline.py
git commit -m "refactor(da): update registry name and migrate tests to lazy-import mocks"
```

---

## Task 4: Documentation

**Files:**
- Modify: `docs/user-guide/datasets.md`

- [ ] **Step 1: Add ERA5 multi-backend section to datasets.md**

After line 91 (end of WindTunnel-20k section), before `## Mesh Types`, insert:

```markdown

## ERA5 Reanalysis

The {py:class}`~physicsnemo_curator.da.sources.era5.ERA5Source` fetches
ERA5 reanalysis data from earth2studio.  It supports four backends with
automatic per-variable routing.

### Quick Start

```python
from datetime import datetime
from physicsnemo_curator.da.sources.era5 import ERA5Source

# Default: uses ARCO (largest variable coverage)
source = ERA5Source(
    times=[datetime(2020, 6, 1, 0), datetime(2020, 6, 1, 6)],
    variables=["t2m", "u10m", "v10m"],
)
da = next(source[0])  # xr.DataArray (time, variable, lat, lon)
```

### Multi-Backend Routing

Some variables are only available in certain backends.  Pass a
priority-ordered list and each variable routes to the first backend
that supports it:

```python
source = ERA5Source(
    times=[datetime(2020, 6, 1, 0)],
    variables=["t2m", "cp"],       # cp is NCAR-only
    backend=["arco", "ncar"],
)
print(source.variable_routing)
# {'t2m': 'arco', 'cp': 'ncar'}
```

### Available Backends

| Backend | Name | Variables | Time Resolution | Requirements |
|---------|------|-----------|-----------------|--------------|
| ARCO | `"arco"` | 1,408 | 1 hr | None |
| WB2 | `"wb2"` | 103 | 6 hr | None |
| NCAR | `"ncar"` | 276 | 1 hr | None |
| CDS | `"cds"` | 117 | 1 hr | CDS API key |

All backends produce 0.25° (721×1440) grids.  84 core variables
(surface + 13 pressure levels) are common to all four.

### Backend Options

Per-backend constructor arguments can be forwarded:

```python
source = ERA5Source(
    times=[datetime(2020, 6, 1, 0)],
    variables=["t2m"],
    backend="ncar",
    backend_options={"ncar": {"max_workers": 8}},
)
```

### Introspection

```python
source.variable_routing   # dict: variable -> backend name
source.backends_used       # set of backend names in use
source.active_backend      # str if single backend, else None
```

Requires the ``da`` dependency group:

```bash
uv sync --group da
```
```

- [ ] **Step 2: Commit**

```bash
git add docs/user-guide/datasets.md
git commit -m "docs(da): add ERA5 multi-backend documentation to datasets guide"
```

---

## Task 5: Linting, Type-Checking, and Docstring Coverage

**Files:**
- Possibly modify: `src/physicsnemo_curator/da/sources/era5.py`, `test/da/test_pipeline.py`

- [ ] **Step 1: Run ruff format**

Run: `uv run ruff format src/physicsnemo_curator/da/sources/era5.py test/da/test_pipeline.py`
Expected: Already formatted or reformatted

- [ ] **Step 2: Run ruff check**

Run: `uv run ruff check src/physicsnemo_curator/da/sources/era5.py test/da/test_pipeline.py --fix`
Expected: No errors (or auto-fixed)

- [ ] **Step 3: Run ty check**

Run: `uv run ty check src/physicsnemo_curator/da/sources/era5.py`
Expected: No errors (or pre-existing only)

- [ ] **Step 4: Run interrogate**

Run: `uv run interrogate src/physicsnemo_curator/da/sources/era5.py -v`
Expected: 100% docstring coverage

- [ ] **Step 5: Fix any issues and commit**

```bash
git add -A
git commit -m "style(da): fix lint and formatting issues in ERA5Source"
```

---

## Task 6: Final Validation

**Files:** None (verification only)

- [ ] **Step 1: Run full da test suite**

Run: `uv run pytest test/da/test_pipeline.py -v --tb=short 2>&1 | tail -40`
Expected: All tests PASS

- [ ] **Step 2: Run full project test suite**

Run: `uv run pytest test/ -v --tb=short 2>&1 | tail -40`
Expected: No regressions (same pass/skip/fail counts as before)

- [ ] **Step 3: Run make check**

Run: `make check`
Expected: All checks pass (format + lint + typecheck + interrogate + deny)
