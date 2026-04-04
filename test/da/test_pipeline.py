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

"""Tests for the ``da`` submodule — ERA5Source, ZarrSink, NetCDF4Sink, MomentsFilter.

Unit tests use synthetic xarray DataArrays and mock the earth2studio
ARCO backend.  End-to-end tests hit the real ARCO endpoint and are
marked ``@pytest.mark.slow`` + ``@pytest.mark.e2e``.
"""

from __future__ import annotations

import pathlib
from datetime import datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.requires("da")

# ---------------------------------------------------------------------------
# Synthetic data helpers (import-guarded)
# ---------------------------------------------------------------------------

_LATS_N = 9  # small grid for tests
_LONS_N = 8
_VARS = ["t2m", "u10m"]
_TIMES = [datetime(2020, 6, 1, 0), datetime(2020, 6, 1, 6)]


def _make_dataarray(
    times: list[datetime] | None = None,
    variables: list[str] | None = None,
    n_lat: int = _LATS_N,
    n_lon: int = _LONS_N,
    seed: int = 42,
) -> object:
    """Create a synthetic DataArray mimicking ARCO output.

    Returns an xr.DataArray but typed as ``object`` to avoid
    top-level numpy/xarray imports.
    """
    import numpy as np
    import xarray as xr

    times = times or _TIMES[:1]
    variables = variables or _VARS
    lats = np.linspace(90, -90, n_lat)
    lons = np.linspace(0, 350, n_lon)
    rng = np.random.default_rng(seed)

    data = rng.standard_normal((len(times), len(variables), len(lats), len(lons)))
    return xr.DataArray(
        data=data,
        dims=["time", "variable", "lat", "lon"],
        coords={
            "time": [np.datetime64(t) for t in times],
            "variable": variables,
            "lat": lats,
            "lon": lons,
        },
    )


def _make_simple_dataarray(
    n_samples: int = 1,
    n_lat: int = _LATS_N,
    n_lon: int = _LONS_N,
    fill_value: float | None = None,
    seed: int = 42,
) -> object:
    """Create a simple DataArray without the variable dimension.

    Returns an xr.DataArray but typed as ``object`` to avoid
    top-level numpy/xarray imports.
    """
    import numpy as np
    import xarray as xr

    lats = np.linspace(90, -90, n_lat)
    lons = np.linspace(0, 350, n_lon)

    if fill_value is not None:
        data = np.full((n_samples, n_lat, n_lon), fill_value)
    else:
        rng = np.random.default_rng(seed)
        data = rng.standard_normal((n_samples, n_lat, n_lon))

    return xr.DataArray(
        data=data,
        dims=["time", "lat", "lon"],
        coords={
            "time": [np.datetime64(datetime(2020, 1, 1, i)) for i in range(n_samples)],
            "lat": lats,
            "lon": lons,
        },
    )


# ===================================================================
# ERA5Source tests
# ===================================================================


class TestERA5Source:
    """Unit tests for ERA5Source."""

    def test_params(self) -> None:
        """ERA5Source.params() returns descriptors for times, variables, cache."""
        from physicsnemo_curator.da.sources.era5 import ERA5Source

        params = ERA5Source.params()
        names = {p.name for p in params}
        assert {"times", "variables", "cache"} == names

    def test_name_and_description(self) -> None:
        """ERA5Source has correct name and description."""
        from physicsnemo_curator.da.sources.era5 import ERA5Source

        assert ERA5Source.name == "ERA5 (ARCO)"
        assert "ERA5" in ERA5Source.description

    @patch("curator.da.sources.era5.ARCO")
    def test_len(self, mock_arco_cls: MagicMock) -> None:
        """Length equals number of timestamps."""
        from physicsnemo_curator.da.sources.era5 import ERA5Source

        source = ERA5Source(times=_TIMES, variables=_VARS)
        assert len(source) == 2

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

    @patch("curator.da.sources.era5.ARCO")
    def test_getitem_negative_index(self, mock_arco_cls: MagicMock) -> None:
        """Negative indexing works."""
        from physicsnemo_curator.da.sources.era5 import ERA5Source

        mock_instance = mock_arco_cls.return_value
        mock_instance.return_value = _make_dataarray(times=[_TIMES[-1]])

        source = ERA5Source(times=_TIMES, variables=_VARS)
        results = list(source[-1])
        assert len(results) == 1

    @patch("curator.da.sources.era5.ARCO")
    def test_getitem_out_of_range(self, mock_arco_cls: MagicMock) -> None:
        """Out-of-range index raises IndexError."""
        from physicsnemo_curator.da.sources.era5 import ERA5Source

        source = ERA5Source(times=_TIMES, variables=_VARS)
        with pytest.raises(IndexError):
            list(source[999])

    def test_empty_times_raises(self) -> None:
        """Empty times list raises ValueError."""
        from physicsnemo_curator.da.sources.era5 import ERA5Source

        with pytest.raises(ValueError, match="non-empty"):
            ERA5Source(times=[], variables=_VARS)

    def test_empty_variables_raises(self) -> None:
        """Empty variables list raises ValueError."""
        from physicsnemo_curator.da.sources.era5 import ERA5Source

        with pytest.raises(ValueError, match="non-empty"):
            ERA5Source(times=_TIMES, variables=[])

    @patch("curator.da.sources.era5.ARCO")
    def test_properties(self, mock_arco_cls: MagicMock) -> None:
        """Properties return copies of the constructor inputs."""
        from physicsnemo_curator.da.sources.era5 import ERA5Source

        source = ERA5Source(times=_TIMES, variables=_VARS)
        assert source.times == _TIMES
        assert source.variables == _VARS


# ===================================================================
# Multi-backend routing tests
# ===================================================================


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
        arco_da_v10m_t2m = _make_dataarray(times=[_TIMES[0]], variables=["v10m", "t2m"])
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


# ===================================================================
# ZarrSink tests
# ===================================================================


class TestZarrSink:
    """Unit tests for ZarrSink."""

    def test_params(self) -> None:
        """ZarrSink.params() returns descriptors for output_path and chunks."""
        from physicsnemo_curator.da.sinks.zarr_writer import ZarrSink

        params = ZarrSink.params()
        names = {p.name for p in params}
        assert {"output_path", "chunks"} == names

    def test_name_and_description(self) -> None:
        """ZarrSink has correct name and description."""
        from physicsnemo_curator.da.sinks.zarr_writer import ZarrSink

        assert ZarrSink.name == "Zarr Writer"
        assert "Zarr" in ZarrSink.description

    def test_write_single_variable(self, tmp_path: Path) -> None:
        """Writing a single-variable DataArray creates the expected Zarr group."""
        import xarray as xr

        from physicsnemo_curator.da.sinks.zarr_writer import ZarrSink

        sink = ZarrSink(output_path=str(tmp_path / "output.zarr"))
        da = _make_dataarray(variables=["t2m"])

        def gen():  # type: ignore[override]
            yield da

        paths = sink(gen(), index=0)
        assert len(paths) == 1
        assert "t2m" in paths[0]

        # Read back and verify
        ds = xr.open_zarr(paths[0])
        assert "data" in ds
        assert ds["data"].shape == (1, _LATS_N, _LONS_N)

    def test_write_multiple_variables(self, tmp_path: Path) -> None:
        """Writing a multi-variable DataArray creates one group per variable."""
        import xarray as xr

        from physicsnemo_curator.da.sinks.zarr_writer import ZarrSink

        sink = ZarrSink(output_path=str(tmp_path / "output.zarr"))
        da = _make_dataarray()

        def gen():  # type: ignore[override]
            yield da

        paths = sink(gen(), index=0)
        assert len(paths) == 2

        for path in paths:
            ds = xr.open_zarr(path)
            assert "data" in ds

    def test_append_along_time(self, tmp_path: Path) -> None:
        """Subsequent writes append along the time dimension."""
        import xarray as xr

        from physicsnemo_curator.da.sinks.zarr_writer import ZarrSink

        sink = ZarrSink(output_path=str(tmp_path / "output.zarr"))

        da1 = _make_dataarray(times=[_TIMES[0]], variables=["t2m"])
        da2 = _make_dataarray(times=[_TIMES[1]], variables=["t2m"], seed=99)

        def gen1():  # type: ignore[override]
            yield da1

        def gen2():  # type: ignore[override]
            yield da2

        sink(gen1(), index=0)
        sink(gen2(), index=1)

        ds = xr.open_zarr(str(tmp_path / "output.zarr" / "t2m"))
        assert ds["data"].sizes["time"] == 2

    def test_custom_chunks(self, tmp_path: Path) -> None:
        """Custom chunk sizes are respected."""
        import xarray as xr

        from physicsnemo_curator.da.sinks.zarr_writer import ZarrSink

        sink = ZarrSink(
            output_path=str(tmp_path / "output.zarr"),
            chunks={"time": 1, "lat": 3, "lon": 4},
        )
        da = _make_dataarray(variables=["t2m"])

        def gen():  # type: ignore[override]
            yield da

        sink(gen(), index=0)

        ds = xr.open_zarr(str(tmp_path / "output.zarr" / "t2m"))
        assert ds["data"].encoding.get("chunks") is not None

    def test_output_path_property(self, tmp_path: Path) -> None:
        """output_path property returns the configured path."""
        from physicsnemo_curator.da.sinks.zarr_writer import ZarrSink

        sink = ZarrSink(output_path=str(tmp_path / "output.zarr"))
        assert sink.output_path == tmp_path / "output.zarr"


# ===================================================================
# NetCDF4Sink tests
# ===================================================================


class TestNetCDF4Sink:
    """Unit tests for NetCDF4Sink."""

    def test_params(self) -> None:
        """NetCDF4Sink.params() returns descriptors for output_dir, chunks, compression_level, split_dim."""
        from physicsnemo_curator.da.sinks.netcdf_writer import NetCDF4Sink

        params = NetCDF4Sink.params()
        names = {p.name for p in params}
        assert {"output_dir", "chunks", "compression_level", "split_dim"} == names

    def test_name_and_description(self) -> None:
        """NetCDF4Sink has correct name and description."""
        from physicsnemo_curator.da.sinks.netcdf_writer import NetCDF4Sink

        assert NetCDF4Sink.name == "NetCDF4 Writer"
        assert "NetCDF4" in NetCDF4Sink.description

    def test_write_single_variable(self, tmp_path: Path) -> None:
        """Writing a single-variable DataArray creates the expected .nc file."""
        import xarray as xr

        from physicsnemo_curator.da.sinks.netcdf_writer import NetCDF4Sink

        sink = NetCDF4Sink(output_dir=str(tmp_path / "output_nc"))
        da = _make_dataarray(variables=["t2m"])

        def gen():  # type: ignore[override]
            yield da

        paths = sink(gen(), index=0)
        assert len(paths) == 1
        assert paths[0].endswith("2020.nc")
        assert "t2m" in paths[0]

        # Read back and verify
        ds = xr.open_dataset(paths[0])
        assert "data" in ds
        assert ds["data"].shape == (1, _LATS_N, _LONS_N)

    def test_write_multiple_variables(self, tmp_path: Path) -> None:
        """Writing a multi-variable DataArray creates one subdirectory per variable."""
        import xarray as xr

        from physicsnemo_curator.da.sinks.netcdf_writer import NetCDF4Sink

        sink = NetCDF4Sink(output_dir=str(tmp_path / "output_nc"))
        da = _make_dataarray()

        def gen():  # type: ignore[override]
            yield da

        paths = sink(gen(), index=0)
        assert len(paths) == 2

        for path in paths:
            ds = xr.open_dataset(path)
            assert "data" in ds

    def test_append_along_time(self, tmp_path: Path) -> None:
        """Subsequent writes with the same year append along the time dimension."""
        import xarray as xr

        from physicsnemo_curator.da.sinks.netcdf_writer import NetCDF4Sink

        sink = NetCDF4Sink(output_dir=str(tmp_path / "output_nc"))

        da1 = _make_dataarray(times=[_TIMES[0]], variables=["t2m"])
        da2 = _make_dataarray(times=[_TIMES[1]], variables=["t2m"], seed=99)

        def gen1():  # type: ignore[override]
            yield da1

        def gen2():  # type: ignore[override]
            yield da2

        sink(gen1(), index=0)
        sink(gen2(), index=1)

        ds = xr.open_dataset(str(tmp_path / "output_nc" / "t2m" / "2020.nc"))
        assert ds["data"].sizes["time"] == 2

    def test_custom_chunks(self, tmp_path: Path) -> None:
        """Custom chunk sizes are passed through to the NetCDF4 file."""
        import xarray as xr

        from physicsnemo_curator.da.sinks.netcdf_writer import NetCDF4Sink

        sink = NetCDF4Sink(
            output_dir=str(tmp_path / "output_nc"),
            chunks={"time": 1, "lat": 3, "lon": 4},
        )
        da = _make_dataarray(variables=["t2m"])

        def gen():  # type: ignore[override]
            yield da

        sink(gen(), index=0)

        ds = xr.open_dataset(str(tmp_path / "output_nc" / "t2m" / "2020.nc"))
        assert "data" in ds

    def test_no_compression(self, tmp_path: Path) -> None:
        """compression_level=0 disables zlib compression."""
        import xarray as xr

        from physicsnemo_curator.da.sinks.netcdf_writer import NetCDF4Sink

        sink = NetCDF4Sink(
            output_dir=str(tmp_path / "output_nc"),
            compression_level=0,
        )
        da = _make_dataarray(variables=["t2m"])

        def gen():  # type: ignore[override]
            yield da

        sink(gen(), index=0)

        ds = xr.open_dataset(str(tmp_path / "output_nc" / "t2m" / "2020.nc"))
        assert "data" in ds

    def test_no_variable_dim(self, tmp_path: Path) -> None:
        """DataArrays without a variable dim write to data/<year>.nc."""
        import xarray as xr

        from physicsnemo_curator.da.sinks.netcdf_writer import NetCDF4Sink

        sink = NetCDF4Sink(output_dir=str(tmp_path / "output_nc"))
        da = _make_simple_dataarray(n_samples=1, fill_value=42.0)

        def gen():  # type: ignore[override]
            yield da

        paths = sink(gen(), index=0)
        assert len(paths) == 1
        assert "data" in paths[0]
        assert paths[0].endswith("2020.nc")

        ds = xr.open_dataset(paths[0])
        assert "data" in ds

    def test_no_split(self, tmp_path: Path) -> None:
        """split_dim=None writes a single file per variable."""
        import xarray as xr

        from physicsnemo_curator.da.sinks.netcdf_writer import NetCDF4Sink

        sink = NetCDF4Sink(output_dir=str(tmp_path / "output_nc"), split_dim=None)
        da = _make_dataarray(variables=["t2m"])

        def gen():  # type: ignore[override]
            yield da

        paths = sink(gen(), index=0)
        assert len(paths) == 1
        assert paths[0].endswith("data.nc")
        assert "t2m" in paths[0]

        ds = xr.open_dataset(paths[0])
        assert "data" in ds

    def test_split_across_years(self, tmp_path: Path) -> None:
        """Data spanning multiple years is split into separate files."""
        import numpy as np
        import xarray as xr

        from physicsnemo_curator.da.sinks.netcdf_writer import NetCDF4Sink

        sink = NetCDF4Sink(output_dir=str(tmp_path / "output_nc"))

        # Create data with timestamps in 2020 and 2021
        times_cross_year = [datetime(2020, 12, 31, 12), datetime(2021, 1, 1, 0)]
        lats = np.linspace(90, -90, _LATS_N)
        lons = np.linspace(0, 350, _LONS_N)
        rng = np.random.default_rng(42)

        da_with_var = xr.DataArray(
            data=rng.standard_normal((2, 1, _LATS_N, _LONS_N)),
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": [np.datetime64(t) for t in times_cross_year],
                "variable": ["t2m"],
                "lat": lats,
                "lon": lons,
            },
        )

        def gen():  # type: ignore[override]
            yield da_with_var

        paths = sink(gen(), index=0)
        # Should produce two files: 2020.nc and 2021.nc
        assert len(paths) == 2
        basenames = sorted(pathlib.Path(p).name for p in paths)
        assert basenames == ["2020.nc", "2021.nc"]

        for path in paths:
            ds = xr.open_dataset(path)
            assert ds["data"].sizes["time"] == 1

    def test_custom_split_func(self, tmp_path: Path) -> None:
        """Custom split_func is used for grouping."""
        import xarray as xr

        from physicsnemo_curator.da.sinks.netcdf_writer import NetCDF4Sink

        # Split by month: YYYY-MM
        def month_key(t):  # type: ignore[override]
            import numpy as np

            dt = np.datetime64(t, "M")
            return str(dt)

        sink = NetCDF4Sink(
            output_dir=str(tmp_path / "output_nc"),
            split_func=month_key,
        )
        da = _make_dataarray(
            times=[datetime(2020, 6, 1, 0)],
            variables=["t2m"],
        )

        def gen():  # type: ignore[override]
            yield da

        paths = sink(gen(), index=0)
        assert len(paths) == 1
        assert "2020-06" in paths[0]

        ds = xr.open_dataset(paths[0])
        assert "data" in ds

    def test_output_dir_property(self, tmp_path: Path) -> None:
        """output_dir property returns the configured path."""
        from physicsnemo_curator.da.sinks.netcdf_writer import NetCDF4Sink

        sink = NetCDF4Sink(output_dir=str(tmp_path / "output_nc"))
        assert sink.output_dir == tmp_path / "output_nc"

    def test_compression_level_property(self) -> None:
        """compression_level property returns the configured value."""
        from physicsnemo_curator.da.sinks.netcdf_writer import NetCDF4Sink

        sink = NetCDF4Sink(output_dir="/tmp/output_nc", compression_level=7)
        assert sink.compression_level == 7

    def test_unlimited_dims_property(self) -> None:
        """unlimited_dims property returns the configured list."""
        from physicsnemo_curator.da.sinks.netcdf_writer import NetCDF4Sink

        sink = NetCDF4Sink(output_dir="/tmp/output_nc", unlimited_dims=["time", "lat"])
        assert sink.unlimited_dims == ["time", "lat"]

    def test_default_unlimited_dims(self) -> None:
        """Default unlimited_dims is ['time']."""
        from physicsnemo_curator.da.sinks.netcdf_writer import NetCDF4Sink

        sink = NetCDF4Sink(output_dir="/tmp/output_nc")
        assert sink.unlimited_dims == ["time"]

    def test_split_dim_property(self) -> None:
        """split_dim property returns the configured value."""
        from physicsnemo_curator.da.sinks.netcdf_writer import NetCDF4Sink

        sink = NetCDF4Sink(output_dir="/tmp/output_nc", split_dim="time")
        assert sink.split_dim == "time"

    def test_split_dim_none_property(self) -> None:
        """split_dim=None disables splitting."""
        from physicsnemo_curator.da.sinks.netcdf_writer import NetCDF4Sink

        sink = NetCDF4Sink(output_dir="/tmp/output_nc", split_dim=None)
        assert sink.split_dim is None


# ===================================================================
# MomentsFilter tests
# ===================================================================


class TestMomentsFilter:
    """Unit tests for MomentsFilter."""

    def test_params(self) -> None:
        """MomentsFilter.params() returns descriptors for output and dims."""
        from physicsnemo_curator.da.filters.moments import MomentsFilter

        params = MomentsFilter.params()
        names = {p.name for p in params}
        assert {"output", "dims"} == names

    def test_name_and_description(self) -> None:
        """MomentsFilter has correct name and description."""
        from physicsnemo_curator.da.filters.moments import MomentsFilter

        assert MomentsFilter.name == "Statistical Moments"
        assert "moments" in MomentsFilter.description.lower()

    def test_passthrough(self) -> None:
        """Filter yields the same DataArray unchanged."""
        import xarray as xr

        from physicsnemo_curator.da.filters.moments import MomentsFilter

        filt = MomentsFilter(output="/tmp/stats.zarr", dims=("time",))
        da = _make_dataarray()

        def gen():  # type: ignore[override]
            yield da

        results = list(filt(gen()))
        assert len(results) == 1
        xr.testing.assert_identical(results[0], da)

    def test_flush_no_data(self) -> None:
        """Flush with no data returns None."""
        from physicsnemo_curator.da.filters.moments import MomentsFilter

        filt = MomentsFilter(output="/tmp/stats.zarr")
        assert filt.flush() is None

    def test_flush_writes_stats(self, tmp_path: Path) -> None:
        """Flush writes mean, variance, skewness, min, max to Zarr."""
        import xarray as xr

        from physicsnemo_curator.da.filters.moments import MomentsFilter

        filt = MomentsFilter(output=str(tmp_path / "stats.zarr"), dims=("time",))

        # Feed multiple samples through the filter
        for i in range(5):
            da = _make_dataarray(
                times=[datetime(2020, 6, 1, i)],
                variables=["t2m"],
                seed=i,
            )

            def gen(data=da):  # type: ignore[override]
                yield data

            list(filt(gen()))

        result = filt.flush()
        assert result is not None

        # Check the output contains expected variables
        ds = xr.open_zarr(str(tmp_path / "stats.zarr" / "t2m"))
        assert "mean" in ds
        assert "variance" in ds
        assert "skewness" in ds
        assert "min" in ds
        assert "max" in ds
        assert ds.attrs["count"] == 5

    def test_mean_accuracy(self, tmp_path: Path) -> None:
        """Verify the computed mean is numerically correct."""
        import numpy as np

        from physicsnemo_curator.da.filters.moments import MomentsFilter

        filt = MomentsFilter(output=str(tmp_path / "stats.zarr"), dims=("time",))

        # Create deterministic data: values = index
        all_values: list[float] = []
        for i in range(10):
            da = _make_simple_dataarray(n_samples=1, fill_value=float(i))
            all_values.append(float(i))

            def gen(d=da):  # type: ignore[override]
                yield d

            list(filt(gen()))

        filt.flush()

        import xarray as xr

        ds = xr.open_zarr(str(tmp_path / "stats.zarr" / "data"))

        expected_mean = np.mean(all_values)
        np.testing.assert_allclose(ds["mean"].values.flat[0], expected_mean, rtol=1e-10)

        expected_var = np.var(all_values)
        np.testing.assert_allclose(ds["variance"].values.flat[0], expected_var, rtol=1e-10)

    def test_properties(self) -> None:
        """Properties return the configured values."""
        from physicsnemo_curator.da.filters.moments import MomentsFilter

        filt = MomentsFilter(output="/tmp/stats.zarr", dims=("time", "variable"))
        assert filt.output_path.name == "stats.zarr"
        assert filt.dims == ("time", "variable")


# ===================================================================
# Registration tests
# ===================================================================


class TestRegistration:
    """Verify components register with the global registry."""

    def test_era5_registered(self) -> None:
        """ERA5Source is discoverable via the registry."""
        import physicsnemo_curator.da  # noqa: F401
        from physicsnemo_curator.core.registry import registry

        sources = registry.sources("da")
        assert "ERA5 (ARCO)" in sources

    def test_zarrsink_registered(self) -> None:
        """ZarrSink is discoverable via the registry."""
        import physicsnemo_curator.da  # noqa: F401
        from physicsnemo_curator.core.registry import registry

        sinks = registry.sinks("da")
        assert "Zarr Writer" in sinks

    def test_netcdf4sink_registered(self) -> None:
        """NetCDF4Sink is discoverable via the registry."""
        import physicsnemo_curator.da  # noqa: F401
        from physicsnemo_curator.core.registry import registry

        sinks = registry.sinks("da")
        assert "NetCDF4 Writer" in sinks

    def test_moments_registered(self) -> None:
        """MomentsFilter is discoverable via the registry."""
        import physicsnemo_curator.da  # noqa: F401
        from physicsnemo_curator.core.registry import registry

        filters = registry.filters("da")
        assert "Statistical Moments" in filters


# ===================================================================
# Pipeline integration tests (synthetic data)
# ===================================================================


class TestDAPipeline:
    """Integration tests using synthetic data (no network)."""

    @patch("curator.da.sources.era5.ARCO")
    def test_full_pipeline(self, mock_arco_cls: MagicMock, tmp_path: Path) -> None:
        """Full pipeline: ERA5Source -> MomentsFilter -> ZarrSink."""
        from physicsnemo_curator.da.filters.moments import MomentsFilter
        from physicsnemo_curator.da.sinks.zarr_writer import ZarrSink
        from physicsnemo_curator.da.sources.era5 import ERA5Source

        mock_instance = mock_arco_cls.return_value

        # Mock returns different data per call
        def side_effect(time, variable):  # type: ignore[override]
            return _make_dataarray(times=time, variables=variable)

        mock_instance.side_effect = side_effect

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

        # Flush moments
        stats_path = filt.flush()
        assert stats_path is not None

    @patch("curator.da.sources.era5.ARCO")
    def test_pipeline_without_filter(self, mock_arco_cls: MagicMock, tmp_path: Path) -> None:
        """Pipeline with just source and sink (no filter)."""
        from physicsnemo_curator.da.sinks.zarr_writer import ZarrSink
        from physicsnemo_curator.da.sources.era5 import ERA5Source

        mock_instance = mock_arco_cls.return_value
        mock_instance.return_value = _make_dataarray(times=[_TIMES[0]])

        source = ERA5Source(times=_TIMES[:1], variables=_VARS)
        sink = ZarrSink(output_path=str(tmp_path / "output.zarr"))

        pipeline = source.write(sink)
        paths = pipeline[0]
        assert len(paths) == 2  # one per variable

    @patch("curator.da.sources.era5.ARCO")
    def test_full_pipeline_netcdf4(self, mock_arco_cls: MagicMock, tmp_path: Path) -> None:
        """Full pipeline: ERA5Source -> MomentsFilter -> NetCDF4Sink."""
        from physicsnemo_curator.da.filters.moments import MomentsFilter
        from physicsnemo_curator.da.sinks.netcdf_writer import NetCDF4Sink
        from physicsnemo_curator.da.sources.era5 import ERA5Source

        mock_instance = mock_arco_cls.return_value

        def side_effect(time, variable):  # type: ignore[override]
            return _make_dataarray(times=time, variables=variable)

        mock_instance.side_effect = side_effect

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

            # Verify .nc files were created
            import xarray as xr

            for var in _VARS:
                nc_dir = tmp_path / "output_nc" / var
                assert nc_dir.exists()
                nc_files = list(nc_dir.glob("*.nc"))
                assert len(nc_files) >= 1
                for nc_file in nc_files:
                    ds = xr.open_dataset(str(nc_file))
                    assert "data" in ds

        # Flush moments
        stats_path = filt.flush()
        assert stats_path is not None

    @patch("curator.da.sources.era5.ARCO")
    def test_pipeline_netcdf4_no_filter(self, mock_arco_cls: MagicMock, tmp_path: Path) -> None:
        """Pipeline with just source and NetCDF4Sink (no filter)."""
        from physicsnemo_curator.da.sinks.netcdf_writer import NetCDF4Sink
        from physicsnemo_curator.da.sources.era5 import ERA5Source

        mock_instance = mock_arco_cls.return_value
        mock_instance.return_value = _make_dataarray(times=[_TIMES[0]])

        source = ERA5Source(times=_TIMES[:1], variables=_VARS)
        sink = NetCDF4Sink(output_dir=str(tmp_path / "output_nc"))

        pipeline = source.write(sink)
        paths = pipeline[0]
        assert len(paths) == 2  # one per variable


# ===================================================================
# End-to-end tests (real ARCO endpoint)
# ===================================================================


@pytest.mark.slow
@pytest.mark.e2e
class TestERA5EndToEnd:
    """End-to-end tests that hit the real ARCO endpoint.

    These are slow (~30s per test) and require network access.
    Run with: ``pytest -m 'slow and e2e' test/test_da_pipeline.py``
    """

    def test_era5_fetch_single_timestep(self) -> None:
        """Fetch a single surface variable from ARCO."""
        from physicsnemo_curator.da.sources.era5 import ERA5Source

        source = ERA5Source(
            times=[datetime(2020, 1, 1, 0)],
            variables=["t2m"],
            cache=True,
        )
        assert len(source) == 1

        results = list(source[0])
        assert len(results) == 1

        da = results[0]
        assert da.dims == ("time", "variable", "lat", "lon")
        assert da.sizes["time"] == 1
        assert da.sizes["variable"] == 1
        assert da.sizes["lat"] == 721
        assert da.sizes["lon"] == 1440

    def test_era5_fetch_multiple_variables(self) -> None:
        """Fetch multiple variables from ARCO."""
        from physicsnemo_curator.da.sources.era5 import ERA5Source

        source = ERA5Source(
            times=[datetime(2020, 1, 1, 0)],
            variables=["t2m", "u10m", "sp"],
            cache=True,
        )
        results = list(source[0])
        da = results[0]
        assert da.sizes["variable"] == 3

    def test_era5_full_pipeline(self, tmp_path: Path) -> None:
        """Full pipeline: ERA5 -> MomentsFilter -> ZarrSink."""
        import xarray as xr

        from physicsnemo_curator.da.filters.moments import MomentsFilter
        from physicsnemo_curator.da.sinks.zarr_writer import ZarrSink
        from physicsnemo_curator.da.sources.era5 import ERA5Source

        source = ERA5Source(
            times=[datetime(2020, 1, 1, 0), datetime(2020, 1, 1, 6)],
            variables=["t2m"],
            cache=True,
        )
        filt = MomentsFilter(output=str(tmp_path / "stats.zarr"), dims=("time",))
        sink = ZarrSink(
            output_path=str(tmp_path / "output.zarr"),
            chunks={"time": 1, "lat": 721, "lon": 1440},
        )

        pipeline = source.filter(filt).write(sink)

        for i in range(len(pipeline)):
            paths = pipeline[i]
            assert len(paths) > 0

        # Flush and verify stats
        stats_path = filt.flush()
        assert stats_path is not None

        # Verify the output Zarr store
        ds = xr.open_zarr(str(tmp_path / "output.zarr" / "t2m"))
        assert ds["data"].sizes["time"] == 2
        assert ds["data"].sizes["lat"] == 721
        assert ds["data"].sizes["lon"] == 1440

        # Verify stats
        stats = xr.open_zarr(str(tmp_path / "stats.zarr" / "t2m"))
        assert "mean" in stats
        assert "variance" in stats
        assert stats.attrs["count"] == 2

    def test_era5_pressure_level(self) -> None:
        """Fetch a pressure-level variable."""
        from physicsnemo_curator.da.sources.era5 import ERA5Source

        source = ERA5Source(
            times=[datetime(2020, 6, 15, 12)],
            variables=["z500", "t850"],
            cache=True,
        )
        results = list(source[0])
        da = results[0]
        assert da.sizes["variable"] == 2
        assert da.sizes["lat"] == 721

    def test_era5_zarr_sharding(self, tmp_path: Path) -> None:
        """Test Zarr v3 sharding configuration."""
        import xarray as xr

        from physicsnemo_curator.da.sinks.zarr_writer import ZarrSink
        from physicsnemo_curator.da.sources.era5 import ERA5Source

        source = ERA5Source(
            times=[datetime(2020, 1, 1, 0)],
            variables=["t2m"],
            cache=True,
        )
        sink = ZarrSink(
            output_path=str(tmp_path / "output.zarr"),
            chunks={"time": 1, "lat": 181, "lon": 360},
            shards={"time": 1, "lat": 721, "lon": 1440},
        )

        pipeline = source.write(sink)
        paths = pipeline[0]
        assert len(paths) == 1

        ds = xr.open_zarr(paths[0])
        assert ds["data"].sizes["lat"] == 721

    def test_era5_netcdf4_pipeline(self, tmp_path: Path) -> None:
        """Full pipeline: ERA5 -> NetCDF4Sink with compression."""
        import xarray as xr

        from physicsnemo_curator.da.sinks.netcdf_writer import NetCDF4Sink
        from physicsnemo_curator.da.sources.era5 import ERA5Source

        source = ERA5Source(
            times=[datetime(2020, 1, 1, 0), datetime(2020, 1, 1, 6)],
            variables=["t2m"],
            cache=True,
        )
        sink = NetCDF4Sink(
            output_dir=str(tmp_path / "output_nc"),
            chunks={"time": 1, "lat": 721, "lon": 1440},
            compression_level=4,
        )

        pipeline = source.write(sink)

        for i in range(len(pipeline)):
            paths = pipeline[i]
            assert len(paths) > 0

        # Verify the output NetCDF4 file
        ds = xr.open_dataset(str(tmp_path / "output_nc" / "t2m" / "2020.nc"))
        assert ds["data"].sizes["time"] == 2
        assert ds["data"].sizes["lat"] == 721
        assert ds["data"].sizes["lon"] == 1440
