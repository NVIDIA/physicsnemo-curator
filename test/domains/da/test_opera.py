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

"""Tests for OPERASource."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.requires("da")


def _make_opera_dataarray() -> object:
    import numpy as np
    import xarray as xr

    ysize, xsize = 4, 5
    lat = np.ones((ysize, xsize), dtype=np.float32)
    lon = np.zeros((ysize, xsize), dtype=np.float32)
    data = np.zeros((1, 1, ysize, xsize), dtype=np.float32)
    return xr.DataArray(
        data=data,
        dims=["time", "variable", "y", "x"],
        coords={
            "time": [np.datetime64("2024-09-01T00:00")],
            "variable": ["refc"],
            "y": np.arange(ysize),
            "x": np.arange(xsize),
        },
    ).assign_coords({"_lat": (("y", "x"), lat), "_lon": (("y", "x"), lon)})


class TestOPERASource:
    """Unit tests for OPERASource."""

    def test_params_list(self) -> None:
        """params() exposes the constructor arguments for the CLI."""
        from physicsnemo_curator.domains.da.sources.opera import OPERASource

        names = [p.name for p in OPERASource.params()]
        assert "times" in names
        assert "variables" in names
        assert "cache" in names

    def test_name_and_description(self) -> None:
        """Registry metadata is populated."""
        from physicsnemo_curator.domains.da.sources.opera import OPERASource

        assert OPERASource.name == "OPERA"
        assert len(OPERASource.description) > 0

    @patch("physicsnemo_curator.domains.da.sources.opera._import_lexicon")
    def test_unknown_variable_raises(self, mock_lexicon: MagicMock) -> None:
        """An identifier absent from the lexicon is rejected."""
        from physicsnemo_curator.domains.da.sources.opera import OPERASource

        mock_lexicon.return_value = {"refc": "DBZH"}
        with pytest.raises(ValueError, match="OPERALexicon"):
            OPERASource(times=[datetime(2024, 9, 1)], variables=["not_a_var"])

    def test_empty_times_raises(self) -> None:
        """An empty time list is rejected."""
        from physicsnemo_curator.domains.da.sources.opera import OPERASource

        with pytest.raises(ValueError, match="times must be"):
            OPERASource(times=[], variables=["refc"])

    def test_empty_variables_raises(self) -> None:
        """An empty variable list is rejected."""
        from physicsnemo_curator.domains.da.sources.opera import OPERASource

        with pytest.raises(ValueError, match="variables must be"):
            OPERASource(times=[datetime(2024, 9, 1)], variables=[])

    @patch("physicsnemo_curator.domains.da.sources.opera._import_lexicon")
    def test_len(self, mock_lexicon: MagicMock) -> None:
        """Length is one pipeline index per requested time."""
        from physicsnemo_curator.domains.da.sources.opera import OPERASource

        mock_lexicon.return_value = {"refc": "DBZH"}
        times = [datetime(2024, 9, 1, 0, 0), datetime(2024, 9, 1, 0, 15)]
        assert len(OPERASource(times=times, variables=["refc"])) == 2

    @patch("physicsnemo_curator.domains.da.sources.opera._import_opera")
    @patch("physicsnemo_curator.domains.da.sources.opera._import_lexicon")
    def test_getitem_returns_dataarray(
        self,
        mock_lexicon: MagicMock,
        mock_import: MagicMock,
    ) -> None:
        from physicsnemo_curator.domains.da.sources.opera import OPERASource

        mock_lexicon.return_value = {"refc": "DBZH"}
        backend = MagicMock(return_value=_make_opera_dataarray())
        mock_import.return_value = backend

        source = OPERASource(times=[datetime(2024, 9, 1, 0, 0)], variables=["refc"])
        da = next(source[0])
        assert da.dims == ("time", "variable", "y", "x")
        backend.assert_called_once()

    @patch("physicsnemo_curator.domains.da.sources.opera._import_opera")
    @patch("physicsnemo_curator.domains.da.sources.opera._import_lexicon")
    def test_index_out_of_bounds(
        self,
        mock_lexicon: MagicMock,
        mock_import: MagicMock,
    ) -> None:
        from physicsnemo_curator.domains.da.sources.opera import OPERASource

        mock_lexicon.return_value = {"refc": "DBZH"}
        mock_import.return_value = MagicMock()
        source = OPERASource(times=[datetime(2024, 9, 1)], variables=["refc"])
        with pytest.raises(IndexError):
            next(source[1])

    @patch("physicsnemo_curator.domains.da.sources.opera._import_opera")
    @patch("physicsnemo_curator.domains.da.sources.opera._import_lexicon")
    def test_negative_index(
        self,
        mock_lexicon: MagicMock,
        mock_import: MagicMock,
    ) -> None:
        from physicsnemo_curator.domains.da.sources.opera import OPERASource

        mock_lexicon.return_value = {"refc": "DBZH"}
        backend = MagicMock(return_value=_make_opera_dataarray())
        mock_import.return_value = backend

        times = [datetime(2024, 9, 1, 0, 0), datetime(2024, 9, 1, 0, 5)]
        source = OPERASource(times=times, variables=["refc"])
        next(source[-1])
        assert backend.call_args.kwargs["time"] == [times[-1]]

    @patch("physicsnemo_curator.domains.da.sources.opera._import_opera")
    @patch("physicsnemo_curator.domains.da.sources.opera._import_lexicon")
    def test_pickle_roundtrip(
        self,
        mock_lexicon: MagicMock,
        mock_import: MagicMock,
    ) -> None:
        """Source survives pickling so worker processes can rebuild the backend."""
        import pickle

        from physicsnemo_curator.domains.da.sources.opera import OPERASource

        mock_lexicon.return_value = {"refc": "DBZH"}
        mock_import.return_value = MagicMock()

        source = OPERASource(times=[datetime(2024, 9, 1)], variables=["refc"], cache=False)
        restored = pickle.loads(pickle.dumps(source))

        assert len(restored) == len(source)
        assert restored.times == source.times
        assert restored.variables == source.variables
        assert restored.cache == source.cache

    def test_source_registered(self) -> None:
        import physicsnemo_curator.domains.da  # noqa: F401 — triggers registration
        from physicsnemo_curator.core.registry import registry

        names = [s.name for s in registry.list_sources("da")]
        assert "OPERA" in names
