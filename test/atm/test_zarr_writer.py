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

"""Tests for AtomicDataZarrSink.

Unit tests use mock AtomicData objects to verify batching, write/append
semantics, and output path handling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    import pathlib


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_mock_atomic_data() -> MagicMock:
    """Create a mock AtomicData instance."""
    mock = MagicMock()
    mock.num_nodes = 10
    mock.atomic_numbers = MagicMock()
    mock.positions = MagicMock()
    return mock


# ---------------------------------------------------------------------------
# Unit tests — metadata and parameters
# ---------------------------------------------------------------------------


@pytest.mark.requires("atm")
class TestAtomicDataZarrSinkUnit:
    """Metadata and parameter tests."""

    def test_params_list(self) -> None:
        from physicsnemo_curator.atm.sinks.zarr_writer import AtomicDataZarrSink

        params = AtomicDataZarrSink.params()
        assert len(params) > 0
        names = [p.name for p in params]
        assert "output_path" in names
        assert "batch_size" in names

    def test_name_and_description(self) -> None:
        from physicsnemo_curator.atm.sinks.zarr_writer import AtomicDataZarrSink

        assert isinstance(AtomicDataZarrSink.name, str)
        assert AtomicDataZarrSink.name == "AtomicData Zarr"
        assert isinstance(AtomicDataZarrSink.description, str)
        assert len(AtomicDataZarrSink.description) > 0

    def test_output_path_property(self, tmp_path: pathlib.Path) -> None:
        from physicsnemo_curator.atm.sinks.zarr_writer import AtomicDataZarrSink

        sink = AtomicDataZarrSink(output_path=str(tmp_path / "out.zarr"))
        assert sink.output_path == tmp_path / "out.zarr"

    def test_batch_size_property(self, tmp_path: pathlib.Path) -> None:
        from physicsnemo_curator.atm.sinks.zarr_writer import AtomicDataZarrSink

        sink = AtomicDataZarrSink(output_path=str(tmp_path / "out.zarr"), batch_size=500)
        assert sink.batch_size == 500

    def test_default_batch_size(self, tmp_path: pathlib.Path) -> None:
        from physicsnemo_curator.atm.sinks.zarr_writer import AtomicDataZarrSink

        sink = AtomicDataZarrSink(output_path=str(tmp_path / "out.zarr"))
        assert sink.batch_size == 1000


# ---------------------------------------------------------------------------
# Write tests (mocked writer)
# ---------------------------------------------------------------------------


@pytest.mark.requires("atm")
class TestAtomicDataZarrSinkWrite:
    """Tests for write/append semantics with mocked AtomicDataZarrWriter."""

    @patch("physicsnemo_curator.atm.sinks.zarr_writer.AtomicDataZarrWriter")
    def test_writes_single_item(self, mock_writer_cls: MagicMock, tmp_path: pathlib.Path) -> None:
        from physicsnemo_curator.atm.sinks.zarr_writer import AtomicDataZarrSink

        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer

        sink = AtomicDataZarrSink(output_path=str(tmp_path / "out.zarr"), batch_size=100)
        items = iter([_make_mock_atomic_data()])
        paths = sink(items, index=0)

        assert len(paths) == 1
        assert paths[0] == str(tmp_path / "out.zarr")
        # First flush should call write (not append) since store doesn't exist.
        mock_writer.write.assert_called_once()

    @patch("physicsnemo_curator.atm.sinks.zarr_writer.AtomicDataZarrWriter")
    def test_writes_multiple_items(self, mock_writer_cls: MagicMock, tmp_path: pathlib.Path) -> None:
        from physicsnemo_curator.atm.sinks.zarr_writer import AtomicDataZarrSink

        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer

        sink = AtomicDataZarrSink(output_path=str(tmp_path / "out.zarr"), batch_size=100)
        items = iter([_make_mock_atomic_data() for _ in range(5)])
        paths = sink(items, index=0)

        assert len(paths) == 1
        mock_writer.write.assert_called_once()

    @patch("physicsnemo_curator.atm.sinks.zarr_writer.AtomicDataZarrWriter")
    def test_batching(self, mock_writer_cls: MagicMock, tmp_path: pathlib.Path) -> None:
        from physicsnemo_curator.atm.sinks.zarr_writer import AtomicDataZarrSink

        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer

        sink = AtomicDataZarrSink(output_path=str(tmp_path / "out.zarr"), batch_size=3)
        items = iter([_make_mock_atomic_data() for _ in range(7)])
        paths = sink(items, index=0)

        assert len(paths) == 1
        # 7 items with batch_size=3: write(3), append(3), append(1)
        mock_writer.write.assert_called_once()
        assert mock_writer.append.call_count == 2

    @patch("physicsnemo_curator.atm.sinks.zarr_writer.AtomicDataZarrWriter")
    def test_append_on_existing_store(self, mock_writer_cls: MagicMock, tmp_path: pathlib.Path) -> None:
        from physicsnemo_curator.atm.sinks.zarr_writer import AtomicDataZarrSink

        # Create the output dir to simulate existing store.
        store_path = tmp_path / "existing.zarr"
        store_path.mkdir()

        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer

        sink = AtomicDataZarrSink(output_path=str(store_path), batch_size=100)
        items = iter([_make_mock_atomic_data()])
        sink(items, index=0)

        # Should call append (not write) since store already exists.
        mock_writer.append.assert_called_once()
        mock_writer.write.assert_not_called()

    @patch("physicsnemo_curator.atm.sinks.zarr_writer.AtomicDataZarrWriter")
    def test_second_index_appends(self, mock_writer_cls: MagicMock, tmp_path: pathlib.Path) -> None:
        from physicsnemo_curator.atm.sinks.zarr_writer import AtomicDataZarrSink

        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer

        sink = AtomicDataZarrSink(output_path=str(tmp_path / "out.zarr"), batch_size=100)

        # First index: should write.
        sink(iter([_make_mock_atomic_data()]), index=0)
        mock_writer.write.assert_called_once()

        # Second index: should append (store now exists in sink's state).
        sink(iter([_make_mock_atomic_data()]), index=1)
        mock_writer.append.assert_called_once()

    @patch("physicsnemo_curator.atm.sinks.zarr_writer.AtomicDataZarrWriter")
    def test_empty_iterator(self, mock_writer_cls: MagicMock, tmp_path: pathlib.Path) -> None:
        from physicsnemo_curator.atm.sinks.zarr_writer import AtomicDataZarrSink

        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer

        sink = AtomicDataZarrSink(output_path=str(tmp_path / "out.zarr"))
        paths = sink(iter([]), index=0)

        assert paths == []
        mock_writer.write.assert_not_called()
        mock_writer.append.assert_not_called()

    @patch("physicsnemo_curator.atm.sinks.zarr_writer.AtomicDataZarrWriter")
    def test_creates_parent_directory(self, mock_writer_cls: MagicMock, tmp_path: pathlib.Path) -> None:
        from physicsnemo_curator.atm.sinks.zarr_writer import AtomicDataZarrSink

        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer

        nested_path = tmp_path / "nested" / "dir" / "out.zarr"
        sink = AtomicDataZarrSink(output_path=str(nested_path))
        sink(iter([_make_mock_atomic_data()]), index=0)

        assert nested_path.parent.exists()


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


@pytest.mark.requires("atm")
class TestAtomicDataZarrSinkRegistry:
    """Test that the sink is registered."""

    def test_sink_registered(self) -> None:
        import physicsnemo_curator.atm  # noqa: F401
        from physicsnemo_curator.core.registry import registry

        sinks = registry.list_sinks("atm")
        sink_names = {s.name for s in sinks}
        assert "AtomicData Zarr" in sink_names
