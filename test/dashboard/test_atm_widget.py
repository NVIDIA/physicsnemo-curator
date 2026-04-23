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

"""Tests for AtomicStatsScatterWidget."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("panel")
pytest.importorskip("holoviews")


@pytest.fixture
def mock_stats_parquet(tmp_path: Path) -> str:
    """Create a mock AtomicStatsFilter parquet file."""
    df = pd.DataFrame(
        {
            "field_key": ["positions", "positions", "positions", "forces", "energies"],
            "level": ["node", "node", "node", "node", "system"],
            "component": [0, 1, 2, 0, -1],
            "n_values": [100, 100, 100, 100, 10],
            "n_components": [3, 3, 3, 3, 1],
            "mean": [0.5, 1.2, -0.3, 0.01, -150.5],
            "std": [0.1, 0.2, 0.15, 0.005, 10.2],
            "var": [0.01, 0.04, 0.0225, 0.000025, 104.04],
            "min": [0.2, 0.8, -0.7, -0.02, -180.0],
            "max": [0.9, 1.6, 0.1, 0.05, -120.0],
            "median": [0.5, 1.2, -0.3, 0.01, -150.0],
            "abs_mean": [0.5, 1.2, 0.3, 0.01, 150.5],
            "abs_max": [0.9, 1.6, 0.7, 0.05, 180.0],
            "skewness": [0.1, -0.2, 0.3, 0.0, -0.5],
            "kurtosis": [0.05, 0.1, -0.1, 0.0, 0.2],
        }
    )
    path = tmp_path / "stats.parquet"
    df.to_parquet(path)
    return str(path)


class TestAtomicStatsScatterWidget:
    """Tests for AtomicStatsScatterWidget."""

    def test_instantiation(self) -> None:
        """Widget can be instantiated."""
        from physicsnemo_curator.dashboard.widgets.atm import AtomicStatsScatterWidget

        widget = AtomicStatsScatterWidget()
        assert widget.name == "Atomic Statistics Scatter"
        assert widget.filter_name == "Atomic Statistics"

    def test_panel_empty_artifacts(self) -> None:
        """Widget returns message when no artifacts provided."""
        import panel as pn

        from physicsnemo_curator.dashboard.widgets.atm import AtomicStatsScatterWidget

        widget = AtomicStatsScatterWidget()
        result = widget.panel([])

        assert isinstance(result, pn.pane.Markdown)
        assert "No Atomic Statistics artifacts" in result.object

    def test_panel_with_data(self, mock_stats_parquet: str) -> None:
        """Widget returns a GridStack with Paper tiles when data is provided."""
        import panel as pn

        from physicsnemo_curator.dashboard.widgets.atm import AtomicStatsScatterWidget

        widget = AtomicStatsScatterWidget()
        result = widget.panel([mock_stats_parquet])

        # Should return a GridStack layout (sidebar + plot area in Paper tiles)
        assert isinstance(result, pn.GridStack)
        assert len(result.objects) == 2  # sidebar and plot area

    def test_panel_contains_scatter_plot(self, mock_stats_parquet: str) -> None:
        """Widget contains a Holoviews scatter plot in a Paper tile."""
        import panel as pn
        import panel_material_ui as pmui

        from physicsnemo_curator.dashboard.widgets.atm import AtomicStatsScatterWidget

        widget = AtomicStatsScatterWidget()
        result = widget.panel([mock_stats_parquet])

        # The GridStack should contain Paper tiles
        assert isinstance(result, pn.GridStack)
        tiles = list(result.objects.values())
        assert len(tiles) == 2
        # Both tiles should be Paper components
        assert all(isinstance(t, pmui.Paper) for t in tiles)
        # Second Paper tile wraps the HoloViews plot pane
        plot_tile = tiles[1]
        assert isinstance(plot_tile.objects[0], (pn.pane.HoloViews, pn.Column))

    def test_layout_hints(self) -> None:
        """Widget declares grid layout hints."""
        from physicsnemo_curator.dashboard.widgets.atm import AtomicStatsScatterWidget

        widget = AtomicStatsScatterWidget()
        hints = widget.layout_hints()

        assert isinstance(hints, dict)
        assert "cols" in hints
        assert "rows" in hints
        assert 1 <= hints["cols"] <= 12
        assert hints["rows"] >= 1


class TestWidgetRegistry:
    """Tests for AtomicStatsScatterWidget registration."""

    def test_widget_registered(self) -> None:
        """AtomicStatsScatterWidget is registered in WidgetRegistry."""
        from physicsnemo_curator.dashboard.widgets import WidgetRegistry

        registry = WidgetRegistry()
        provider = registry.get("Atomic Statistics")

        assert provider is not None
        assert provider.name == "Atomic Statistics Scatter"


class TestMeanFilterWidget:
    """Tests for MeanFilterWidget."""

    def test_layout_hints(self) -> None:
        """Widget declares grid layout hints."""
        from physicsnemo_curator.dashboard.widgets.mesh import MeanFilterWidget

        widget = MeanFilterWidget()
        hints = widget.layout_hints()

        assert isinstance(hints, dict)
        assert hints["cols"] == 6
        assert hints["rows"] == 2
