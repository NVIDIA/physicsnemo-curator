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

try:
    import torch  # noqa: F401

    _has_torch = True
except ModuleNotFoundError:
    _has_torch = False


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


class TestWidgetRegistry:
    """Tests for the refactored WidgetRegistry."""

    @pytest.mark.skipif(not _has_torch, reason="torch not installed")
    def test_auto_discovers_atomic_stats(self) -> None:
        """Registry discovers AtomicStatsFilter."""
        from physicsnemo_curator.dashboard.widgets import WidgetRegistry

        registry = WidgetRegistry()
        result = registry.get_panel("Atomic Statistics", [])
        # Should return a Markdown (empty artifacts) rather than None (no widget)
        assert result is not None

    def test_returns_none_for_unknown(self) -> None:
        """Registry returns None for unknown filter names."""
        from physicsnemo_curator.dashboard.widgets import WidgetRegistry

        registry = WidgetRegistry()
        result = registry.get_panel("NonExistentFilter", [])
        assert result is None

    @pytest.mark.skipif(not _has_torch, reason="torch not installed")
    def test_get_layout_hints(self) -> None:
        """Registry returns layout hints for known filters."""
        from physicsnemo_curator.dashboard.widgets import WidgetRegistry

        registry = WidgetRegistry()
        hints = registry.get_layout_hints("Atomic Statistics")
        assert hints == {"cols": 12, "rows": 3}

    def test_get_layout_hints_default(self) -> None:
        """Registry returns default hints for unknown filters."""
        from physicsnemo_curator.dashboard.widgets import WidgetRegistry

        registry = WidgetRegistry()
        hints = registry.get_layout_hints("UnknownFilter")
        assert hints == {"cols": 6, "rows": 2}


@pytest.mark.skipif(not _has_torch, reason="torch not installed")
class TestAtomicStatsFilterDashboard:
    """Tests for AtomicStatsFilter dashboard classmethods."""

    def test_dashboard_panel_empty_artifacts(self) -> None:
        """Returns Markdown message when no artifacts provided."""
        import panel as pn

        from physicsnemo_curator.domains.atm.filters.stats import AtomicStatsFilter

        result = AtomicStatsFilter.dashboard_panel([])
        assert isinstance(result, pn.pane.Markdown)
        assert "No Atomic Statistics artifacts" in result.object

    def test_dashboard_panel_with_data(self, mock_stats_parquet: str) -> None:
        """Returns a GridStack with Paper tiles when data is provided."""
        import panel as pn

        from physicsnemo_curator.domains.atm.filters.stats import AtomicStatsFilter

        result = AtomicStatsFilter.dashboard_panel([mock_stats_parquet])
        assert isinstance(result, pn.GridStack)
        assert len(result.objects) == 2

    def test_dashboard_layout_hints(self) -> None:
        """Returns full-width, 3-row layout."""
        from physicsnemo_curator.domains.atm.filters.stats import AtomicStatsFilter

        hints = AtomicStatsFilter.dashboard_layout_hints()
        assert hints == {"cols": 12, "rows": 3}


class TestFilterDashboardDefaults:
    """Tests for Filter base class dashboard defaults."""

    def test_dashboard_panel_returns_none(self) -> None:
        """Base Filter.dashboard_panel() returns None."""
        from physicsnemo_curator.core.base import Filter

        result = Filter.dashboard_panel([], selected_index=None)
        assert result is None

    def test_dashboard_layout_hints_defaults(self) -> None:
        """Base Filter.dashboard_layout_hints() returns sensible defaults."""
        from physicsnemo_curator.core.base import Filter

        hints = Filter.dashboard_layout_hints()
        assert hints == {"cols": 6, "rows": 2}


@pytest.fixture
def mock_mean_parquet(tmp_path: Path) -> str:
    """Create a mock MeanFilter parquet file."""
    df = pd.DataFrame(
        {
            "n_points": [100, 200],
            "n_cells": [50, 100],
            "point_data/velocity": [1.5, 2.3],
            "point_data/pressure": [101.3, 99.8],
        }
    )
    path = tmp_path / "means.parquet"
    df.to_parquet(path)
    return str(path)


@pytest.mark.skipif(not _has_torch, reason="torch not installed")
class TestMeanFilterDashboard:
    """Tests for MeanFilter dashboard classmethods."""

    def test_dashboard_panel_empty_artifacts(self) -> None:
        """Returns Markdown message when no artifacts provided."""
        import panel as pn

        from physicsnemo_curator.domains.mesh.filters.mean import MeanFilter

        result = MeanFilter.dashboard_panel([])
        assert isinstance(result, pn.pane.Markdown)
        assert "No Mean Statistics artifacts" in result.object

    def test_dashboard_panel_overview_mode(self, mock_mean_parquet: str) -> None:
        """Returns a Column with header and table in overview mode."""
        import panel as pn

        from physicsnemo_curator.domains.mesh.filters.mean import MeanFilter

        result = MeanFilter.dashboard_panel([mock_mean_parquet])
        assert isinstance(result, pn.Column)

    def test_dashboard_layout_hints(self) -> None:
        """Returns half-width, 2-row layout."""
        from physicsnemo_curator.domains.mesh.filters.mean import MeanFilter

        hints = MeanFilter.dashboard_layout_hints()
        assert hints == {"cols": 6, "rows": 2}
