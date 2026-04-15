# AtomicStatsScatterWidget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an interactive scatter plot widget to visualize AtomicStatsFilter
parquet output in the PhysicsNeMo Curator dashboard.

**Architecture:** Single widget class (`AtomicStatsScatterWidget`) following the
existing `WidgetProvider` protocol. Uses Holoviews + Bokeh for the interactive
scatter plot with Panel widgets for controls in a sidebar layout.

**Tech Stack:** Panel, Holoviews, Bokeh, pandas, pyarrow

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/physicsnemo_curator/dashboard/widgets/atm.py` | NEW: Widget implementation |
| `src/physicsnemo_curator/dashboard/widgets/__init__.py` | MODIFY: Register widget |
| `test/dashboard/test_atm_widget.py` | NEW: Widget tests |

---

## Task 1: Create test file with basic instantiation test

**Files:**

- Create: `test/dashboard/test_atm_widget.py`

- [ ] **Step 1: Create test file with instantiation test**

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

"""Tests for AtomicStatsScatterWidget."""

from __future__ import annotations

import pytest

pytest.importorskip("panel")
pytest.importorskip("holoviews")


class TestAtomicStatsScatterWidget:
    """Tests for AtomicStatsScatterWidget."""

    def test_instantiation(self) -> None:
        """Widget can be instantiated."""
        from physicsnemo_curator.dashboard.widgets.atm import AtomicStatsScatterWidget

        widget = AtomicStatsScatterWidget()
        assert widget.name == "Atomic Statistics Scatter"
        assert widget.filter_name == "AtomicStatsFilter"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/dashboard/test_atm_widget.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'physicsnemo_curator.dashboard.widgets.atm'`

---

## Task 2: Create widget file with class skeleton

**Files:**

- Create: `src/physicsnemo_curator/dashboard/widgets/atm.py`

- [ ] **Step 1: Create widget file with class skeleton**

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

"""AtomicStatsFilter artifact visualization widget."""

from __future__ import annotations

import panel as pn


class AtomicStatsScatterWidget:
    """Interactive scatter plot widget for AtomicStatsFilter parquet output.

    Displays a scatter plot where users can select X/Y axes from available
    statistics columns, color points by categorical dimensions, and filter
    by level (node/edge/system/extra).
    """

    name: str = "Atomic Statistics Scatter"
    filter_name: str = "AtomicStatsFilter"

    def panel(
        self,
        artifact_paths: list[str],
        selected_index: int | None = None,
    ) -> pn.viewable.Viewable:
        """Return a Panel component visualizing AtomicStatsFilter artifacts.

        Parameters
        ----------
        artifact_paths : list[str]
            Paths to Parquet files produced by AtomicStatsFilter.
        selected_index : int or None
            Currently selected pipeline index, if any.

        Returns
        -------
        pn.viewable.Viewable
            A Panel Row containing sidebar controls and scatter plot.
        """
        return pn.pane.Markdown("*Widget not yet implemented.*")
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest test/dashboard/test_atm_widget.py -v`

Expected: PASS

- [ ] **Step 3: Run linting**

Run: `uv run ruff check src/physicsnemo_curator/dashboard/widgets/atm.py`

Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add src/physicsnemo_curator/dashboard/widgets/atm.py test/dashboard/test_atm_widget.py
git commit -m "feat(dashboard): add AtomicStatsScatterWidget skeleton"
```

---

## Task 3: Add test for empty artifacts handling

**Files:**

- Modify: `test/dashboard/test_atm_widget.py`

- [ ] **Step 1: Add test for empty artifacts**

Add to `TestAtomicStatsScatterWidget` class:

```python
    def test_panel_empty_artifacts(self) -> None:
        """Widget returns message when no artifacts provided."""
        import panel as pn

        from physicsnemo_curator.dashboard.widgets.atm import AtomicStatsScatterWidget

        widget = AtomicStatsScatterWidget()
        result = widget.panel([])

        assert isinstance(result, pn.pane.Markdown)
        assert "No AtomicStatsFilter artifacts" in result.object
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/dashboard/test_atm_widget.py::TestAtomicStatsScatterWidget::test_panel_empty_artifacts -v`

Expected: FAIL with assertion error (current implementation returns different text)

- [ ] **Step 3: Update implementation to handle empty artifacts**

In `src/physicsnemo_curator/dashboard/widgets/atm.py`, update the `panel` method:

```python
    def panel(
        self,
        artifact_paths: list[str],
        selected_index: int | None = None,
    ) -> pn.viewable.Viewable:
        """Return a Panel component visualizing AtomicStatsFilter artifacts.

        Parameters
        ----------
        artifact_paths : list[str]
            Paths to Parquet files produced by AtomicStatsFilter.
        selected_index : int or None
            Currently selected pipeline index, if any.

        Returns
        -------
        pn.viewable.Viewable
            A Panel Row containing sidebar controls and scatter plot.
        """
        if not artifact_paths:
            return pn.pane.Markdown("*No AtomicStatsFilter artifacts found.*")

        return pn.pane.Markdown("*Widget not yet implemented.*")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/dashboard/test_atm_widget.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/physicsnemo_curator/dashboard/widgets/atm.py test/dashboard/test_atm_widget.py
git commit -m "feat(dashboard): handle empty artifacts in AtomicStatsScatterWidget"
```

---

## Task 4: Add test for data loading and basic scatter plot

**Files:**

- Modify: `test/dashboard/test_atm_widget.py`
- Modify: `src/physicsnemo_curator/dashboard/widgets/atm.py`

- [ ] **Step 1: Add fixture for mock parquet data**

Add at module level in `test/dashboard/test_atm_widget.py` after the imports:

```python
import tempfile
from pathlib import Path

import pandas as pd


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
```

- [ ] **Step 2: Add test for basic panel with data**

Add to `TestAtomicStatsScatterWidget` class:

```python
    def test_panel_with_data(self, mock_stats_parquet: str) -> None:
        """Widget returns a Row with sidebar and plot when data is provided."""
        import panel as pn

        from physicsnemo_curator.dashboard.widgets.atm import AtomicStatsScatterWidget

        widget = AtomicStatsScatterWidget()
        result = widget.panel([mock_stats_parquet])

        # Should return a Row layout (sidebar + plot)
        assert isinstance(result, pn.Row)
        assert len(result) == 2  # sidebar and plot area
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest test/dashboard/test_atm_widget.py::TestAtomicStatsScatterWidget::test_panel_with_data -v`

Expected: FAIL with assertion error (current implementation returns Markdown, not Row)

- [ ] **Step 4: Implement data loading and basic layout**

Update `src/physicsnemo_curator/dashboard/widgets/atm.py`:

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

"""AtomicStatsFilter artifact visualization widget."""

from __future__ import annotations

import panel as pn

# Statistics columns available for scatter plot axes
STAT_COLUMNS: list[str] = [
    "mean",
    "std",
    "var",
    "min",
    "max",
    "median",
    "abs_mean",
    "abs_max",
    "skewness",
    "kurtosis",
    "n_values",
    "n_components",
]

# Categorical columns available for color-by
COLOR_BY_OPTIONS: list[str] = ["level", "field_key", "component"]


class AtomicStatsScatterWidget:
    """Interactive scatter plot widget for AtomicStatsFilter parquet output.

    Displays a scatter plot where users can select X/Y axes from available
    statistics columns, color points by categorical dimensions, and filter
    by level (node/edge/system/extra).
    """

    name: str = "Atomic Statistics Scatter"
    filter_name: str = "AtomicStatsFilter"

    def panel(
        self,
        artifact_paths: list[str],
        selected_index: int | None = None,
    ) -> pn.viewable.Viewable:
        """Return a Panel component visualizing AtomicStatsFilter artifacts.

        Parameters
        ----------
        artifact_paths : list[str]
            Paths to Parquet files produced by AtomicStatsFilter.
        selected_index : int or None
            Currently selected pipeline index, if any.

        Returns
        -------
        pn.viewable.Viewable
            A Panel Row containing sidebar controls and scatter plot.
        """
        if not artifact_paths:
            return pn.pane.Markdown("*No AtomicStatsFilter artifacts found.*")

        # Load data
        df = self._load_data(artifact_paths)
        if df is None or df.empty:
            return pn.pane.Markdown("*Could not read any AtomicStatsFilter artifacts.*")

        # Create sidebar with controls
        sidebar = self._create_sidebar(df)

        # Create placeholder for plot area
        plot_area = pn.pane.Markdown("*Scatter plot placeholder*")

        return pn.Row(sidebar, plot_area, sizing_mode="stretch_both")

    def _load_data(self, artifact_paths: list[str]) -> "pd.DataFrame | None":
        """Load and concatenate parquet files.

        Parameters
        ----------
        artifact_paths : list[str]
            Paths to Parquet files.

        Returns
        -------
        pd.DataFrame or None
            Concatenated DataFrame, or None if no files could be read.
        """
        import pandas as pd

        frames = []
        for path in artifact_paths:
            try:
                frames.append(pd.read_parquet(path))
            except Exception:  # noqa: BLE001
                continue

        if not frames:
            return None

        return pd.concat(frames, ignore_index=True)

    def _create_sidebar(self, df: "pd.DataFrame") -> pn.viewable.Viewable:
        """Create sidebar with axis selectors and filters.

        Parameters
        ----------
        df : pd.DataFrame
            The loaded data.

        Returns
        -------
        pn.viewable.Viewable
            A Panel Column containing control widgets.
        """
        # X-axis selector
        x_select = pn.widgets.Select(
            name="X-Axis",
            options=STAT_COLUMNS,
            value="mean",
        )

        # Y-axis selector
        y_select = pn.widgets.Select(
            name="Y-Axis",
            options=STAT_COLUMNS,
            value="std",
        )

        # Color-by selector
        color_select = pn.widgets.Select(
            name="Color by",
            options=COLOR_BY_OPTIONS,
            value="level",
        )

        # Level filter checkboxes
        available_levels = df["level"].unique().tolist() if "level" in df.columns else []
        level_filter = pn.widgets.CheckBoxGroup(
            name="Filter Levels",
            options=available_levels,
            value=available_levels,
        )

        return pn.Column(
            "### Controls",
            x_select,
            y_select,
            color_select,
            "---",
            level_filter,
            width=180,
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest test/dashboard/test_atm_widget.py -v`

Expected: PASS

- [ ] **Step 6: Run linting**

Run: `uv run ruff check src/physicsnemo_curator/dashboard/widgets/atm.py`

Expected: No errors

- [ ] **Step 7: Commit**

```bash
git add src/physicsnemo_curator/dashboard/widgets/atm.py test/dashboard/test_atm_widget.py
git commit -m "feat(dashboard): add data loading and sidebar controls"
```

---

## Task 5: Add interactive scatter plot with Holoviews

**Files:**

- Modify: `test/dashboard/test_atm_widget.py`
- Modify: `src/physicsnemo_curator/dashboard/widgets/atm.py`

- [ ] **Step 1: Add test for scatter plot presence**

Add to `TestAtomicStatsScatterWidget` class:

```python
    def test_panel_contains_scatter_plot(self, mock_stats_parquet: str) -> None:
        """Widget contains a Holoviews scatter plot."""
        import holoviews as hv
        import panel as pn

        from physicsnemo_curator.dashboard.widgets.atm import AtomicStatsScatterWidget

        widget = AtomicStatsScatterWidget()
        result = widget.panel([mock_stats_parquet])

        # The second element should be a Column containing the plot
        assert isinstance(result, pn.Row)
        plot_area = result[1]
        # Should contain a HoloViews pane
        assert isinstance(plot_area, (pn.pane.HoloViews, pn.Column))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/dashboard/test_atm_widget.py::TestAtomicStatsScatterWidget::test_panel_contains_scatter_plot -v`

Expected: FAIL (plot_area is currently a Markdown pane)

- [ ] **Step 3: Implement scatter plot with Holoviews**

Update `src/physicsnemo_curator/dashboard/widgets/atm.py` - replace the entire file:

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

"""AtomicStatsFilter artifact visualization widget."""

from __future__ import annotations

from typing import TYPE_CHECKING

import holoviews as hv
import panel as pn

if TYPE_CHECKING:
    import pandas as pd

# Initialize Holoviews with Bokeh backend
hv.extension("bokeh")

# Statistics columns available for scatter plot axes
STAT_COLUMNS: list[str] = [
    "mean",
    "std",
    "var",
    "min",
    "max",
    "median",
    "abs_mean",
    "abs_max",
    "skewness",
    "kurtosis",
    "n_values",
    "n_components",
]

# Categorical columns available for color-by
COLOR_BY_OPTIONS: list[str] = ["level", "field_key", "component"]

# All columns to include in hover tooltip
TOOLTIP_COLUMNS: list[str] = [
    "field_key",
    "level",
    "component",
    "n_values",
    "n_components",
    "mean",
    "std",
    "var",
    "min",
    "max",
    "median",
    "abs_mean",
    "abs_max",
    "skewness",
    "kurtosis",
]


class AtomicStatsScatterWidget:
    """Interactive scatter plot widget for AtomicStatsFilter parquet output.

    Displays a scatter plot where users can select X/Y axes from available
    statistics columns, color points by categorical dimensions, and filter
    by level (node/edge/system/extra).
    """

    name: str = "Atomic Statistics Scatter"
    filter_name: str = "AtomicStatsFilter"

    def panel(
        self,
        artifact_paths: list[str],
        selected_index: int | None = None,
    ) -> pn.viewable.Viewable:
        """Return a Panel component visualizing AtomicStatsFilter artifacts.

        Parameters
        ----------
        artifact_paths : list[str]
            Paths to Parquet files produced by AtomicStatsFilter.
        selected_index : int or None
            Currently selected pipeline index, if any.

        Returns
        -------
        pn.viewable.Viewable
            A Panel Row containing sidebar controls and scatter plot.
        """
        if not artifact_paths:
            return pn.pane.Markdown("*No AtomicStatsFilter artifacts found.*")

        # Load data
        df = self._load_data(artifact_paths)
        if df is None or df.empty:
            return pn.pane.Markdown("*Could not read any AtomicStatsFilter artifacts.*")

        # Create widgets
        x_select = pn.widgets.Select(
            name="X-Axis",
            options=STAT_COLUMNS,
            value="mean",
        )
        y_select = pn.widgets.Select(
            name="Y-Axis",
            options=STAT_COLUMNS,
            value="std",
        )
        color_select = pn.widgets.Select(
            name="Color by",
            options=COLOR_BY_OPTIONS,
            value="level",
        )
        available_levels = df["level"].unique().tolist() if "level" in df.columns else []
        level_filter = pn.widgets.CheckBoxGroup(
            name="Filter Levels",
            options=available_levels,
            value=available_levels,
        )

        # Create sidebar
        sidebar = pn.Column(
            "### Controls",
            x_select,
            y_select,
            color_select,
            "---",
            level_filter,
            width=180,
        )

        # Create reactive plot
        @pn.depends(
            x_select.param.value,
            y_select.param.value,
            color_select.param.value,
            level_filter.param.value,
        )
        def update_plot(
            x_col: str,
            y_col: str,
            color_col: str,
            selected_levels: list[str],
        ) -> hv.Points:
            """Update scatter plot based on widget selections."""
            # Filter by selected levels
            if selected_levels and "level" in df.columns:
                filtered_df = df[df["level"].isin(selected_levels)]
            else:
                filtered_df = df

            if filtered_df.empty:
                return hv.Points([]).opts(title="No data matches filters")

            # Build hover tooltip columns
            hover_cols = [c for c in TOOLTIP_COLUMNS if c in filtered_df.columns]

            # Create scatter plot
            points = hv.Points(
                filtered_df,
                kdims=[x_col, y_col],
                vdims=[color_col] + [c for c in hover_cols if c not in [x_col, y_col, color_col]],
            )

            # Apply styling
            points = points.opts(
                color=color_col,
                cmap="Category10",
                size=8,
                tools=["hover", "pan", "wheel_zoom", "box_zoom", "reset"],
                width=600,
                height=450,
                xlabel=x_col,
                ylabel=y_col,
                title=f"{y_col} vs {x_col}",
                legend_position="right",
                show_legend=True,
            )

            return points

        # Wrap in HoloViews pane
        plot_pane = pn.pane.HoloViews(update_plot, sizing_mode="stretch_both")

        return pn.Row(sidebar, plot_pane, sizing_mode="stretch_both")

    def _load_data(self, artifact_paths: list[str]) -> "pd.DataFrame | None":
        """Load and concatenate parquet files.

        Parameters
        ----------
        artifact_paths : list[str]
            Paths to Parquet files.

        Returns
        -------
        pd.DataFrame or None
            Concatenated DataFrame, or None if no files could be read.
        """
        import pandas as pd

        frames = []
        for path in artifact_paths:
            try:
                frames.append(pd.read_parquet(path))
            except Exception:  # noqa: BLE001
                continue

        if not frames:
            return None

        return pd.concat(frames, ignore_index=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/dashboard/test_atm_widget.py -v`

Expected: PASS

- [ ] **Step 5: Run linting and type checking**

Run: `uv run ruff check src/physicsnemo_curator/dashboard/widgets/atm.py && uv run ty check src/physicsnemo_curator/dashboard/widgets/atm.py`

Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add src/physicsnemo_curator/dashboard/widgets/atm.py test/dashboard/test_atm_widget.py
git commit -m "feat(dashboard): add interactive Holoviews scatter plot"
```

---

## Task 6: Register widget in registry

**Files:**

- Modify: `src/physicsnemo_curator/dashboard/widgets/__init__.py`
- Modify: `test/dashboard/test_atm_widget.py`

- [ ] **Step 1: Add test for widget registration**

Add to `test/dashboard/test_atm_widget.py`:

```python
class TestWidgetRegistry:
    """Tests for AtomicStatsScatterWidget registration."""

    def test_widget_registered(self) -> None:
        """AtomicStatsScatterWidget is registered in WidgetRegistry."""
        from physicsnemo_curator.dashboard.widgets import WidgetRegistry

        registry = WidgetRegistry()
        provider = registry.get("AtomicStatsFilter")

        assert provider is not None
        assert provider.name == "Atomic Statistics Scatter"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/dashboard/test_atm_widget.py::TestWidgetRegistry::test_widget_registered -v`

Expected: FAIL with `AssertionError: assert None is not None`

- [ ] **Step 3: Register widget in `__init__.py`**

In `src/physicsnemo_curator/dashboard/widgets/__init__.py`, add after the MeanFilterWidget
registration block (around line 87):

```python
        try:
            from physicsnemo_curator.dashboard.widgets.atm import AtomicStatsScatterWidget

            self.register(AtomicStatsScatterWidget())
        except Exception:  # noqa: BLE001
            logger.debug("AtomicStatsScatterWidget not available", exc_info=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/dashboard/test_atm_widget.py -v`

Expected: PASS

- [ ] **Step 5: Run linting**

Run: `uv run ruff check src/physicsnemo_curator/dashboard/widgets/__init__.py`

Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add src/physicsnemo_curator/dashboard/widgets/__init__.py test/dashboard/test_atm_widget.py
git commit -m "feat(dashboard): register AtomicStatsScatterWidget in registry"
```

---

## Task 7: Verify docstring coverage and run full test suite

**Files:**

- All modified files

- [ ] **Step 1: Check docstring coverage**

Run: `uv run interrogate src/physicsnemo_curator/dashboard/widgets/atm.py -v`

Expected: 100% coverage (or fix any missing docstrings)

- [ ] **Step 2: Run full dashboard test suite**

Run: `uv run pytest test/dashboard/ -v`

Expected: All tests pass

- [ ] **Step 3: Run type checking on widget module**

Run: `uv run ty check src/physicsnemo_curator/dashboard/widgets/`

Expected: No errors

- [ ] **Step 4: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "chore(dashboard): fix docstrings and type hints"
```

---

## Summary

After completing all tasks, you will have:

1. `src/physicsnemo_curator/dashboard/widgets/atm.py` - New widget with:
   - `AtomicStatsScatterWidget` class implementing `WidgetProvider` protocol
   - Interactive Holoviews scatter plot with Bokeh backend
   - Sidebar with X/Y axis selectors, color-by dropdown, level filter checkboxes
   - Hover tooltips showing all statistics

2. `src/physicsnemo_curator/dashboard/widgets/__init__.py` - Updated to register the new widget

3. `test/dashboard/test_atm_widget.py` - Tests for:
   - Widget instantiation
   - Empty artifacts handling
   - Data loading and layout
   - Scatter plot presence
   - Widget registration
