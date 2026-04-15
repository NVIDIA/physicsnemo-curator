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
hv.extension("bokeh")  # ty: ignore[too-many-positional-arguments]

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
        @pn.depends(  # ty: ignore[invalid-argument-type]
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
            filtered_df = df[df["level"].isin(selected_levels)] if selected_levels and "level" in df.columns else df

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

    def _load_data(self, artifact_paths: list[str]) -> pd.DataFrame | None:
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
