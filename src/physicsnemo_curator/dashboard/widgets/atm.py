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
from bokeh.models import HoverTool

if TYPE_CHECKING:
    import pandas as pd

# Initialize Holoviews with Bokeh backend
hv.extension("bokeh")  # ty: ignore[too-many-positional-arguments]

# Statistics columns available for scatter plot axes (index is added dynamically)
STAT_COLUMNS: list[str] = [
    "index",
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

# Minimal columns for hover tooltip (just axis values and index for lookup)
TOOLTIP_COLUMNS: list[str] = [
    "index",
    "field_key",
]


class AtomicStatsScatterWidget:
    """Interactive scatter plot widget for AtomicStatsFilter parquet output.

    Displays a scatter plot where users can select X/Y axes from available
    statistics columns and filter by level (node/edge/system/extra) and field.
    """

    name: str = "Atomic Statistics Scatter"
    filter_name: str = "Atomic Statistics"

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
            return pn.pane.Markdown("*No Atomic Statistics artifacts found.*")

        # Load data
        df = self._load_data(artifact_paths)
        if df is None or df.empty:
            return pn.pane.Markdown("*Could not read any Atomic Statistics artifacts.*")

        # Add index column from DataFrame index for lookup
        df = df.reset_index(drop=True)
        df["index"] = df.index

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
        available_levels = df["level"].unique().tolist() if "level" in df.columns else []
        level_filter = pn.widgets.CheckBoxGroup(
            name="Filter Levels",
            options=available_levels,
            value=available_levels,
        )

        # Field key filter dropdown
        available_fields = sorted(df["field_key"].unique().tolist()) if "field_key" in df.columns else []
        field_filter = pn.widgets.MultiSelect(
            name="Fields",
            options=available_fields,
            value=available_fields,
            size=min(8, len(available_fields)),
        )

        # Create sidebar with scrolling to prevent overlap
        sidebar = pn.Column(
            "### Controls",
            x_select,
            y_select,
            "---",
            "### Filter by Level",
            level_filter,
            "---",
            "### Filter by Field",
            field_filter,
            width=220,
            sizing_mode="fixed",
            scroll=True,
        )

        # Create reactive plot
        @pn.depends(  # ty: ignore[invalid-argument-type]
            x_select.param.value,
            y_select.param.value,
            level_filter.param.value,
            field_filter.param.value,
        )
        def update_plot(
            x_col: str,
            y_col: str,
            selected_levels: list[str],
            selected_fields: list[str],
        ) -> hv.Points:
            """Update scatter plot based on widget selections."""
            # Filter by selected levels
            filtered_df = df[df["level"].isin(selected_levels)] if selected_levels and "level" in df.columns else df

            # Filter by selected fields
            if selected_fields and "field_key" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["field_key"].isin(selected_fields)]

            if filtered_df.empty:
                return hv.Points([]).opts(title="No data matches filters")

            # Create scatter plot with index and field_key for tooltip
            points = hv.Points(
                filtered_df,
                kdims=[x_col, y_col],
                vdims=["index", "field_key"],
            )

            # Custom hover tool showing just x/y values and indices
            hover = HoverTool(
                tooltips=[
                    (x_col, "@{" + x_col + "}{0.4f}"),
                    (y_col, "@{" + y_col + "}{0.4f}"),
                    ("index", "@index"),
                    ("field", "@field_key"),
                ],
                point_policy="follow_mouse",
            )

            # Apply styling - use responsive sizing
            points = points.opts(
                color="#1f77b4",  # Solid blue color
                size=8,
                tools=[hover, "pan", "wheel_zoom", "box_zoom", "reset"],
                responsive=True,
                min_height=400,
                xlabel=x_col,
                ylabel=y_col,
                title=f"{y_col} vs {x_col}",
            )

            return points

        # Wrap in HoloViews pane
        plot_pane = pn.pane.HoloViews(update_plot, sizing_mode="stretch_both")

        # Use FlexBox layout to prevent overlap
        return pn.Row(
            sidebar,
            plot_pane,
            sizing_mode="stretch_both",
        )

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
