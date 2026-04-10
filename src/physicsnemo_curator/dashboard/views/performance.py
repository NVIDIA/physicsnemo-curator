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

"""Performance tab — timing scatter, stage breakdown, memory analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import holoviews as hv
import numpy as np
import panel as pn

if TYPE_CHECKING:
    from physicsnemo_curator.dashboard.data import DashboardStore

hv.extension("bokeh")


def _timeline_scatter(store: DashboardStore) -> pn.Column:
    """Build the timeline scatter plot.

    Parameters
    ----------
    store : DashboardStore
        The dashboard data store.

    Returns
    -------
    pn.Column
        Column with scatter plot and memory toggle.
    """
    df = store.index_df
    if df.empty:
        return pn.Column(pn.pane.Markdown("*No data to plot.*"))

    # Color by status
    color_map = {"completed": "#4caf50", "error": "#f44336"}
    df = df.copy()
    df["color"] = df["status"].map(color_map).fillna("#999999")

    # Build scatter
    scatter = hv.Scatter(
        df,
        kdims=["index"],
        vdims=["wall_time_s", "status", "peak_memory_mb", "color"],
    ).opts(
        color="color",
        size=6,
        tools=["tap", "hover"],
        width=800,
        height=350,
        xlabel="Index",
        ylabel="Wall Time (s)",
        title="Index Processing Time",
    )

    # Memory overlay toggle
    show_memory = pn.widgets.Toggle(name="Show Memory", value=False, width=120)

    plot_pane = pn.pane.HoloViews(scatter, sizing_mode="stretch_width")

    def _toggle_memory(event: object) -> None:  # noqa: ARG001
        """Toggle memory overlay on the scatter plot."""
        if show_memory.value and not df.empty:
            mem_scatter = hv.Scatter(
                df,
                kdims=["index"],
                vdims=["peak_memory_mb"],
            ).opts(
                color="#2196f3",
                size=4,
                alpha=0.5,
                ylabel="Peak Memory (MB)",
                width=800,
                height=350,
            )
            combined = scatter + mem_scatter
            plot_pane.object = combined.cols(1)
        else:
            plot_pane.object = scatter

    show_memory.param.watch(_toggle_memory, "value")

    # Tap selection callback
    selection = hv.streams.Selection1D(source=scatter)

    def _on_select(index: list[int]) -> None:
        """Handle tap selection on scatter points."""
        if index and len(index) > 0:
            selected_row = df.iloc[index[0]]
            store.selected_index = int(selected_row["index"])

    selection.param.watch(lambda event: _on_select(event.new), "index")

    return pn.Column(
        pn.pane.Markdown("### Timeline"),
        pn.Row(show_memory),
        plot_pane,
    )


def _stage_breakdown(store: DashboardStore) -> pn.Column:
    """Build the stage breakdown visualization.

    Parameters
    ----------
    store : DashboardStore
        The dashboard data store.

    Returns
    -------
    pn.Column
        Column with stacked bar chart and stats table.
    """
    import pandas as pd

    stage_df = store.stage_df
    if stage_df.empty:
        return pn.Column(pn.pane.Markdown("*No stage data available.*"))

    # Pivot for stacked bar
    pivot = stage_df.pivot_table(
        index="index",
        columns="stage_name",
        values="wall_time_s",
        aggfunc="sum",
        fill_value=0,
    )

    # Stacked bar chart
    bars = hv.Bars(
        pivot.reset_index().melt(id_vars="index", var_name="stage", value_name="time_s"),
        kdims=["index", "stage"],
        vdims=["time_s"],
    ).opts(
        stacked=True,
        width=800,
        height=300,
        xlabel="Index",
        ylabel="Wall Time (s)",
        title="Stage Breakdown",
        legend_position="right",
    )

    # Stage filter dropdown
    stage_names = sorted(stage_df["stage_name"].unique())
    stage_select = pn.widgets.Select(
        name="Stage Filter",
        options=["(all)"] + stage_names,
        value="(all)",
        width=200,
    )

    # Stats table
    stats_rows = []
    for stage in stage_names:
        stage_times = stage_df[stage_df["stage_name"] == stage]["wall_time_s"]
        stats_rows.append(
            {
                "stage": stage,
                "mean_s": round(stage_times.mean(), 4),
                "median_s": round(stage_times.median(), 4),
                "p95_s": round(float(np.percentile(stage_times, 95)), 4),
                "max_s": round(stage_times.max(), 4),
            }
        )
    stats_df = pd.DataFrame(stats_rows)

    bars_pane = pn.pane.HoloViews(bars, sizing_mode="stretch_width")

    def _filter_stage(event: object) -> None:  # noqa: ARG001
        """Filter the bar chart to a single stage."""
        selected = stage_select.value
        if selected == "(all)":
            bars_pane.object = bars
        else:
            filtered = stage_df[stage_df["stage_name"] == selected]
            single = hv.Bars(
                filtered,
                kdims=["index"],
                vdims=["wall_time_s"],
            ).opts(
                width=800,
                height=300,
                xlabel="Index",
                ylabel="Wall Time (s)",
                title=f"{selected} — Per-Index Time",
                color="#2196f3",
            )
            bars_pane.object = single

    stage_select.param.watch(_filter_stage, "value")

    return pn.Column(
        pn.pane.Markdown("### Stage Breakdown"),
        pn.Row(stage_select),
        bars_pane,
        pn.pane.Markdown("#### Stage Statistics"),
        pn.pane.DataFrame(stats_df, index=False, sizing_mode="stretch_width"),
    )


def _resource_summary(store: DashboardStore) -> pn.Column:
    """Build the resource summary section.

    Parameters
    ----------
    store : DashboardStore
        The dashboard data store.

    Returns
    -------
    pn.Column
        Column with memory histograms and slowest-indices table.
    """
    df = store.index_df
    if df.empty:
        return pn.Column(pn.pane.Markdown("*No data.*"))

    components: list[pn.viewable.Viewable] = []
    components.append(pn.pane.Markdown("### Resource Summary"))

    # Memory histogram
    completed = df[df["status"] == "completed"]
    if not completed.empty and completed["peak_memory_mb"].sum() > 0:
        mem_hist = hv.Histogram(
            np.histogram(completed["peak_memory_mb"].dropna(), bins=30),
        ).opts(
            width=400,
            height=250,
            xlabel="Peak Memory (MB)",
            ylabel="Count",
            title="Memory Distribution",
            color="#4caf50",
        )
        components.append(pn.pane.HoloViews(mem_hist))

    # GPU memory histogram (if tracked)
    if not completed.empty and completed["gpu_memory_mb"].sum() > 0:
        gpu_hist = hv.Histogram(
            np.histogram(completed["gpu_memory_mb"].dropna(), bins=30),
        ).opts(
            width=400,
            height=250,
            xlabel="GPU Memory (MB)",
            ylabel="Count",
            title="GPU Memory Distribution",
            color="#ff9800",
        )
        components.append(pn.pane.HoloViews(gpu_hist))

    # Slowest indices table
    slowest = df.nlargest(10, "wall_time_s")[["index", "wall_time_s", "peak_memory_mb", "status"]]
    components.append(pn.pane.Markdown("#### Slowest Indices (Top 10)"))
    components.append(pn.pane.DataFrame(slowest.round(3), index=False, sizing_mode="stretch_width"))

    return pn.Column(*components, sizing_mode="stretch_width")


def performance_tab(store: DashboardStore) -> pn.Column:
    """Build the Performance tab layout.

    Parameters
    ----------
    store : DashboardStore
        The dashboard data store.

    Returns
    -------
    pn.Column
        Complete performance tab content.
    """
    return pn.Column(
        _timeline_scatter(store),
        pn.layout.Divider(),
        _stage_breakdown(store),
        pn.layout.Divider(),
        _resource_summary(store),
        sizing_mode="stretch_width",
    )
