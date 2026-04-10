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

"""Pipeline tab — structure, index query, artifact inspection, widgets."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pandas as pd
import panel as pn

if TYPE_CHECKING:
    from physicsnemo_curator.dashboard.data import DashboardStore
    from physicsnemo_curator.dashboard.widgets import WidgetRegistry


def _parse_index_query(query: str, max_index: int) -> list[int]:
    """Parse an index query string into a list of indices.

    Supports comma-separated values, ranges (``10-20``), and ``all``.

    Parameters
    ----------
    query : str
        Index query string.
    max_index : int
        Maximum valid index (exclusive).

    Returns
    -------
    list[int]
        Sorted list of unique indices.
    """
    query = query.strip().lower()
    if not query or query == "all":
        return list(range(max_index))

    indices: set[int] = set()
    for part in query.split(","):
        part = part.strip()
        range_match = re.match(r"^(\d+)\s*-\s*(\d+)$", part)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            indices.update(range(start, end + 1))
        elif part.isdigit():
            indices.add(int(part))
    return sorted(i for i in indices if 0 <= i < max_index)


def _pipeline_structure(store: DashboardStore) -> pn.Row:
    """Build the pipeline structure flow visualization.

    Parameters
    ----------
    store : DashboardStore
        The dashboard data store.

    Returns
    -------
    pn.Row
        Horizontal flow of pipeline components.
    """
    config = store.pipeline_config

    source_name = config.get("source", {}).get("name", "Unknown")
    filters = config.get("filters", [])
    sink_name = config.get("sink", {}).get("name", "None")

    cards = []

    # Source card
    source_params = config.get("source", {}).get("params", {})
    source_detail = ", ".join(f"{k}={v}" for k, v in source_params.items()) if source_params else ""
    cards.append(
        pn.Card(
            pn.pane.Markdown(f"**Source**\n\n`{source_name}`\n\n{source_detail}"),
            title="Source",
            styles={"background": "#e8f5e9"},
            width=200,
        )
    )

    # Filter cards
    for f in filters:
        fname = f.get("name", "?")
        fparams = f.get("params", {})
        fdetail = ", ".join(f"{k}={v}" for k, v in fparams.items()) if fparams else ""
        cards.append(pn.pane.Markdown("→", styles={"font-size": "24px"}, align="center"))
        cards.append(
            pn.Card(
                pn.pane.Markdown(f"**Filter**\n\n`{fname}`\n\n{fdetail}"),
                title=fname,
                styles={"background": "#e3f2fd"},
                width=200,
            )
        )

    # Sink card
    cards.append(pn.pane.Markdown("→", styles={"font-size": "24px"}, align="center"))
    sink_params = config.get("sink", {}).get("params", {})
    sink_detail = ", ".join(f"{k}={v}" for k, v in sink_params.items()) if sink_params else ""
    cards.append(
        pn.Card(
            pn.pane.Markdown(f"**Sink**\n\n`{sink_name}`\n\n{sink_detail}"),
            title="Sink",
            styles={"background": "#fff3e0"},
            width=200,
        )
    )

    return pn.Row(*cards, sizing_mode="stretch_width")


def _artifact_detail(
    store: DashboardStore,
    selected_index: int,
    registry: WidgetRegistry,
) -> pn.Column:
    """Build the artifact inspection panel for a selected index.

    Parameters
    ----------
    store : DashboardStore
        The dashboard data store.
    selected_index : int
        The selected pipeline index.
    registry : WidgetRegistry
        Widget registry for filter-specific visualizations.

    Returns
    -------
    pn.Column
        Column with output files, artifacts, and widgets.
    """
    components: list[pn.viewable.Viewable] = []
    components.append(pn.pane.Markdown(f"### Index {selected_index} Details"))

    # Output files
    paths = store.output_paths(selected_index)
    if paths:
        paths_md = "\n".join(f"- `{p}`" for p in paths)
        components.append(pn.pane.Markdown(f"**Output Files:**\n{paths_md}"))
    else:
        components.append(pn.pane.Markdown("*No output files.*"))

    # Artifacts by filter
    artifacts = store.artifacts(selected_index)
    if artifacts:
        for filter_name, artifact_paths in artifacts.items():
            components.append(pn.pane.Markdown(f"**{filter_name} Artifacts:**"))
            paths_md = "\n".join(f"- `{p}`" for p in artifact_paths)
            components.append(pn.pane.Markdown(paths_md))

            # Render widget if available
            widget = registry.get(filter_name)
            if widget is not None:
                try:
                    components.append(widget.panel(artifact_paths, selected_index=selected_index))
                except Exception as exc:  # noqa: BLE001
                    components.append(pn.pane.Markdown(f"*Widget error: {exc}*"))
    else:
        components.append(pn.pane.Markdown("*No filter artifacts.*"))

    return pn.Column(*components, sizing_mode="stretch_width")


def _aggregate_artifacts(
    store: DashboardStore,
    registry: WidgetRegistry,
) -> pn.Column:
    """Build the aggregate artifact browser (no index selected).

    Parameters
    ----------
    store : DashboardStore
        The dashboard data store.
    registry : WidgetRegistry
        Widget registry for filter-specific visualizations.

    Returns
    -------
    pn.Column
        Column with artifact browser and preview.
    """
    all_artifacts = store.all_artifacts()

    if not all_artifacts:
        return pn.Column(pn.pane.Markdown("*No artifacts recorded.*"))

    components: list[pn.viewable.Viewable] = []
    components.append(pn.pane.Markdown("### All Artifacts"))

    for filter_name, paths in all_artifacts.items():
        components.append(pn.pane.Markdown(f"**{filter_name}** ({len(paths)} files)"))

        # Show a preview table of artifact paths
        artifact_df = pd.DataFrame({"path": paths})
        components.append(pn.pane.DataFrame(artifact_df, index=False, sizing_mode="stretch_width"))

        # Preview content for Parquet files
        parquet_paths = [p for p in paths if p.endswith(".parquet")]
        if parquet_paths:
            try:
                preview = pd.read_parquet(parquet_paths[0]).head(20)
                components.append(pn.pane.Markdown(f"*Preview of `{parquet_paths[0]}`* (first 20 rows):"))
                components.append(pn.pane.DataFrame(preview, index=False, sizing_mode="stretch_width"))
            except Exception:  # noqa: BLE001
                pass

        # Render widget if available
        widget = registry.get(filter_name)
        if widget is not None:
            try:
                components.append(widget.panel(paths, selected_index=None))
            except Exception as exc:  # noqa: BLE001
                components.append(pn.pane.Markdown(f"*Widget error: {exc}*"))

    return pn.Column(*components, sizing_mode="stretch_width")


def pipeline_tab(store: DashboardStore, registry: WidgetRegistry) -> pn.Column:
    """Build the Pipeline tab layout.

    Parameters
    ----------
    store : DashboardStore
        The dashboard data store.
    registry : WidgetRegistry
        Widget registry for filter-specific visualizations.

    Returns
    -------
    pn.Column
        Complete pipeline tab content.
    """
    # Pipeline structure
    structure = _pipeline_structure(store)

    # Index query controls
    query_input = pn.widgets.TextInput(
        name="Index Query",
        placeholder='e.g. "10-20", "1,5,10", or "all"',
        value="all",
        width=300,
    )
    status_filter = pn.widgets.Select(
        name="Status",
        options=["all", "completed", "error"],
        value="all",
        width=150,
    )

    # Reactive table and detail area
    table_pane = pn.Column(sizing_mode="stretch_width")
    detail_pane = pn.Column(sizing_mode="stretch_width")

    def _update(event: object = None) -> None:  # noqa: ARG001
        """Update the index table and detail view based on query."""
        df = store.index_df
        if df.empty:
            table_pane.clear()
            table_pane.append(pn.pane.Markdown("*No data available.*"))
            detail_pane.clear()
            return

        # Parse index query
        query = query_input.value or "all"
        valid_indices = _parse_index_query(query, int(df["index"].max()) + 1 if len(df) > 0 else 0)

        # Apply filters
        filtered = df[df["index"].isin(valid_indices)]
        status = status_filter.value
        if status != "all":
            filtered = filtered[filtered["status"] == status]

        # Update table
        table_pane.clear()
        if filtered.empty:
            table_pane.append(pn.pane.Markdown("*No matching indices.*"))
        else:
            display_cols = ["index", "status", "wall_time_s", "peak_memory_mb"]
            table_pane.append(
                pn.pane.DataFrame(
                    filtered[display_cols].round(3),
                    index=False,
                    sizing_mode="stretch_width",
                )
            )

        # Update detail view
        detail_pane.clear()
        sel = store.selected_index
        if sel is not None and sel >= 0:
            detail_pane.append(_artifact_detail(store, sel, registry))
        else:
            detail_pane.append(_aggregate_artifacts(store, registry))

    # Wire up callbacks
    query_input.param.watch(_update, "value")
    status_filter.param.watch(_update, "value")
    store.param.watch(_update, "selected_index")
    store.param.watch(_update, "refresh")

    # Initial render
    _update()

    return pn.Column(
        pn.pane.Markdown("## Pipeline"),
        structure,
        pn.layout.Divider(),
        pn.Row(query_input, status_filter),
        table_pane,
        pn.layout.Divider(),
        detail_pane,
        sizing_mode="stretch_width",
    )
