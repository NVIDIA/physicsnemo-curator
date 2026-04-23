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
import panel_material_ui as pmui

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
            try:
                widget_panel = registry.get_panel(filter_name, artifact_paths, selected_index=selected_index)
                if widget_panel is not None:
                    components.append(widget_panel)
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
        # Get unique file names only (not full paths which may have duplicates per index)
        unique_paths = sorted(set(paths))
        components.append(pn.pane.Markdown(f"**{filter_name}** ({len(unique_paths)} unique files)"))

        # Show unique artifact paths only
        artifact_df = pd.DataFrame({"path": unique_paths})
        components.append(pn.pane.DataFrame(artifact_df, index=False, sizing_mode="stretch_width"))

        # Preview content for Parquet files
        parquet_paths = [p for p in unique_paths if p.endswith(".parquet")]
        if parquet_paths:
            try:
                preview = pd.read_parquet(parquet_paths[0]).head(20)
                components.append(pn.pane.Markdown(f"*Preview of `{parquet_paths[0]}`* (first 20 rows):"))
                components.append(pn.pane.DataFrame(preview, index=False, sizing_mode="stretch_width"))
            except Exception:  # noqa: BLE001
                pass

        # Render widget if available
        try:
            widget_panel = registry.get_panel(filter_name, unique_paths, selected_index=None)
            if widget_panel is not None:
                components.append(widget_panel)
        except Exception as exc:  # noqa: BLE001
            components.append(pn.pane.Markdown(f"*Widget error: {exc}*"))

    return pn.Column(*components, sizing_mode="stretch_width")


def pipeline_tab(store: DashboardStore, registry: WidgetRegistry) -> pn.GridStack:
    """Build the Pipeline tab layout.

    Parameters
    ----------
    store : DashboardStore
        The dashboard data store.
    registry : WidgetRegistry
        Widget registry for filter-specific visualizations.

    Returns
    -------
    pn.GridStack
        GridStack layout with draggable, resizable tiles.
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

    # Pagination controls
    page_size_select = pn.widgets.Select(
        name="Page Size",
        options=[20, 50, 100],
        value=20,
        width=100,
    )
    page_input = pn.widgets.IntInput(
        name="Page",
        value=1,
        start=1,
        end=1,
        width=80,
    )
    page_info = pn.pane.Markdown("", width=200)

    # Reactive table and detail area
    table_pane = pn.Column(sizing_mode="stretch_width")
    detail_pane = pn.Column(sizing_mode="stretch_width")

    # Track filtered data for pagination
    _state: dict[str, object] = {"filtered_df": None, "total_pages": 1}

    def _update_table() -> None:
        """Update the table display with current page."""
        filtered = _state["filtered_df"]
        if filtered is None or not hasattr(filtered, "empty") or filtered.empty:
            table_pane.clear()
            table_pane.append(pn.pane.Markdown("*No matching indices.*"))
            page_info.object = ""
            return

        # Calculate pagination
        page_size = page_size_select.value
        total_rows = len(filtered)  # ty: ignore[invalid-argument-type]
        total_pages = max(1, (total_rows + page_size - 1) // page_size)
        _state["total_pages"] = total_pages

        # Clamp page to valid range
        page_input.end = total_pages
        current_page = min(max(1, page_input.value), total_pages)
        if page_input.value != current_page:
            page_input.value = current_page

        # Slice for current page
        start_idx = (current_page - 1) * page_size
        end_idx = start_idx + page_size
        page_df = filtered.iloc[start_idx:end_idx]  # ty: ignore[unresolved-attribute]

        # Update info
        start_row = start_idx + 1
        end_row = min(end_idx, total_rows)
        page_info.object = f"**{start_row}-{end_row}** of **{total_rows}**"

        # Update table
        table_pane.clear()
        display_cols = ["index", "status", "wall_time_s", "peak_memory_mb"]
        table_pane.append(
            pn.pane.DataFrame(
                page_df[display_cols].round(3),
                index=False,
                sizing_mode="stretch_width",
            )
        )

    def _update_filter(event: object = None) -> None:  # noqa: ARG001
        """Update filtered data based on query and status."""
        df = store.index_df
        if df.empty:
            _state["filtered_df"] = None
            table_pane.clear()
            table_pane.append(pn.pane.Markdown("*No data available.*"))
            detail_pane.clear()
            page_info.object = ""
            return

        # Parse index query
        query = query_input.value or "all"
        valid_indices = _parse_index_query(query, int(df["index"].max()) + 1 if len(df) > 0 else 0)

        # Apply filters
        filtered = df[df["index"].isin(valid_indices)]
        status = status_filter.value
        if status != "all":
            filtered = filtered[filtered["status"] == status]

        _state["filtered_df"] = filtered

        # Reset to page 1 when filter changes
        page_input.value = 1
        _update_table()

        # Update detail view
        detail_pane.clear()
        sel = store.selected_index
        if sel is not None and sel >= 0:
            detail_pane.append(_artifact_detail(store, sel, registry))
        else:
            detail_pane.append(_aggregate_artifacts(store, registry))

    def _on_page_change(event: object = None) -> None:  # noqa: ARG001
        """Handle page or page size changes."""
        _update_table()

    # Wire up callbacks
    query_input.param.watch(_update_filter, "value")
    status_filter.param.watch(_update_filter, "value")
    store.param.watch(_update_filter, "selected_index")
    store.param.watch(_update_filter, "refresh")
    page_size_select.param.watch(_on_page_change, "value")
    page_input.param.watch(_on_page_change, "value")

    # Initial render
    _update_filter()

    # Pagination buttons
    prev_btn = pn.widgets.Button(name="<", width=40, button_type="default")
    next_btn = pn.widgets.Button(name=">", width=40, button_type="default")

    # Pagination row
    pagination_row = pn.Row(
        page_size_select,
        prev_btn,
        page_input,
        next_btn,
        page_info,
        align="center",
    )

    # Wire up prev/next buttons
    def _prev_page(event: object) -> None:  # noqa: ARG001
        if page_input.value > 1:
            page_input.value -= 1

    def _next_page(event: object) -> None:  # noqa: ARG001
        if page_input.value < _state["total_pages"]:
            page_input.value += 1

    prev_btn.on_click(_prev_page)
    next_btn.on_click(_next_page)

    # Build query controls column
    controls = pn.Column(
        pn.Row(query_input, status_filter),
        pagination_row,
    )

    # Build GridStack
    gstack = pn.GridStack(sizing_mode="stretch_both", min_height=800, allow_drag=True, allow_resize=True)

    gstack[0:2, 0:12] = pmui.Paper(structure, elevation=2)
    gstack[2:3, 0:12] = pmui.Paper(controls, elevation=2)
    gstack[3:5, 0:8] = pmui.Paper(table_pane, elevation=2)
    gstack[3:5, 8:12] = pmui.Paper(detail_pane, elevation=2)

    # Dynamic filter widget tiles
    next_row = 5
    all_artifacts = store.all_artifacts()
    for filter_name, paths in all_artifacts.items():
        try:
            tile = registry.get_panel(filter_name, sorted(set(paths)), selected_index=None)
            if tile is not None:
                hints = registry.get_layout_hints(filter_name)
                cols = hints.get("cols", 12)
                rows = hints.get("rows", 2)
                gstack[next_row : next_row + rows, 0:cols] = pmui.Paper(tile, elevation=2)
                next_row += rows
        except Exception:  # noqa: BLE001
            pass

    return gstack
