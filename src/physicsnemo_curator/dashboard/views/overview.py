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

"""Overview tab — progress, runner status, pipeline info, recent files."""

from __future__ import annotations

from typing import TYPE_CHECKING

import panel as pn
import panel_material_ui as pmui

if TYPE_CHECKING:
    from physicsnemo_curator.dashboard.data import DashboardStore


def _summary_cards(store: DashboardStore) -> pn.Row:
    """Build the top-row summary cards.

    Parameters
    ----------
    store : DashboardStore
        The dashboard data store.

    Returns
    -------
    pn.Row
        Row of summary indicator cards.
    """
    summary = store.summary
    completed = summary.get("completed", 0)
    total = summary.get("total", 0)
    failed = summary.get("failed", 0)
    remaining = summary.get("remaining", 0)
    elapsed = summary.get("total_elapsed_s", 0.0)

    pct = (completed / total * 100) if total > 0 else 0

    # Format elapsed time
    if elapsed >= 3600:
        elapsed_str = f"{elapsed / 3600:.1f}h"
    elif elapsed >= 60:
        elapsed_str = f"{elapsed / 60:.1f}m"
    else:
        elapsed_str = f"{elapsed:.1f}s"

    progress_card = pn.indicators.Number(
        name="Completed",
        value=completed,
        format=f"{{value}} / {total} ({pct:.0f}%)",
        default_color="green" if completed == total else "blue",
        font_size="24pt",
        title_size="12pt",
    )
    failed_card = pn.indicators.Number(
        name="Failed",
        value=failed,
        default_color="red" if failed > 0 else "gray",
        font_size="24pt",
        title_size="12pt",
    )
    remaining_card = pn.indicators.Number(
        name="Remaining",
        value=remaining,
        default_color="orange" if remaining > 0 else "gray",
        font_size="24pt",
        title_size="12pt",
    )
    elapsed_card = pn.indicators.Number(
        name="Elapsed",
        value=0,
        format=elapsed_str,
        default_color="black",
        font_size="24pt",
        title_size="12pt",
    )

    return pn.Row(progress_card, failed_card, remaining_card, elapsed_card, sizing_mode="stretch_width")


def _worker_table(store: DashboardStore) -> pn.Column:
    """Build the worker status table.

    Parameters
    ----------
    store : DashboardStore
        The dashboard data store.

    Returns
    -------
    pn.Column
        Column with header and worker table.
    """
    df = store.workers_df
    if df.empty:
        return pn.Column(
            pn.pane.Markdown("### Workers"),
            pn.pane.Markdown("*No workers registered.*"),
        )
    return pn.Column(
        pn.pane.Markdown(f"### Workers ({len(df)})"),
        pn.pane.DataFrame(df, index=False, sizing_mode="stretch_width"),
    )


def _pipeline_info(store: DashboardStore) -> pn.Column:
    """Build the pipeline configuration summary.

    Parameters
    ----------
    store : DashboardStore
        The dashboard data store.

    Returns
    -------
    pn.Column
        Column showing pipeline structure.
    """
    config = store.pipeline_config

    source_name = config.get("source", {}).get("name", "Unknown")
    filters = config.get("filters", [])
    filter_names = [f.get("name", "?") for f in filters] if filters else ["(none)"]
    sink_name = config.get("sink", {}).get("name", "None")

    chain = f"**{source_name}** → " + " → ".join(f"*{n}*" for n in filter_names) + f" → **{sink_name}**"

    components: list[pn.viewable.Viewable] = [
        pn.pane.Markdown("### Pipeline"),
        pn.pane.Markdown(chain),
    ]

    # Add stage timing summary if data is available
    stage_df = store.stage_df
    if not stage_df.empty:
        # Aggregate stage timings across all indices
        stage_summary = stage_df.groupby("stage_name", sort=False)["wall_time_s"].agg(["mean", "sum", "count"])
        stage_summary = stage_summary.rename(columns={"mean": "Mean (s)", "sum": "Total (s)", "count": "Indices"})
        stage_summary.index.name = "Stage"
        stage_summary = stage_summary.reset_index()
        # Sort by stage_order
        order_map = stage_df.drop_duplicates("stage_name").set_index("stage_name")["stage_order"]
        stage_summary["_order"] = stage_summary["Stage"].map(order_map)
        stage_summary = stage_summary.sort_values("_order").drop(columns="_order")
        stage_summary["Mean (s)"] = stage_summary["Mean (s)"].round(2)
        stage_summary["Total (s)"] = stage_summary["Total (s)"].round(2)

        components.append(pn.pane.Markdown("#### Stage Timings"))
        components.append(pn.pane.DataFrame(stage_summary, index=False, sizing_mode="stretch_width"))

    return pn.Column(*components)


def _recent_files(store: DashboardStore) -> pn.Column:
    """Build the recent output files section.

    Parameters
    ----------
    store : DashboardStore
        The dashboard data store.

    Returns
    -------
    pn.Column
        Column showing recently produced output files.
    """
    df = store.index_df
    if df.empty:
        return pn.Column(
            pn.pane.Markdown("### Recent Output Files"),
            pn.pane.Markdown("*No completed indices yet.*"),
        )

    completed = df[df["status"] == "completed"].tail(20)
    rows = []
    for _, row in completed.iterrows():
        paths = store.output_paths(int(row["index"]))
        for p in paths:
            rows.append({"index": int(row["index"]), "path": p})

    if not rows:
        return pn.Column(
            pn.pane.Markdown("### Recent Output Files"),
            pn.pane.Markdown("*No output files recorded.*"),
        )

    import pandas as pd

    files_df = pd.DataFrame(rows).tail(20)
    return pn.Column(
        pn.pane.Markdown(f"### Recent Output Files ({len(files_df)})"),
        pn.pane.DataFrame(files_df, index=False, sizing_mode="stretch_width"),
    )


def _error_log(store: DashboardStore) -> pn.Column:
    """Build the error log section.

    Parameters
    ----------
    store : DashboardStore
        The dashboard data store.

    Returns
    -------
    pn.Column
        Column showing recent errors, or empty if none.
    """
    df = store.index_df
    errors = df[df["status"] == "error"]
    if errors.empty:
        return pn.Column()

    display = errors[["index", "error"]].tail(10)
    return pn.Column(
        pn.pane.Markdown(f"### Errors ({len(errors)} total)"),
        pn.pane.DataFrame(display, index=False, sizing_mode="stretch_width"),
    )


def overview_tab(store: DashboardStore) -> pn.Column:
    """Build the Overview tab layout with auto-refresh support.

    The tab content rebuilds each time ``store.refresh`` is triggered,
    re-querying the database for fresh metrics.

    Parameters
    ----------
    store : DashboardStore
        The dashboard data store.

    Returns
    -------
    pn.Column
        Reactive vertical layout that updates on refresh.
    """
    content_area = pn.Column(sizing_mode="stretch_width", scroll=True)

    def _rebuild() -> None:
        """Rebuild all overview panels from current database state."""
        content_area.clear()
        content_area.extend(
            [
                pmui.Paper(_summary_cards(store), elevation=2),
                pmui.Paper(_worker_table(store), elevation=2),
                pmui.Paper(
                    pn.Column(_pipeline_info(store), _recent_files(store)),
                    elevation=2,
                ),
                pmui.Paper(_error_log(store), elevation=2),
            ]
        )

    # Watch for refresh events to rebuild
    store.param.watch(lambda event: _rebuild(), "refresh")

    # Initial render
    _rebuild()

    return content_area
