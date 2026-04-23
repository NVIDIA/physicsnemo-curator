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

"""MeanFilter artifact visualization widget."""

from __future__ import annotations

import panel as pn


class MeanFilterWidget:
    """Widget for visualizing MeanFilter Parquet artifacts.

    Reads the merged Parquet artifact and displays a bar chart of
    per-field mean values.  In overview mode (no selected index), shows
    a table of all rows.  In drill-down mode, highlights the selected
    index row and shows a bar chart for that index.
    """

    name: str = "Mesh Mean Statistics"
    filter_name: str = "MeanFilter"

    def panel(
        self,
        artifact_paths: list[str],
        selected_index: int | None = None,
    ) -> pn.viewable.Viewable:
        """Return a Panel component visualizing MeanFilter artifacts.

        Parameters
        ----------
        artifact_paths : list[str]
            Paths to Parquet files produced by MeanFilter.
        selected_index : int or None
            Currently selected pipeline index, if any.

        Returns
        -------
        pn.viewable.Viewable
            A Panel Column containing the visualization.
        """
        import pandas as pd

        if not artifact_paths:
            return pn.pane.Markdown("*No MeanFilter artifacts found.*")

        # Read and concatenate all Parquet files
        frames = []
        for path in artifact_paths:
            try:
                frames.append(pd.read_parquet(path))
            except Exception:  # noqa: BLE001
                continue

        if not frames:
            return pn.pane.Markdown("*Could not read any MeanFilter artifacts.*")

        df = pd.concat(frames, ignore_index=True)

        # Identify field columns (point_data/* and cell_data/*)
        field_cols = [c for c in df.columns if "/" in c]

        if selected_index is not None and selected_index < len(df):
            # Drill-down mode: bar chart for the selected index
            row = df.iloc[selected_index]
            field_values = {col: row[col] for col in field_cols if pd.notna(row[col])}

            if field_values:
                bar_df = pd.DataFrame({"field": list(field_values.keys()), "mean": list(field_values.values())})
                bar_plot = pn.pane.DataFrame(bar_df, index=False, sizing_mode="stretch_width")
            else:
                bar_plot = pn.pane.Markdown("*No field data for this index.*")

            header = pn.pane.Markdown(f"### MeanFilter — Index {selected_index}")
            table = pn.pane.DataFrame(
                df.style.apply(  # type: ignore[arg-type]
                    lambda x: ["background: #e6f3ff" if x.name == selected_index else "" for _ in x],
                    axis=1,
                ),
                sizing_mode="stretch_width",
            )
            return pn.Column(header, bar_plot, "---", "#### All Indices", table)

        # Overview mode: show full table
        header = pn.pane.Markdown(f"### MeanFilter — All Indices ({len(df)} rows)")
        table = pn.pane.DataFrame(df, index=False, sizing_mode="stretch_width")
        return pn.Column(header, table)

    def layout_hints(self) -> dict[str, int]:
        """Declare grid space preferences.

        Returns
        -------
        dict[str, int]
            Grid column and row span.
        """
        return {"cols": 6, "rows": 2}
