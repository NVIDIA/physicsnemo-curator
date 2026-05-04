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

"""Main dashboard application with tab layout and manual refresh."""

from __future__ import annotations

from datetime import UTC, datetime

import panel as pn

try:
    import panel_material_ui as pmui
except ImportError as exc:
    raise ImportError(
        "panel-material-ui is required for the dashboard. Install it with: pip install 'physicsnemo-curator[dashboard]'"
    ) from exc

from physicsnemo_curator.dashboard.data import DashboardStore
from physicsnemo_curator.dashboard.views.overview import overview_tab
from physicsnemo_curator.dashboard.views.performance import performance_tab
from physicsnemo_curator.dashboard.views.pipeline import pipeline_tab
from physicsnemo_curator.dashboard.widgets import WidgetRegistry


class DashboardApp:
    """Interactive pipeline metrics dashboard.

    Creates a 3-tab Panel application (Overview, Pipeline, Performance)
    backed by a :class:`DashboardStore`.  Data is refreshed on demand
    via a toolbar button.  Uses ``panel-material-ui`` for Material
    Design theming.
    """

    def __init__(self, db_path: str) -> None:
        """Initialize the dashboard application.

        Parameters
        ----------
        db_path : str
            Path to an existing PipelineStore SQLite database.
        """
        self.store = DashboardStore(db_path)
        self.widget_registry = WidgetRegistry()
        self._page: pmui.Page | None = None

    def _build_app(self) -> pmui.Page:
        """Build the application shell with Material UI theming.

        Returns
        -------
        pmui.Page
            The themed dashboard page.
        """
        pn.extension("bokeh")

        tabs = pmui.Tabs(
            ("Overview", overview_tab(self.store)),
            ("Pipeline", pipeline_tab(self.store, self.widget_registry)),
            ("Performance", performance_tab(self.store)),
            sizing_mode="stretch_both",
        )

        # Refresh button and timestamp label
        timestamp_label = pn.pane.Markdown(
            self._timestamp_text(),
            styles={"margin": "auto 0", "font-size": "12px", "color": "#666"},
        )
        refresh_btn = pn.widgets.Button(
            name="Refresh",
            button_type="primary",
            width=90,
        )

        def _on_refresh(event: object) -> None:  # noqa: ARG001
            self.store.param.trigger("refresh")
            timestamp_label.object = self._timestamp_text()

        refresh_btn.on_click(_on_refresh)

        toolbar = pn.Row(
            refresh_btn,
            timestamp_label,
            sizing_mode="stretch_width",
            styles={"padding": "4px 16px", "background": "#f5f5f5"},
        )

        page = pmui.Page(
            main=[toolbar, tabs],
            title="PhysicsNeMo Curator",
            theme_toggle=True,
            theme_config={
                "palette": {
                    "primary": {"main": "#76b900"},  # NVIDIA green
                }
            },
        )

        return page

    @staticmethod
    def _timestamp_text() -> str:
        """Return a formatted 'Last refreshed' markdown string.

        Returns
        -------
        str
            Markdown text with the current UTC timestamp.
        """
        now = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        return f"*Last loaded: {now}*"

    def servable(self) -> pmui.Page:
        """Return the Page object for embedding in notebooks.

        Returns
        -------
        pmui.Page
            The dashboard page, ready for ``panel.servable()``.
        """
        if self._page is None:
            self._page = self._build_app()
        return self._page

    def serve(self, port: int = 5006, open_browser: bool = True) -> None:
        """Start the Panel server.

        Parameters
        ----------
        port : int
            Port number for the server.
        open_browser : bool
            Whether to open a browser window on launch.
        """
        if self._page is None:
            self._page = self._build_app()

        pn.serve(
            self._page,
            port=port,
            show=open_browser,
            title="PhysicsNeMo Curator Dashboard",
        )
