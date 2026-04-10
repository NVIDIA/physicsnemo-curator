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

"""Main dashboard application with tab layout and auto-refresh."""

from __future__ import annotations

import panel as pn

from physicsnemo_curator.dashboard.data import DashboardStore
from physicsnemo_curator.dashboard.views.overview import overview_tab
from physicsnemo_curator.dashboard.views.performance import performance_tab
from physicsnemo_curator.dashboard.views.pipeline import pipeline_tab
from physicsnemo_curator.dashboard.widgets import WidgetRegistry


class DashboardApp:
    """Interactive pipeline metrics dashboard.

    Creates a 3-tab Panel application (Overview, Pipeline, Performance)
    backed by a :class:`DashboardStore` and auto-refreshes on a timer.

    Parameters
    ----------
    db_path : str
        Path to the PipelineStore SQLite database.
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
        self._tabs: pn.Tabs | None = None
        self._periodic: pn.state.PeriodicCallback | None = None  # type: ignore[name-defined]

    def _build_tabs(self) -> pn.Tabs:
        """Build the tab layout.

        Returns
        -------
        pn.Tabs
            The 3-tab dashboard layout.
        """
        return pn.Tabs(
            ("Overview", overview_tab(self.store)),
            ("Pipeline", pipeline_tab(self.store, self.widget_registry)),
            ("Performance", performance_tab(self.store)),
            sizing_mode="stretch_width",
        )

    def servable(self) -> pn.Tabs:
        """Return the Panel Tabs object for embedding in notebooks.

        Returns
        -------
        pn.Tabs
            The dashboard tabs, ready for ``panel.servable()``.
        """
        if self._tabs is None:
            self._tabs = self._build_tabs()
        return self._tabs

    def serve(self, port: int = 5006, open_browser: bool = True) -> None:
        """Start the Panel server.

        Parameters
        ----------
        port : int
            Port number for the server.
        open_browser : bool
            Whether to open a browser window on launch.
        """
        if self._tabs is None:
            self._tabs = self._build_tabs()

        # Set up auto-refresh
        def _refresh() -> None:
            self.store.param.trigger("refresh")

        pn.serve(
            self._tabs,
            port=port,
            show=open_browser,
            title="PhysicsNeMo Curator Dashboard",
        )
