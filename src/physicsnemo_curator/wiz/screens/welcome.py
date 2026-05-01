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

"""Welcome screen for the pipeline wizard.

Offers four modes: build a new pipeline, load an existing one,
open the metrics dashboard, or manage the pipeline database cache.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Input, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from physicsnemo_curator.wiz.app import CuratorApp


class WelcomeScreen(Screen[None]):
    """Landing screen with mode selection buttons.

    Modes:
    - **Build** — push :class:`SubmoduleScreen`
    - **Load** — prompt for file path, load pipeline, push :class:`SummaryScreen`
    - **Dashboard** — push :class:`DashboardScreen`
    - **Cache** — push :class:`CacheScreen`
    """

    DEFAULT_CSS = """
    WelcomeScreen {
        align: center middle;
    }
    WelcomeScreen > Vertical {
        width: auto;
        height: auto;
        align: center middle;
    }
    #banner {
        text-align: center;
        padding: 1 2;
        text-style: bold;
        color: #76B900;
    }
    #subtitle {
        text-align: center;
        color: $text-muted;
        margin-bottom: 2;
    }
    .btn-row {
        width: auto;
        height: auto;
        align: center middle;
    }
    #load-input {
        display: none;
        width: 40;
        margin: 1 0;
    }
    .welcome-btn {
        width: 24;
        margin: 1 1;
    }
    #quit-row {
        width: 100%;
        height: auto;
        align: center middle;
        margin-top: 1;
    }
    #quit-btn {
        width: 24;
    }
    """

    def compose(self) -> ComposeResult:
        """Yield the banner, mode buttons in a 2x2 grid, and quit button."""
        with Vertical():
            yield Static("PhysicsNeMo Curator", id="banner")
            yield Static("Interactive ETL Pipeline Wizard", id="subtitle")
            with Horizontal(classes="btn-row"):
                yield Button("Build pipeline", id="build-btn", classes="welcome-btn")
                yield Button("Load pipeline", id="load-btn", classes="welcome-btn")
            with Horizontal(classes="btn-row"):
                yield Button("Open dashboard", id="dashboard-btn", classes="welcome-btn")
                yield Button("Manage cache", id="cache-btn", classes="welcome-btn")
            yield Input(placeholder="Path to pipeline file (YAML / JSON)", id="load-input")
            with Horizontal(id="quit-row"):
                yield Button("Quit", id="quit-btn", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle mode selection."""
        app: CuratorApp = self.app  # type: ignore[assignment]  # ty: ignore[invalid-assignment]

        if event.button.id == "build-btn":
            app.state.mode = "build"
            from physicsnemo_curator.wiz.screens.submodule import SubmoduleScreen

            app.push_screen(SubmoduleScreen())

        elif event.button.id == "load-btn":
            load_input = self.query_one("#load-input", Input)
            if load_input.styles.display == "none":
                load_input.styles.display = "block"
                load_input.focus()
            else:
                load_input.styles.display = "none"  # ty: ignore[invalid-assignment]

        elif event.button.id == "cache-btn":
            from physicsnemo_curator.wiz.screens.cache import CacheScreen

            app.push_screen(CacheScreen())

        elif event.button.id == "dashboard-btn":
            from physicsnemo_curator.wiz.screens.dashboard import DashboardScreen

            app.push_screen(DashboardScreen())

        elif event.button.id == "quit-btn":
            self.app.exit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Load a pipeline file and push the summary screen."""
        from physicsnemo_curator.core.serialization import load_pipeline

        app: CuratorApp = self.app  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        path = event.value.strip()
        if not path:
            return

        try:
            pipeline = load_pipeline(path)
        except Exception as exc:  # noqa: BLE001
            self.notify(f"Failed to load: {exc}", severity="error")
            return

        app.state.mode = "load"
        app.state.pipeline = pipeline
        from physicsnemo_curator.wiz.screens.summary import SummaryScreen

        app.push_screen(SummaryScreen())
