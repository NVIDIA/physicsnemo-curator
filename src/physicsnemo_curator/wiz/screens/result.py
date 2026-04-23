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

"""Result screen — final summary after pipeline execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.containers import Center, Vertical
from textual.screen import Screen
from textual.widgets import Button, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from physicsnemo_curator.wiz.app import CuratorApp


class ResultScreen(Screen[None]):
    """Display execution results and offer restart / quit options.

    Parameters
    ----------
    all_paths : list[list[str]]
        Output paths from each pipeline index.
    """

    DEFAULT_CSS = """
    ResultScreen {
        align: center middle;
    }
    #result-title {
        text-style: bold;
        text-align: center;
        color: $success;
        margin-bottom: 1;
    }
    #result-stats {
        text-align: center;
        margin-bottom: 2;
    }
    .result-btn {
        width: 30;
        margin: 1 0;
    }
    """

    def __init__(self, all_paths: list[list[str]] | None = None) -> None:
        super().__init__()
        self._all_paths = all_paths or []

    def compose(self) -> ComposeResult:
        """Yield result summary and action buttons."""
        app: CuratorApp = self.app  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        pipeline = app.state.pipeline
        n = len(pipeline) if pipeline else 0
        total_outputs = sum(len(p) for p in self._all_paths)

        lines: list[str] = [
            f"Source items processed: {n}",
            f"Outputs written: {total_outputs}",
        ]
        if pipeline and pipeline.track_metrics and pipeline.db_path is not None:
            lines.append(f"Pipeline DB: {pipeline.db_path}")

        with Center(), Vertical():
            yield Static("Complete", id="result-title")
            yield Static("\n".join(lines), id="result-stats")
            yield Button("New Pipeline", id="new-btn", classes="result-btn")
            yield Button("Quit", id="quit-btn", classes="result-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle new pipeline or quit."""
        if event.button.id == "new-btn":
            from physicsnemo_curator.wiz.screens.welcome import WelcomeScreen

            app: CuratorApp = self.app  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
            app.state.reset()
            # Clear the screen stack and start fresh
            while len(self.app.screen_stack) > 1:
                self.app.pop_screen()
            app.push_screen(WelcomeScreen())

        elif event.button.id == "quit-btn":
            self.app.exit()
