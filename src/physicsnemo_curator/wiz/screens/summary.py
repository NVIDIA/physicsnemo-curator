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

"""Summary screen — review pipeline, optionally save, then execute."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.containers import Center, Vertical
from textual.screen import Screen
from textual.widgets import Button, Input, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from physicsnemo_curator.wiz.app import CuratorApp


class SummaryScreen(Screen[None]):
    """Display the assembled pipeline chain and offer save / execute options.

    Shows: ``Source -> Filter1 -> Filter2 -> Sink`` with item count.
    Buttons: "Save to YAML", "Save to JSON", "Execute".
    """

    BINDINGS = [("escape", "go_back", "Back")]

    DEFAULT_CSS = """
    SummaryScreen {
        align: center middle;
    }
    #chain-label {
        text-align: center;
        text-style: bold;
        margin: 1 0;
    }
    #item-count {
        text-align: center;
        color: $text-muted;
        margin-bottom: 2;
    }
    .action-btn {
        width: 30;
        margin: 1 0;
    }
    #save-input {
        display: none;
        margin: 1 4;
        width: 50;
    }
    """

    def compose(self) -> ComposeResult:
        """Yield the pipeline chain, item count, and action buttons."""
        app: CuratorApp = self.app  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        pipeline = app.state.pipeline

        # Build chain string
        parts: list[str] = []
        if pipeline is not None:
            parts.append(pipeline.source.name)
            for f in pipeline.filters:
                parts.append(f.name)
            parts.append(pipeline.sink.name)  # ty: ignore[unresolved-attribute]

        chain_str = " -> ".join(parts) if parts else "(empty pipeline)"
        item_count = len(pipeline) if pipeline is not None else 0

        with Center(), Vertical():
            yield Static("Pipeline Summary", id="chain-label")
            yield Static(chain_str)
            yield Static(f"{item_count} items", id="item-count")
            yield Button("Save to YAML", id="save-yaml-btn", classes="action-btn")
            yield Button("Save to JSON", id="save-json-btn", classes="action-btn")
            yield Input(placeholder="Save path", id="save-input")
            yield Button("Execute", id="execute-btn", classes="action-btn", variant="primary")
            yield Button("Quit", id="quit-btn", classes="action-btn", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle save and execute actions."""
        if event.button.id == "save-yaml-btn":
            save_input = self.query_one("#save-input", Input)
            save_input.value = "pipeline.yaml"
            save_input.styles.display = "block"
            save_input.focus()
            save_input.name = "yaml"  # ty: ignore[invalid-assignment]

        elif event.button.id == "save-json-btn":
            save_input = self.query_one("#save-input", Input)
            save_input.value = "pipeline.json"
            save_input.styles.display = "block"
            save_input.focus()
            save_input.name = "json"  # ty: ignore[invalid-assignment]

        elif event.button.id == "execute-btn":
            from physicsnemo_curator.wiz.screens.execution import ExecutionScreen

            app: CuratorApp = self.app  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
            app.push_screen(ExecutionScreen())

        elif event.button.id == "quit-btn":
            self.app.exit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Save the pipeline to the specified path."""
        from physicsnemo_curator.core.serialization import save_pipeline

        app: CuratorApp = self.app  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        path = event.value.strip()
        if not path:
            return

        try:
            save_pipeline(app.state.pipeline, path)
            self.notify(f"Saved to {path}", severity="information")
        except Exception as exc:  # noqa: BLE001
            self.notify(f"Save failed: {exc}", severity="error")

        event.input.styles.display = "none"  # ty: ignore[invalid-assignment]

    def action_go_back(self) -> None:
        """Pop this screen."""
        self.app.pop_screen()
