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

"""Step 2/4 — source/reader selection and parameter configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Input, Label, Select, Static

from physicsnemo_curator.core.registry import registry

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from physicsnemo_curator.core.base import Param
    from physicsnemo_curator.wiz.app import CuratorApp


class SourceScreen(Screen[None]):
    """Step 2/4: select a data source and configure its parameters.

    The parameter form rebuilds dynamically when the source selection changes.
    """

    BINDINGS = [("escape", "go_back", "Back")]

    DEFAULT_CSS = """
    SourceScreen {
        layout: vertical;
        padding: 1 2;
    }
    #step-label {
        text-style: bold;
        margin-bottom: 1;
    }
    #source-select {
        width: 60;
        margin: 1 0;
    }
    #param-container {
        height: auto;
        max-height: 70%;
        margin: 1 0;
    }
    .param-label {
        margin-top: 1;
    }
    .nav-row {
        layout: horizontal;
        height: auto;
    }
    .nav-btn {
        margin: 1 1;
    }
    """

    def compose(self) -> ComposeResult:
        """Yield step label, source selector, param form, and nav buttons."""
        app: CuratorApp = self.app  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        submodule = app.state.submodule
        sources = registry.sources(submodule)

        options = [(f"{cls.name} — {cls.description}", cls_name) for cls_name, cls in sources.items()]

        yield Static("Step 2/4: Select Source / Reader", id="step-label")
        yield Select(options, prompt="Choose a source", id="source-select")
        yield VerticalScroll(id="param-container")
        with Vertical(classes="nav-row"):
            yield Button("← Back", id="back-btn", classes="nav-btn")
            yield Button("Next →", id="next-btn", classes="nav-btn")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Rebuild the parameter form when source selection changes."""
        if event.select.id != "source-select":
            return

        app: CuratorApp = self.app  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        container = self.query_one("#param-container", VerticalScroll)
        container.remove_children()

        if event.value is Select.BLANK:
            return

        sources = registry.sources(app.state.submodule)
        source_cls = sources[str(event.value)]
        params: list[Param] = source_cls.params()

        for param in params:
            label_text = param.description
            if not param.required:
                label_text += f" [{param.default}]"
            container.mount(Label(label_text, classes="param-label"))

            if param.choices:
                param_options = [(c, c) for c in param.choices]
                widget = Select(param_options, prompt=f"Select {param.name}", id=f"param-{param.name}")
            else:
                placeholder = "" if param.required else str(param.default)
                widget = Input(placeholder=placeholder, id=f"param-{param.name}")
            container.mount(widget)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle Back / Next navigation."""
        if event.button.id == "back-btn":
            self.app.pop_screen()
            return

        if event.button.id != "next-btn":
            return

        app: CuratorApp = self.app  # type: ignore[assignment]  # ty: ignore[invalid-assignment]

        select = self.query_one("#source-select", Select)
        if select.value is Select.BLANK:
            self.notify("Please select a source", severity="warning")
            return

        sources = registry.sources(app.state.submodule)
        source_cls = sources[str(select.value)]
        params: list[Param] = source_cls.params()
        kwargs: dict[str, Any] = {}

        for param in params:
            widget = self.query_one(f"#param-{param.name}")
            if isinstance(widget, Select):
                val = widget.value
                if val is Select.BLANK and param.required:
                    self.notify(f"'{param.name}' is required", severity="warning")
                    return
                kwargs[param.name] = str(val) if val is not Select.BLANK else param.default
            elif isinstance(widget, Input):
                text = widget.value.strip()
                if text == "" and param.required:
                    self.notify(f"'{param.name}' is required", severity="warning")
                    return
                kwargs[param.name] = text if text else param.default

        try:
            source_instance = source_cls(**kwargs)
        except Exception as exc:  # noqa: BLE001
            self.notify(f"Error creating source: {exc}", severity="error")
            return

        app.state.source_cls = source_cls
        app.state.source_kwargs = kwargs
        app.state.source_instance = source_instance

        from physicsnemo_curator.wiz.screens.filters import FilterScreen

        app.push_screen(FilterScreen())

    def action_go_back(self) -> None:
        """Pop this screen."""
        self.app.pop_screen()
