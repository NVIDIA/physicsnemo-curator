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

"""Step 3/4 — filter selection and configuration."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Collapsible, Input, Label, Select, SelectionList, Static

from physicsnemo_curator.core.base import Param
from physicsnemo_curator.core.registry import registry


class FilterScreen(Screen[None]):
    """Step 3/4: select and configure zero or more filters.

    Uses a :class:`~textual.widgets.SelectionList` for multi-select.  Each
    selected filter gets a collapsible parameter form.
    """

    BINDINGS = [("escape", "go_back", "Back")]

    DEFAULT_CSS = """
    FilterScreen {
        layout: vertical;
        padding: 1 2;
    }
    #step-label {
        text-style: bold;
        margin-bottom: 1;
    }
    #filter-list {
        height: auto;
        max-height: 30%;
        margin: 1 0;
    }
    #filter-params {
        height: auto;
        max-height: 50%;
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
        """Yield filter selection list, param area, and nav buttons."""
        from physicsnemo_curator.wiz.app import CuratorApp

        app: CuratorApp = self.app  # type: ignore[assignment]
        submodule = app.state.submodule
        filters = registry.filters(submodule)

        items = [(f"{cls.name} — {cls.description}", cls_name) for cls_name, cls in filters.items()]

        yield Static("Step 3/4: Select Filters (space to toggle)", id="step-label")

        if items:
            yield SelectionList(*items, id="filter-list")
        else:
            yield Static("No filters available for this submodule", id="filter-list")

        yield VerticalScroll(id="filter-params")
        with Vertical(classes="nav-row"):
            yield Button("← Back", id="back-btn", classes="nav-btn")
            yield Button("Next →", id="next-btn", classes="nav-btn")

    def on_selection_list_selected_changed(self, event: SelectionList.SelectedChanged) -> None:
        """Rebuild parameter forms when selection changes."""
        from physicsnemo_curator.wiz.app import CuratorApp

        app: CuratorApp = self.app  # type: ignore[assignment]
        container = self.query_one("#filter-params", VerticalScroll)
        container.remove_children()

        try:
            sel_list = self.query_one("#filter-list", SelectionList)
        except Exception:  # noqa: BLE001
            return

        selected_values = sel_list.selected
        filters = registry.filters(app.state.submodule)

        for filter_name in selected_values:
            filter_cls = filters[str(filter_name)]
            params: list[Param] = filter_cls.params()

            collapsible = Collapsible(title=f"Configure {filter_cls.name}", collapsed=False)
            for param in params:
                label_text = param.description
                if not param.required:
                    label_text += f" [{param.default}]"
                collapsible.compose_add_child(Label(label_text, classes="param-label"))

                if param.choices:
                    param_options = [(c, c) for c in param.choices]
                    widget = Select(
                        param_options,
                        prompt=f"Select {param.name}",
                        id=f"fparam-{filter_name}-{param.name}",
                    )
                else:
                    placeholder = "" if param.required else str(param.default)
                    widget = Input(placeholder=placeholder, id=f"fparam-{filter_name}-{param.name}")
                collapsible.compose_add_child(widget)

            container.mount(collapsible)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle Back / Next."""
        if event.button.id == "back-btn":
            self.app.pop_screen()
            return

        if event.button.id != "next-btn":
            return

        from physicsnemo_curator.wiz.app import CuratorApp

        app: CuratorApp = self.app  # type: ignore[assignment]
        filters = registry.filters(app.state.submodule)

        # Get selected filters (may be empty — that's fine)
        selected_values: list[str] = []
        try:
            sel_list = self.query_one("#filter-list", SelectionList)
            selected_values = [str(v) for v in sel_list.selected]
        except Exception:  # noqa: BLE001
            pass  # No filters available

        filter_classes: list[type] = []
        filter_kwargs_list: list[dict[str, Any]] = []
        filter_instances: list[Any] = []

        for filter_name in selected_values:
            filter_cls = filters[filter_name]
            params: list[Param] = filter_cls.params()
            kwargs: dict[str, Any] = {}

            for param in params:
                widget_id = f"#fparam-{filter_name}-{param.name}"
                try:
                    widget = self.query_one(widget_id)
                except Exception:  # noqa: BLE001
                    if param.required:
                        self.notify(f"Missing param '{param.name}' for {filter_cls.name}", severity="error")
                        return
                    kwargs[param.name] = param.default
                    continue

                if isinstance(widget, Select):
                    val = widget.value
                    if val is Select.BLANK and param.required:
                        self.notify(f"'{param.name}' is required for {filter_cls.name}", severity="warning")
                        return
                    kwargs[param.name] = str(val) if val is not Select.BLANK else param.default
                elif isinstance(widget, Input):
                    text = widget.value.strip()
                    if text == "" and param.required:
                        self.notify(f"'{param.name}' is required for {filter_cls.name}", severity="warning")
                        return
                    kwargs[param.name] = text if text else param.default

            try:
                instance = filter_cls(**kwargs)
            except Exception as exc:  # noqa: BLE001
                self.notify(f"Error creating {filter_cls.name}: {exc}", severity="error")
                return

            filter_classes.append(filter_cls)
            filter_kwargs_list.append(kwargs)
            filter_instances.append(instance)

        app.state.filter_classes = filter_classes
        app.state.filter_kwargs = filter_kwargs_list
        app.state.filter_instances = filter_instances

        from physicsnemo_curator.wiz.screens.sink import SinkScreen

        app.push_screen(SinkScreen())

    def action_go_back(self) -> None:
        """Pop this screen."""
        self.app.pop_screen()
