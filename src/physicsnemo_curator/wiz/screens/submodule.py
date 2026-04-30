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

"""Step 1/4 — submodule selection screen."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Select, Static

from physicsnemo_curator.core.registry import registry

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from physicsnemo_curator.wiz.app import CuratorApp

# Map submodule names to their Python module paths so we can import them
# on demand (triggering component registration).
_SUBMODULE_IMPORTS: dict[str, str] = {
    "mesh": "physicsnemo_curator.domains.mesh",
    "da": "physicsnemo_curator.domains.da",
    "atm": "physicsnemo_curator.domains.atm",
}


def _ensure_submodules_registered() -> None:
    """Import submodule packages so they register with the global registry.

    Only submodules whose dependencies are installed will succeed; others
    are registered as placeholders with availability info.
    """
    for name, module_path in _SUBMODULE_IMPORTS.items():
        try:
            importlib.import_module(module_path)
        except ImportError:
            dep_map = {"mesh": "physicsnemo.mesh", "da": "xarray", "atm": "nvalchemi.data"}
            desc_map = {
                "mesh": "Mesh data curation (physicsnemo.mesh.Mesh)",
                "da": "DataArray data curation (xarray.DataArray)",
                "atm": "Atomic data curation (nvalchemi.data.AtomicData)",
            }
            registry.register_submodule(
                name,
                desc_map.get(name, name),
                dep_map.get(name, name),
            )


class SubmoduleScreen(Screen[None]):
    """Step 1/4: select a domain submodule.

    Discovers available submodules via the global registry and presents
    them in a :class:`~textual.widgets.Select` dropdown.
    """

    BINDINGS = [("escape", "go_back", "Back")]

    DEFAULT_CSS = """
    SubmoduleScreen {
        align: center middle;
    }
    #step-label {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    #submodule-select {
        width: 60;
        margin: 1 0;
    }
    #submodule-select .option-list--option {
        padding: 1 1;
        border-bottom: solid $surface-darken-1;
    }
    .nav-btn {
        margin: 1 1;
    }
    """

    def compose(self) -> ComposeResult:
        """Yield step label, submodule selector, and navigation buttons."""
        _ensure_submodules_registered()
        submodules = registry.submodules()

        options: list[tuple[str, str]] = []
        for name, entry in submodules.items():
            label = f"{name} — {entry.description}"
            if not entry.available:
                label += " (not installed)"
            options.append((label, name))

        with Vertical():
            yield Static("Step 1/4: Select Submodule", id="step-label")
            yield Select(options, prompt="Choose a submodule", id="submodule-select")
            yield Button("Next →", id="next-btn", classes="nav-btn")
            yield Button("Quit", id="quit-btn", classes="nav-btn", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle Next button — validate and push SourceScreen."""
        if event.button.id == "quit-btn":
            self.app.exit()
            return

        if event.button.id != "next-btn":
            return

        app: CuratorApp = self.app  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        select = self.query_one("#submodule-select", Select)
        value = select.value

        if value is Select.BLANK:
            self.notify("Please select a submodule", severity="warning")
            return

        # Check availability
        submodules = registry.submodules()
        entry = submodules.get(str(value))
        if entry and not entry.available:
            self.notify(f"Submodule '{value}' is not installed", severity="error")
            return

        app.state.submodule = str(value)

        from physicsnemo_curator.wiz.screens.source import SourceScreen

        app.push_screen(SourceScreen())

    def action_go_back(self) -> None:
        """Pop this screen to return to Welcome."""
        self.app.pop_screen()
