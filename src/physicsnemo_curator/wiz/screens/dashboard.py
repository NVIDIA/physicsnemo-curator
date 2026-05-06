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

"""Dashboard launch screen — select a pipeline database and open the metrics dashboard."""

from __future__ import annotations

import atexit
import contextlib
import subprocess
import sys
from typing import TYPE_CHECKING

from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Input, Select, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult


class DashboardScreen(Screen[None]):
    """Select a pipeline database and launch the metrics dashboard.

    Displays a dropdown of available pipeline databases from the cache
    directory, a port input field, and launch/stop buttons that manage
    the Panel dashboard subprocess.
    """

    BINDINGS = [("escape", "go_back", "Back")]

    DEFAULT_CSS = """
    DashboardScreen {
        align: center middle;
    }
    DashboardScreen > Vertical {
        width: 70;
        height: auto;
        padding: 1 2;
    }
    #dash-title {
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
    }
    #dash-subtitle {
        color: $text-muted;
        text-align: center;
        margin-bottom: 2;
    }
    #db-select .option-list--option {
        padding: 0 1 1 1;
        border-bottom: solid $surface-darken-1;
    }
    #port-row {
        height: auto;
        margin: 1 0;
    }
    #port-label {
        width: 10;
        padding: 1 0;
    }
    #port-input {
        width: 20;
    }
    #status-label {
        margin: 1 0;
        color: $success;
        text-align: center;
    }
    .dash-btn {
        width: 100%;
        margin: 1 0;
    }
    #stop-btn {
        display: none;
    }
    """

    def __init__(self) -> None:
        """Initialize the dashboard screen."""
        super().__init__()
        self._process: subprocess.Popen[bytes] | None = None
        self._atexit_registered: bool = False

    def compose(self) -> ComposeResult:
        """Yield the database selector, port input, and action buttons."""
        from physicsnemo_curator.core.cache import default_cache_dir, list_databases

        cache_dir = default_cache_dir()
        databases = list_databases(cache_dir)

        options: list[tuple[str, str]] = []
        for db in databases:
            label = (
                f"[{db.created.strftime('%Y-%m-%d %H:%M')}] "
                f"{db.source_name} \u2192 {db.sink_name} "
                f"({db.completed}/{db.total})"
            )
            options.append((label, str(db.path)))

        with Vertical():
            yield Static("Open Dashboard", id="dash-title")
            yield Static("Select a pipeline database to visualize", id="dash-subtitle")
            if options:
                yield Select(
                    options,
                    prompt="Select database...",
                    id="db-select",
                    allow_blank=len(options) > 1,
                    value=options[0][1] if len(options) == 1 else Select.BLANK,
                )
            else:
                yield Select(
                    [("No databases found", "")],
                    prompt="No databases found",
                    id="db-select",
                )
            with Horizontal(id="port-row"):
                yield Static("Port:", id="port-label")
                yield Input(value="5006", id="port-input", type="integer")
            yield Static("", id="status-label")
            yield Button("Launch Dashboard", id="launch-btn", classes="dash-btn", variant="success")
            yield Button("Stop Dashboard", id="stop-btn", classes="dash-btn", variant="error")
            yield Button("\u2190 Back", id="back-btn", classes="dash-btn")

    def on_mount(self) -> None:
        """No-op — databases are loaded during compose."""

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle launch, stop, and back button presses."""
        if event.button.id == "back-btn":
            self._stop_dashboard()
            self.app.pop_screen()
            return

        if event.button.id == "launch-btn":
            self._launch_dashboard()

        elif event.button.id == "stop-btn":
            self._stop_dashboard()

    def _launch_dashboard(self) -> None:
        """Launch the dashboard in a subprocess."""
        if self._process is not None and self._process.poll() is None:
            self.notify("Dashboard is already running", severity="warning")
            return

        select = self.query_one("#db-select", Select)
        port_input = self.query_one("#port-input", Input)
        status = self.query_one("#status-label", Static)

        db_path = select.value
        if not db_path or db_path == Select.BLANK:
            self.notify("Please select a database first", severity="warning")
            return

        port_str = port_input.value.strip()
        if not port_str.isdigit():
            self.notify("Port must be a number", severity="error")
            return
        port = int(port_str)

        # Launch dashboard as a subprocess so the wizard stays responsive.
        # Capture stderr via PIPE so we can report errors if the process
        # exits immediately (e.g. missing dependencies).
        try:
            self._process = subprocess.Popen(  # noqa: S603
                [
                    sys.executable,
                    "-c",
                    f"from physicsnemo_curator.dashboard import launch; launch({db_path!r}, port={port})",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

            # Give the subprocess a moment to fail on import errors
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._process.wait(timeout=2)

            if self._process.poll() is not None:
                # Process exited early — report the error
                stderr_output = ""
                if self._process.stderr:
                    stderr_output = self._process.stderr.read().decode(errors="replace").strip()
                    self._process.stderr.close()
                # Extract the last meaningful line (usually the exception message)
                error_line = stderr_output.rsplit("\n", 1)[-1] if stderr_output else "unknown error"
                # Add install hint for missing-dependency errors
                hint = ""
                if "ModuleNotFoundError" in stderr_output or "ImportError" in stderr_output:
                    hint = "\nHint: run [bold]uv sync --extra dashboard[/] to install dependencies"
                status.update(f"[red]Dashboard failed: {error_line}[/]{hint}")
                self.notify(f"Dashboard exited: {error_line}", severity="error")
                self._process = None
                return

            # Process is running — close the stderr pipe so it doesn't block
            if self._process.stderr:
                self._process.stderr.close()

            status.update(f"Dashboard running at http://localhost:{port}")
            self.notify(f"Dashboard launched at http://localhost:{port}", severity="information")

            # Register atexit handler to kill dashboard if wizard exits
            if not self._atexit_registered:
                atexit.register(self._kill_process)
                self._atexit_registered = True

            # Show stop button, hide launch button
            self.query_one("#launch-btn", Button).styles.display = "none"  # ty: ignore[invalid-assignment]
            self.query_one("#stop-btn", Button).styles.display = "block"

        except OSError as exc:
            self.notify(f"Failed to launch dashboard: {exc}", severity="error")

    def _kill_process(self) -> None:
        """Terminate the dashboard subprocess unconditionally (atexit handler)."""
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()

    def _stop_dashboard(self) -> None:
        """Terminate the dashboard subprocess if running."""
        status = self.query_one("#status-label", Static)

        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            self._process.wait(timeout=5)
            self._process = None
            status.update("")
            self.notify("Dashboard stopped", severity="information")

            # Show launch button, hide stop button
            self.query_one("#launch-btn", Button).styles.display = "block"
            self.query_one("#stop-btn", Button).styles.display = "none"  # ty: ignore[invalid-assignment]

    def action_go_back(self) -> None:
        """Stop the dashboard and pop this screen."""
        self._stop_dashboard()
        self.app.pop_screen()
