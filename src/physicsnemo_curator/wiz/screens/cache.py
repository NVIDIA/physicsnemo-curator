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

"""Cache management screen — list, inspect, and remove pipeline databases."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult


def _human_size(n_bytes: int) -> str:
    """Format bytes as a human-readable string.

    Parameters
    ----------
    n_bytes : int
        Number of bytes.

    Returns
    -------
    str
        Formatted size string (e.g. ``"1.2MB"``).
    """
    size: float = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size) < 1024:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


class CacheScreen(Screen[None]):
    """List and manage cached pipeline databases.

    Uses a :class:`~textual.widgets.DataTable` to display all cached
    databases.  Supports row selection, removal by hash, and full cache
    clear.
    """

    BINDINGS = [("escape", "go_back", "Back")]

    DEFAULT_CSS = """
    CacheScreen {
        layout: vertical;
        padding: 1 2;
    }
    #cache-title {
        text-style: bold;
        margin-bottom: 1;
    }
    #cache-table {
        height: 70%;
        margin: 1 0;
    }
    .nav-row {
        layout: horizontal;
        height: auto;
    }
    .action-btn {
        margin: 1 1;
    }
    """

    def compose(self) -> ComposeResult:
        """Yield title, data table, and action buttons."""
        yield Static("Pipeline Database Cache", id="cache-title")
        yield DataTable(id="cache-table", cursor_type="row")
        with Vertical(classes="nav-row"):
            yield Button("← Back", id="back-btn", classes="action-btn")
            yield Button("Remove Selected", id="rm-btn", classes="action-btn", variant="warning")
            yield Button("Remove All", id="rm-all-btn", classes="action-btn", variant="error")
            yield Button("Quit", id="quit-btn", classes="action-btn", variant="error")

    def on_mount(self) -> None:
        """Populate the data table with cached databases."""
        self._refresh_table()

    def _refresh_table(self) -> None:
        """Reload the data table from the cache directory."""
        from physicsnemo_curator.core.cache import default_cache_dir, list_databases

        table = self.query_one("#cache-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Hash", "Created", "Size", "Done", "Source", "Sink")

        cache_dir = default_cache_dir()
        databases = list_databases(cache_dir)

        for db in databases:
            # Display short hash but use full stem as unique row key
            display_hash = db.hash_prefix[:8]
            # Append timestamp portion if present for disambiguation
            if "_" in db.hash_prefix:
                ts_suffix = db.hash_prefix.rsplit("_", 1)[-1][-6:]
                display_hash = f"{display_hash}..{ts_suffix}"
            table.add_row(
                display_hash,
                db.created.strftime("%Y-%m-%d %H:%M"),
                _human_size(db.size_bytes),
                f"{db.completed}/{db.total}",
                db.source_name,
                db.sink_name,
                key=db.hash_prefix,
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle back, remove selected, remove all, and quit."""
        if event.button.id == "back-btn":
            self.app.pop_screen()
            return

        if event.button.id == "quit-btn":
            self.app.exit()
            return

        if event.button.id == "rm-btn":
            self._remove_selected()

        elif event.button.id == "rm-all-btn":
            self._remove_all()

    def _remove_selected(self) -> None:
        """Remove the database at the currently highlighted row."""
        from physicsnemo_curator.core.cache import remove_databases

        table = self.query_one("#cache-table", DataTable)
        if table.cursor_row is None or table.row_count == 0:
            self.notify("No row selected", severity="warning")
            return

        # Get the row key (full file stem) via coordinate_to_cell_key
        from textual.widgets._data_table import Coordinate

        cell_key = table.coordinate_to_cell_key(Coordinate(table.cursor_row, 0))
        full_stem = str(cell_key.row_key.value or "")

        try:
            removed = remove_databases([full_stem])
            self.notify(f"Removed {removed} database(s)")
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return

        self._refresh_table()

    def _remove_all(self) -> None:
        """Remove all cached databases."""
        from physicsnemo_curator.core.cache import clear_cache

        removed = clear_cache()
        self.notify(f"Removed {removed} database(s)")
        self._refresh_table()

    def action_go_back(self) -> None:
        """Pop this screen."""
        self.app.pop_screen()
