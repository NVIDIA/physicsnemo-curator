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

import atexit
from typing import TYPE_CHECKING

from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.widgets._data_table import ColumnKey

# Module-level list of hashes to print on exit (works over SSH)
_pending_prints: list[str] = []
_atexit_registered = False

_CHECK = "\u2714"  # checkmark
_EMPTY = " "


def _flush_pending_prints() -> None:
    """Print any copied hashes to stdout after the TUI exits."""
    for line in _pending_prints:
        print(line)  # noqa: T201
    _pending_prints.clear()


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
    databases.  Supports multi-row selection (Space to toggle, Shift+Space
    for range), removal, and full cache clear.
    """

    BINDINGS = [
        ("escape", "go_back", "Back"),
        ("space", "toggle_select", "Select"),
        ("shift+space", "range_select", "Range select"),
    ]

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

    def __init__(self) -> None:
        """Initialize selection state."""
        super().__init__()
        self._selected: set[str] = set()  # row keys (hash_prefix stems)
        self._row_keys: list[str] = []  # ordered list of row keys
        self._last_toggled: int | None = None  # last toggled row index
        self._sel_col_key: ColumnKey | None = None  # ColumnKey for the "Sel" column

    def compose(self) -> ComposeResult:
        """Yield title, data table, and action buttons."""
        yield Static("Pipeline Database Cache", id="cache-title")
        yield DataTable(id="cache-table", cursor_type="row")
        with Vertical(classes="nav-row"):
            yield Button("\u2190 Back", id="back-btn", classes="action-btn")
            yield Button("Copy Hash", id="copy-btn", classes="action-btn", variant="primary")
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
        col_keys = table.add_columns("Sel", "Hash", "Created", "Size", "Done", "Source", "Sink")
        self._sel_col_key = col_keys[0]

        cache_dir = default_cache_dir()
        databases = list_databases(cache_dir)

        self._row_keys.clear()
        self._selected.clear()
        self._last_toggled = None

        for db in databases:
            # Display short hash but use full stem as unique row key
            display_hash = db.hash_prefix[:8]
            # Append timestamp portion if present for disambiguation
            if "_" in db.hash_prefix:
                ts_suffix = db.hash_prefix.rsplit("_", 1)[-1][-6:]
                display_hash = f"{display_hash}..{ts_suffix}"
            table.add_row(
                _EMPTY,
                display_hash,
                db.created.strftime("%Y-%m-%d %H:%M"),
                _human_size(db.size_bytes),
                f"{db.completed}/{db.total}",
                db.source_name,
                db.sink_name,
                key=db.hash_prefix,
            )
            self._row_keys.append(db.hash_prefix)

    def _get_current_row_key(self) -> str | None:
        """Return the row key for the cursor row, or None."""
        table = self.query_one("#cache-table", DataTable)
        if table.cursor_row is None or table.row_count == 0:
            return None
        from textual.widgets._data_table import Coordinate

        cell_key = table.coordinate_to_cell_key(Coordinate(table.cursor_row, 0))
        return str(cell_key.row_key.value or "") or None

    def _update_sel_column(self, key: str) -> None:
        """Update the 'Sel' column for a given row key."""
        if self._sel_col_key is None:
            return
        table = self.query_one("#cache-table", DataTable)
        from textual.widgets._data_table import RowKey

        mark = _CHECK if key in self._selected else _EMPTY
        table.update_cell(RowKey(key), self._sel_col_key, mark)

    def action_toggle_select(self) -> None:
        """Toggle selection of the current row (Space key)."""
        table = self.query_one("#cache-table", DataTable)
        key = self._get_current_row_key()
        if not key:
            return

        if key in self._selected:
            self._selected.discard(key)
        else:
            self._selected.add(key)

        self._update_sel_column(key)
        self._last_toggled = table.cursor_row
        self._update_status()

    def action_range_select(self) -> None:
        """Select range from last toggle to current row (Shift+Space)."""
        table = self.query_one("#cache-table", DataTable)
        if table.cursor_row is None or table.row_count == 0:
            return

        current = table.cursor_row
        anchor = self._last_toggled if self._last_toggled is not None else 0

        start = min(anchor, current)
        end = max(anchor, current)

        for row_idx in range(start, end + 1):
            if row_idx < len(self._row_keys):
                key = self._row_keys[row_idx]
                self._selected.add(key)
                self._update_sel_column(key)

        self._last_toggled = current
        self._update_status()

    def _update_status(self) -> None:
        """Update the title to show selection count."""
        title = self.query_one("#cache-title", Static)
        n = len(self._selected)
        if n > 0:
            title.update(f"Pipeline Database Cache ({n} selected)")
        else:
            title.update("Pipeline Database Cache")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle back, copy hash, remove selected, remove all, and quit."""
        if event.button.id == "back-btn":
            self.app.pop_screen()
            return

        if event.button.id == "quit-btn":
            self.app.exit()
            return

        if event.button.id == "copy-btn":
            self._copy_hash()

        elif event.button.id == "rm-btn":
            self._remove_selected()

        elif event.button.id == "rm-all-btn":
            self._remove_all()

    def _copy_hash(self) -> None:
        """Stage the hash prefix for printing on TUI exit (works over SSH)."""
        global _atexit_registered  # noqa: PLW0603

        table = self.query_one("#cache-table", DataTable)
        if table.cursor_row is None or table.row_count == 0:
            self.notify("No row selected", severity="warning")
            return

        key = self._get_current_row_key()
        if not key:
            self.notify("No hash available", severity="warning")
            return

        # Register atexit handler once to print on exit
        if not _atexit_registered:
            atexit.register(_flush_pending_prints)
            _atexit_registered = True

        _pending_prints.append(key)
        self.notify(f"Hash: {key} (will print on exit)")

    def _remove_selected(self) -> None:
        """Remove all selected databases (or cursor row if none selected)."""
        from physicsnemo_curator.core.cache import remove_databases

        targets = list(self._selected) if self._selected else []

        # Fall back to cursor row if nothing explicitly selected
        if not targets:
            key = self._get_current_row_key()
            if key:
                targets = [key]

        if not targets:
            self.notify("No rows selected", severity="warning")
            return

        try:
            removed = remove_databases(targets)
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
