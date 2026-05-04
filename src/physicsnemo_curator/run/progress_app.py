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

"""Textual TUI application for pipeline progress display.

Renders a full-screen terminal UI with an overall progress bar, a grid
of per-worker progress tiles, and a live log panel.  Polls the SQLite
database every 0.5 seconds for live updates.  Print statements and
Python logging output are captured and displayed in the log panel.
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.containers import Grid, Vertical
from textual.widgets import Footer, Header, ProgressBar, RichLog, Static

if TYPE_CHECKING:
    from threading import Event

    from textual.events import Print

    from physicsnemo_curator.core.pipeline_store import PipelineStore


class _TUILogHandler(logging.Handler):
    """Logging handler that routes records into a Textual RichLog widget.

    Uses :meth:`App.call_from_thread` so it is safe to call from any
    thread (main thread, worker threads, or the Textual event-loop
    thread).

    Parameters
    ----------
    app : PipelineProgressApp
        The running Textual app instance.
    """

    def __init__(self, app: PipelineProgressApp) -> None:
        """Initialise the handler with a reference to the app."""
        super().__init__()
        self._app = app

    def emit(self, record: logging.LogRecord) -> None:
        """Format and route a log record to the TUI log panel.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to display.
        """
        try:
            msg = self.format(record)
            self._app.call_from_thread(self._app.append_log, msg)
        except Exception:  # noqa: BLE001
            self.handleError(record)


class WorkerTile(Static):
    """A single worker's progress tile.

    Displays the worker label, a progress bar, and a status line.
    """

    DEFAULT_CSS = """
    WorkerTile {
        border: solid $accent;
        padding: 1;
        height: auto;
    }
    WorkerTile .worker-label {
        text-style: bold;
        margin-bottom: 1;
    }
    WorkerTile .worker-status {
        color: $text-muted;
    }
    """

    def __init__(self, worker_id: str, pid: int, per_worker_total: int) -> None:
        """Initialise a worker tile."""
        super().__init__()
        self._worker_id = worker_id
        self._pid = pid
        self._per_worker_total = max(per_worker_total, 1)

    def compose(self) -> ComposeResult:
        """Build the worker tile widgets."""
        yield Static(f"Worker (PID {self._pid})", classes="worker-label")
        yield ProgressBar(total=self._per_worker_total, show_eta=False)
        yield Static("Idle", classes="worker-status")

    def update_progress(self, completed: int, current_index: int | None) -> None:
        """Update the worker's progress bar and status line.

        Parameters
        ----------
        completed : int
            Number of items this worker has completed.
        current_index : int | None
            Index currently being processed, or None if idle.
        """
        bar = self.query_one(ProgressBar)
        bar.update(progress=completed)
        status = self.query_one(".worker-status", Static)
        if current_index is not None:
            status.update(f"Processing index {current_index}")
        else:
            status.update("Idle")


class PipelineProgressApp(App[None]):
    """Full-screen Textual app for pipeline progress monitoring.

    Displays an overall progress bar, a grid of per-worker tiles, and
    a scrolling log panel that captures ``print()`` output and Python
    ``logging`` messages.

    Parameters
    ----------
    store : PipelineStore
        Pipeline store instance for polling progress data.
    total : int
        Total number of indices to process.
    n_workers : int
        Number of expected workers.
    stop_event : Event
        Threading event signalling pipeline completion.
    invocation_id : str | None
        If set, only show workers from this invocation.
    """

    TITLE = "PhysicsNeMo Curator — Pipeline Progress"

    CSS = """
    #overall-container {
        height: auto;
        padding: 1 2;
    }
    #overall-label {
        margin-bottom: 1;
    }
    #worker-grid {
        grid-size: 4;
        grid-gutter: 1 2;
        padding: 1 2;
        height: 1fr;
    }
    #log-panel {
        height: 12;
        border: solid $accent;
        margin: 0 2 1 2;
    }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def __init__(
        self,
        store: PipelineStore,
        total: int,
        n_workers: int,
        stop_event: Event,
        invocation_id: str | None = None,
    ) -> None:
        """Initialise the progress app."""
        super().__init__()
        self._store = store
        self._total = total
        self._n_workers = n_workers
        self._stop_event = stop_event
        self._invocation_id = invocation_id
        self._start_time = time.monotonic()
        self._worker_tiles: dict[str, WorkerTile] = {}
        self._log_handler: _TUILogHandler | None = None

    def compose(self) -> ComposeResult:
        """Build the top-level layout."""
        yield Header()
        yield Vertical(
            ProgressBar(total=self._total, show_eta=True, id="overall-bar"),
            Static(
                "Completed: 0 | Failed: 0 | Remaining: 0 | Elapsed: 0s",
                id="overall-label",
            ),
            id="overall-container",
        )
        yield Grid(id="worker-grid")
        yield RichLog(id="log-panel", max_lines=500, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        """Start polling and set up print/logging capture."""
        self.set_interval(0.5, self._poll)

        # Capture print() calls (stdout + stderr) into Print events
        self.begin_capture_print(self, stdout=True, stderr=True)

        # Attach a logging handler to the root logger so library log
        # messages appear in the TUI log panel instead of corrupting
        # the alternate screen buffer.
        self._log_handler = _TUILogHandler(self)
        self._log_handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
        )
        logging.getLogger().addHandler(self._log_handler)

        log_panel = self.query_one("#log-panel", RichLog)
        log_panel.border_title = "Log"

    def on_print(self, event: Print) -> None:
        """Handle captured print() output by writing it to the log panel.

        Parameters
        ----------
        event : Print
            Textual event containing the captured text.
        """
        text = event.text.rstrip("\n")
        if text:
            self.append_log(text)

    def append_log(self, text: str) -> None:
        """Append a line of text to the log panel.

        Parameters
        ----------
        text : str
            Text to write.
        """
        try:
            log_panel = self.query_one("#log-panel", RichLog)
            log_panel.write(text)
        except Exception:  # noqa: BLE001
            pass

    def _poll(self) -> None:
        """Poll the database and update all widgets."""
        summary = self._store.summary(self._total)
        workers = self._store.active_workers(invocation_id=self._invocation_id)

        # Overall bar
        bar = self.query_one("#overall-bar", ProgressBar)
        bar.update(progress=summary["completed"])

        elapsed = time.monotonic() - self._start_time
        label = self.query_one("#overall-label", Static)
        label.update(
            f"Completed: {summary['completed']} | "
            f"Failed: {summary['failed']} | "
            f"Remaining: {summary['remaining']} | "
            f"Elapsed: {elapsed:.1f}s"
        )

        # Worker tiles
        grid = self.query_one("#worker-grid", Grid)
        per_worker_total = math.ceil(self._total / max(len(workers), 1))

        seen_ids: set[str] = set()
        for w in workers:
            wid = w["worker_id"]
            seen_ids.add(wid)
            if wid not in self._worker_tiles:
                tile = WorkerTile(wid, w["pid"], per_worker_total)
                self._worker_tiles[wid] = tile
                grid.mount(tile)
            else:
                tile = self._worker_tiles[wid]
            tile.update_progress(w["completed_count"], w["current_index"])

        # Remove tiles for workers no longer present
        for wid in list(self._worker_tiles):
            if wid not in seen_ids:
                self._worker_tiles[wid].remove()
                del self._worker_tiles[wid]

        # Check if pipeline is done
        if self._stop_event.is_set():
            self._cleanup_logging()
            self.exit()

    def _cleanup_logging(self) -> None:
        """Remove the TUI log handler from the root logger."""
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler = None
