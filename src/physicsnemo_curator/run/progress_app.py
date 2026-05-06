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

Renders a compact full-screen terminal UI with an overall progress bar,
a 2x4 grid of single-line per-worker status indicators (paginated for
>8 workers), and a live log panel.  Polls the SQLite database every
0.5 seconds for live updates.
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

_WORKERS_PER_PAGE = 8  # 2 columns x 4 rows
_BAR_WIDTH = 16  # Character width of the inline progress bar


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


def _render_worker_line(worker_id: str, completed: int, total: int, current_index: int | None) -> str:
    """Render a single compact worker status line.

    Parameters
    ----------
    worker_id : str
        Short worker identifier.
    completed : int
        Number of items completed by this worker.
    total : int
        Per-worker total items.
    current_index : int | None
        Index currently being processed, or None if idle.

    Returns
    -------
    str
        Formatted single-line string like: ``W1 ▓▓▓▓░░░░ 25% idx:7``
    """
    pct = min(completed / max(total, 1), 1.0)
    filled = int(pct * _BAR_WIDTH)
    bar = "▓" * filled + "░" * (_BAR_WIDTH - filled)

    # Short worker label (last 4 chars of ID or thread/pid suffix)
    label = worker_id[-6:] if len(worker_id) > 6 else worker_id

    status = f"idx:{current_index}" if current_index is not None else "idle"

    return f"{label:>6} {bar} {pct:>4.0%} {status}"


class PipelineProgressApp(App[None]):
    """Compact full-screen Textual app for pipeline progress monitoring.

    Displays an overall progress bar, a 2x4 grid of single-line
    per-worker status indicators with pagination for >8 workers, and a
    scrolling log panel that captures ``print()`` output and Python
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
        padding: 0 2;
    }
    #overall-label {
        margin-bottom: 0;
        color: $text-muted;
    }
    #worker-grid {
        grid-size: 2;
        grid-gutter: 0 2;
        padding: 0 2;
        height: auto;
        max-height: 6;
    }
    #page-nav {
        height: 1;
        padding: 0 2;
        color: $text-muted;
    }
    #log-panel {
        height: 1fr;
        min-height: 6;
        border: solid $accent;
        margin: 0 2 0 2;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("[", "prev_page", "Prev"),
        ("]", "next_page", "Next"),
    ]

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
        self._log_handler: _TUILogHandler | None = None
        self._loguru_sink_id: int | None = None
        self._page = 0
        self._workers_data: list[dict] = []

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
        yield Static("", id="page-nav")
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

        # Capture loguru output (used by earth2studio and other libs)
        # into the TUI log panel when loguru is installed.
        self._setup_loguru_sink()

        # Pre-populate 8 slots in the grid
        grid = self.query_one("#worker-grid", Grid)
        for i in range(_WORKERS_PER_PAGE):
            grid.mount(Static("", id=f"worker-slot-{i}"))

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

    def action_prev_page(self) -> None:
        """Navigate to the previous page of workers."""
        if self._page > 0:
            self._page -= 1
            self._render_workers()

    def action_next_page(self) -> None:
        """Navigate to the next page of workers."""
        max_page = max(0, math.ceil(len(self._workers_data) / _WORKERS_PER_PAGE) - 1)
        if self._page < max_page:
            self._page += 1
            self._render_workers()

    def _render_workers(self) -> None:
        """Render the current page of workers into the grid slots."""
        per_worker_total = math.ceil(self._total / max(len(self._workers_data), 1))
        start = self._page * _WORKERS_PER_PAGE
        page_workers = self._workers_data[start : start + _WORKERS_PER_PAGE]

        for i in range(_WORKERS_PER_PAGE):
            slot = self.query_one(f"#worker-slot-{i}", Static)
            if i < len(page_workers):
                w = page_workers[i]
                line = _render_worker_line(w["worker_id"], w["completed_count"], per_worker_total, w["current_index"])
                slot.update(line)
            else:
                slot.update("")

        # Update page navigation
        total_pages = max(1, math.ceil(len(self._workers_data) / _WORKERS_PER_PAGE))
        nav = self.query_one("#page-nav", Static)
        if total_pages > 1:
            nav.update(f"  ◀ [{self._page + 1}/{total_pages}] ▶   ([/] to navigate)")
        else:
            nav.update("")

    def _poll(self) -> None:
        """Poll the database and update all widgets."""
        summary = self._store.summary(self._total)
        self._workers_data = self._store.active_workers(invocation_id=self._invocation_id)

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

        # Clamp page if workers reduced
        max_page = max(0, math.ceil(len(self._workers_data) / _WORKERS_PER_PAGE) - 1)
        if self._page > max_page:
            self._page = max_page

        self._render_workers()

        # Check if pipeline is done
        if self._stop_event.is_set():
            self._cleanup_logging()
            self.exit()

    def _cleanup_logging(self) -> None:
        """Remove the TUI log handler from the root logger and loguru sink."""
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler = None
        if self._loguru_sink_id is not None:
            try:
                from loguru import logger

                logger.remove(self._loguru_sink_id)
            except Exception:  # noqa: BLE001
                pass
            self._loguru_sink_id = None

    def _setup_loguru_sink(self) -> None:
        """Add a loguru sink that routes messages to the TUI log panel.

        If loguru is not installed, this is a no-op.  The sink is
        removed on cleanup so it does not outlive the TUI session.
        """
        try:
            from loguru import logger

            def _tui_sink(message: object) -> None:
                """Route a loguru message record to the TUI."""
                text = str(message).rstrip("\n")
                if text:
                    self.call_from_thread(self.append_log, text)

            # Remove default stderr sink to prevent mangling the TUI,
            # then add our custom sink.
            logger.remove()
            self._loguru_sink_id = logger.add(
                _tui_sink,
                format="{time:HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
                level="DEBUG",
                colorize=False,
            )
        except ImportError:
            pass
