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

"""Execution screen — runs the pipeline with a progress bar."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from textual.screen import Screen
from textual.widgets import ProgressBar, Static
from textual.worker import Worker, WorkerState

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from physicsnemo_curator.wiz.app import CuratorApp


class ExecutionScreen(Screen[None]):
    """Execute the pipeline in a background worker, showing live progress.

    Uses a Textual :class:`~textual.worker.Worker` so the UI stays
    responsive during processing.
    """

    DEFAULT_CSS = """
    ExecutionScreen {
        align: center middle;
        layout: vertical;
        padding: 2 4;
    }
    #exec-title {
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
    }
    #exec-bar {
        margin: 1 0;
    }
    #exec-status {
        text-align: center;
        color: $text-muted;
    }
    """

    def compose(self) -> ComposeResult:
        """Yield title, progress bar, and status label."""
        app: CuratorApp = self.app  # type: ignore[assignment]
        total = len(app.state.pipeline) if app.state.pipeline else 0

        yield Static("Executing Pipeline...", id="exec-title")
        yield ProgressBar(total=total, id="exec-bar")
        yield Static(f"0 / {total}", id="exec-status")

    def on_mount(self) -> None:
        """Start the pipeline worker on mount."""
        self.run_worker(self._execute_pipeline, thread=True)

    def _execute_pipeline(self) -> list[list[str]]:
        """Run the pipeline in a background thread.

        Returns
        -------
        list[list[str]]
            Output paths from each pipeline index.
        """
        app: CuratorApp = self.app  # type: ignore[assignment]
        pipeline = app.state.pipeline
        n = len(pipeline)
        all_paths: list[list[str]] = []
        start = time.monotonic()

        for i in range(n):
            paths = pipeline[i]
            all_paths.append(paths)
            elapsed = time.monotonic() - start
            # Post UI updates via call_from_thread
            self.app.call_from_thread(self._update_progress, i + 1, n, elapsed)

        # Flush stateful filters
        for f in pipeline.filters:
            if hasattr(f, "flush"):
                flush = getattr(f, "flush")  # noqa: B009
                flush()

        return all_paths

    def _update_progress(self, completed: int, total: int, elapsed: float) -> None:
        """Update the progress bar and status label (called on main thread).

        Parameters
        ----------
        completed : int
            Number of completed items.
        total : int
            Total number of items.
        elapsed : float
            Elapsed seconds.
        """
        bar = self.query_one("#exec-bar", ProgressBar)
        bar.update(progress=completed)
        status = self.query_one("#exec-status", Static)
        status.update(f"{completed} / {total}  ({elapsed:.1f}s)")

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Navigate to ResultScreen when the worker finishes."""
        if event.state == WorkerState.SUCCESS:
            from physicsnemo_curator.wiz.screens.result import ResultScreen

            app: CuratorApp = self.app  # type: ignore[assignment]
            all_paths = event.worker.result
            app.push_screen(ResultScreen(all_paths=all_paths))

        elif event.state == WorkerState.ERROR:
            self.notify(f"Pipeline failed: {event.worker.error}", severity="error")
