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

"""Progress monitor for pipeline execution.

Provides a context manager that runs a :class:`PipelineProgressApp`
Textual TUI in a daemon thread, polling the SQLite database for live
progress updates.
"""

from __future__ import annotations

import signal
import sys
import threading
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from physicsnemo_curator.core.base import Pipeline
    from physicsnemo_curator.run.base import RunConfig


class _NoOpMonitor:
    """No-op progress monitor used when progress display is disabled."""

    def stop(self) -> None:
        """No-op stop."""

    def __enter__(self) -> _NoOpMonitor:
        """Enter context."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context."""


class ProgressMonitor:
    """Context manager that runs a Textual TUI in a daemon thread.

    The TUI polls the pipeline's SQLite database and displays an overall
    progress bar plus per-worker progress tiles.

    Parameters
    ----------
    store : PipelineStore
        Pipeline store instance (used for read-only polling).
    total : int
        Total number of indices to process.
    n_workers : int
        Number of parallel workers.
    invocation_id : str | None
        If set, filter workers by this invocation ID.
    """

    def __init__(self, store: Any, total: int, n_workers: int, invocation_id: str | None = None) -> None:
        """Initialise the progress monitor."""
        from physicsnemo_curator.run.progress_app import PipelineProgressApp

        self._stop_event = threading.Event()
        self._app = PipelineProgressApp(
            store=store,
            total=total,
            n_workers=n_workers,
            stop_event=self._stop_event,
            invocation_id=invocation_id,
        )
        self._thread: threading.Thread | None = None

    @staticmethod
    @contextmanager
    def _suppress_signals():
        """Suppress signal registration when running off the main thread.

        Textual's LinuxDriver registers SIGTSTP/SIGCONT handlers in its
        ``__init__``, but Python only allows signal handlers on the main
        thread.  This context manager temporarily patches ``signal.signal``
        to silently skip those registrations when called from a non-main
        thread.
        """
        if threading.current_thread() is threading.main_thread():
            yield
            return

        original_signal = signal.signal

        def _safe_signal(signalnum, handler):  # noqa: ANN001, ANN202
            try:
                return original_signal(signalnum, handler)
            except ValueError:
                # "signal only works in main thread" — return the
                # current handler as a no-op acknowledgement.
                return signal.getsignal(signalnum)

        signal.signal = _safe_signal  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        try:
            yield
        finally:
            signal.signal = original_signal  # type: ignore[assignment]

    def _run_app(self) -> None:
        """Run the Textual app (called in daemon thread)."""
        with self._suppress_signals():
            self._app.run()

    def start(self) -> None:
        """Start the Textual TUI in a daemon thread."""
        self._thread = threading.Thread(target=self._run_app, daemon=True, name="progress-tui")
        self._thread.start()

    def stop(self) -> None:
        """Signal the TUI to exit and wait for the thread to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def __enter__(self) -> ProgressMonitor:
        """Start the monitor on context entry."""
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop the monitor on context exit."""
        self.stop()


def start_progress_monitor(
    pipeline: Pipeline[Any],
    config: RunConfig,
) -> ProgressMonitor | _NoOpMonitor:
    """Create and return a progress monitor for pipeline execution.

    Returns a :class:`ProgressMonitor` that runs a Textual TUI in a
    daemon thread, or a no-op monitor when progress display is disabled
    or the terminal is non-interactive.

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline being executed.
    config : RunConfig
        Execution configuration.

    Returns
    -------
    ProgressMonitor | _NoOpMonitor
        An active progress monitor, or a no-op if disabled.
    """
    if not config.progress or not sys.stdout.isatty():
        return _NoOpMonitor()

    # Generate a unique invocation ID and set it on the pipeline so
    # workers created by Pipeline.__getitem__ will carry this ID.
    invocation_id = uuid.uuid4().hex
    pipeline.invocation_id = invocation_id

    store = pipeline._get_store()
    indices = config.indices if config.indices is not None else list(range(len(pipeline)))
    total = len(indices)
    n_workers = config.resolved_n_jobs

    return ProgressMonitor(store=store, total=total, n_workers=n_workers, invocation_id=invocation_id)
