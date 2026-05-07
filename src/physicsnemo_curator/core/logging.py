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

"""Logging utilities for PhysicsNeMo Curator.

Provides a consistent logging interface for sources, filters, and sinks that:

1. Uses standard Python logging (captured by TUI when in main process)
2. Automatically includes process/worker identification
3. Provides structured log messages with timing information
4. Supports database-backed logging for cross-process log aggregation

Usage in Sources/Filters
------------------------
.. code-block:: python

    from physicsnemo_curator.core.logging import get_logger

    class MySource(Source):
        def __init__(self, ...):
            self._log = get_logger(self)

        def __getitem__(self, index: int) -> Generator[Mesh]:
            self._log.info("Reading index %d", index)
            ...
            self._log.debug("Loaded %d points", n_points)

Log Format
----------
Messages are formatted as::

    [ProcessName] ClassName: message

For example::
    [Worker-1] AhmedMLSource: Reading index 5
    [MainProcess] MeshStatsFilter: Computing statistics

Database Logging
----------------
For cross-process logging (e.g., in worker processes), use
:class:`DatabaseLogHandler` which buffers logs and writes them
to the pipeline store in batches to minimize database lock contention.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import multiprocessing
import os
import threading
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from physicsnemo_curator.core.base import Filter, Sink, Source
    from physicsnemo_curator.core.pipeline_store import PipelineStore

__all__ = ["get_logger", "configure_logging", "DatabaseLogHandler", "setup_worker_logging"]


class _ProcessAwareFormatter(logging.Formatter):
    """Formatter that includes process identification in log messages."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with process name prefix."""
        # Get process name - works in main and worker processes
        process_name = multiprocessing.current_process().name

        # For loky workers, extract worker number from process name
        # loky uses names like "LokyProcess-1"
        if process_name.startswith("Loky"):
            process_name = process_name.replace("LokyProcess", "Worker")
        elif process_name.startswith("Fork") or process_name.startswith("Spawn"):
            # multiprocessing Pool uses ForkPoolWorker-N or SpawnPoolWorker-N
            process_name = process_name.replace("PoolWorker", "")

        # Add PID for disambiguation when needed
        pid = os.getpid()
        record.process_info = f"{process_name}:{pid}"

        return super().format(record)


class _ComponentLogger:
    """Logger wrapper for pipeline components.

    Wraps a standard Python logger with convenience methods that
    automatically include component context.
    """

    def __init__(self, logger: logging.Logger, component_name: str) -> None:
        self._logger = logger
        self._component_name = component_name

    def _format_message(self, msg: str) -> str:
        """Prepend component name to message."""
        return f"{self._component_name}: {msg}"

    def debug(self, msg: str, *args: object) -> None:
        """Log debug message."""
        self._logger.debug(self._format_message(msg), *args)

    def info(self, msg: str, *args: object) -> None:
        """Log info message."""
        self._logger.info(self._format_message(msg), *args)

    def warning(self, msg: str, *args: object) -> None:
        """Log warning message."""
        self._logger.warning(self._format_message(msg), *args)

    def error(self, msg: str, *args: object) -> None:
        """Log error message."""
        self._logger.error(self._format_message(msg), *args)

    def exception(self, msg: str, *args: object) -> None:
        """Log exception with traceback."""
        self._logger.exception(self._format_message(msg), *args)


class DatabaseLogHandler(logging.Handler):
    """Logging handler that writes to the pipeline store database.

    Buffers log records and flushes them to the database periodically
    or when the buffer reaches a threshold. This minimizes database
    lock contention in multi-process scenarios.

    Call :meth:`flush` explicitly at key points (e.g., after source reads)
    to ensure logs appear promptly during long operations.

    Parameters
    ----------
    store : PipelineStore
        The pipeline store to write logs to.
    worker_id : str | None
        Identifier for the current worker (e.g., "Worker-1").
    buffer_size : int
        Number of records to buffer before flushing (default: 50).
    flush_interval : float
        Maximum seconds between flushes (default: 2.0).
    """

    def __init__(
        self,
        store: PipelineStore,
        worker_id: str | None = None,
        buffer_size: int = 50,
        flush_interval: float = 2.0,
    ) -> None:
        super().__init__()
        self._store = store
        self._worker_id = worker_id
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval

        self._buffer: list[tuple[str, int, str, str, str, str | None, int | None]] = []
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()
        self._current_index: int | None = None

        # Register cleanup on exit
        atexit.register(self.flush)

    def set_current_index(self, index: int | None) -> None:
        """Set the current index being processed (for log context)."""
        self._current_index = index

    def emit(self, record: logging.LogRecord) -> None:
        """Buffer a log record for later database write."""
        try:
            timestamp = datetime.now(tz=UTC).isoformat()
            message = self.format(record)

            entry = (
                timestamp,
                record.levelno,
                record.levelname,
                record.name,
                message,
                self._worker_id,
                self._current_index,
            )

            with self._lock:
                self._buffer.append(entry)

                # Flush if buffer is full or interval exceeded
                now = time.monotonic()
                if len(self._buffer) >= self._buffer_size or (now - self._last_flush) >= self._flush_interval:
                    self._flush_buffer()

        except Exception:  # noqa: BLE001
            self.handleError(record)

    def flush(self) -> None:
        """Flush buffered logs to the database."""
        with self._lock:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Flush buffered logs to the database (must hold lock)."""
        if not self._buffer:
            return

        with contextlib.suppress(Exception):
            self._store.record_logs(self._buffer)

        self._buffer.clear()
        self._last_flush = time.monotonic()

    def close(self) -> None:
        """Flush and close the handler."""
        self.flush()
        super().close()


def get_logger(component: Source | Filter | Sink | str) -> _ComponentLogger:
    """Get a logger for a pipeline component.

    Parameters
    ----------
    component : Source, Filter, Sink, or str
        The pipeline component instance or a string name.

    Returns
    -------
    _ComponentLogger
        A logger wrapper with process-aware formatting.

    Examples
    --------
    >>> from physicsnemo_curator.core.logging import get_logger
    >>> log = get_logger("MySource")
    >>> log.info("Processing index %d", 42)
    [MainProcess:12345] MySource: Processing index 42
    """
    component_name = component if isinstance(component, str) else type(component).__name__

    # Get logger for the curator namespace
    logger = logging.getLogger(f"physicsnemo_curator.{component_name}")

    return _ComponentLogger(logger, component_name)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure logging for physicsnemo_curator.

    Sets up a console handler with process-aware formatting.
    Call this at the start of your script if you want to see
    log output. The TUI automatically configures logging when
    it starts.

    Parameters
    ----------
    level : int, optional
        Logging level (default: logging.INFO).

    Examples
    --------
    >>> from physicsnemo_curator.core.logging import configure_logging
    >>> configure_logging(logging.DEBUG)
    """
    root_logger = logging.getLogger("physicsnemo_curator")

    # Don't add handlers if already configured
    if root_logger.handlers:
        root_logger.setLevel(level)
        return

    root_logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setLevel(level)

    formatter = _ProcessAwareFormatter(
        fmt="[%(process_info)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)

    root_logger.addHandler(handler)


def setup_worker_logging(
    store: PipelineStore,
    level: int = logging.INFO,
) -> DatabaseLogHandler:
    """Configure logging for a worker process to write to the database.

    Call this at the start of each worker process to enable database-backed
    logging. Logs are buffered and written in batches to minimize lock
    contention.

    Parameters
    ----------
    store : PipelineStore
        The pipeline store to write logs to.
    level : int
        Logging level (default: logging.INFO).

    Returns
    -------
    DatabaseLogHandler
        The handler instance (useful for setting current_index).

    Examples
    --------
    >>> from physicsnemo_curator.core.logging import setup_worker_logging
    >>> handler = setup_worker_logging(store, level=logging.DEBUG)
    >>> handler.set_current_index(42)  # Set context for current task
    """
    # Build worker ID from process name
    process_name = multiprocessing.current_process().name
    if process_name.startswith("Loky"):
        worker_id = process_name.replace("LokyProcess", "Worker")
    elif process_name.startswith("Fork") or process_name.startswith("Spawn"):
        worker_id = process_name.replace("PoolWorker", "")
    else:
        worker_id = f"{process_name}:{os.getpid()}"

    root_logger = logging.getLogger("physicsnemo_curator")
    root_logger.setLevel(level)

    # Create database handler
    handler = DatabaseLogHandler(store, worker_id=worker_id)
    handler.setLevel(level)
    handler.setFormatter(_ProcessAwareFormatter(fmt="%(message)s"))

    root_logger.addHandler(handler)

    return handler


def flush_logs() -> None:
    """Flush all database log handlers.

    Call this before long blocking operations to ensure logs
    appear promptly in the TUI/dashboard.

    Examples
    --------
    >>> from physicsnemo_curator.core.logging import flush_logs, get_logger
    >>> log = get_logger("MySource")
    >>> log.info("Starting long read...")
    >>> flush_logs()  # Ensure log appears before blocking
    >>> data = read_large_file()  # Long operation
    """
    root_logger = logging.getLogger("physicsnemo_curator")
    for handler in root_logger.handlers:
        if isinstance(handler, DatabaseLogHandler):
            handler.flush()
