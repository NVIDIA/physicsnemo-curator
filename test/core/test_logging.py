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

"""Tests for the logging module."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, ClassVar
from unittest.mock import MagicMock

import pytest

from physicsnemo_curator.core.base import Param, Source
from physicsnemo_curator.core.logging import (
    DatabaseLogHandler,
    _ComponentLogger,
    _ProcessAwareFormatter,
    configure_logging,
    get_logger,
    setup_worker_logging,
)
from physicsnemo_curator.core.pipeline_store import PipelineStore

if TYPE_CHECKING:
    from collections.abc import Generator

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class DummySource(Source[int]):
    """Dummy source for testing get_logger."""

    name: ClassVar[str] = "DummySource"
    description: ClassVar[str] = "A dummy source"

    @classmethod
    def params(cls) -> list[Param]:
        """Return empty params."""
        return []

    def __len__(self) -> int:
        """Return length."""
        return 1

    def __getitem__(self, index: int) -> Generator[int]:
        """Yield a value."""
        yield index


@pytest.fixture
def store(tmp_path):
    """Create a fresh PipelineStore for each test."""
    db = tmp_path / "test.db"
    config = {"source": "test", "filters": [], "sink": "test"}
    return PipelineStore(db, config, "testhash")


# ---------------------------------------------------------------------------
# Test _ProcessAwareFormatter
# ---------------------------------------------------------------------------


class TestProcessAwareFormatter:
    """Tests for _ProcessAwareFormatter."""

    def test_format_includes_process_info(self) -> None:
        """Formatter adds process_info to log records."""
        formatter = _ProcessAwareFormatter(fmt="[%(process_info)s] %(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        # Should contain process name and PID
        assert "[" in formatted
        assert ":" in formatted  # PID separator
        assert "Test message" in formatted


# ---------------------------------------------------------------------------
# Test _ComponentLogger
# ---------------------------------------------------------------------------


class TestComponentLogger:
    """Tests for _ComponentLogger."""

    def test_formats_message_with_component_name(self) -> None:
        """Messages are prefixed with component name."""
        mock_logger = MagicMock()
        comp_logger = _ComponentLogger(mock_logger, "MyComponent")

        comp_logger.info("Test %s", "value")

        mock_logger.info.assert_called_once_with("MyComponent: Test %s", "value")

    def test_all_log_levels(self) -> None:
        """All log level methods work correctly."""
        mock_logger = MagicMock()
        comp_logger = _ComponentLogger(mock_logger, "Comp")

        comp_logger.debug("debug")
        comp_logger.info("info")
        comp_logger.warning("warning")
        comp_logger.error("error")
        comp_logger.exception("exception")

        mock_logger.debug.assert_called_once()
        mock_logger.info.assert_called_once()
        mock_logger.warning.assert_called_once()
        mock_logger.error.assert_called_once()
        mock_logger.exception.assert_called_once()


# ---------------------------------------------------------------------------
# Test get_logger
# ---------------------------------------------------------------------------


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_with_string(self) -> None:
        """get_logger accepts a string name."""
        logger = get_logger("TestName")
        assert isinstance(logger, _ComponentLogger)
        assert logger._component_name == "TestName"

    def test_get_logger_with_component(self) -> None:
        """get_logger extracts class name from component instance."""
        source = DummySource()
        logger = get_logger(source)
        assert isinstance(logger, _ComponentLogger)
        assert logger._component_name == "DummySource"

    def test_logger_uses_curator_namespace(self) -> None:
        """Logger is in the physicsnemo_curator namespace."""
        logger = get_logger("TestComp")
        assert logger._logger.name == "physicsnemo_curator.TestComp"


# ---------------------------------------------------------------------------
# Test configure_logging
# ---------------------------------------------------------------------------


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_logging_adds_handler(self) -> None:
        """configure_logging adds a StreamHandler to the root curator logger."""
        root = logging.getLogger("physicsnemo_curator")
        # Clear any existing handlers
        root.handlers.clear()

        configure_logging(level=logging.DEBUG)

        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], logging.StreamHandler)
        assert root.level == logging.DEBUG

        # Clean up
        root.handlers.clear()

    def test_configure_logging_idempotent(self) -> None:
        """Calling configure_logging twice doesn't add duplicate handlers."""
        root = logging.getLogger("physicsnemo_curator")
        root.handlers.clear()

        configure_logging(level=logging.INFO)
        configure_logging(level=logging.DEBUG)

        assert len(root.handlers) == 1
        # Level should be updated
        assert root.level == logging.DEBUG

        # Clean up
        root.handlers.clear()


# ---------------------------------------------------------------------------
# Test DatabaseLogHandler
# ---------------------------------------------------------------------------


class TestDatabaseLogHandler:
    """Tests for DatabaseLogHandler."""

    def test_emit_buffers_records(self, store) -> None:
        """emit() buffers records without immediate write."""
        handler = DatabaseLogHandler(store, worker_id="Worker-1", buffer_size=10)
        handler.setFormatter(logging.Formatter("%(message)s"))

        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,  # Use DEBUG to test buffering (INFO+ flushes immediately)
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        handler.emit(record)

        # Should be buffered, not yet in database (DEBUG level doesn't trigger immediate flush)
        assert len(handler._buffer) == 1
        assert store.get_logs() == []

        handler.close()

    def test_flush_writes_to_database(self, store) -> None:
        """flush() writes buffered records to database."""
        handler = DatabaseLogHandler(store, worker_id="Worker-1", buffer_size=100)
        handler.setFormatter(logging.Formatter("%(message)s"))

        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        handler.emit(record)
        handler.flush()

        logs = store.get_logs()
        assert len(logs) == 1
        assert logs[0]["message"] == "Test message"
        assert logs[0]["worker_id"] == "Worker-1"
        assert logs[0]["level_name"] == "INFO"

        handler.close()

    def test_buffer_size_triggers_flush(self, store) -> None:
        """Reaching buffer_size triggers automatic flush."""
        handler = DatabaseLogHandler(store, worker_id="Worker-1", buffer_size=3, flush_interval=999)
        handler.setFormatter(logging.Formatter("%(message)s"))

        for i in range(3):
            record = logging.LogRecord(
                name="test",
                level=logging.DEBUG,
                pathname="",
                lineno=0,
                msg=f"Message {i}",
                args=(),
                exc_info=None,
            )
            handler.emit(record)

        # Buffer should have been flushed
        assert len(handler._buffer) == 0
        assert len(store.get_logs()) == 3

        handler.close()

    def test_flush_interval_triggers_flush(self, store) -> None:
        """Exceeding flush_interval triggers automatic flush on next emit."""
        handler = DatabaseLogHandler(store, worker_id="Worker-1", buffer_size=100, flush_interval=0.05)
        handler.setFormatter(logging.Formatter("%(message)s"))

        # Reset last_flush so the interval hasn't elapsed yet
        handler._last_flush = time.monotonic()

        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="",
            lineno=0,
            msg="First message",
            args=(),
            exc_info=None,
        )
        handler.emit(record)

        # Initially buffered (interval has not elapsed)
        assert len(handler._buffer) == 1

        # Wait for interval to pass
        time.sleep(0.1)

        # Next emit should trigger flush due to elapsed time
        record2 = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="",
            lineno=0,
            msg="Second message",
            args=(),
            exc_info=None,
        )
        handler.emit(record2)

        # Both should be flushed
        logs = store.get_logs()
        assert len(logs) == 2

        handler.close()

    def test_set_current_index(self, store) -> None:
        """set_current_index records the index in log entries."""
        handler = DatabaseLogHandler(store, worker_id="Worker-1")
        handler.setFormatter(logging.Formatter("%(message)s"))

        handler.set_current_index(42)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="With index",
            args=(),
            exc_info=None,
        )
        handler.emit(record)
        handler.flush()

        logs = store.get_logs()
        assert len(logs) == 1
        assert logs[0]["idx"] == 42

        handler.close()

    def test_close_flushes_buffer(self, store) -> None:
        """close() flushes any remaining buffered records."""
        handler = DatabaseLogHandler(store, worker_id="Worker-1", buffer_size=100)
        handler.setFormatter(logging.Formatter("%(message)s"))

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Before close",
            args=(),
            exc_info=None,
        )
        handler.emit(record)

        assert len(handler._buffer) == 1
        handler.close()

        # Should be flushed now
        logs = store.get_logs()
        assert len(logs) == 1


# ---------------------------------------------------------------------------
# Test setup_worker_logging
# ---------------------------------------------------------------------------


class TestSetupWorkerLogging:
    """Tests for setup_worker_logging function."""

    def test_setup_worker_logging_returns_handler(self, store) -> None:
        """setup_worker_logging returns a DatabaseLogHandler."""
        root = logging.getLogger("physicsnemo_curator")
        original_handlers = root.handlers.copy()

        handler = setup_worker_logging(store, level=logging.DEBUG)

        try:
            assert isinstance(handler, DatabaseLogHandler)
            assert handler in root.handlers
            assert root.level == logging.DEBUG
        finally:
            # Clean up
            root.handlers = original_handlers
            handler.close()

    def test_setup_worker_logging_with_custom_level(self, store) -> None:
        """setup_worker_logging respects the level parameter."""
        root = logging.getLogger("physicsnemo_curator")
        original_handlers = root.handlers.copy()

        handler = setup_worker_logging(store, level=logging.WARNING)

        try:
            assert handler.level == logging.WARNING
        finally:
            root.handlers = original_handlers
            handler.close()


# ---------------------------------------------------------------------------
# Test flush_logs
# ---------------------------------------------------------------------------


class TestFlushLogs:
    """Tests for flush_logs function."""

    def test_flush_logs_flushes_all_db_handlers(self, store) -> None:
        """flush_logs flushes all DatabaseLogHandler instances."""
        from physicsnemo_curator.core.logging import flush_logs

        root = logging.getLogger("physicsnemo_curator")
        original_handlers = root.handlers.copy()

        handler = DatabaseLogHandler(store, worker_id="Worker-1", buffer_size=100, flush_interval=999)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)

        try:
            # Emit a record (should be buffered)
            record = logging.LogRecord(
                name="test",
                level=logging.DEBUG,
                pathname="",
                lineno=0,
                msg="Buffered message",
                args=(),
                exc_info=None,
            )
            handler.emit(record)
            assert len(handler._buffer) == 1
            assert store.get_logs() == []

            # Call flush_logs
            flush_logs()

            # Should be flushed now
            assert len(handler._buffer) == 0
            logs = store.get_logs()
            assert len(logs) == 1
            assert logs[0]["message"] == "Buffered message"
        finally:
            root.handlers = original_handlers
            handler.close()
