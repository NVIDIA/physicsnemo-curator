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

"""Tests for :mod:`physicsnemo.curator.core.checkpoint`."""

from __future__ import annotations

import json
import pathlib
import sqlite3
from typing import TYPE_CHECKING, ClassVar

import pytest

from physicsnemo.curator.core.base import Filter, Param, Pipeline, Sink, Source
from physicsnemo.curator.core.checkpoint import (
    CheckpointedPipeline,
    _component_config,
    _config_hash,
    _pipeline_config,
)
from physicsnemo.curator.run import run_pipeline

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator


# ---------------------------------------------------------------------------
# Test pipeline components (module-level for pickling)
# ---------------------------------------------------------------------------


class _CountSource(Source[int]):
    """Source that yields integers 0..n-1."""

    name: ClassVar[str] = "CountSource"
    description: ClassVar[str] = "Yields integers"

    @classmethod
    def params(cls) -> list[Param]:
        """Return params."""
        return [Param(name="count", description="Number of items", type=int)]

    def __init__(self, count: int = 5) -> None:
        """Initialize with count."""
        self._count = count

    def __len__(self) -> int:
        """Return count."""
        return self._count

    def __getitem__(self, index: int) -> Generator[int]:
        """Yield the index."""
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            msg = f"Index {index} out of range for source with {len(self)} items."
            raise IndexError(msg)
        yield index


class _DoubleFilter(Filter[int]):
    """Filter that doubles values."""

    name: ClassVar[str] = "Double"
    description: ClassVar[str] = "Doubles values"

    @classmethod
    def params(cls) -> list[Param]:
        """Return empty params."""
        return []

    def __call__(self, items: Generator[int]) -> Generator[int]:
        """Double each item."""
        for item in items:
            yield item * 2


class _PathSink(Sink[int]):
    """Sink that writes items to files."""

    name: ClassVar[str] = "PathSink"
    description: ClassVar[str] = "Writes to files"

    @classmethod
    def params(cls) -> list[Param]:
        """Return params."""
        return [Param(name="output_dir", description="Output directory", type=str)]

    def __init__(self, output_dir: str) -> None:
        """Initialize with output dir."""
        self._output_dir = pathlib.Path(output_dir)

    def __call__(self, items: Iterator[int], index: int) -> list[str]:
        """Write items and return paths."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for seq, val in enumerate(items):
            p = self._output_dir / f"item_{index:04d}_{seq}.txt"
            p.write_text(str(val))
            paths.append(str(p))
        return paths


class _FailingSource(Source[int]):
    """Source that raises on specific indices."""

    name: ClassVar[str] = "FailingSource"
    description: ClassVar[str] = "Fails on specified indices"

    @classmethod
    def params(cls) -> list[Param]:
        """Return params."""
        return [Param(name="count", description="Total items", type=int)]

    def __init__(self, count: int = 5, *, fail_on: set[int] | None = None) -> None:
        """Initialize."""
        self._count = count
        self._fail_on = fail_on or set()

    def __len__(self) -> int:
        """Return count."""
        return self._count

    def __getitem__(self, index: int) -> Generator[int]:
        """Yield or raise."""
        if index in self._fail_on:
            msg = f"Simulated failure on index {index}"
            raise RuntimeError(msg)
        yield index


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline(tmp_path: pathlib.Path, count: int = 5) -> Pipeline[int]:
    """Create a simple test pipeline."""
    return Pipeline(
        source=_CountSource(count=count),
        filters=[_DoubleFilter()],
        sink=_PathSink(output_dir=str(tmp_path / "output")),
    )


# ---------------------------------------------------------------------------
# Config serialization tests
# ---------------------------------------------------------------------------


class TestConfigSerialization:
    """Tests for pipeline config serialization and hashing."""

    def test_component_config_captures_class_info(self) -> None:
        """Component config includes class name and module."""
        source = _CountSource(count=10)
        config = _component_config(source)
        assert config["class"] == "_CountSource"
        assert "module" in config
        assert config["name"] == "CountSource"

    def test_component_config_captures_params(self) -> None:
        """Component config includes parameter values."""
        source = _CountSource(count=42)
        config = _component_config(source)
        assert config["params"]["count"] == 42

    def test_pipeline_config_structure(self, tmp_path: pathlib.Path) -> None:
        """Full pipeline config has source, filters, and sink."""
        pipeline = _make_pipeline(tmp_path)
        config = _pipeline_config(pipeline)
        assert "source" in config
        assert "filters" in config
        assert "sink" in config
        assert len(config["filters"]) == 1
        assert config["filters"][0]["class"] == "_DoubleFilter"

    def test_config_hash_is_stable(self, tmp_path: pathlib.Path) -> None:
        """Same pipeline config produces the same hash."""
        p1 = _make_pipeline(tmp_path)
        p2 = _make_pipeline(tmp_path)
        c1 = _pipeline_config(p1)
        c2 = _pipeline_config(p2)
        assert _config_hash(c1) == _config_hash(c2)

    def test_config_hash_changes_with_params(self, tmp_path: pathlib.Path) -> None:
        """Different params produce different hashes."""
        p1 = Pipeline(
            source=_CountSource(count=5),
            filters=[],
            sink=_PathSink(output_dir=str(tmp_path / "out1")),
        )
        p2 = Pipeline(
            source=_CountSource(count=10),
            filters=[],
            sink=_PathSink(output_dir=str(tmp_path / "out2")),
        )
        h1 = _config_hash(_pipeline_config(p1))
        h2 = _config_hash(_pipeline_config(p2))
        assert h1 != h2


# ---------------------------------------------------------------------------
# Database schema tests
# ---------------------------------------------------------------------------


class TestCheckpointDB:
    """Tests for SQLite database initialization and schema."""

    def test_creates_db_file(self, tmp_path: pathlib.Path) -> None:
        """Checkpoint creates the SQLite database file."""
        pipeline = _make_pipeline(tmp_path)
        db = tmp_path / "checkpoint.db"
        CheckpointedPipeline(pipeline, db_path=db)
        assert db.exists()

    def test_creates_parent_dirs(self, tmp_path: pathlib.Path) -> None:
        """Checkpoint creates parent directories for the DB path."""
        pipeline = _make_pipeline(tmp_path)
        db = tmp_path / "nested" / "deep" / "checkpoint.db"
        CheckpointedPipeline(pipeline, db_path=db)
        assert db.exists()

    def test_schema_has_expected_tables(self, tmp_path: pathlib.Path) -> None:
        """Database has pipeline_runs and completed_indices tables."""
        pipeline = _make_pipeline(tmp_path)
        db = tmp_path / "checkpoint.db"
        CheckpointedPipeline(pipeline, db_path=db)

        conn = sqlite3.connect(str(db))
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        conn.close()

        table_names = {t[0] for t in tables}
        assert "pipeline_runs" in table_names
        assert "completed_indices" in table_names

    def test_registers_pipeline_run(self, tmp_path: pathlib.Path) -> None:
        """Creating a CheckpointedPipeline registers a run."""
        pipeline = _make_pipeline(tmp_path)
        db = tmp_path / "checkpoint.db"
        cp = CheckpointedPipeline(pipeline, db_path=db)

        conn = sqlite3.connect(str(db))
        rows = conn.execute("SELECT config_hash FROM pipeline_runs").fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0][0] == cp.config_hash

    def test_wal_mode_enabled(self, tmp_path: pathlib.Path) -> None:
        """Database uses WAL journal mode."""
        pipeline = _make_pipeline(tmp_path)
        db = tmp_path / "checkpoint.db"
        CheckpointedPipeline(pipeline, db_path=db)

        conn = sqlite3.connect(str(db))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()

        assert mode == "wal"


# ---------------------------------------------------------------------------
# Skip logic tests
# ---------------------------------------------------------------------------


class TestSkipLogic:
    """Tests for index skipping on restart."""

    def test_first_run_executes_all(self, tmp_path: pathlib.Path) -> None:
        """On first run, all indices are executed."""
        pipeline = _make_pipeline(tmp_path, count=3)
        cp = CheckpointedPipeline(pipeline, db_path=tmp_path / "cp.db")

        for i in range(3):
            paths = cp[i]
            assert len(paths) == 1
            assert pathlib.Path(paths[0]).exists()

        assert cp.completed_indices == {0, 1, 2}

    def test_second_run_skips_completed(self, tmp_path: pathlib.Path) -> None:
        """On restart, completed indices return cached paths without running."""
        pipeline = _make_pipeline(tmp_path, count=3)
        db = tmp_path / "cp.db"
        cp1 = CheckpointedPipeline(pipeline, db_path=db)

        # First run
        original_paths = [cp1[i] for i in range(3)]

        # Second run — same pipeline, same DB
        pipeline2 = _make_pipeline(tmp_path, count=3)
        cp2 = CheckpointedPipeline(pipeline2, db_path=db)

        for i in range(3):
            cached = cp2[i]
            assert cached == original_paths[i]

    def test_skip_does_not_call_inner_pipeline(self, tmp_path: pathlib.Path) -> None:
        """Cached indices do not call the inner pipeline's __getitem__."""
        pipeline = _make_pipeline(tmp_path, count=2)
        db = tmp_path / "cp.db"
        cp = CheckpointedPipeline(pipeline, db_path=db)

        # Process index 0
        cp[0]

        # Track calls via wrapper
        original_getitem = Pipeline.__getitem__
        call_count = 0

        def tracked_getitem(self_inner: Pipeline[int], idx: int) -> list[str]:
            nonlocal call_count
            call_count += 1
            return original_getitem(self_inner, idx)

        from unittest.mock import patch

        with patch.object(Pipeline, "__getitem__", tracked_getitem):
            # This should use the cache
            cp[0]
            assert call_count == 0

            # This should call the pipeline
            cp[1]
            assert call_count == 1

    def test_partial_resume(self, tmp_path: pathlib.Path) -> None:
        """Only unprocessed indices run on restart."""
        pipeline = _make_pipeline(tmp_path, count=5)
        db = tmp_path / "cp.db"
        cp = CheckpointedPipeline(pipeline, db_path=db)

        # Process only first 3
        for i in range(3):
            cp[i]

        assert cp.completed_indices == {0, 1, 2}
        assert cp.remaining_indices == [3, 4]


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error recording and re-raise behavior."""

    def test_failed_index_is_not_completed(self, tmp_path: pathlib.Path) -> None:
        """A failed index is not marked as completed."""
        pipeline = Pipeline(
            source=_FailingSource(count=3, fail_on={1}),
            filters=[],
            sink=_PathSink(output_dir=str(tmp_path / "output")),
        )
        db = tmp_path / "cp.db"
        cp = CheckpointedPipeline(pipeline, db_path=db)

        cp[0]  # success
        with pytest.raises(RuntimeError, match="Simulated failure"):
            cp[1]  # failure
        cp[2]  # success

        assert cp.completed_indices == {0, 2}
        assert 1 in cp.failed_indices

    def test_failed_index_records_error_message(self, tmp_path: pathlib.Path) -> None:
        """Error message is stored in the database."""
        pipeline = Pipeline(
            source=_FailingSource(count=2, fail_on={0}),
            filters=[],
            sink=_PathSink(output_dir=str(tmp_path / "output")),
        )
        cp = CheckpointedPipeline(pipeline, db_path=tmp_path / "cp.db")

        with pytest.raises(RuntimeError):
            cp[0]

        errors = cp.failed_indices
        assert 0 in errors
        assert "Simulated failure" in errors[0]

    def test_failed_index_retried_on_restart(self, tmp_path: pathlib.Path) -> None:
        """A failed index is retried (not skipped) on restart."""
        # First run with failure
        pipeline1 = Pipeline(
            source=_FailingSource(count=2, fail_on={0}),
            filters=[],
            sink=_PathSink(output_dir=str(tmp_path / "output")),
        )
        db = tmp_path / "cp.db"
        cp1 = CheckpointedPipeline(pipeline1, db_path=db)
        with pytest.raises(RuntimeError):
            cp1[0]

        # Second run — source fixed, should succeed
        pipeline2 = Pipeline(
            source=_CountSource(count=2),
            filters=[],
            sink=_PathSink(output_dir=str(tmp_path / "output")),
        )
        cp2 = CheckpointedPipeline(pipeline2, db_path=db)
        result = cp2[0]
        assert len(result) == 1
        assert cp2.completed_indices == {0}


# ---------------------------------------------------------------------------
# Provenance tests
# ---------------------------------------------------------------------------


class TestProvenance:
    """Tests for config drift detection and provenance storage."""

    def test_same_config_reuses_run_id(self, tmp_path: pathlib.Path) -> None:
        """Same pipeline config produces the same run_id."""
        pipeline = _make_pipeline(tmp_path)
        db = tmp_path / "cp.db"

        cp1 = CheckpointedPipeline(pipeline, db_path=db)
        run_id_1 = cp1._run_id

        cp2 = CheckpointedPipeline(_make_pipeline(tmp_path), db_path=db)
        run_id_2 = cp2._run_id

        assert run_id_1 == run_id_2

    def test_config_drift_warns(self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture) -> None:
        """Different pipeline config logs a warning but proceeds."""
        import logging

        db = tmp_path / "cp.db"

        # First pipeline
        p1 = Pipeline(
            source=_CountSource(count=5),
            filters=[],
            sink=_PathSink(output_dir=str(tmp_path / "out1")),
        )
        CheckpointedPipeline(p1, db_path=db)

        # Second pipeline — different config
        p2 = Pipeline(
            source=_CountSource(count=10),
            filters=[_DoubleFilter()],  # ty: ignore[invalid-argument-type]
            sink=_PathSink(output_dir=str(tmp_path / "out2")),
        )
        with caplog.at_level(logging.WARNING):
            cp2 = CheckpointedPipeline(p2, db_path=db)

        assert "config has changed" in caplog.text
        assert cp2._run_id is not None

    def test_config_stored_in_db(self, tmp_path: pathlib.Path) -> None:
        """Pipeline config JSON is stored in the pipeline_runs table."""
        pipeline = _make_pipeline(tmp_path)
        db = tmp_path / "cp.db"
        cp = CheckpointedPipeline(pipeline, db_path=db)

        conn = sqlite3.connect(str(db))
        row = conn.execute("SELECT config_json FROM pipeline_runs WHERE run_id = ?", (cp._run_id,)).fetchone()
        conn.close()

        config = json.loads(row[0])
        assert config["source"]["class"] == "_CountSource"
        assert "filters" in config
        assert config["sink"]["class"] == "_PathSink"


# ---------------------------------------------------------------------------
# Query API tests
# ---------------------------------------------------------------------------


class TestQueryAPI:
    """Tests for the checkpoint query API."""

    def test_completed_indices_empty_initially(self, tmp_path: pathlib.Path) -> None:
        """No indices completed initially."""
        cp = CheckpointedPipeline(_make_pipeline(tmp_path), db_path=tmp_path / "cp.db")
        assert cp.completed_indices == set()

    def test_remaining_indices_all_initially(self, tmp_path: pathlib.Path) -> None:
        """All indices remaining initially."""
        cp = CheckpointedPipeline(_make_pipeline(tmp_path, count=3), db_path=tmp_path / "cp.db")
        assert cp.remaining_indices == [0, 1, 2]

    def test_summary(self, tmp_path: pathlib.Path) -> None:
        """Summary returns correct counts."""
        cp = CheckpointedPipeline(_make_pipeline(tmp_path, count=5), db_path=tmp_path / "cp.db")
        cp[0]
        cp[1]

        s = cp.summary()
        assert s["total"] == 5
        assert s["completed"] == 2
        assert s["remaining"] == 3
        assert s["failed"] == 0
        assert s["config_hash"] == cp.config_hash
        assert s["total_elapsed_s"] >= 0

    def test_db_path_property(self, tmp_path: pathlib.Path) -> None:
        """db_path property returns the configured path."""
        db = tmp_path / "my.db"
        cp = CheckpointedPipeline(_make_pipeline(tmp_path), db_path=db)
        assert cp.db_path == db


# ---------------------------------------------------------------------------
# Reset tests
# ---------------------------------------------------------------------------


class TestReset:
    """Tests for checkpoint reset."""

    def test_reset_clears_completions(self, tmp_path: pathlib.Path) -> None:
        """Reset clears all completion records."""
        cp = CheckpointedPipeline(_make_pipeline(tmp_path, count=3), db_path=tmp_path / "cp.db")
        for i in range(3):
            cp[i]
        assert len(cp.completed_indices) == 3

        cp.reset()
        assert len(cp.completed_indices) == 0
        assert len(cp.remaining_indices) == 3

    def test_reset_allows_reprocessing(self, tmp_path: pathlib.Path) -> None:
        """After reset, indices are reprocessed."""
        pipeline = _make_pipeline(tmp_path, count=2)
        cp = CheckpointedPipeline(pipeline, db_path=tmp_path / "cp.db")
        cp[0]
        cp.reset()

        # Should execute again (not skip)
        original = Pipeline.__getitem__
        call_count = 0

        def tracked(self_inner: Pipeline[int], idx: int) -> list[str]:
            nonlocal call_count
            call_count += 1
            return original(self_inner, idx)

        from unittest.mock import patch

        with patch.object(Pipeline, "__getitem__", tracked):
            cp[0]
            assert call_count == 1


# ---------------------------------------------------------------------------
# Duck-type protocol tests
# ---------------------------------------------------------------------------


class TestDuckType:
    """Tests that CheckpointedPipeline is duck-type compatible with Pipeline."""

    def test_has_source(self, tmp_path: pathlib.Path) -> None:
        """Exposes the inner pipeline's source."""
        pipeline = _make_pipeline(tmp_path)
        cp = CheckpointedPipeline(pipeline, db_path=tmp_path / "cp.db")
        assert cp.source is pipeline.source

    def test_has_filters(self, tmp_path: pathlib.Path) -> None:
        """Exposes the inner pipeline's filters."""
        pipeline = _make_pipeline(tmp_path)
        cp = CheckpointedPipeline(pipeline, db_path=tmp_path / "cp.db")
        assert cp.filters is pipeline.filters

    def test_has_sink(self, tmp_path: pathlib.Path) -> None:
        """Exposes the inner pipeline's sink."""
        pipeline = _make_pipeline(tmp_path)
        cp = CheckpointedPipeline(pipeline, db_path=tmp_path / "cp.db")
        assert cp.sink is pipeline.sink

    def test_len(self, tmp_path: pathlib.Path) -> None:
        """Length delegates to inner pipeline."""
        cp = CheckpointedPipeline(_make_pipeline(tmp_path, count=7), db_path=tmp_path / "cp.db")
        assert len(cp) == 7


# ---------------------------------------------------------------------------
# Composability tests
# ---------------------------------------------------------------------------


class TestComposability:
    """Tests for composing CheckpointedPipeline with ProfiledPipeline."""

    def test_wraps_profiled_pipeline(self, tmp_path: pathlib.Path) -> None:
        """CheckpointedPipeline can wrap a ProfiledPipeline."""
        from physicsnemo.curator.core.profiling import ProfiledPipeline

        pipeline = _make_pipeline(tmp_path, count=2)
        profiled = ProfiledPipeline(pipeline)
        cp = CheckpointedPipeline(profiled, db_path=tmp_path / "cp.db")

        result = cp[0]
        assert len(result) == 1
        assert cp.completed_indices == {0}


# ---------------------------------------------------------------------------
# Concurrency test
# ---------------------------------------------------------------------------


class TestConcurrency:
    """Tests for concurrent access with WAL mode."""

    def test_concurrent_writes_from_threads(self, tmp_path: pathlib.Path) -> None:
        """Multiple threads can write checkpoints concurrently."""
        import concurrent.futures

        pipeline = _make_pipeline(tmp_path, count=10)
        db = tmp_path / "cp.db"
        cp = CheckpointedPipeline(pipeline, db_path=db)

        def process(idx: int) -> list[str]:
            return cp[idx]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process, i) for i in range(10)]
            results = [f.result() for f in futures]

        assert len(cp.completed_indices) == 10
        assert all(len(r) == 1 for r in results)


# ---------------------------------------------------------------------------
# Integration with run_pipeline
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRunPipelineIntegration:
    """Tests with the actual run_pipeline function."""

    def test_sequential_with_checkpoint(self, tmp_path: pathlib.Path) -> None:
        """CheckpointedPipeline works with sequential backend."""
        pipeline = _make_pipeline(tmp_path, count=3)
        cp = CheckpointedPipeline(pipeline, db_path=tmp_path / "cp.db")

        results = run_pipeline(cp, n_jobs=1, backend="sequential")
        assert len(results) == 3
        assert cp.completed_indices == {0, 1, 2}

    def test_sequential_resume(self, tmp_path: pathlib.Path) -> None:
        """Sequential backend correctly resumes from checkpoint."""
        pipeline = _make_pipeline(tmp_path, count=5)
        db = tmp_path / "cp.db"

        # First run: process first 3
        cp1 = CheckpointedPipeline(pipeline, db_path=db)
        run_pipeline(cp1, n_jobs=1, backend="sequential", indices=range(3))

        # Second run: process all 5 (first 3 should be skipped)
        pipeline2 = _make_pipeline(tmp_path, count=5)
        cp2 = CheckpointedPipeline(pipeline2, db_path=db)
        results = run_pipeline(cp2, n_jobs=1, backend="sequential")
        assert len(results) == 5
        assert cp2.completed_indices == {0, 1, 2, 3, 4}

    def test_thread_pool_with_checkpoint(self, tmp_path: pathlib.Path) -> None:
        """CheckpointedPipeline works with thread_pool backend."""
        pipeline = _make_pipeline(tmp_path, count=5)
        cp = CheckpointedPipeline(pipeline, db_path=tmp_path / "cp.db")

        results = run_pipeline(cp, n_jobs=2, backend="thread_pool")
        assert len(results) == 5
        assert cp.completed_indices == {0, 1, 2, 3, 4}
