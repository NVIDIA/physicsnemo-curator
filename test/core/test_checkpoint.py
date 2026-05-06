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

"""Tests for Pipeline checkpoint behavior (track_metrics=True)."""

from __future__ import annotations

import json
import pathlib
import sqlite3
from typing import TYPE_CHECKING, ClassVar

import pytest

from physicsnemo_curator.core.base import Filter, Param, Pipeline, Sink, Source
from physicsnemo_curator.core.pipeline_store import (
    _component_config,
    _config_hash,
    _pipeline_config,
)
from physicsnemo_curator.run import run_pipeline

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator


pytestmark = pytest.mark.unit

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
    """Create a simple test pipeline with metrics enabled."""
    return Pipeline(
        source=_CountSource(count=count),
        filters=[_DoubleFilter()],
        sink=_PathSink(output_dir=str(tmp_path / "output")),
        track_metrics=True,
        db_dir=tmp_path / ".pnc",
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
            track_metrics=False,
        )
        p2 = Pipeline(
            source=_CountSource(count=10),
            filters=[],
            sink=_PathSink(output_dir=str(tmp_path / "out2")),
            track_metrics=False,
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
        """Pipeline with track_metrics creates the SQLite database on first access."""
        pipeline = _make_pipeline(tmp_path)
        pipeline[0]  # force store creation
        store = pipeline._get_store()  # noqa: SLF001
        assert store._db_path.exists()  # noqa: SLF001

    def test_schema_has_expected_tables(self, tmp_path: pathlib.Path) -> None:
        """Database has pipeline_runs, index_results, and stage_metrics tables."""
        pipeline = _make_pipeline(tmp_path)
        pipeline[0]  # force store creation
        store = pipeline._get_store()  # noqa: SLF001

        conn = sqlite3.connect(str(store._db_path))  # noqa: SLF001
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        conn.close()

        table_names = {t[0] for t in tables}
        assert "pipeline_runs" in table_names
        assert "index_results" in table_names
        assert "stage_metrics" in table_names

    def test_registers_pipeline_run(self, tmp_path: pathlib.Path) -> None:
        """First access registers a run in pipeline_runs."""
        pipeline = _make_pipeline(tmp_path)
        pipeline[0]  # force store creation
        store = pipeline._get_store()  # noqa: SLF001

        conn = sqlite3.connect(str(store._db_path))  # noqa: SLF001
        rows = conn.execute("SELECT config_hash FROM pipeline_runs").fetchall()
        conn.close()

        assert len(rows) == 1

    def test_wal_mode_enabled(self, tmp_path: pathlib.Path) -> None:
        """Database uses WAL journal mode."""
        pipeline = _make_pipeline(tmp_path)
        pipeline[0]  # force store creation
        store = pipeline._get_store()  # noqa: SLF001

        conn = sqlite3.connect(str(store._db_path))  # noqa: SLF001
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

        for i in range(3):
            paths = pipeline[i]
            assert len(paths) == 1
            assert pathlib.Path(paths[0]).exists()

        assert pipeline.completed_indices == {0, 1, 2}

    def test_second_run_reprocesses_all(self, tmp_path: pathlib.Path) -> None:
        """By default, a new pipeline gets a fresh DB and reprocesses all indices."""
        db_dir = tmp_path / ".pnc"
        pipeline = Pipeline(
            source=_CountSource(count=3),
            filters=[_DoubleFilter()],  # ty: ignore[invalid-argument-type]
            sink=_PathSink(output_dir=str(tmp_path / "output")),
            track_metrics=True,
            db_dir=db_dir,
        )

        # First run
        for i in range(3):
            pipeline[i]

        assert pipeline.completed_indices == {0, 1, 2}

        # Second pipeline with same config gets a fresh DB
        pipeline2 = Pipeline(
            source=_CountSource(count=3),
            filters=[_DoubleFilter()],  # ty: ignore[invalid-argument-type]
            sink=_PathSink(output_dir=str(tmp_path / "output")),
            track_metrics=True,
            db_dir=db_dir,
        )

        # Store is fresh — no completed indices
        assert pipeline2.completed_indices == set()

    def test_resume_skips_completed(self, tmp_path: pathlib.Path) -> None:
        """With resume=True, completed indices return cached paths without re-running."""
        db_dir = tmp_path / ".pnc"
        pipeline = Pipeline(
            source=_CountSource(count=3),
            filters=[_DoubleFilter()],  # ty: ignore[invalid-argument-type]
            sink=_PathSink(output_dir=str(tmp_path / "output")),
            track_metrics=True,
            db_dir=db_dir,
            resume=True,
        )

        # First run
        original_paths = [pipeline[i] for i in range(3)]

        # Second pipeline with resume=True reuses the same DB
        pipeline2 = Pipeline(
            source=_CountSource(count=3),
            filters=[_DoubleFilter()],  # ty: ignore[invalid-argument-type]
            sink=_PathSink(output_dir=str(tmp_path / "output")),
            track_metrics=True,
            db_dir=db_dir,
            resume=True,
        )

        for i in range(3):
            cached = pipeline2[i]
            assert cached == original_paths[i]

    def test_partial_resume(self, tmp_path: pathlib.Path) -> None:
        """Only unprocessed indices run on restart."""
        pipeline = _make_pipeline(tmp_path, count=5)

        # Process only first 3
        for i in range(3):
            pipeline[i]

        assert pipeline.completed_indices == {0, 1, 2}
        assert pipeline.remaining_indices() == [3, 4]


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
            track_metrics=True,
            db_dir=tmp_path / ".pnc",
        )

        pipeline[0]  # success
        with pytest.raises(RuntimeError, match="Simulated failure"):
            pipeline[1]  # failure
        pipeline[2]  # success

        assert pipeline.completed_indices == {0, 2}
        assert 1 in pipeline.failed_indices

    def test_failed_index_records_error_message(self, tmp_path: pathlib.Path) -> None:
        """Error message is stored in the database."""
        pipeline = Pipeline(
            source=_FailingSource(count=2, fail_on={0}),
            filters=[],
            sink=_PathSink(output_dir=str(tmp_path / "output")),
            track_metrics=True,
            db_dir=tmp_path / ".pnc",
        )

        with pytest.raises(RuntimeError):
            pipeline[0]

        errors = pipeline.failed_indices
        assert 0 in errors
        assert "Simulated failure" in errors[0]

    def test_failed_index_retried_on_restart(self, tmp_path: pathlib.Path) -> None:
        """A failed index is retried (not skipped) on restart."""
        db_dir = tmp_path / ".pnc"

        # First run with failure
        pipeline1 = Pipeline(
            source=_FailingSource(count=2, fail_on={0}),
            filters=[],
            sink=_PathSink(output_dir=str(tmp_path / "output")),
            track_metrics=True,
            db_dir=db_dir,
        )
        with pytest.raises(RuntimeError):
            pipeline1[0]

        # Second run — source fixed, should succeed (different config hash though)
        pipeline2 = Pipeline(
            source=_CountSource(count=2),
            filters=[],
            sink=_PathSink(output_dir=str(tmp_path / "output")),
            track_metrics=True,
            db_dir=db_dir,
        )
        result = pipeline2[0]
        assert len(result) == 1
        assert pipeline2.completed_indices == {0}


# ---------------------------------------------------------------------------
# Provenance tests
# ---------------------------------------------------------------------------


class TestProvenance:
    """Tests for config drift detection and provenance storage."""

    def test_same_config_gets_separate_dbs(self, tmp_path: pathlib.Path) -> None:
        """Each pipeline construction creates a separate DB file by default."""
        db_dir = tmp_path / ".pnc"
        pipeline1 = Pipeline(
            source=_CountSource(count=5),
            filters=[_DoubleFilter()],  # ty: ignore[invalid-argument-type]
            sink=_PathSink(output_dir=str(tmp_path / "output")),
            track_metrics=True,
            db_dir=db_dir,
        )
        pipeline1[0]
        db_path_1 = pipeline1._get_store()._db_path  # noqa: SLF001

        pipeline2 = Pipeline(
            source=_CountSource(count=5),
            filters=[_DoubleFilter()],  # ty: ignore[invalid-argument-type]
            sink=_PathSink(output_dir=str(tmp_path / "output")),
            track_metrics=True,
            db_dir=db_dir,
        )
        pipeline2[0]
        db_path_2 = pipeline2._get_store()._db_path  # noqa: SLF001

        assert db_path_1 != db_path_2

    def test_resume_reuses_db(self, tmp_path: pathlib.Path) -> None:
        """With resume=True, same config reuses the same DB file and run_id."""
        db_dir = tmp_path / ".pnc"
        pipeline1 = Pipeline(
            source=_CountSource(count=5),
            filters=[_DoubleFilter()],  # ty: ignore[invalid-argument-type]
            sink=_PathSink(output_dir=str(tmp_path / "output")),
            track_metrics=True,
            db_dir=db_dir,
            resume=True,
        )
        pipeline1[0]
        run_id_1 = pipeline1._get_store()._run_id  # noqa: SLF001
        db_path_1 = pipeline1._get_store()._db_path  # noqa: SLF001

        pipeline2 = Pipeline(
            source=_CountSource(count=5),
            filters=[_DoubleFilter()],  # ty: ignore[invalid-argument-type]
            sink=_PathSink(output_dir=str(tmp_path / "output")),
            track_metrics=True,
            db_dir=db_dir,
            resume=True,
        )
        pipeline2[0]
        run_id_2 = pipeline2._get_store()._run_id  # noqa: SLF001
        db_path_2 = pipeline2._get_store()._db_path  # noqa: SLF001

        assert db_path_1 == db_path_2
        assert run_id_1 == run_id_2

    def test_config_stored_in_db(self, tmp_path: pathlib.Path) -> None:
        """Pipeline config JSON is stored in the pipeline_runs table."""
        pipeline = _make_pipeline(tmp_path)
        pipeline[0]  # force store
        store = pipeline._get_store()  # noqa: SLF001

        conn = sqlite3.connect(str(store._db_path))  # noqa: SLF001
        row = conn.execute(
            "SELECT config_json FROM pipeline_runs WHERE run_id = ?",
            (store._run_id,),  # noqa: SLF001
        ).fetchone()
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
        pipeline = _make_pipeline(tmp_path)
        assert pipeline.completed_indices == set()

    def test_remaining_indices_all_initially(self, tmp_path: pathlib.Path) -> None:
        """All indices remaining initially."""
        pipeline = _make_pipeline(tmp_path, count=3)
        assert pipeline.remaining_indices() == [0, 1, 2]

    def test_summary(self, tmp_path: pathlib.Path) -> None:
        """Summary returns correct counts."""
        pipeline = _make_pipeline(tmp_path, count=5)
        pipeline[0]
        pipeline[1]

        s = pipeline.summary()
        assert s["total"] == 5
        assert s["completed"] == 2
        assert s["remaining"] == 3
        assert s["failed"] == 0
        assert s["total_elapsed_s"] >= 0

    def test_metrics_disabled_raises(self) -> None:
        """Query API raises RuntimeError when track_metrics=False."""
        pipeline = Pipeline(
            source=_CountSource(count=3),
            track_metrics=False,
        )
        with pytest.raises(RuntimeError, match="track_metrics"):
            _ = pipeline.completed_indices


# ---------------------------------------------------------------------------
# Reset tests
# ---------------------------------------------------------------------------


class TestReset:
    """Tests for checkpoint reset."""

    def test_reset_clears_completions(self, tmp_path: pathlib.Path) -> None:
        """Reset clears all completion records."""
        pipeline = _make_pipeline(tmp_path, count=3)
        for i in range(3):
            pipeline[i]
        assert len(pipeline.completed_indices) == 3

        pipeline.reset()
        assert len(pipeline.completed_indices) == 0
        assert len(pipeline.remaining_indices()) == 3

    def test_reset_allows_reprocessing(self, tmp_path: pathlib.Path) -> None:
        """After reset, indices are reprocessed."""
        pipeline = _make_pipeline(tmp_path, count=2)
        pipeline[0]
        pipeline.reset()

        # Should execute again (not skip) — verify by checking it succeeds
        second_paths = pipeline[0]
        assert len(second_paths) == 1


# ---------------------------------------------------------------------------
# Concurrency test
# ---------------------------------------------------------------------------


class TestConcurrency:
    """Tests for concurrent access with WAL mode."""

    def test_concurrent_writes_from_threads(self, tmp_path: pathlib.Path) -> None:
        """Multiple threads can write checkpoints concurrently."""
        import concurrent.futures

        pipeline = _make_pipeline(tmp_path, count=10)

        def process(idx: int) -> list[str]:
            return pipeline[idx]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process, i) for i in range(10)]
            results = [f.result() for f in futures]

        assert len(pipeline.completed_indices) == 10
        assert all(len(r) == 1 for r in results)


# ---------------------------------------------------------------------------
# Integration with run_pipeline
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRunPipelineIntegration:
    """Tests with the actual run_pipeline function."""

    def test_sequential_with_checkpoint(self, tmp_path: pathlib.Path) -> None:
        """Pipeline with track_metrics works with sequential backend."""
        pipeline = _make_pipeline(tmp_path, count=3)

        results = run_pipeline(pipeline, n_jobs=1, backend="sequential", progress=False)
        assert len(results) == 3
        assert pipeline.completed_indices == {0, 1, 2}

    def test_sequential_no_resume_across_pipelines(self, tmp_path: pathlib.Path) -> None:
        """A new pipeline does not resume from a previous pipeline's checkpoint."""
        db_dir = tmp_path / ".pnc"

        pipeline = Pipeline(
            source=_CountSource(count=5),
            filters=[_DoubleFilter()],  # ty: ignore[invalid-argument-type]
            sink=_PathSink(output_dir=str(tmp_path / "output")),
            track_metrics=True,
            db_dir=db_dir,
        )

        # First run: process first 3
        run_pipeline(pipeline, n_jobs=1, backend="sequential", indices=range(3), progress=False)
        assert pipeline.completed_indices == {0, 1, 2}

        # Second pipeline gets a fresh DB — processes all 5 from scratch
        pipeline2 = Pipeline(
            source=_CountSource(count=5),
            filters=[_DoubleFilter()],  # ty: ignore[invalid-argument-type]
            sink=_PathSink(output_dir=str(tmp_path / "output")),
            track_metrics=True,
            db_dir=db_dir,
        )
        results = run_pipeline(pipeline2, n_jobs=1, backend="sequential", progress=False)
        assert len(results) == 5
        assert pipeline2.completed_indices == {0, 1, 2, 3, 4}

    def test_sequential_resume_with_flag(self, tmp_path: pathlib.Path) -> None:
        """With resume=True, sequential backend resumes from checkpoint."""
        db_dir = tmp_path / ".pnc"

        pipeline = Pipeline(
            source=_CountSource(count=5),
            filters=[_DoubleFilter()],  # ty: ignore[invalid-argument-type]
            sink=_PathSink(output_dir=str(tmp_path / "output")),
            track_metrics=True,
            db_dir=db_dir,
            resume=True,
        )

        # First run: process first 3
        run_pipeline(pipeline, n_jobs=1, backend="sequential", indices=range(3), progress=False)
        assert pipeline.completed_indices == {0, 1, 2}

        # Second pipeline with resume=True reuses the same DB
        pipeline2 = Pipeline(
            source=_CountSource(count=5),
            filters=[_DoubleFilter()],  # ty: ignore[invalid-argument-type]
            sink=_PathSink(output_dir=str(tmp_path / "output")),
            track_metrics=True,
            db_dir=db_dir,
            resume=True,
        )
        results = run_pipeline(pipeline2, n_jobs=1, backend="sequential", progress=False)
        assert len(results) == 5
        assert pipeline2.completed_indices == {0, 1, 2, 3, 4}


class TestWorkerTracking:
    """Tests for worker progress tracking via Pipeline.__getitem__."""

    pytestmark = pytest.mark.unit

    def test_workers_registered_after_execution(self, tmp_path: pathlib.Path) -> None:
        """Pipeline execution registers at least one worker."""
        pipeline = _make_pipeline(tmp_path, count=3)
        for i in range(3):
            pipeline[i]
        workers = pipeline.active_workers
        assert len(workers) >= 1
        w = workers[0]
        assert "worker_id" in w
        assert "pid" in w
        assert "hostname" in w
        assert w["current_index"] is None  # finished

    def test_worker_cleared_after_checkpoint_hit(self, tmp_path: pathlib.Path) -> None:
        """Worker current_index is cleared even on checkpoint hits."""
        pipeline = _make_pipeline(tmp_path, count=2)
        pipeline[0]
        pipeline[0]  # checkpoint hit
        workers = pipeline.active_workers
        assert all(w["current_index"] is None for w in workers)
