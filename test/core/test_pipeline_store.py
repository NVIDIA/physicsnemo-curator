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

"""Tests for PipelineStore, metrics dataclasses, _TimedGenerator, and provenance helpers."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

import pytest

from physicsnemo_curator.core.base import Filter, Param, Sink, Source
from physicsnemo_curator.core.pipeline_store import (
    IndexMetrics,
    PipelineMetrics,
    PipelineStore,
    StageMetrics,
    _component_config,
    _config_hash,
    _pipeline_config,
    _TimedGenerator,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Test components (Source, Filter, Sink for provenance / PipelineStore tests)
# ---------------------------------------------------------------------------


class IntSource(Source[int]):
    """Test source that yields integers from a list."""

    name: ClassVar[str] = "Int Source"
    description: ClassVar[str] = "Yields integers for testing"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameters for this source."""
        return [Param(name="values", description="List of ints", type=str)]

    def __init__(self, values: list[int]) -> None:
        """Initialize with a list of values."""
        self._values = values

    def __len__(self) -> int:
        """Return the number of values."""
        return len(self._values)

    def __getitem__(self, index: int) -> Generator[int]:
        """Yield the value at the given index."""
        yield self._values[index]


class DoubleFilter(Filter[int]):
    """Test filter that doubles each value."""

    name: ClassVar[str] = "Double"
    description: ClassVar[str] = "Doubles each integer"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameters for this filter."""
        return []

    def __call__(self, items: Generator[int]) -> Generator[int]:
        """Double each item in the stream."""
        for item in items:
            yield item * 2


class ScaleFilter(Filter[int]):
    """Test filter that multiplies each value by a configurable factor."""

    name: ClassVar[str] = "Scale"
    description: ClassVar[str] = "Scales each integer by a factor"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameters for this filter."""
        return [Param(name="factor", description="Scale factor", type=int, default=1)]

    def __init__(self, factor: int = 1) -> None:
        """Initialize with a scale factor."""
        self._factor = factor

    def __call__(self, items: Generator[int]) -> Generator[int]:
        """Scale each item in the stream."""
        for item in items:
            yield item * self._factor


class CollectSink(Sink[int]):
    """Test sink that collects items and returns string representations."""

    name: ClassVar[str] = "Collector"
    description: ClassVar[str] = "Collects items for testing"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameters for this sink."""
        return []

    def __init__(self) -> None:
        """Initialize the collector."""
        self.collected: list[list[int]] = []

    def __call__(self, items: Iterator[int], index: int) -> list[str]:
        """Consume items and return string paths."""
        values = list(items)
        self.collected.append(values)
        return [str(v) for v in values]


# ---------------------------------------------------------------------------
# StageMetrics tests
# ---------------------------------------------------------------------------


class TestStageMetrics:
    """Tests for :class:`StageMetrics`."""

    def test_creation(self) -> None:
        """StageMetrics can be created with name and wall_time_ns."""
        sm = StageMetrics(name="source", wall_time_ns=1_000_000)
        assert sm.name == "source"
        assert sm.wall_time_ns == 1_000_000

    def test_to_dict(self) -> None:
        """to_dict returns a plain dictionary with expected keys."""
        sm = StageMetrics(name="filter_a", wall_time_ns=500)
        d = sm.to_dict()
        assert d == {"name": "filter_a", "wall_time_ns": 500}


# ---------------------------------------------------------------------------
# IndexMetrics tests
# ---------------------------------------------------------------------------


class TestIndexMetrics:
    """Tests for :class:`IndexMetrics`."""

    def test_creation(self) -> None:
        """IndexMetrics can be created with all fields."""
        stages = [StageMetrics(name="source", wall_time_ns=100)]
        im = IndexMetrics(index=0, stages=stages, wall_time_ns=200, peak_memory_bytes=4096, gpu_memory_bytes=8192)
        assert im.index == 0
        assert len(im.stages) == 1
        assert im.wall_time_ns == 200
        assert im.peak_memory_bytes == 4096
        assert im.gpu_memory_bytes == 8192

    def test_to_dict_from_dict_roundtrip(self) -> None:
        """to_dict/from_dict produces identical objects."""
        stages = [
            StageMetrics(name="source", wall_time_ns=100),
            StageMetrics(name="Double", wall_time_ns=200),
            StageMetrics(name="sink", wall_time_ns=50),
        ]
        original = IndexMetrics(index=3, stages=stages, wall_time_ns=350, peak_memory_bytes=2048, gpu_memory_bytes=512)
        d = original.to_dict()
        restored = IndexMetrics.from_dict(d)

        assert restored.index == original.index
        assert restored.wall_time_ns == original.wall_time_ns
        assert restored.peak_memory_bytes == original.peak_memory_bytes
        assert restored.gpu_memory_bytes == original.gpu_memory_bytes
        assert len(restored.stages) == len(original.stages)
        for r_s, o_s in zip(restored.stages, original.stages, strict=True):
            assert r_s.name == o_s.name
            assert r_s.wall_time_ns == o_s.wall_time_ns


# ---------------------------------------------------------------------------
# PipelineMetrics tests
# ---------------------------------------------------------------------------


class TestPipelineMetrics:
    """Tests for :class:`PipelineMetrics`."""

    def test_computed_properties(self) -> None:
        """Verify total_wall_time_ns, mean_index_time_ns, total_peak_memory_bytes."""
        indices = [
            IndexMetrics(index=0, stages=[], wall_time_ns=1000, peak_memory_bytes=100, gpu_memory_bytes=None),
            IndexMetrics(index=1, stages=[], wall_time_ns=3000, peak_memory_bytes=300, gpu_memory_bytes=None),
        ]
        pm = PipelineMetrics(indices=indices)

        assert pm.total_wall_time_ns == 4000
        assert pm.mean_index_time_ns == 2000.0
        # total_peak_memory_bytes is max, not sum
        assert pm.total_peak_memory_bytes == 300

    def test_empty_metrics(self) -> None:
        """Empty PipelineMetrics returns safe defaults."""
        pm = PipelineMetrics()
        assert pm.total_wall_time_ns == 0
        assert pm.mean_index_time_ns == 0.0
        assert pm.total_peak_memory_bytes == 0


# ---------------------------------------------------------------------------
# _TimedGenerator tests
# ---------------------------------------------------------------------------


class TestTimedGenerator:
    """Tests for :class:`_TimedGenerator`."""

    def test_iterates_correctly(self) -> None:
        """_TimedGenerator yields all items from inner iterator."""
        inner = iter([1, 2, 3])
        tg = _TimedGenerator(inner)
        result = list(tg)
        assert result == [1, 2, 3]

    def test_accumulates_time(self) -> None:
        """elapsed_ns is positive after iterating through items."""

        def slow_gen():
            """Generate values with a small delay."""
            for i in range(3):
                time.sleep(0.001)  # 1ms
                yield i

        tg = _TimedGenerator(slow_gen())
        list(tg)  # consume
        # Should have accumulated at least some nanoseconds
        assert tg.elapsed_ns > 0

    def test_empty_generator(self) -> None:
        """_TimedGenerator handles empty iterators."""
        tg = _TimedGenerator(iter([]))
        result = list(tg)
        assert result == []
        # elapsed should be non-negative (even for empty)
        assert tg.elapsed_ns >= 0


# ---------------------------------------------------------------------------
# Config serialization tests
# ---------------------------------------------------------------------------


class TestConfigSerialization:
    """Tests for provenance helpers."""

    def test_component_config(self) -> None:
        """_component_config captures class, module, and params."""
        source = IntSource(values=[1, 2, 3])
        config = _component_config(source)

        assert config["class"] == "IntSource"
        assert "module" in config
        assert config["name"] == "Int Source"
        assert "params" in config

    def test_pipeline_config_structure(self) -> None:
        """_pipeline_config returns dict with source, filters, sink keys."""
        pipeline = IntSource(values=[1]).filter(DoubleFilter()).write(CollectSink())
        config = _pipeline_config(pipeline)

        assert "source" in config
        assert "filters" in config
        assert isinstance(config["filters"], list)
        assert len(config["filters"]) == 1
        assert "sink" in config

    def test_hash_stability(self) -> None:
        """Same pipeline config produces the same hash."""
        pipeline = IntSource(values=[1]).filter(DoubleFilter()).write(CollectSink())
        config1 = _pipeline_config(pipeline)
        config2 = _pipeline_config(pipeline)

        assert _config_hash(config1) == _config_hash(config2)

    def test_hash_changes_with_params(self) -> None:
        """Different parameters produce different hashes."""
        p1 = IntSource(values=[1]).filter(ScaleFilter(factor=2)).write(CollectSink())
        p2 = IntSource(values=[1]).filter(ScaleFilter(factor=10)).write(CollectSink())

        h1 = _config_hash(_pipeline_config(p1))
        h2 = _config_hash(_pipeline_config(p2))
        assert h1 != h2


# ---------------------------------------------------------------------------
# PipelineStore tests
# ---------------------------------------------------------------------------


class TestPipelineStore:
    """Tests for :class:`PipelineStore`."""

    @pytest.fixture()
    def store(self, tmp_path) -> PipelineStore:
        """Create a PipelineStore with a test pipeline config."""
        pipeline = IntSource(values=[1, 2, 3]).filter(DoubleFilter()).write(CollectSink())
        config = _pipeline_config(pipeline)
        chash = _config_hash(config)
        return PipelineStore(db_path=tmp_path / "test.db", pipeline_config=config, config_hash=chash)

    def test_creates_db_file(self, tmp_path) -> None:
        """PipelineStore creates the database file on init."""
        pipeline = IntSource(values=[1]).write(CollectSink())
        config = _pipeline_config(pipeline)
        chash = _config_hash(config)
        db_path = tmp_path / "subdir" / "store.db"
        PipelineStore(db_path=db_path, pipeline_config=config, config_hash=chash)
        assert db_path.exists()

    def test_schema_tables(self, store) -> None:
        """Database contains the three expected tables."""
        import sqlite3

        conn = sqlite3.connect(str(store._db_path))
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        conn.close()
        assert "pipeline_runs" in tables
        assert "index_results" in tables
        assert "stage_metrics" in tables

    def test_is_completed_initially_none(self, store) -> None:
        """is_completed returns None for an index that has not been processed."""
        assert store.is_completed(0) is None

    def test_record_success_and_is_completed(self, store) -> None:
        """After record_success, is_completed returns the output paths."""
        stages = [StageMetrics(name="source", wall_time_ns=100), StageMetrics(name="sink", wall_time_ns=200)]
        store.record_success(
            index=0,
            output_paths=["/out/0.vtk"],
            wall_time_ns=300,
            peak_memory_bytes=4096,
            gpu_memory_bytes=None,
            stages=stages,
        )
        result = store.is_completed(0)
        assert result == ["/out/0.vtk"]

    def test_record_error_not_completed(self, store) -> None:
        """An index with an error is not considered completed."""
        store.record_error(index=1, error="boom", wall_time_ns=100)
        assert store.is_completed(1) is None

    def test_completed_indices(self, store) -> None:
        """completed_indices returns all successfully completed indices."""
        store.record_success(0, ["/a"], 100, 50, None, [])
        store.record_success(2, ["/c"], 200, 60, None, [])
        store.record_error(1, "fail", 50)
        assert store.completed_indices() == {0, 2}

    def test_failed_indices(self, store) -> None:
        """failed_indices returns indices with their error messages."""
        store.record_error(1, "timeout", 100)
        store.record_error(3, "oom", 200)
        store.record_success(0, ["/a"], 50, 30, None, [])
        failed = store.failed_indices()
        assert failed == {1: "timeout", 3: "oom"}

    def test_remaining_indices(self, store) -> None:
        """remaining_indices returns indices not yet completed or failed."""
        store.record_success(0, ["/a"], 100, 50, None, [])
        store.record_error(2, "fail", 50)
        remaining = store.remaining_indices(total=5)
        assert remaining == [1, 3, 4]

    def test_summary(self, store) -> None:
        """summary returns expected counts and elapsed time."""
        store.record_success(0, ["/a"], 1_000_000, 50, None, [])
        store.record_success(1, ["/b"], 2_000_000, 60, None, [])
        store.record_error(2, "fail", 500)
        s = store.summary(total=5)
        assert s["total"] == 5
        assert s["completed"] == 2
        assert s["failed"] == 1
        assert s["remaining"] == 2
        assert s["total_elapsed_s"] == pytest.approx(3_000_000 / 1e9)

    def test_reset_clears_all(self, store) -> None:
        """reset clears all records and allows fresh recording."""
        store.record_success(0, ["/a"], 100, 50, None, [])
        store.record_error(1, "fail", 50)
        store.reset()
        assert store.completed_indices() == set()
        assert store.failed_indices() == {}

    def test_reset_index(self, store) -> None:
        """reset_index removes only the specified index."""
        store.record_success(0, ["/a"], 100, 50, None, [StageMetrics("s", 50)])
        store.record_success(1, ["/b"], 200, 60, None, [StageMetrics("s", 80)])
        store.reset_index(0)
        assert store.is_completed(0) is None
        assert store.is_completed(1) == ["/b"]

    def test_metrics_query(self, store) -> None:
        """metrics() returns PipelineMetrics built from the database."""
        stages_0 = [StageMetrics(name="source", wall_time_ns=100), StageMetrics(name="sink", wall_time_ns=200)]
        stages_1 = [StageMetrics(name="source", wall_time_ns=150), StageMetrics(name="sink", wall_time_ns=250)]
        store.record_success(0, ["/a"], 300, 4096, None, stages_0)
        store.record_success(1, ["/b"], 400, 8192, None, stages_1)

        pm = store.metrics()
        assert len(pm.indices) == 2
        assert pm.total_wall_time_ns == 700
        assert pm.total_peak_memory_bytes == 8192
        # Check stages were stored and retrieved
        assert len(pm.indices[0].stages) == 2
        assert pm.indices[0].stages[0].name == "source"

    def test_index_metrics(self, store) -> None:
        """index_metrics returns metrics for a single index."""
        stages = [StageMetrics(name="source", wall_time_ns=500)]
        store.record_success(5, ["/e"], 1000, 2048, 1024, stages)
        im = store.index_metrics(5)
        assert im is not None
        assert im.index == 5
        assert im.wall_time_ns == 1000
        assert im.peak_memory_bytes == 2048
        assert im.gpu_memory_bytes == 1024
        assert len(im.stages) == 1

    def test_index_metrics_not_found(self, store) -> None:
        """index_metrics returns None for a non-existent index."""
        assert store.index_metrics(99) is None

    def test_wal_mode_enabled(self, store) -> None:
        """The database connection uses WAL journal mode."""
        conn = store._connect()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode == "wal"

    def test_resume_same_config(self, tmp_path) -> None:
        """Creating a second PipelineStore with the same config resumes the same run."""
        pipeline = IntSource(values=[1]).write(CollectSink())
        config = _pipeline_config(pipeline)
        chash = _config_hash(config)
        db_path = tmp_path / "resume.db"

        store1 = PipelineStore(db_path=db_path, pipeline_config=config, config_hash=chash)
        store1.record_success(0, ["/a"], 100, 50, None, [])
        run_id_1 = store1._run_id

        # Second store should resume the same run
        store2 = PipelineStore(db_path=db_path, pipeline_config=config, config_hash=chash)
        assert store2._run_id == run_id_1
        assert store2.is_completed(0) == ["/a"]
