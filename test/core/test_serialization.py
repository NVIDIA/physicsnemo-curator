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

"""Tests for :mod:`physicsnemo_curator.core.serialization`."""

from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING, ClassVar

import pytest

from physicsnemo_curator.core.base import Filter, Param, Pipeline, Sink, Source
from physicsnemo_curator.core.serialization import (
    _collect_metadata,
    deserialize_pipeline,
    load_pipeline,
    save_pipeline,
    serialize_pipeline,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Test pipeline components (module-level for importability during deser)
# ---------------------------------------------------------------------------


class _IntSource(Source[int]):
    """Source that yields integers."""

    name: ClassVar[str] = "IntSource"
    description: ClassVar[str] = "Yields integers"

    @classmethod
    def params(cls) -> list[Param]:
        """Return params."""
        return [Param(name="count", description="Number of items", type=int)]

    def __init__(self, count: int = 5) -> None:
        """Initialize with count."""
        self._count = count

    def __len__(self) -> int:
        """Return item count."""
        return self._count

    def __getitem__(self, index: int) -> Generator[int]:
        """Yield the index value."""
        yield index


class _AddFilter(Filter[int]):
    """Filter that adds a constant."""

    name: ClassVar[str] = "Add"
    description: ClassVar[str] = "Adds a constant"

    @classmethod
    def params(cls) -> list[Param]:
        """Return params."""
        return [Param(name="amount", description="Amount to add", type=int, default=1)]

    def __init__(self, amount: int = 1) -> None:
        """Initialize with amount."""
        self._amount = amount

    def __call__(self, items: Generator[int]) -> Generator[int]:
        """Add amount to each item."""
        for item in items:
            yield item + self._amount


class _FileSink(Sink[int]):
    """Sink that writes integers to text files."""

    name: ClassVar[str] = "FileSink"
    description: ClassVar[str] = "Writes to files"

    @classmethod
    def params(cls) -> list[Param]:
        """Return params."""
        return [Param(name="output_dir", description="Output directory", type=str)]

    def __init__(self, output_dir: str) -> None:
        """Initialize with output directory."""
        self._output_dir = pathlib.Path(output_dir)

    def __call__(self, items: Iterator[int], index: int) -> list[str]:
        """Write items to files."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for seq, val in enumerate(items):
            p = self._output_dir / f"item_{index:04d}_{seq}.txt"
            p.write_text(str(val))
            paths.append(str(p))
        return paths


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_pipeline(tmp_path: pathlib.Path) -> Pipeline[int]:
    """Create a test pipeline with source, filter, and sink."""
    return Pipeline(
        source=_IntSource(count=3),
        filters=[_AddFilter(amount=10)],
        sink=_FileSink(output_dir=str(tmp_path / "output")),
    )


# ---------------------------------------------------------------------------
# TestSerializePipeline
# ---------------------------------------------------------------------------


class TestSerializePipeline:
    """Tests for serialize_pipeline."""

    def test_returns_dict_with_expected_keys(self, tmp_path: pathlib.Path) -> None:
        """Serialized dict has version, source, filters, and sink keys."""
        data = serialize_pipeline(_make_pipeline(tmp_path))
        assert "version" in data
        assert "source" in data
        assert "filters" in data
        assert "sink" in data

    def test_source_has_class_and_module(self, tmp_path: pathlib.Path) -> None:
        """Source config includes class and module identifiers."""
        data = serialize_pipeline(_make_pipeline(tmp_path))
        assert data["source"]["class"] == "_IntSource"
        assert "module" in data["source"]

    def test_source_params_captured(self, tmp_path: pathlib.Path) -> None:
        """Source params reflect the configured count value."""
        data = serialize_pipeline(_make_pipeline(tmp_path))
        assert data["source"]["params"]["count"] == 3

    def test_filter_params_captured(self, tmp_path: pathlib.Path) -> None:
        """Filter params reflect the configured amount value."""
        data = serialize_pipeline(_make_pipeline(tmp_path))
        assert data["filters"][0]["params"]["amount"] == 10

    def test_sink_params_captured(self, tmp_path: pathlib.Path) -> None:
        """Sink params include the output_dir key."""
        data = serialize_pipeline(_make_pipeline(tmp_path))
        assert "output_dir" in data["sink"]["params"]

    def test_pipeline_without_sink(self) -> None:
        """Pipeline with no sink serializes sink as None."""
        pipeline: Pipeline[int] = Pipeline(source=_IntSource(count=2), filters=[])
        data = serialize_pipeline(pipeline)
        assert data["sink"] is None

    def test_pipeline_without_filters(self, tmp_path: pathlib.Path) -> None:
        """Pipeline with no filters serializes filters as empty list."""
        pipeline: Pipeline[int] = Pipeline(
            source=_IntSource(count=2),
            filters=[],
            sink=_FileSink(output_dir=str(tmp_path / "out")),
        )
        data = serialize_pipeline(pipeline)
        assert data["filters"] == []


# ---------------------------------------------------------------------------
# TestSavePipeline
# ---------------------------------------------------------------------------


class TestSavePipeline:
    """Tests for save_pipeline."""

    def test_save_json(self, tmp_path: pathlib.Path) -> None:
        """Saving as JSON produces a valid JSON file with expected keys."""
        path = tmp_path / "pipeline.json"
        save_pipeline(_make_pipeline(tmp_path), path)
        data = json.loads(path.read_text())
        assert data["version"] == 1
        assert data["source"]["class"] == "_IntSource"

    def test_save_yaml(self, tmp_path: pathlib.Path) -> None:
        """Saving as YAML produces a valid YAML file."""
        import yaml

        path = tmp_path / "pipeline.yaml"
        save_pipeline(_make_pipeline(tmp_path), path)
        data = yaml.safe_load(path.read_text())
        assert data["version"] == 1

    def test_save_yml_extension(self, tmp_path: pathlib.Path) -> None:
        """The .yml extension is accepted as YAML."""
        import yaml

        path = tmp_path / "pipeline.yml"
        save_pipeline(_make_pipeline(tmp_path), path)
        data = yaml.safe_load(path.read_text())
        assert data["version"] == 1

    def test_save_creates_parent_dirs(self, tmp_path: pathlib.Path) -> None:
        """Saving to a nested path creates intermediate directories."""
        path = tmp_path / "nested" / "deep" / "pipeline.yaml"
        save_pipeline(_make_pipeline(tmp_path), path)
        assert path.exists()

    def test_save_unknown_extension_raises(self, tmp_path: pathlib.Path) -> None:
        """Saving with an unsupported extension raises ValueError."""
        path = tmp_path / "pipeline.toml"
        with pytest.raises(ValueError, match="Unsupported"):
            save_pipeline(_make_pipeline(tmp_path), path)


# ---------------------------------------------------------------------------
# TestDeserializePipeline
# ---------------------------------------------------------------------------


class TestDeserializePipeline:
    """Tests for deserialize_pipeline."""

    def test_round_trip_produces_functional_pipeline(self, tmp_path: pathlib.Path) -> None:
        """Serialize then deserialize produces a pipeline that can execute."""
        pipeline = _make_pipeline(tmp_path)
        data = serialize_pipeline(pipeline)
        restored = deserialize_pipeline(data)
        result = restored[0]
        assert len(result) > 0

    def test_round_trip_preserves_source_params(self) -> None:
        """Deserialized source preserves the count parameter."""
        pipeline: Pipeline[int] = Pipeline(source=_IntSource(count=7), filters=[])
        data = serialize_pipeline(pipeline)
        restored = deserialize_pipeline(data)
        assert len(restored.source) == 7

    def test_round_trip_preserves_filter_params(self) -> None:
        """Deserialized filter preserves the amount parameter."""
        pipeline: Pipeline[int] = Pipeline(
            source=_IntSource(count=1),
            filters=[_AddFilter(amount=42)],
        )
        data = serialize_pipeline(pipeline)
        restored = deserialize_pipeline(data)
        assert restored.filters[0]._amount == 42

    def test_round_trip_without_sink(self) -> None:
        """Deserialized pipeline with no sink has sink=None."""
        pipeline: Pipeline[int] = Pipeline(source=_IntSource(count=2), filters=[])
        data = serialize_pipeline(pipeline)
        restored = deserialize_pipeline(data)
        assert restored.sink is None

    def test_round_trip_without_filters(self, tmp_path: pathlib.Path) -> None:
        """Deserialized pipeline with no filters has empty filter list."""
        pipeline: Pipeline[int] = Pipeline(
            source=_IntSource(count=1),
            filters=[],
            sink=_FileSink(output_dir=str(tmp_path / "out")),
        )
        data = serialize_pipeline(pipeline)
        restored = deserialize_pipeline(data)
        assert restored.filters == []

    def test_multiple_filters_preserved_in_order(self) -> None:
        """Multiple filters are deserialized in the correct order."""
        pipeline: Pipeline[int] = Pipeline(
            source=_IntSource(count=1),
            filters=[_AddFilter(amount=1), _AddFilter(amount=2), _AddFilter(amount=3)],
        )
        data = serialize_pipeline(pipeline)
        restored = deserialize_pipeline(data)
        assert [f._amount for f in restored.filters] == [1, 2, 3]

    def test_unknown_class_raises(self) -> None:
        """Deserializing a config with a fake module raises an import error."""
        data = {
            "version": 1,
            "source": {
                "class": "NoSuchClass",
                "module": "no.such.module",
                "params": {},
            },
            "filters": [],
            "sink": None,
        }
        with pytest.raises((ImportError, ModuleNotFoundError)):
            deserialize_pipeline(data)


# ---------------------------------------------------------------------------
# TestLoadPipeline
# ---------------------------------------------------------------------------


class TestLoadPipeline:
    """Tests for load_pipeline."""

    def test_load_json(self, tmp_path: pathlib.Path) -> None:
        """Loading a saved JSON file returns a Pipeline."""
        path = tmp_path / "pipeline.json"
        save_pipeline(_make_pipeline(tmp_path), path)
        restored = load_pipeline(path)
        assert isinstance(restored, Pipeline)
        assert len(restored.source) == 3

    def test_load_yaml(self, tmp_path: pathlib.Path) -> None:
        """Loading a saved YAML file returns a Pipeline."""
        path = tmp_path / "pipeline.yaml"
        save_pipeline(_make_pipeline(tmp_path), path)
        restored = load_pipeline(path)
        assert isinstance(restored, Pipeline)
        assert len(restored.source) == 3

    def test_full_round_trip_json(self, tmp_path: pathlib.Path) -> None:
        """Save JSON, load, execute — produced file contains expected value."""
        out_dir = tmp_path / "output"
        pipeline = Pipeline(
            source=_IntSource(count=1),
            filters=[_AddFilter(amount=10)],  # ty: ignore[invalid-argument-type]
            sink=_FileSink(output_dir=str(out_dir)),
        )
        path = tmp_path / "pipeline.json"
        save_pipeline(pipeline, path)
        restored = load_pipeline(path)
        result = restored[0]
        content = pathlib.Path(result[0]).read_text()
        assert content == "10"

    def test_full_round_trip_yaml(self, tmp_path: pathlib.Path) -> None:
        """Save YAML, load, execute — produced file contains expected value."""
        out_dir = tmp_path / "output"
        pipeline = Pipeline(
            source=_IntSource(count=1),
            filters=[_AddFilter(amount=10)],  # ty: ignore[invalid-argument-type]
            sink=_FileSink(output_dir=str(out_dir)),
        )
        path = tmp_path / "pipeline.yaml"
        save_pipeline(pipeline, path)
        restored = load_pipeline(path)
        result = restored[0]
        content = pathlib.Path(result[0]).read_text()
        assert content == "10"

    def test_load_nonexistent_raises(self, tmp_path: pathlib.Path) -> None:
        """Loading a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_pipeline(tmp_path / "does_not_exist.json")

    def test_load_unknown_extension_raises(self, tmp_path: pathlib.Path) -> None:
        """Loading a file with unsupported extension raises ValueError."""
        path = tmp_path / "pipeline.toml"
        path.write_text("")
        with pytest.raises(ValueError, match="Unsupported"):
            load_pipeline(path)


# ---------------------------------------------------------------------------
# Pipeline convenience methods
# ---------------------------------------------------------------------------


class TestPipelineConvenienceMethods:
    """Tests for Pipeline.save() and Pipeline.load()."""

    def test_pipeline_save_method(self, tmp_path: pathlib.Path) -> None:
        """Pipeline.save() delegates to save_pipeline."""
        pipeline = _make_pipeline(tmp_path)
        path = tmp_path / "pipeline.yaml"
        pipeline.save(path)

        assert path.exists()

    def test_pipeline_load_classmethod(self, tmp_path: pathlib.Path) -> None:
        """Pipeline.load() restores a functional pipeline."""
        pipeline = _make_pipeline(tmp_path)
        path = tmp_path / "pipeline.yaml"
        pipeline.save(path)

        restored = Pipeline.load(path)
        assert len(restored) == 3
        assert restored.sink is not None

    def test_save_load_round_trip(self, tmp_path: pathlib.Path) -> None:
        """Full Pipeline.save() -> Pipeline.load() -> execute round-trip."""
        pipeline = _make_pipeline(tmp_path)
        path = tmp_path / "pipeline.json"
        pipeline.save(path)

        restored = Pipeline.load(path)
        result = restored[0]
        assert len(result) == 1
        content = pathlib.Path(result[0]).read_text()
        assert content == "10"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    """Tests that serialization preserves metadata and produces executable pipelines."""

    def test_serialize_includes_name_for_registry_lookup(self, tmp_path: pathlib.Path) -> None:
        """Serialized config includes 'name' field for registry lookup."""
        pipeline = _make_pipeline(tmp_path)
        data = serialize_pipeline(pipeline)

        assert data["source"]["name"] == "IntSource"
        assert data["filters"][0]["name"] == "Add"
        assert data["sink"]["name"] == "FileSink"

    def test_version_field_is_present(self, tmp_path: pathlib.Path) -> None:
        """Version field is present for forward compatibility."""
        pipeline = _make_pipeline(tmp_path)
        data = serialize_pipeline(pipeline)

        assert data["version"] == 1

    def test_deserialized_pipeline_is_executable(self, tmp_path: pathlib.Path) -> None:
        """Deserialized pipeline can process all indices."""
        pipeline = _make_pipeline(tmp_path)
        data = serialize_pipeline(pipeline)
        restored = deserialize_pipeline(data)

        # Execute all 3 indices
        for i in range(len(restored)):
            result = restored[i]
            assert len(result) == 1
            content = pathlib.Path(result[0]).read_text()
            # Source yields i, filter adds 10 -> i + 10
            assert content == str(i + 10)


# ---------------------------------------------------------------------------
# Metadata tests
# ---------------------------------------------------------------------------


class TestMetadata:
    """Tests for pipeline serialization metadata."""

    def test_serialize_includes_metadata_key(self, tmp_path: pathlib.Path) -> None:
        """Serialized output contains a metadata section."""
        data = serialize_pipeline(_make_pipeline(tmp_path))
        assert "metadata" in data

    def test_metadata_has_required_fields(self, tmp_path: pathlib.Path) -> None:
        """Metadata includes all expected provenance fields."""
        data = serialize_pipeline(_make_pipeline(tmp_path))
        meta = data["metadata"]
        assert "psnc_version" in meta
        assert "rust_extension" in meta
        assert "python_version" in meta
        assert "platform" in meta
        assert "created_utc" in meta
        assert "git_hash" in meta
        assert "git_dirty" in meta

    def test_metadata_psnc_version_matches(self, tmp_path: pathlib.Path) -> None:
        """psnc_version matches the package __version__."""
        from physicsnemo_curator import __version__

        data = serialize_pipeline(_make_pipeline(tmp_path))
        assert data["metadata"]["psnc_version"] == __version__

    def test_metadata_created_utc_is_iso_format(self, tmp_path: pathlib.Path) -> None:
        """created_utc is a valid ISO 8601 UTC timestamp."""
        from datetime import datetime, timezone

        data = serialize_pipeline(_make_pipeline(tmp_path))
        ts = data["metadata"]["created_utc"]
        # Should parse without error
        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
        assert dt.year >= 2025

    def test_metadata_python_version_is_string(self, tmp_path: pathlib.Path) -> None:
        """python_version is a non-empty string."""
        data = serialize_pipeline(_make_pipeline(tmp_path))
        pv = data["metadata"]["python_version"]
        assert isinstance(pv, str)
        assert len(pv) > 0

    def test_collect_metadata_returns_dict(self) -> None:
        """_collect_metadata returns a dict with expected keys."""
        meta = _collect_metadata()
        assert isinstance(meta, dict)
        assert set(meta.keys()) == {
            "psnc_version",
            "rust_extension",
            "python_version",
            "platform",
            "created_utc",
            "git_hash",
            "git_dirty",
        }

    def test_metadata_ignored_on_deserialize(self, tmp_path: pathlib.Path) -> None:
        """Metadata is ignored during deserialization — round-trip works."""
        pipeline = _make_pipeline(tmp_path)
        data = serialize_pipeline(pipeline)
        # Metadata is present but should not affect deserialization
        assert "metadata" in data
        restored = deserialize_pipeline(data)
        assert len(restored) == 3
        result = restored[0]
        content = pathlib.Path(result[0]).read_text()
        assert content == "10"

    def test_metadata_not_in_config_hash(self, tmp_path: pathlib.Path) -> None:
        """Metadata does not affect the pipeline config hash."""
        from physicsnemo_curator.core.pipeline_store import _config_hash, _pipeline_config

        pipeline = _make_pipeline(tmp_path)
        config = _pipeline_config(pipeline)
        # _pipeline_config does not include metadata
        assert "metadata" not in config
        # Hash is stable (not affected by time-varying metadata)
        h1 = _config_hash(config)
        h2 = _config_hash(config)
        assert h1 == h2

    def test_save_load_preserves_metadata_in_file(self, tmp_path: pathlib.Path) -> None:
        """Metadata is written to disk and readable from the saved file."""
        path = tmp_path / "pipeline.json"
        save_pipeline(_make_pipeline(tmp_path), path)
        data = json.loads(path.read_text())
        assert "metadata" in data
        assert "psnc_version" in data["metadata"]
        assert "created_utc" in data["metadata"]

    def test_git_fields_nullable(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Git fields are None when git is unavailable."""
        import subprocess

        original_run = subprocess.run

        def _failing_git(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            cmd = args[0] if args else kwargs.get("args", [])
            if isinstance(cmd, list) and cmd[0] == "git":
                raise FileNotFoundError("git not found")
            return original_run(*args, **kwargs)

        monkeypatch.setattr(subprocess, "run", _failing_git)
        meta = _collect_metadata()
        assert meta["git_hash"] is None
        assert meta["git_dirty"] is None
