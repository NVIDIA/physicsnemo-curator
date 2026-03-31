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

"""Tests for the core pipeline framework (Source, Filter, Sink, Pipeline)."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

import pytest

from curator.core.base import REQUIRED, Filter, Param, Pipeline, Sink, Source

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Concrete test implementations
# ---------------------------------------------------------------------------


class IntSource(Source[int]):
    """Test source that yields integers from a list."""

    name: ClassVar[str] = "Int Source"
    description: ClassVar[str] = "Yields integers for testing"

    @classmethod
    def params(cls) -> list[Param]:
        return [Param(name="values", description="Comma-separated ints", type=str)]

    def __init__(self, values: list[int]) -> None:
        self._values = values

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> Generator[int]:
        yield self._values[index]


class DoubleFilter(Filter[int]):
    """Test filter that doubles each value."""

    name: ClassVar[str] = "Double"
    description: ClassVar[str] = "Doubles each integer"

    @classmethod
    def params(cls) -> list[Param]:
        return []

    def __call__(self, items: Generator[int]) -> Generator[int]:
        for item in items:
            yield item * 2


class ExpandFilter(Filter[int]):
    """Test filter that expands each value into two items."""

    name: ClassVar[str] = "Expand"
    description: ClassVar[str] = "Yields original and original+100"

    @classmethod
    def params(cls) -> list[Param]:
        return []

    def __call__(self, items: Generator[int]) -> Generator[int]:
        for item in items:
            yield item
            yield item + 100


class DropFilter(Filter[int]):
    """Test filter that drops items below a threshold."""

    name: ClassVar[str] = "Drop"
    description: ClassVar[str] = "Drops items below threshold"

    @classmethod
    def params(cls) -> list[Param]:
        return [Param(name="threshold", description="Min value", type=int, default=50)]

    def __init__(self, threshold: int = 50) -> None:
        self._threshold = threshold

    def __call__(self, items: Generator[int]) -> Generator[int]:
        for item in items:
            if item >= self._threshold:
                yield item


class CollectSink(Sink[int]):
    """Test sink that collects items and returns string representations."""

    name: ClassVar[str] = "Collector"
    description: ClassVar[str] = "Collects items for testing"

    @classmethod
    def params(cls) -> list[Param]:
        return []

    def __init__(self) -> None:
        self.collected: list[list[int]] = []

    def __call__(self, items: Iterator[int], index: int) -> list[str]:
        values = list(items)
        self.collected.append(values)
        return [str(v) for v in values]


# ---------------------------------------------------------------------------
# Param tests
# ---------------------------------------------------------------------------


class TestParam:
    def test_required_param(self):
        p = Param(name="x", description="test")
        assert p.required is True
        assert p.default is REQUIRED

    def test_optional_param(self):
        p = Param(name="x", description="test", default="hello")
        assert p.required is False
        assert p.default == "hello"

    def test_choices(self):
        p = Param(name="x", description="test", choices=["a", "b", "c"])
        assert p.choices == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Source tests
# ---------------------------------------------------------------------------


class TestSource:
    def test_len(self):
        source = IntSource(values=[10, 20, 30])
        assert len(source) == 3

    def test_getitem_yields(self):
        source = IntSource(values=[10, 20, 30])
        result = list(source[1])
        assert result == [20]

    def test_filter_returns_pipeline(self):
        source = IntSource(values=[1])
        pipeline = source.filter(DoubleFilter())
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.filters) == 1

    def test_write_returns_pipeline(self):
        source = IntSource(values=[1])
        pipeline = source.write(CollectSink())
        assert isinstance(pipeline, Pipeline)
        assert pipeline.sink is not None
        assert len(pipeline.filters) == 0


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_basic_pipeline(self):
        source = IntSource(values=[5, 10, 15])
        sink = CollectSink()
        pipeline = source.filter(DoubleFilter()).write(sink)

        result = pipeline[0]
        assert result == ["10"]

        result = pipeline[2]
        assert result == ["30"]

    def test_chained_filters(self):
        source = IntSource(values=[3])
        sink = CollectSink()
        pipeline = source.filter(DoubleFilter()).filter(DoubleFilter()).write(sink)

        result = pipeline[0]
        assert result == ["12"]  # 3 * 2 * 2

    def test_expand_filter(self):
        source = IntSource(values=[5])
        sink = CollectSink()
        pipeline = source.filter(ExpandFilter()).write(sink)

        result = pipeline[0]
        assert result == ["5", "105"]

    def test_contract_filter(self):
        source = IntSource(values=[5])
        sink = CollectSink()
        # Expand → Drop: 5 and 105 → only 105 passes threshold of 50
        pipeline = source.filter(ExpandFilter()).filter(DropFilter(threshold=50)).write(sink)

        result = pipeline[0]
        assert result == ["105"]

    def test_len_delegates_to_source(self):
        source = IntSource(values=[1, 2, 3, 4])
        pipeline = source.filter(DoubleFilter()).write(CollectSink())
        assert len(pipeline) == 4

    def test_no_sink_raises(self):
        source = IntSource(values=[1])
        pipeline = source.filter(DoubleFilter())
        with pytest.raises(RuntimeError, match="no sink"):
            pipeline[0]

    def test_index_out_of_range(self):
        source = IntSource(values=[1, 2])
        pipeline = source.filter(DoubleFilter()).write(CollectSink())
        with pytest.raises(IndexError):
            pipeline[5]

    def test_negative_index(self):
        source = IntSource(values=[10, 20, 30])
        sink = CollectSink()
        pipeline = source.filter(DoubleFilter()).write(sink)

        result = pipeline[-1]
        assert result == ["60"]  # 30 * 2

    def test_pipeline_immutability(self):
        source = IntSource(values=[1])
        p1 = source.filter(DoubleFilter())
        p2 = p1.filter(DoubleFilter())

        assert len(p1.filters) == 1
        assert len(p2.filters) == 2
        assert p1 is not p2

    def test_sink_receives_index(self):
        source = IntSource(values=[10, 20])
        sink = CollectSink()
        pipeline = source.write(sink)

        pipeline[0]
        pipeline[1]

        assert sink.collected == [[10], [20]]


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_register_and_query(self):
        from curator.core.registry import Registry

        reg = Registry()
        reg.register_submodule("test", "Test submodule", "builtins")
        reg.register_source("test", IntSource)
        reg.register_filter("test", DoubleFilter)
        reg.register_sink("test", CollectSink)

        assert "test" in reg.submodules()
        assert "Int Source" in reg.sources("test")
        assert "Double" in reg.filters("test")
        assert "Collector" in reg.sinks("test")

    def test_submodule_available(self):
        from curator.core.registry import Registry

        reg = Registry()
        reg.register_submodule("test", "Test", "builtins")
        assert reg.submodules()["test"].available is True

    def test_submodule_not_available(self):
        from curator.core.registry import Registry

        reg = Registry()
        reg.register_submodule("test", "Test", "nonexistent.module.xyz")
        assert reg.submodules()["test"].available is False

    def test_unregistered_submodule_raises(self):
        from curator.core.registry import Registry

        reg = Registry()
        with pytest.raises(KeyError, match="not registered"):
            reg.register_source("nope", IntSource)

    def test_repr(self):
        from curator.core.registry import Registry

        reg = Registry()
        reg.register_submodule("test", "Test", "builtins")
        reg.register_source("test", IntSource)
        text = repr(reg)
        assert "test" in text
        assert "1 sources" in text
        assert "0 stores" in text

    def test_register_store(self):
        from curator.core.registry import Registry

        reg = Registry()
        reg.register_submodule("test", "Test", "builtins")

        class DummyStore:
            def __len__(self):
                return 0

            def __getitem__(self, index):
                raise IndexError

        reg.register_store("test", "Dummy store", DummyStore)
        stores = reg.stores("test")
        assert "Dummy store" in stores
        assert stores["Dummy store"] is DummyStore

    def test_register_multiple_stores(self):
        from curator.core.registry import Registry

        reg = Registry()
        reg.register_submodule("test", "Test", "builtins")

        class StoreA:
            def __len__(self):
                return 0

            def __getitem__(self, index):
                raise IndexError

        class StoreB:
            def __len__(self):
                return 0

            def __getitem__(self, index):
                raise IndexError

        reg.register_store("test", "Store A", StoreA)
        reg.register_store("test", "Store B", StoreB)
        stores = reg.stores("test")
        assert len(stores) == 2
        assert "Store A" in stores
        assert "Store B" in stores

    def test_stores_empty_by_default(self):
        from curator.core.registry import Registry

        reg = Registry()
        reg.register_submodule("test", "Test", "builtins")
        assert reg.stores("test") == {}

    def test_register_store_unregistered_submodule_raises(self):
        from curator.core.registry import Registry

        class DummyStore:
            def __len__(self):
                return 0

            def __getitem__(self, index):
                raise IndexError

        reg = Registry()
        with pytest.raises(KeyError, match="not registered"):
            reg.register_store("nope", "Dummy", DummyStore)
