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

"""Abstract base classes for pipeline components and the Pipeline builder."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

# Sentinel for required parameters (no default).
_REQUIRED = object()
REQUIRED: Any = _REQUIRED
"""Sentinel value indicating a :class:`Param` has no default and must be provided."""


@dataclass(frozen=True)
class Param:
    """Descriptor for a configurable parameter on a pipeline component.

    Parameters
    ----------
    name : str
        Parameter name (should match the ``__init__`` keyword argument).
    description : str
        Human-readable help text shown in the interactive CLI.
    type : type
        Expected Python type (``str``, ``int``, ``float``, ``pathlib.Path``, …).
    default : Any
        Default value.  Use :data:`REQUIRED` (the default) to indicate the
        parameter must be supplied by the user.
    choices : list[str] | None
        If not *None*, the CLI will present a selection prompt instead of
        free-text input.
    """

    name: str
    description: str
    type: type = str  # ty: ignore[invalid-type-form]
    default: Any = REQUIRED
    choices: list[str] | None = None

    @property
    def required(self) -> bool:
        """Return ``True`` if this parameter has no default value."""
        return self.default is _REQUIRED


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


class Source[T](ABC):
    """Abstract data source that yields items of type *T*.

    A source represents a collection of data items (e.g. files on disk).
    Each item is accessed by integer index and may yield one or more *T*
    objects (generator semantics allow a single source item to expand into
    multiple outputs).

    Subclasses must set the class-level :attr:`name` and :attr:`description`
    attributes and implement :meth:`params`, :meth:`__len__`, and
    :meth:`__getitem__`.

    Examples
    --------
    >>> pipeline = MySource(path="/data").filter(MyFilter()).write(MySink())
    >>> pipeline[0]  # process first source item lazily
    """

    name: ClassVar[str]
    """Human-readable display name for the interactive CLI."""
    description: ClassVar[str]
    """Short description shown in the interactive CLI."""

    @classmethod
    @abstractmethod
    def params(cls) -> list[Param]:
        """Declare the configurable parameters for this source.

        Returns
        -------
        list[Param]
            Ordered list of parameter descriptors.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of items available in this source."""
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> Generator[T]:
        """Yield one or more *T* items for the given *index*.

        Parameters
        ----------
        index : int
            Zero-based index into the source's item collection.

        Yields
        ------
        T
            Data item(s) produced from the source at *index*.
        """
        ...

    # -- Convenience builder methods -----------------------------------------

    def filter(self, f: Filter[T]) -> Pipeline[T]:
        """Create a :class:`Pipeline` with this source and a single filter.

        Parameters
        ----------
        f : Filter[T]
            The filter to append.

        Returns
        -------
        Pipeline[T]
            A new pipeline containing this source and the given filter.
        """
        return Pipeline(source=self, filters=[f])

    def write(self, s: Sink[T]) -> Pipeline[T]:
        """Create a :class:`Pipeline` with this source and a sink (no filters).

        Parameters
        ----------
        s : Sink[T]
            The sink to attach.

        Returns
        -------
        Pipeline[T]
            A new pipeline containing this source and the given sink.
        """
        return Pipeline(source=self, sink=s)


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


class Filter[T](ABC):
    """Abstract filter/transform that processes a stream of *T* items.

    Filters receive a generator of items and yield zero or more items per
    input (full generator semantics — can expand, contract, or pass through).

    Subclasses must set :attr:`name` and :attr:`description` and implement
    :meth:`params` and :meth:`__call__`.
    """

    name: ClassVar[str]
    """Human-readable display name for the interactive CLI."""
    description: ClassVar[str]
    """Short description shown in the interactive CLI."""

    @classmethod
    @abstractmethod
    def params(cls) -> list[Param]:
        """Declare the configurable parameters for this filter.

        Returns
        -------
        list[Param]
            Ordered list of parameter descriptors.
        """
        ...

    @abstractmethod
    def __call__(self, items: Generator[T]) -> Generator[T]:
        """Process a stream of items, yielding transformed results.

        Parameters
        ----------
        items : Generator[T]
            Incoming stream of data items.

        Yields
        ------
        T
            Transformed data item(s).
        """
        ...


# ---------------------------------------------------------------------------
# Sink
# ---------------------------------------------------------------------------


class Sink[T](ABC):
    """Abstract sink that persists items and returns output file paths.

    The sink consumes a generator of items and writes each one to storage,
    returning the file paths of the written outputs.

    Subclasses must set :attr:`name` and :attr:`description` and implement
    :meth:`params` and :meth:`__call__`.
    """

    name: ClassVar[str]
    """Human-readable display name for the interactive CLI."""
    description: ClassVar[str]
    """Short description shown in the interactive CLI."""

    @classmethod
    @abstractmethod
    def params(cls) -> list[Param]:
        """Declare the configurable parameters for this sink.

        Returns
        -------
        list[Param]
            Ordered list of parameter descriptors.
        """
        ...

    @abstractmethod
    def __call__(self, items: Iterator[T], index: int) -> list[str]:
        """Consume items and persist them to storage.

        Parameters
        ----------
        items : Iterator[T]
            Stream of data items to write.
        index : int
            Source index being processed (useful for naming output files).

        Returns
        -------
        list[str]
            Paths of the files written.
        """
        ...


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class Pipeline[T]:
    """Lazy pipeline that chains a source through filters into a sink.

    The pipeline is built incrementally using the :meth:`filter` and
    :meth:`write` builder methods.  Execution is deferred until the
    pipeline is indexed with ``pipeline[i]``, which processes only
    the *i*-th source item.

    Parameters
    ----------
    source : Source[T]
        The data source.
    filters : list[Filter[T]]
        Ordered list of filters to apply.
    sink : Sink[T] | None
        Optional sink for writing output.

    Examples
    --------
    >>> pipeline = (
    ...     MySource(path="/data")
    ...     .filter(FilterA())
    ...     .filter(FilterB())
    ...     .write(MySink(output="/out"))
    ... )
    >>> pipeline[0]   # lazily process source item 0
    ['/out/item_0']
    """

    source: Source[T]
    filters: list[Filter[T]] = field(default_factory=list)
    sink: Sink[T] | None = None

    def filter(self, f: Filter[T]) -> Pipeline[T]:
        """Return a new pipeline with an additional filter appended.

        Parameters
        ----------
        f : Filter[T]
            The filter to append.

        Returns
        -------
        Pipeline[T]
            A new pipeline instance (the original is unchanged).
        """
        return Pipeline(
            source=self.source,
            filters=[*self.filters, f],
            sink=self.sink,
        )

    def write(self, s: Sink[T]) -> Pipeline[T]:
        """Return a new pipeline with the given sink attached.

        Parameters
        ----------
        s : Sink[T]
            The sink to attach.

        Returns
        -------
        Pipeline[T]
            A new pipeline instance (the original is unchanged).
        """
        return Pipeline(
            source=self.source,
            filters=list(self.filters),
            sink=s,
        )

    def __len__(self) -> int:
        """Return the number of items in the source."""
        return len(self.source)

    def __getitem__(self, index: int) -> list[str]:
        """Lazily process the *index*-th source item through the full chain.

        Parameters
        ----------
        index : int
            Zero-based index into the source.

        Returns
        -------
        list[str]
            File paths produced by the sink.

        Raises
        ------
        RuntimeError
            If no sink has been attached to the pipeline.
        IndexError
            If *index* is out of range.
        """
        if self.sink is None:
            msg = "Pipeline has no sink. Call .write(sink) before indexing."
            raise RuntimeError(msg)

        n = len(self.source)
        if index < 0:
            index += n
        if index < 0 or index >= n:
            msg = f"Index {index} out of range for source with {n} items."
            raise IndexError(msg)

        # Start with the source generator for this index.
        stream: Generator[T] = self.source[index]

        # Chain through each filter.
        for f in self.filters:
            stream = f(stream)

        # Feed into the sink.
        return self.sink(stream, index)
