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

"""AtomicData Zarr writer sink for atomic/molecular pipelines.

Persists :class:`~nvalchemi.data.AtomicData` objects to a structured Zarr
store using :class:`~nvalchemi.data.datapipes.backends.zarr.AtomicDataZarrWriter`.

Items are collected into batches of configurable size before being flushed
to the store for efficient I/O.  The first batch creates the store via
:meth:`write`, and subsequent batches extend it via :meth:`append`.

When a *naming_template* is provided and the pipeline's source exposes a
``relative_path(index)`` method, the sink can mirror the input directory
structure — each source index writes to a separate Zarr store whose path
is derived from the source file layout.

Examples
--------
>>> sink = AtomicDataZarrSink(output_path="./output.zarr")  # doctest: +SKIP
>>> paths = sink(atomic_data_iterator, index=0)  # doctest: +SKIP
"""

from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING, Any, ClassVar

from physicsnemo_curator.core.base import Param, Sink, Source

if TYPE_CHECKING:
    from collections.abc import Iterator

    from nvalchemi.data import AtomicData

logger = logging.getLogger(__name__)


class AtomicDataZarrSink(Sink["AtomicData"]):
    """Write :class:`~nvalchemi.data.AtomicData` objects to a Zarr store.

    Items are batched in memory (up to *batch_size*) and flushed to the
    Zarr store using :class:`~nvalchemi.data.datapipes.backends.zarr.AtomicDataZarrWriter`.
    The first flush creates the store; all subsequent flushes append to it.

    **Default mode** (no *naming_template*): all pipeline indices write to
    the **same** store via append semantics, producing a single consolidated
    output.

    **Directory-mirroring mode** (*naming_template* provided): each pipeline
    index writes to a separate Zarr store whose name is derived from the
    template.  When the pipeline's source exposes a ``relative_path(index)``
    method (e.g.
    :class:`~physicsnemo_curator.domains.atm.sources.aselmdb.ASELMDBSource`),
    the ``{relpath}`` and ``{stem}`` placeholders resolve to the source's
    directory structure, enabling output layouts that mirror the input.

    Parameters
    ----------
    output_path : str
        Base directory for output Zarr store(s).
    naming_template : str or None
        Python format string for per-index store naming.  The placeholders
        ``{index}`` (source index) is always available.  When the source
        supports it, ``{relpath}`` (parent directory relative to source
        root) and ``{stem}`` (filename stem without extension) are also
        available.  When ``None`` (default), all indices write to a single
        store at *output_path*.
    batch_size : int
        Number of :class:`AtomicData` items to accumulate before flushing
        to the store.  Larger batches reduce I/O overhead.

    Examples
    --------
    Default (single store):

    >>> sink = AtomicDataZarrSink(output_path="./output.zarr")  # doctest: +SKIP
    >>> paths = sink(atomic_data_iterator, index=0)  # doctest: +SKIP
    >>> paths
    ['./output.zarr']

    Directory mirroring:

    >>> sink = AtomicDataZarrSink(  # doctest: +SKIP
    ...     output_path="./output/",
    ...     naming_template="{relpath}/{stem}.zarr",
    ... )
    >>> # Input:  ./data/split_a/run_01.aselmdb
    >>> # Output: ./output/split_a/run_01.zarr
    """

    name: ClassVar[str] = "AtomicData Zarr"
    description: ClassVar[str] = "Write AtomicData to a Zarr store using nvalchemi"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for this sink.

        Returns
        -------
        list[Param]
            The ``output_path``, ``naming_template``, and ``batch_size``
            parameters.
        """
        return [
            Param(
                name="output_path",
                description="Base path for the output Zarr store(s)",
                type=str,
            ),
            Param(
                name="naming_template",
                description=(
                    "Per-index naming template (e.g. '{relpath}/{stem}.zarr'). Leave empty for single-store mode."
                ),
                type=str,
                default="",
            ),
            Param(
                name="batch_size",
                description="Items per write batch (larger = fewer I/O calls)",
                type=int,
                default=1000,
            ),
        ]

    def __init__(
        self,
        output_path: str,
        naming_template: str | None = None,
        batch_size: int = 1000,
    ) -> None:
        self._output_path = pathlib.Path(output_path)
        self._naming_template = naming_template or None
        self._batch_size = batch_size
        self._source: Source[AtomicData] | None = None

        # Track which store paths have been created (for write vs append).
        # In single-store mode, check existence at construction time so that
        # all workers (which receive pickled copies) start in append mode.
        self._existing_stores: set[str] = set()
        if self._naming_template is None and self._output_path.exists():
            self._existing_stores.add(str(self._output_path))
            logger.warning(
                "Zarr store already exists at %s; new data will be appended. "
                "Delete the store first if you want a fresh dataset.",
                self._output_path,
            )

    def set_source(self, source: Source[AtomicData]) -> None:
        """Inject the pipeline source for ``{relpath}``/``{stem}`` resolution.

        Called automatically by the :class:`~physicsnemo_curator.core.base.Pipeline`
        when the sink is attached via :meth:`Pipeline.write`.

        Parameters
        ----------
        source : Source[AtomicData]
            The pipeline source.  If it exposes a ``relative_path(index)``
            method, the sink will use it to resolve naming placeholders.
        """
        self._source = source

    def __call__(self, items: Iterator[AtomicData], index: int) -> list[str]:
        """Consume atomic data items and write them to the Zarr store.

        Parameters
        ----------
        items : Iterator[AtomicData]
            Stream of :class:`AtomicData` objects to persist.
        index : int
            Source index.  In single-store mode this is used only for
            logging.  In directory-mirroring mode it determines the
            output store path via the naming template.

        Returns
        -------
        list[str]
            Single-element list containing the store path, or empty list
            if no items were consumed.
        """
        from nvalchemi.data.datapipes.backends.zarr import AtomicDataZarrWriter

        store_path = self._resolve_store_path(index)
        store_path.parent.mkdir(parents=True, exist_ok=True)

        writer = AtomicDataZarrWriter(str(store_path))
        wrote_any = False
        batch: list[AtomicData] = []

        for item in items:
            batch.append(item)
            if len(batch) >= self._batch_size:
                self._flush(writer, batch, str(store_path))
                wrote_any = True
                batch = []

        # Flush remaining items.
        if batch:
            self._flush(writer, batch, str(store_path))
            wrote_any = True

        if wrote_any:
            logger.info("Wrote index %d to %s", index, store_path)
            return [str(store_path)]

        return []

    def _resolve_store_path(self, index: int) -> pathlib.Path:
        """Resolve the output Zarr store path for the given index.

        Parameters
        ----------
        index : int
            Source index.

        Returns
        -------
        pathlib.Path
            The resolved store path.
        """
        if self._naming_template is None:
            return self._output_path

        # Resolve relpath / stem from source if available.
        relpath = ""
        stem = ""
        if self._source is not None and hasattr(self._source, "relative_path"):
            rel = self._source.relative_path(index)  # ty: ignore[call-non-callable]
            rel_path = pathlib.PurePosixPath(rel)
            stem = rel_path.stem
            relpath = str(rel_path.parent) if str(rel_path.parent) != "." else ""

        name = self._naming_template.format(
            index=index,
            relpath=relpath,
            stem=stem,
        )
        return self._output_path / name

    def _flush(self, writer: Any, batch: list[AtomicData], store_key: str) -> None:
        """Write or append a batch of items to the store.

        Parameters
        ----------
        writer : AtomicDataZarrWriter
            The nvalchemi Zarr writer instance.
        batch : list[AtomicData]
            Items to flush.
        store_key : str
            String key identifying which store this batch targets,
            used to track write-vs-append state.
        """
        # Check filesystem to handle multi-worker race conditions.
        # In-memory tracking (_existing_stores) is per-process and can miss
        # stores created by other workers. Fall back to append if store exists.
        store_exists = store_key in self._existing_stores or pathlib.Path(store_key).exists()
        if store_exists:
            writer.append(batch)
            self._existing_stores.add(store_key)
        else:
            try:
                writer.write(batch)
                self._existing_stores.add(store_key)
            except FileExistsError:
                # Another worker created the store between our check and write.
                # Fall back to append mode.
                logger.debug("Store %s created by another worker; switching to append", store_key)
                writer.append(batch)
                self._existing_stores.add(store_key)
        logger.debug("Flushed batch of %d items to %s", len(batch), store_key)

    # -- Properties -----------------------------------------------------------

    @property
    def output_path(self) -> pathlib.Path:
        """Return the output Zarr store path."""
        return self._output_path

    @property
    def naming_template(self) -> str | None:
        """Return the naming template, or ``None`` for single-store mode."""
        return self._naming_template

    @property
    def batch_size(self) -> int:
        """Return the configured batch size."""
        return self._batch_size
