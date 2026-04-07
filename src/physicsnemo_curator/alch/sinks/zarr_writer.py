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

Examples
--------
>>> sink = AtomicDataZarrSink(output_path="./output.zarr")  # doctest: +SKIP
>>> paths = sink(atomic_data_iterator, index=0)  # doctest: +SKIP
"""

from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING, Any, ClassVar

from physicsnemo_curator.core.base import Param, Sink

if TYPE_CHECKING:
    from collections.abc import Iterator

    from nvalchemi.data import AtomicData

logger = logging.getLogger(__name__)


class AtomicDataZarrSink(Sink["AtomicData"]):
    """Write :class:`~nvalchemi.data.AtomicData` objects to a Zarr store.

    Items are batched in memory (up to *batch_size*) and flushed to the
    Zarr store using :class:`~nvalchemi.data.datapipes.backends.zarr.AtomicDataZarrWriter`.
    The first flush creates the store; all subsequent flushes append to it.

    Multiple pipeline indices write to the **same** store via append
    semantics, producing a single consolidated output.

    Parameters
    ----------
    output_path : str
        Path for the output Zarr store directory.
    batch_size : int
        Number of :class:`AtomicData` items to accumulate before flushing
        to the store.  Larger batches reduce I/O overhead.

    Examples
    --------
    >>> sink = AtomicDataZarrSink(output_path="./output.zarr")  # doctest: +SKIP
    >>> paths = sink(atomic_data_iterator, index=0)  # doctest: +SKIP
    >>> paths
    ['./output.zarr']
    """

    name: ClassVar[str] = "AtomicData Zarr"
    description: ClassVar[str] = "Write AtomicData to a Zarr store using nvalchemi"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for this sink.

        Returns
        -------
        list[Param]
            The ``output_path`` and ``batch_size`` parameters.
        """
        return [
            Param(
                name="output_path",
                description="Path for the output Zarr store",
                type=str,
            ),
            Param(
                name="batch_size",
                description="Items per write batch (larger = fewer I/O calls)",
                type=int,
                default=1000,
            ),
        ]

    def __init__(self, output_path: str, batch_size: int = 1000) -> None:
        self._output_path = pathlib.Path(output_path)
        self._batch_size = batch_size
        self._store_exists = self._output_path.exists()

    def __call__(self, items: Iterator[AtomicData], index: int) -> list[str]:
        """Consume atomic data items and write them to the Zarr store.

        Parameters
        ----------
        items : Iterator[AtomicData]
            Stream of :class:`AtomicData` objects to persist.
        index : int
            Source index (used for logging; all indices write to the same
            store).

        Returns
        -------
        list[str]
            Single-element list containing the store path, or empty list
            if no items were consumed.
        """
        from nvalchemi.data.datapipes.backends.zarr import AtomicDataZarrWriter

        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        writer = AtomicDataZarrWriter(str(self._output_path))
        wrote_any = False
        batch: list[AtomicData] = []

        for item in items:
            batch.append(item)
            if len(batch) >= self._batch_size:
                self._flush(writer, batch)
                wrote_any = True
                batch = []

        # Flush remaining items.
        if batch:
            self._flush(writer, batch)
            wrote_any = True

        if wrote_any:
            logger.info("Wrote index %d to %s", index, self._output_path)
            return [str(self._output_path)]

        return []

    def _flush(self, writer: Any, batch: list[AtomicData]) -> None:
        """Write or append a batch of items to the store.

        Parameters
        ----------
        writer : AtomicDataZarrWriter
            The nvalchemi Zarr writer instance.
        batch : list[AtomicData]
            Items to flush.
        """
        if self._store_exists:
            writer.append(batch)
        else:
            writer.write(batch)
            self._store_exists = True
        logger.debug("Flushed batch of %d items (append=%s)", len(batch), self._store_exists)

    # -- Properties -----------------------------------------------------------

    @property
    def output_path(self) -> pathlib.Path:
        """Return the output Zarr store path."""
        return self._output_path

    @property
    def batch_size(self) -> int:
        """Return the configured batch size."""
        return self._batch_size
