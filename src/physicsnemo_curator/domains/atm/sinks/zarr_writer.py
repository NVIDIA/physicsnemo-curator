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
store.  Supports two modes of operation:

**Sequential mode** (default): Uses
:class:`~nvalchemi.data.datapipes.backends.zarr.AtomicDataZarrWriter` with
write/append semantics.  Suitable for single-process pipelines.

**Pre-allocated parallel mode**: When *natoms* and *schema* are provided at
construction, the store is pre-allocated to its full size upfront.  Workers
can then write to non-overlapping regions concurrently without
synchronization.  This enables safe parallel writes via the ``process_pool``
backend.

Examples
--------
Sequential (single store):

>>> sink = AtomicDataZarrSink(output_path="./output.zarr")  # doctest: +SKIP
>>> paths = sink(atomic_data_iterator, index=0)  # doctest: +SKIP

Parallel (pre-allocated):

>>> import numpy as np
>>> from nvalchemi.data import AtomicData
>>> source = ASELMDBSource(data_dir="input/")
>>> sample = next(source[0])
>>> sink = AtomicDataZarrSink(  # doctest: +SKIP
...     output_path="./output.zarr",
...     natoms=source.metadata["natoms"],
...     schema=sample,
...     chunk_size=1024,
... )
"""

from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np

from physicsnemo_curator.core.base import Param, Sink, Source

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from nvalchemi.data import AtomicData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Field-level classification helpers
# ---------------------------------------------------------------------------

# Known node-level (per-atom) fields and their trailing shapes.
_NODE_FIELDS: dict[str, tuple[int, ...]] = {
    "positions": (3,),
    "atomic_numbers": (),
    "atomic_masses": (),
    "forces": (3,),
    "velocities": (3,),
    "momenta": (3,),
    "kinetic_energies": (),
    "node_charges": (),
    "node_spins": (),
    "atom_categories": (),
}

# Known edge-level (per-bond) fields.
_EDGE_FIELDS: dict[str, tuple[int, ...]] = {
    "edge_index": (),  # shape (2, n_edges) — special case
    "shifts": (3,),
    "unit_shifts": (3,),
}

# Known system-level (per-structure) fields.
_SYSTEM_FIELDS: dict[str, tuple[int, ...]] = {
    "energies": (1,),
    "stresses": (3, 3),
    "virials": (3, 3),
    "dipoles": (3,),
    "graph_charges": (1,),
    "graph_spins": (1,),
    "cell": (3, 3),
    "pbc": (3,),
}


def _classify_field(name: str) -> str:
    """Return 'node', 'edge', or 'system' for a known field name.

    Parameters
    ----------
    name : str
        The field name.

    Returns
    -------
    str
        One of ``"node"``, ``"edge"``, ``"system"``.

    Raises
    ------
    ValueError
        If the field name is not recognized.
    """
    if name in _NODE_FIELDS:
        return "node"
    if name in _EDGE_FIELDS:
        return "edge"
    if name in _SYSTEM_FIELDS:
        return "system"
    msg = f"Unknown field {name!r}; cannot classify as node/edge/system."
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Schema extraction from a sample AtomicData
# ---------------------------------------------------------------------------


def _extract_schema(
    sample: AtomicData,
) -> dict[str, tuple[str, np.dtype, tuple[int, ...]]]:
    """Extract field schema from a single AtomicData sample.

    Parameters
    ----------
    sample : AtomicData
        A representative atomic data object.

    Returns
    -------
    dict[str, tuple[str, np.dtype, tuple[int, ...]]]
        Mapping of ``field_name -> (level, dtype, trailing_shape)``.
        *level* is ``"node"``, ``"edge"``, or ``"system"``.
    """
    import torch

    schema: dict[str, tuple[str, np.dtype, tuple[int, ...]]] = {}

    # Iterate known field sets.
    for field_set, level in [(_NODE_FIELDS, "node"), (_EDGE_FIELDS, "edge"), (_SYSTEM_FIELDS, "system")]:
        for name in field_set:
            val = getattr(sample, name, None)
            if val is None:
                continue
            if not isinstance(val, torch.Tensor):
                continue

            # Convert torch dtype to numpy dtype.
            np_dtype = np.dtype(str(val.dtype).replace("torch.", ""))

            # Determine trailing shape (everything after the leading dim).
            if name == "edge_index":
                # edge_index is (2, n_edges) — trailing shape is (2,) per edge
                # We store transposed: (n_edges, 2)
                schema[name] = (level, np_dtype, (2,))
            elif level == "system":
                # System fields have shape (1, ...) — trailing is everything after dim 0
                trail = tuple(val.shape[1:])
                schema[name] = (level, np_dtype, trail)
            else:
                # Node/edge fields: first dim is count, rest is trailing
                trail = tuple(val.shape[1:])
                schema[name] = (level, np_dtype, trail)

    # Check for extra_data fields (dict of tensors).
    extra = getattr(sample, "extra_data", None) or {}
    for name, val in extra.items():
        if not isinstance(val, torch.Tensor):
            continue
        np_dtype = np.dtype(str(val.dtype).replace("torch.", ""))
        # Heuristic: if shape[0] matches num_nodes → node level
        n_nodes = sample.positions.shape[0]
        if val.shape[0] == n_nodes:
            schema[name] = ("node", np_dtype, tuple(val.shape[1:]))
        else:
            schema[name] = ("system", np_dtype, tuple(val.shape[1:]))

    return schema


# ---------------------------------------------------------------------------
# Main sink class
# ---------------------------------------------------------------------------


class AtomicDataZarrSink(Sink["AtomicData"]):
    """Write :class:`~nvalchemi.data.AtomicData` objects to a Zarr store.

    **Sequential mode** (default): Items are batched and flushed using
    nvalchemi's ``AtomicDataZarrWriter``.  The first flush creates the
    store; subsequent flushes append.

    **Pre-allocated parallel mode**: When *natoms* and *schema* are
    provided, the store is fully pre-allocated at construction time.
    Each call to :meth:`__call__` writes data at a fixed offset
    determined by the index.  No locking or coordination is needed
    because different indices map to non-overlapping array regions.

    Parameters
    ----------
    output_path : str
        Base directory for output Zarr store(s).
    naming_template : str or None
        Per-index store naming template (sequential mode only).
    batch_size : int
        Items per write batch (sequential mode).
    natoms : array-like or None
        Per-structure atom counts.  When provided together with *schema*,
        enables pre-allocated parallel mode.
    nedges : array-like or None
        Per-structure edge counts (optional).  Required for pre-allocating
        edge-level arrays.  If ``None``, edge arrays are skipped in
        pre-allocation.
    schema : AtomicData or None
        A representative sample used to discover field names, dtypes, and
        shapes.  Provide any single :class:`AtomicData` instance (e.g.
        ``next(source[0])``).
    chunk_size : int
        Number of atoms per Zarr chunk along the leading dimension of
        node-level arrays.  Also determines index partitioning for
        parallel dispatch.

    Examples
    --------
    Sequential (backward-compatible):

    >>> sink = AtomicDataZarrSink(output_path="./output.zarr")  # doctest: +SKIP

    Parallel with pre-allocation:

    >>> source = ASELMDBSource(data_dir="input/")  # doctest: +SKIP
    >>> sink = AtomicDataZarrSink(  # doctest: +SKIP
    ...     output_path="./output.zarr",
    ...     natoms=source.metadata["natoms"],
    ...     schema=next(source[0]),
    ...     chunk_size=2048,
    ... )
    """

    name: ClassVar[str] = "AtomicData Zarr"
    description: ClassVar[str] = "Write AtomicData to a Zarr store using nvalchemi"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for this sink.

        Returns
        -------
        list[Param]
            The configurable parameters.
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
            Param(
                name="chunk_size",
                description="Atoms per Zarr chunk (controls parallel partitioning)",
                type=int,
                default=1024,
            ),
        ]

    def __init__(
        self,
        output_path: str,
        naming_template: str | None = None,
        batch_size: int = 1000,
        natoms: np.ndarray | Sequence[int] | None = None,
        nedges: np.ndarray | Sequence[int] | None = None,
        schema: AtomicData | None = None,
        chunk_size: int = 1024,
    ) -> None:
        self._output_path = pathlib.Path(output_path)
        self._naming_template = naming_template or None
        self._batch_size = batch_size
        self._chunk_size = chunk_size
        self._source: Source[AtomicData] | None = None

        # --- Pre-allocated parallel mode ---
        self._parallel = natoms is not None and schema is not None
        self._natoms: np.ndarray | None = None
        self._nedges: np.ndarray | None = None
        self._atom_offsets: np.ndarray | None = None
        self._edge_offsets: np.ndarray | None = None
        self._field_schema: dict[str, tuple[str, np.dtype, tuple[int, ...]]] = {}
        self._chunk_groups: list[list[int]] | None = None

        if self._parallel:
            self._natoms = np.asarray(natoms, dtype=np.int64)
            if nedges is not None:
                self._nedges = np.asarray(nedges, dtype=np.int64)
            self._field_schema = _extract_schema(schema)  # type: ignore[arg-type]
            self._preallocate()
        else:
            # Sequential fallback state.
            self._existing_stores: set[str] = set()
            if self._naming_template is None and self._output_path.exists():
                self._existing_stores.add(str(self._output_path))
                logger.warning(
                    "Zarr store already exists at %s; new data will be appended. "
                    "Delete the store first if you want a fresh dataset.",
                    self._output_path,
                )

    # ------------------------------------------------------------------
    # Pre-allocation
    # ------------------------------------------------------------------

    def _preallocate(self) -> None:
        """Create the full Zarr store with all arrays pre-allocated.

        This creates empty arrays at their final sizes so that workers
        can write to non-overlapping regions without coordination.
        """
        import zarr

        assert self._natoms is not None  # noqa: S101

        n_structures = len(self._natoms)
        total_atoms = int(self._natoms.sum())
        self._atom_offsets = np.zeros(n_structures + 1, dtype=np.int64)
        np.cumsum(self._natoms, out=self._atom_offsets[1:])

        total_edges = 0
        if self._nedges is not None:
            total_edges = int(self._nedges.sum())
            self._edge_offsets = np.zeros(n_structures + 1, dtype=np.int64)
            np.cumsum(self._nedges, out=self._edge_offsets[1:])

        # Create store directory.
        self._output_path.mkdir(parents=True, exist_ok=True)
        store: Any = cast("Any", zarr.open(str(self._output_path), mode="w"))

        # --- Meta arrays ---
        # atoms_ptr: cumulative atom offsets (length N+1).
        store.create_array(
            "meta/atoms_ptr",
            data=self._atom_offsets,
            overwrite=True,
        )

        if self._edge_offsets is not None:
            store.create_array(
                "meta/edges_ptr",
                data=self._edge_offsets,
                overwrite=True,
            )

        # n_structures metadata attribute.
        store.attrs["n_structures"] = n_structures
        store.attrs["total_atoms"] = total_atoms
        store.attrs["total_edges"] = total_edges

        # --- Core arrays (one per field) ---
        for field_name, (level, dtype, trail) in self._field_schema.items():
            if level == "node":
                shape = (total_atoms, *trail)
                c0 = min(self._chunk_size, total_atoms)
                chunks = (c0, *trail) if trail else (c0,)
            elif level == "edge":
                if total_edges == 0:
                    # Skip edge arrays if no edge info provided.
                    continue
                shape = (total_edges, *trail)
                c0 = min(self._chunk_size, total_edges)
                chunks = (c0, *trail) if trail else (c0,)
            else:  # system
                shape = (n_structures, *trail)
                c0 = min(self._chunk_size, n_structures)
                chunks = (c0, *trail) if trail else (c0,)

            store.create_array(
                f"core/{field_name}",
                shape=shape,
                dtype=dtype,
                chunks=chunks,
                fill_value=0,
                overwrite=True,
            )

        logger.info(
            "Pre-allocated Zarr store at %s: %d structures, %d atoms, %d edges, %d fields",
            self._output_path,
            n_structures,
            total_atoms,
            total_edges,
            len(self._field_schema),
        )

        # Compute chunk partition groups.
        self._compute_partition()

    def _compute_partition(self) -> None:
        """Compute index→chunk groups for parallel dispatch.

        Groups indices by which chunk their atom data starts in.
        Each group can be processed by a single worker without
        overlapping any other group's array regions.
        """
        assert self._atom_offsets is not None  # noqa: S101

        n_structures = len(self._atom_offsets) - 1
        # Assign each structure to a chunk based on its start offset.
        chunk_ids = self._atom_offsets[:-1] // self._chunk_size

        # Group indices by chunk_id.
        groups: dict[int, list[int]] = {}
        for idx in range(n_structures):
            cid = int(chunk_ids[idx])
            groups.setdefault(cid, []).append(idx)

        # Sort groups by chunk_id for deterministic ordering.
        self._chunk_groups = [groups[k] for k in sorted(groups.keys())]

        logger.debug(
            "Partition: %d structures → %d chunk groups (chunk_size=%d atoms)",
            n_structures,
            len(self._chunk_groups),
            self._chunk_size,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def partition_indices(self, indices: list[int] | None = None) -> list[list[int]] | None:
        """Return chunk-aligned index groups for parallel dispatch.

        When the sink is in pre-allocated mode, this returns groups of
        indices that can each be processed by a single worker without
        overlapping.  The runner uses this to assign work.

        Parameters
        ----------
        indices : list[int] or None
            Specific indices to partition.  If ``None``, returns all
            groups.

        Returns
        -------
        list[list[int]] or None
            Groups of indices, or ``None`` if not in parallel mode.
        """
        if not self._parallel or self._chunk_groups is None:
            return None

        if indices is None:
            return self._chunk_groups

        # Filter groups to only include requested indices.
        idx_set = set(indices)
        filtered = []
        for group in self._chunk_groups:
            subset = [i for i in group if i in idx_set]
            if subset:
                filtered.append(subset)
        return filtered if filtered else None

    def set_source(self, source: Source[AtomicData]) -> None:
        """Inject the pipeline source for ``{relpath}``/``{stem}`` resolution.

        Called automatically by the :class:`~physicsnemo_curator.core.base.Pipeline`
        when the sink is attached via :meth:`Pipeline.write`.

        Parameters
        ----------
        source : Source[AtomicData]
            The pipeline source.
        """
        self._source = source

    def __call__(self, items: Iterator[AtomicData], index: int) -> list[str]:
        """Consume atomic data items and write them to the Zarr store.

        In pre-allocated mode, writes directly to the pre-computed offset.
        In sequential mode, uses batched write/append via nvalchemi.

        Parameters
        ----------
        items : Iterator[AtomicData]
            Stream of :class:`AtomicData` objects to persist.
        index : int
            Source index determining the write location.

        Returns
        -------
        list[str]
            Single-element list containing the store path, or empty list
            if no items were consumed.
        """
        if self._parallel:
            return self._write_parallel(items, index)
        return self._write_sequential(items, index)

    # ------------------------------------------------------------------
    # Parallel write path
    # ------------------------------------------------------------------

    def _write_parallel(self, items: Iterator[AtomicData], index: int) -> list[str]:
        """Write items at pre-computed offsets (no locking needed).

        Parameters
        ----------
        items : Iterator[AtomicData]
            Items for this index (typically exactly one for ASELMDBSource).
        index : int
            Structure index.

        Returns
        -------
        list[str]
            Single-element list with store path, or empty if no items.
        """
        import torch
        import zarr

        assert self._atom_offsets is not None  # noqa: S101
        assert self._natoms is not None  # noqa: S101

        store: Any = cast("Any", zarr.open(str(self._output_path), mode="r+"))
        wrote_any = False

        # Each index corresponds to exactly one structure.
        atom_start = int(self._atom_offsets[index])
        atom_end = int(self._atom_offsets[index + 1])

        edge_start = 0
        edge_end = 0
        if self._edge_offsets is not None:
            edge_start = int(self._edge_offsets[index])
            edge_end = int(self._edge_offsets[index + 1])

        for item in items:
            wrote_any = True

            for field_name, (level, _dtype, _trail) in self._field_schema.items():
                val = getattr(item, field_name, None)
                if val is None:
                    continue

                # Convert torch tensor to numpy.
                arr = val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else np.asarray(val)

                array_path = f"core/{field_name}"

                if level == "node":
                    store[array_path][atom_start:atom_end] = arr
                elif level == "edge":
                    if field_name == "edge_index":
                        # edge_index is (2, n_edges) → transpose to (n_edges, 2)
                        arr = arr.T if arr.shape[0] == 2 else arr
                    store[array_path][edge_start:edge_end] = arr
                else:  # system
                    # System fields have shape (1, ...) — squeeze first dim.
                    if arr.ndim > 0 and arr.shape[0] == 1:
                        arr = arr[0]
                    store[array_path][index] = arr

        if wrote_any:
            logger.debug("Wrote index %d at atom offset [%d:%d]", index, atom_start, atom_end)
            return [str(self._output_path)]

        return []

    # ------------------------------------------------------------------
    # Sequential write path (backward-compatible)
    # ------------------------------------------------------------------

    def _write_sequential(self, items: Iterator[AtomicData], index: int) -> list[str]:
        """Batched write/append via nvalchemi's AtomicDataZarrWriter.

        Parameters
        ----------
        items : Iterator[AtomicData]
            Stream of :class:`AtomicData` objects.
        index : int
            Source index.

        Returns
        -------
        list[str]
            Single-element list with store path, or empty if no items.
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
            String key identifying which store this batch targets.
        """
        store_exists = store_key in self._existing_stores or pathlib.Path(store_key).exists()
        if store_exists:
            writer.append(batch)
            self._existing_stores.add(store_key)
        else:
            try:
                writer.write(batch)
                self._existing_stores.add(store_key)
            except FileExistsError:
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

    @property
    def chunk_size(self) -> int:
        """Return the configured chunk size (atoms per chunk)."""
        return self._chunk_size

    @property
    def parallel(self) -> bool:
        """Return whether the sink is in pre-allocated parallel mode."""
        return self._parallel
