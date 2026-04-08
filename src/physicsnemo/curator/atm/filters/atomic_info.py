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

"""Atomic information logging filter for atomic data pipelines.

Logs atomic-data metadata (atom count, edge count, PBC, cell, field
inventory with shapes/dtypes/memory) to the Python logger and optionally
writes structured records to a JSON-lines file for post-analysis.
The item is yielded unchanged (pass-through).
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import IO, TYPE_CHECKING, ClassVar, TypedDict

import torch

from physicsnemo.curator.core.base import Filter, Param

if TYPE_CHECKING:
    from collections.abc import Generator

    from nvalchemi.data import AtomicData

logger = logging.getLogger(__name__)

# AtomicData tensor fields grouped by their semantic level.
_NODE_FIELDS: tuple[str, ...] = (
    "positions",
    "atomic_numbers",
    "atomic_masses",
    "atom_categories",
    "forces",
    "velocities",
    "momenta",
    "kinetic_energies",
    "node_charges",
    "node_attrs",
    "node_spins",
    "node_embeddings",
)

_EDGE_FIELDS: tuple[str, ...] = (
    "edge_index",
    "shifts",
    "unit_shifts",
    "edge_embeddings",
)

_SYSTEM_FIELDS: tuple[str, ...] = (
    "energies",
    "stresses",
    "virials",
    "dipoles",
    "graph_charges",
    "graph_spins",
    "cell",
    "pbc",
)


class _FieldInfo(TypedDict):
    """Typed dictionary for a single tensor field metadata entry."""

    name: str
    level: str
    shape: list[int]
    dtype: str
    nbytes: int


def _extract_field_info(data: AtomicData) -> list[_FieldInfo]:
    """Extract tensor field metadata from an AtomicData object.

    Parameters
    ----------
    data : AtomicData
        The atomic data object.

    Returns
    -------
    list[_FieldInfo]
        List of field info dicts with name, level, shape, dtype, nbytes.
    """
    fields: list[_FieldInfo] = []

    for field_names, level in (
        (_NODE_FIELDS, "node"),
        (_EDGE_FIELDS, "edge"),
        (_SYSTEM_FIELDS, "system"),
    ):
        for field_name in field_names:
            val = getattr(data, field_name, None)
            if val is not None and isinstance(val, torch.Tensor):
                fields.append(
                    _FieldInfo(
                        name=field_name,
                        level=level,
                        shape=list(val.shape),
                        dtype=str(val.dtype),
                        nbytes=val.numel() * val.element_size(),
                    )
                )

    # Extra data fields (dynamically set via Pydantic extra="allow").
    extra_data: dict[str, object] | None = getattr(data, "extra_data", None)
    if extra_data:
        for field_name, val in extra_data.items():
            if isinstance(val, torch.Tensor):
                fields.append(
                    _FieldInfo(
                        name=f"extra/{field_name}",
                        level="extra",
                        shape=list(val.shape),
                        dtype=str(val.dtype),
                        nbytes=val.numel() * val.element_size(),
                    )
                )

    return fields


class AtomicInfoFilter(Filter["AtomicData"]):
    """Log atomic data information and optionally write to a JSON-lines file.

    For each incoming :class:`~nvalchemi.data.AtomicData` the filter
    extracts metadata (atom count, edge count, periodic boundary
    conditions, cell dimensions, and a field inventory with shapes,
    dtypes, and memory usage) and logs it.  Optionally writes structured
    records to a JSON-lines file for post-analysis.

    The item is yielded unchanged so downstream filters and sinks receive
    the full data.

    Parameters
    ----------
    output : str or None
        Optional path for JSON-lines output file.  If provided, each
        item's metadata is appended as a JSON object on a new line.
    log_level : str
        Logging level for console output: ``"info"`` or ``"debug"``.
        Default is ``"info"``.
    include_fields : bool
        Whether to include detailed field information (names, levels,
        shapes, dtypes).  Default is ``True``.

    Examples
    --------
    Log atomic info to console only:

    >>> filt = AtomicInfoFilter()  # doctest: +SKIP
    >>> pipeline = source.filter(filt).write(sink)  # doctest: +SKIP

    Log atomic info and write to JSON-lines file:

    >>> filt = AtomicInfoFilter(output="atomic_info.jsonl", log_level="debug")  # doctest: +SKIP
    >>> pipeline = source.filter(filt).write(sink)  # doctest: +SKIP
    """

    name: ClassVar[str] = "Atomic Info Logger"
    description: ClassVar[str] = "Log atomic data metadata and optionally write to JSON-lines file"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the atomic info filter.

        Returns
        -------
        list[Param]
            Parameters for ``output``, ``log_level``, and ``include_fields``.
        """
        return [
            Param(
                name="output",
                description="Optional JSON-lines output file path",
                type=str,
                default=None,
            ),
            Param(
                name="log_level",
                description="Logging level for console output",
                type=str,
                default="info",
                choices=["info", "debug"],
            ),
            Param(
                name="include_fields",
                description="Include detailed field information",
                type=bool,
                default=True,
            ),
        ]

    def __init__(
        self,
        output: str | None = None,
        log_level: str = "info",
        include_fields: bool = True,
    ) -> None:
        """Initialize the atomic info filter.

        Parameters
        ----------
        output : str or None
            Optional path for JSON-lines output file.
        log_level : str
            Logging level: ``"info"`` or ``"debug"``.
        include_fields : bool
            Whether to include detailed field information.
        """
        self._output_path = pathlib.Path(output) if output else None
        self._log_level = logging.INFO if log_level == "info" else logging.DEBUG
        self._include_fields = include_fields
        self._item_index = 0
        self._file_handle: IO[str] | None = None

    def __call__(self, items: Generator[AtomicData]) -> Generator[AtomicData]:
        """Extract and log atomic data info, then yield unchanged.

        Parameters
        ----------
        items : Generator[AtomicData]
            Stream of incoming :class:`AtomicData` objects.

        Yields
        ------
        AtomicData
            The same item, unmodified.
        """
        for data in items:
            info = self._extract_info(data)
            self._log_info(info)
            self._write_to_file(info)
            self._item_index += 1
            yield data

    def flush(self) -> str | None:
        """Close the output file if open.

        Returns
        -------
        str or None
            The path of the output file, or ``None`` if no file was used.
        """
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
        return str(self._output_path) if self._output_path else None

    def _extract_info(self, data: AtomicData) -> dict[str, object]:
        """Extract metadata from an AtomicData object.

        Parameters
        ----------
        data : AtomicData
            The input atomic data.

        Returns
        -------
        dict[str, object]
            Atomic data metadata dictionary.
        """
        info: dict[str, object] = {
            "item_index": self._item_index,
        }

        # Atom (node) count.
        n_atoms = 0
        positions = getattr(data, "positions", None)
        if positions is not None and isinstance(positions, torch.Tensor):
            n_atoms = positions.shape[0]
        info["n_atoms"] = n_atoms

        # Edge count.
        n_edges = 0
        edge_index = getattr(data, "edge_index", None)
        if edge_index is not None and isinstance(edge_index, torch.Tensor):
            n_edges = edge_index.shape[1] if edge_index.ndim >= 2 else edge_index.shape[0]
        info["n_edges"] = n_edges

        # Periodic boundary conditions.
        pbc = getattr(data, "pbc", None)
        if pbc is not None and isinstance(pbc, torch.Tensor):
            info["pbc"] = pbc.tolist()
        else:
            info["pbc"] = None

        # Unit cell.
        cell = getattr(data, "cell", None)
        if cell is not None and isinstance(cell, torch.Tensor):
            info["cell_shape"] = list(cell.shape)
        else:
            info["cell_shape"] = None

        # Field inventory.
        field_infos = _extract_field_info(data)
        total_memory = sum(f["nbytes"] for f in field_infos)
        info["n_fields"] = len(field_infos)
        info["memory_estimate_bytes"] = total_memory

        if self._include_fields:
            info["fields"] = field_infos

        return info

    def _log_info(self, info: dict[str, object]) -> None:
        """Log atomic data info to the Python logger.

        Parameters
        ----------
        info : dict[str, object]
            Atomic data metadata dictionary.
        """
        memory_mb = int(info.get("memory_estimate_bytes", 0)) / (1024 * 1024)  # ty: ignore[invalid-argument-type]
        msg = (
            f"AtomicData {info['item_index']}: "
            f"{info['n_atoms']} atoms, {info['n_edges']} edges, "
            f"{info['n_fields']} fields, "
            f"{memory_mb:.2f} MB"
        )
        logger.log(self._log_level, msg)

        if self._include_fields and self._log_level <= logging.DEBUG:
            field_infos: list[_FieldInfo] = info.get("fields", [])  # ty: ignore[invalid-assignment]
            for fld in field_infos:
                logger.debug(
                    "  %s/%s: shape=%s, dtype=%s",
                    fld["level"],
                    fld["name"],
                    fld["shape"],
                    fld["dtype"],
                )

    def _write_to_file(self, info: dict[str, object]) -> None:
        """Write atomic data info to the JSON-lines file.

        Parameters
        ----------
        info : dict[str, object]
            Atomic data metadata dictionary.
        """
        if self._output_path is None:
            return

        # Lazy open file on first write.
        if self._file_handle is None:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = self._output_path.open("w")

        self._file_handle.write(json.dumps(info) + "\n")
        self._file_handle.flush()
