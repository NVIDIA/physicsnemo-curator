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

"""Mesh information logging filter for mesh pipelines.

Logs mesh metadata (points, cells, field information) to the Python logger
and optionally writes structured records to a JSON-lines file for post-analysis.
The mesh is yielded unchanged (pass-through).
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

    from physicsnemo.mesh import Mesh

logger = logging.getLogger(__name__)


class _FieldInfo(TypedDict):
    """Typed dictionary for a single field metadata entry."""

    name: str
    shape: list[int]
    dtype: str
    nbytes: int


def _extract_field_info(td: object, prefix: str = "") -> list[_FieldInfo]:
    """Extract field metadata from a TensorDict-like object.

    Parameters
    ----------
    td : object
        A TensorDict (``point_data`` or ``cell_data``).
    prefix : str
        Dotted prefix for nested field names.

    Returns
    -------
    list[_FieldInfo]
        List of field info dicts with name, shape, dtype, nbytes.
    """
    from tensordict import TensorDictBase

    if not isinstance(td, TensorDictBase):
        return []

    fields: list[_FieldInfo] = []

    for key in td.keys():  # noqa: SIM118
        child = td[key]
        full_name = f"{prefix}{key}" if prefix else key

        if isinstance(child, TensorDictBase):
            # Recurse into nested TensorDicts
            fields.extend(_extract_field_info(child, prefix=f"{full_name}/"))
        elif isinstance(child, torch.Tensor):
            fields.append(
                _FieldInfo(
                    name=full_name,
                    shape=list(child.shape),
                    dtype=str(child.dtype),
                    nbytes=child.numel() * child.element_size(),
                )
            )

    return fields


class MeshInfoFilter(Filter["Mesh"]):
    """Log mesh information and optionally write to a JSON-lines file.

    For each incoming mesh the filter extracts metadata (point/cell counts,
    field names, shapes, dtypes, memory usage) and logs it.  Optionally
    writes structured records to a JSON-lines file for post-analysis.

    The mesh is yielded unchanged so downstream filters and sinks receive
    the full data.

    Parameters
    ----------
    output : str or None
        Optional path for JSON-lines output file.  If provided, each mesh's
        metadata is appended as a JSON object on a new line.
    log_level : str
        Logging level for console output: ``"info"`` or ``"debug"``.
        Default is ``"info"``.
    include_fields : bool
        Whether to include detailed field information (names, shapes, dtypes).
        Default is ``True``.

    Examples
    --------
    Log mesh info to console only:

    >>> filt = MeshInfoFilter()
    >>> pipeline = source.filter(filt).write(sink)

    Log mesh info and write to JSON-lines file:

    >>> filt = MeshInfoFilter(output="mesh_info.jsonl", log_level="debug")
    >>> pipeline = source.filter(filt).write(sink)
    """

    name: ClassVar[str] = "Mesh Info Logger"
    description: ClassVar[str] = "Log mesh metadata and optionally write to JSON-lines file"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the mesh info filter.

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
        self._output_path = pathlib.Path(output) if output else None
        self._log_level = logging.INFO if log_level == "info" else logging.DEBUG
        self._include_fields = include_fields
        self._mesh_index = 0
        self._file_handle: IO[str] | None = None

    def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
        """Extract and log mesh info, then yield the mesh unchanged.

        Parameters
        ----------
        items : Generator[Mesh]
            Stream of incoming meshes.

        Yields
        ------
        Mesh
            The same mesh, unmodified.
        """
        for mesh in items:
            info = self._extract_mesh_info(mesh)
            self._log_info(info)
            self._write_to_file(info)
            self._mesh_index += 1
            yield mesh

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

    def _extract_mesh_info(self, mesh: Mesh) -> dict[str, object]:
        """Extract metadata from a mesh.

        Parameters
        ----------
        mesh : Mesh
            The input mesh.

        Returns
        -------
        dict[str, object]
            Mesh metadata dictionary.
        """
        info: dict[str, object] = {
            "mesh_index": self._mesh_index,
            "n_points": mesh.n_points,
            "n_cells": mesh.n_cells,
        }

        point_fields: list[_FieldInfo] = []
        cell_fields: list[_FieldInfo] = []
        total_memory = 0

        if mesh.point_data is not None:
            point_fields = _extract_field_info(mesh.point_data, prefix="")
            total_memory += sum(f["nbytes"] for f in point_fields)

        if mesh.cell_data is not None:
            cell_fields = _extract_field_info(mesh.cell_data, prefix="")
            total_memory += sum(f["nbytes"] for f in cell_fields)

        # Add points tensor memory
        if mesh.points is not None:
            total_memory += mesh.points.numel() * mesh.points.element_size()

        info["n_point_fields"] = len(point_fields)
        info["n_cell_fields"] = len(cell_fields)
        info["memory_estimate_bytes"] = total_memory

        if self._include_fields:
            info["point_data_fields"] = point_fields
            info["cell_data_fields"] = cell_fields

        return info

    def _log_info(self, info: dict[str, object]) -> None:
        """Log mesh info to the Python logger.

        Parameters
        ----------
        info : dict[str, object]
            Mesh metadata dictionary.
        """
        memory_mb = int(info.get("memory_estimate_bytes", 0)) / (1024 * 1024)  # ty: ignore[invalid-argument-type]
        msg = (
            f"Mesh {info['mesh_index']}: "
            f"{info['n_points']} points, {info['n_cells']} cells, "
            f"{info['n_point_fields']} point fields, {info['n_cell_fields']} cell fields, "
            f"{memory_mb:.2f} MB"
        )
        logger.log(self._log_level, msg)

        if self._include_fields and self._log_level <= logging.DEBUG:
            point_fields: list[_FieldInfo] = info.get("point_data_fields", [])  # ty: ignore[invalid-assignment]
            cell_fields: list[_FieldInfo] = info.get("cell_data_fields", [])  # ty: ignore[invalid-assignment]
            for fld in point_fields:
                logger.debug("  point_data/%s: shape=%s, dtype=%s", fld["name"], fld["shape"], fld["dtype"])
            for fld in cell_fields:
                logger.debug("  cell_data/%s: shape=%s, dtype=%s", fld["name"], fld["shape"], fld["dtype"])

    def _write_to_file(self, info: dict[str, object]) -> None:
        """Write mesh info to the JSON-lines file.

        Parameters
        ----------
        info : dict[str, object]
            Mesh metadata dictionary.
        """
        if self._output_path is None:
            return

        # Lazy open file on first write
        if self._file_handle is None:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = self._output_path.open("w")

        self._file_handle.write(json.dumps(info) + "\n")
        self._file_handle.flush()
