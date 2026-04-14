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

"""Mean statistics filter for mesh pipelines.

Computes the spatial mean of every field in ``point_data`` and ``cell_data``
for each mesh flowing through, appends a summary row to an in-memory table,
and writes the accumulated statistics to a Parquet file.  The mesh itself
is yielded unchanged (pass-through).
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, ClassVar

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from physicsnemo_curator.core.base import Filter, Param

if TYPE_CHECKING:
    from collections.abc import Generator

    from physicsnemo.mesh import Mesh


class MeanFilter(Filter["Mesh"]):
    """Compute per-field spatial means and accumulate to a Parquet table.

    For each incoming mesh the filter computes the arithmetic mean of every
    tensor stored in ``point_data`` and ``cell_data``.  The results are
    accumulated in memory and flushed to a Parquet file when :meth:`flush`
    is called (or automatically via the CLI after the pipeline finishes).

    The mesh is yielded unchanged so downstream filters and sinks still
    receive the full data.

    Parameters
    ----------
    output : str
        File path for the output Parquet file.

    Examples
    --------
    >>> filt = MeanFilter(output="stats.parquet")
    >>> pipeline = source.filter(filt).write(sink)
    >>> for i in range(len(pipeline)):
    ...     pipeline[i]
    >>> filt.flush()  # write accumulated stats
    """

    name: ClassVar[str] = "Mean Statistics"
    description: ClassVar[str] = "Compute spatial means of mesh fields and save summary to Parquet"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the mean filter.

        Returns
        -------
        list[Param]
            The ``output`` parameter (required Parquet path).
        """
        return [
            Param(name="output", description="Output Parquet file path for statistics", type=str),
        ]

    def __init__(self, output: str) -> None:
        self._output_path = pathlib.Path(output)
        self._rows: list[dict[str, object]] = []
        self._last_artifacts: list[str] = []

    def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
        """Compute means for each mesh and yield it unchanged.

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
            row = self._compute_means(mesh)
            self._rows.append(row)
            yield mesh

    def flush(self) -> str | None:
        """Write accumulated statistics to the Parquet file.

        If the output file already exists (e.g., from a previous flush in
        the same worker process), the new rows are appended to the existing
        file. This enables worker-level aggregation when running pipelines
        in parallel.

        Returns
        -------
        str or None
            The path of the written Parquet file, or ``None`` if there are
            no rows to write.
        """
        if not self._rows:
            return None

        # Build a PyArrow table from the collected rows.
        # Each row may have different keys (meshes with different field names),
        # so we union all column names and fill missing values with None.
        all_columns: dict[str, list[object]] = {}
        for row in self._rows:
            for key in row:
                if key not in all_columns:
                    all_columns[key] = [None] * (len(self._rows))
        for i, row in enumerate(self._rows):
            for key in all_columns:
                all_columns[key][i] = row.get(key)

        new_table = pa.table(all_columns)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        # Append to existing file if it exists (worker-level aggregation)
        if self._output_path.exists():
            existing_table = pq.read_table(str(self._output_path))
            combined_table = pa.concat_tables([existing_table, new_table])
            pq.write_table(combined_table, str(self._output_path))
        else:
            pq.write_table(new_table, str(self._output_path))

        path = str(self._output_path)
        self._rows.clear()
        self._last_artifacts = [path]
        return path

    def artifacts(self) -> list[str]:
        """Return paths written by the last :meth:`flush` call.

        Returns
        -------
        list[str]
            Paths of files written since the last call, or ``[]``.
        """
        paths = self._last_artifacts
        self._last_artifacts = []
        return paths

    @staticmethod
    def merge(parquet_paths: list[str], output: str) -> str:
        """Merge mean-statistics Parquet files produced by parallel workers.

        Each worker writes one row per mesh.  This method concatenates all
        rows into a single Parquet file, preserving column order.  The
        resulting table has a contiguous row range (0 … N-1).

        Parameters
        ----------
        parquet_paths : list[str]
            Paths to per-worker mean-statistics Parquet files.
        output : str
            Path for the merged output Parquet file.

        Returns
        -------
        str
            The path of the written merged Parquet file.

        Raises
        ------
        ValueError
            If *parquet_paths* is empty.

        Examples
        --------
        >>> paths = ["worker_0/means.parquet", "worker_1/means.parquet"]
        >>> MeanFilter.merge(paths, output="merged_means.parquet")  # doctest: +SKIP
        'merged_means.parquet'
        """
        if not parquet_paths:
            msg = "parquet_paths must be a non-empty list."
            raise ValueError(msg)

        tables = [pq.read_table(p) for p in parquet_paths]
        merged = pa.concat_tables(tables, promote_options="default")

        out_path = pathlib.Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(merged, str(out_path))
        return str(out_path)

    # -- Internal helpers ----------------------------------------------------

    @staticmethod
    def _compute_means(mesh: Mesh) -> dict[str, object]:
        """Compute the mean of every field in point_data and cell_data.

        Parameters
        ----------
        mesh : Mesh
            The input mesh.

        Returns
        -------
        dict[str, object]
            Mapping of ``"{data_type}/{field_name}"`` to the scalar mean
            value.  Also includes ``n_points`` and ``n_cells`` counts.
        """
        row: dict[str, object] = {
            "n_points": mesh.n_points,
            "n_cells": mesh.n_cells,
        }

        # Process point_data fields.
        # NOTE: TensorDict.__iter__ iterates over the batch dimension (points/cells),
        # not field names.  We must use .keys() explicitly.
        if mesh.point_data is not None:
            for key in mesh.point_data.keys():  # noqa: SIM118
                tensor = mesh.point_data[key]
                if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
                    mean_val = tensor.float().mean().item()
                    row[f"point_data/{key}"] = mean_val

        # Process cell_data fields.
        if mesh.cell_data is not None:
            for key in mesh.cell_data.keys():  # noqa: SIM118
                tensor = mesh.cell_data[key]
                if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
                    mean_val = tensor.float().mean().item()
                    row[f"cell_data/{key}"] = mean_val

        return row
