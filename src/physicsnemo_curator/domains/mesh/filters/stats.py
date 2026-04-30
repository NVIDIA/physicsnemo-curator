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

"""Comprehensive statistics filter for mesh pipelines.

Computes per-field statistics (mean, std, min, max, skewness, kurtosis, etc.)
for each mesh flowing through, accumulates results using numerically stable
Welford accumulators, and writes to a Parquet file.  The Welford state is
preserved to enable cross-file aggregation without re-reading raw data.

The mesh is yielded unchanged (pass-through).
"""

from __future__ import annotations

import math
import pathlib
from typing import TYPE_CHECKING, ClassVar, TypedDict

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from physicsnemo_curator.core.base import Filter, Param

if TYPE_CHECKING:
    from collections.abc import Generator

    from physicsnemo.mesh import Mesh


class _StatsRow(TypedDict, total=False):
    """Typed dictionary for a single statistics row.

    All numeric fields are ``float`` or ``int`` to avoid ``object``
    arithmetic errors in the type checker.
    """

    field_key: str
    component: int
    n_spatial: int
    n_components: int
    mean: float
    std: float
    var: float
    min: float
    max: float
    median: float
    abs_mean: float
    abs_max: float
    skewness: float
    kurtosis: float
    welford_n: int
    welford_mean: float
    welford_m2: float
    welford_m3: float
    welford_m4: float
    welford_abs_sum: float


# Schema for per-file statistics Parquet output.
# Includes both human-readable summary stats and Welford accumulator state
# for cross-file merging.
_STATS_SCHEMA = pa.schema(
    [
        # Identity
        ("field_key", pa.string()),
        ("component", pa.int32()),  # -1 for scalar, 0/1/2/... for vector components
        # Shape info
        ("n_spatial", pa.int64()),  # number of points/cells
        ("n_components", pa.int32()),
        # Summary statistics (human-readable)
        ("mean", pa.float64()),
        ("std", pa.float64()),
        ("var", pa.float64()),
        ("min", pa.float64()),
        ("max", pa.float64()),
        ("median", pa.float64()),
        ("abs_mean", pa.float64()),
        ("abs_max", pa.float64()),
        ("skewness", pa.float64()),
        ("kurtosis", pa.float64()),
        # Welford accumulator state (for cross-file merging)
        ("welford_n", pa.int64()),
        ("welford_mean", pa.float64()),
        ("welford_m2", pa.float64()),  # sum of squared deviations
        ("welford_m3", pa.float64()),  # 3rd central moment (for skewness)
        ("welford_m4", pa.float64()),  # 4th central moment (for kurtosis)
        ("welford_abs_sum", pa.float64()),  # sum of absolute values
    ]
)


def _extract_leaf_tensors(td: object, prefix: str = "") -> list[tuple[str, torch.Tensor]]:
    """Recursively extract leaf tensors from a TensorDict.

    Parameters
    ----------
    td : object
        A TensorDict (``point_data`` or ``cell_data``).
    prefix : str
        Dotted prefix for nested field names.

    Returns
    -------
    list[tuple[str, torch.Tensor]]
        List of (dotted_key, tensor) pairs.
    """
    from tensordict import TensorDictBase

    if not isinstance(td, TensorDictBase):
        return []

    results: list[tuple[str, torch.Tensor]] = []

    for key in td.keys():  # noqa: SIM118
        child = td[key]
        full_key = f"{prefix}{key}" if prefix else key

        if isinstance(child, TensorDictBase):
            results.extend(_extract_leaf_tensors(child, prefix=f"{full_key}/"))
        elif isinstance(child, torch.Tensor):
            results.append((full_key, child))

    return results


def _compute_component_stats(field_key: str, tensor: torch.Tensor) -> list[_StatsRow]:
    """Compute statistics for each component of a tensor.

    Parameters
    ----------
    field_key : str
        The field name/key.
    tensor : torch.Tensor
        The tensor to compute statistics for.  Shape is (n_spatial,) for
        scalars or (n_spatial, n_components) for vectors.

    Returns
    -------
    list[_StatsRow]
        One stats dict per component (or one dict for scalars).
    """
    if tensor.numel() == 0:
        return []

    # Convert to float64 for numerical stability
    tensor = tensor.to(torch.float64)

    # Handle shape: (n_spatial,) or (n_spatial, n_components)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(1)
        n_components = 1
        is_scalar = True
    else:
        n_components = tensor.shape[1]
        is_scalar = False

    n_spatial = tensor.shape[0]
    rows: list[_StatsRow] = []

    for comp_idx in range(n_components):
        comp_data = tensor[:, comp_idx]
        comp_data_flat = comp_data.flatten()

        # Basic statistics
        n = comp_data_flat.numel()
        mean_val: float = comp_data_flat.mean().item()
        var_val: float = comp_data_flat.var(unbiased=False).item()  # population variance
        std_val = math.sqrt(var_val)
        min_val: float = comp_data_flat.min().item()
        max_val: float = comp_data_flat.max().item()
        median_val: float = comp_data_flat.median().item()
        abs_data = comp_data_flat.abs()
        abs_mean_val: float = abs_data.mean().item()
        abs_max_val: float = abs_data.max().item()

        # Higher-order moments (skewness and kurtosis)
        # Using the formulas for population skewness and kurtosis
        centered = comp_data_flat - mean_val
        m2: float = (centered**2).sum().item()  # sum of squared deviations
        m3: float = (centered**3).sum().item()  # sum of cubed deviations
        m4: float = (centered**4).sum().item()  # sum of 4th power deviations

        # Skewness = E[(X-μ)³] / σ³
        skewness_val = (m3 / n) / (std_val**3) if std_val > 0 else 0.0

        # Kurtosis = E[(X-μ)⁴] / σ⁴ - 3 (excess kurtosis)
        kurtosis_val = (m4 / n) / (std_val**4) - 3.0 if std_val > 0 else 0.0

        rows.append(
            _StatsRow(
                field_key=field_key,
                component=-1 if is_scalar else comp_idx,
                n_spatial=n_spatial,
                n_components=1 if is_scalar else n_components,
                mean=mean_val,
                std=std_val,
                var=var_val,
                min=min_val,
                max=max_val,
                median=median_val,
                abs_mean=abs_mean_val,
                abs_max=abs_max_val,
                skewness=skewness_val,
                kurtosis=kurtosis_val,
                welford_n=n,
                welford_mean=mean_val,
                welford_m2=m2,
                welford_m3=m3,
                welford_m4=m4,
                welford_abs_sum=abs_data.sum().item(),
            )
        )

    return rows


class StatsFilter(Filter["Mesh"]):
    """Compute comprehensive per-field statistics and accumulate to Parquet.

    For each incoming mesh the filter computes statistics for every tensor
    stored in ``point_data`` and ``cell_data``.  Statistics include mean,
    std, variance, min, max, median, skewness, kurtosis, and absolute value
    metrics.

    The Welford accumulator state is also stored to enable exact cross-file
    aggregation without re-reading raw tensor data.

    The mesh is yielded unchanged so downstream filters and sinks receive
    the full data.

    Parameters
    ----------
    output : str
        File path for the output Parquet file.
    per_component : bool
        Whether to compute separate statistics for each vector component.
        Default is ``True``.

    Examples
    --------
    >>> filt = StatsFilter(output="stats.parquet")
    >>> pipeline = source.filter(filt).write(sink)
    >>> for i in range(len(pipeline)):
    ...     pipeline[i]
    >>> filt.flush()  # write accumulated stats
    """

    name: ClassVar[str] = "Comprehensive Statistics"
    description: ClassVar[str] = (
        "Compute per-field statistics (mean, std, skewness, kurtosis, etc.) and save to Parquet"
    )

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the stats filter.

        Returns
        -------
        list[Param]
            The ``output`` and ``per_component`` parameters.
        """
        return [
            Param(name="output", description="Output Parquet file path for statistics", type=str),
            Param(
                name="per_component",
                description="Compute separate statistics for each vector component",
                type=bool,
                default=True,
            ),
        ]

    def __init__(self, output: str, per_component: bool = True) -> None:
        self._output_path = pathlib.Path(output)
        self._per_component = per_component
        self._rows: list[_StatsRow] = []
        self._last_artifacts: list[str] = []

    def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
        """Compute statistics for each mesh and yield it unchanged.

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
            mesh_rows = self._compute_mesh_stats(mesh)
            self._rows.extend(mesh_rows)
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

        # Build PyArrow table from accumulated rows
        columns: dict[str, list[object]] = {field.name: [] for field in _STATS_SCHEMA}

        for row in self._rows:
            for col_name in columns:
                columns[col_name].append(row.get(col_name))  # type: ignore[arg-type]

        new_table = pa.table(columns, schema=_STATS_SCHEMA)
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
        """Merge statistics Parquet files produced by parallel workers.

        Uses the parallel Welford algorithm (Chan et al., 1979) to combine
        per-worker statistics into exact aggregate statistics without
        re-reading the raw tensor data.  This is a convenience wrapper
        around :func:`merge_welford_stats` that also writes the result.

        Parameters
        ----------
        parquet_paths : list[str]
            Paths to per-worker statistics Parquet files.
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

        Notes
        -----
        Median and absolute-max cannot be recovered from Welford state
        alone, so the merged table will contain ``NaN`` for those columns.

        Examples
        --------
        >>> paths = ["worker_0/stats.parquet", "worker_1/stats.parquet"]
        >>> StatsFilter.merge(paths, output="merged_stats.parquet")  # doctest: +SKIP
        'merged_stats.parquet'
        """
        if not parquet_paths:
            msg = "parquet_paths must be a non-empty list."
            raise ValueError(msg)

        merged = merge_welford_stats(parquet_paths)

        out_path = pathlib.Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(merged, str(out_path))
        return str(out_path)

    def _compute_mesh_stats(self, mesh: Mesh) -> list[_StatsRow]:
        """Compute statistics for all fields in a mesh or domain mesh.

        Parameters
        ----------
        mesh : Mesh or DomainMesh
            The input mesh.

        Returns
        -------
        list[_StatsRow]
            List of stats rows (one per field/component).
        """
        from physicsnemo.mesh.domain_mesh import DomainMesh as _DomainMesh

        if isinstance(mesh, _DomainMesh):
            return self._compute_domain_mesh_stats(mesh)

        rows: list[_StatsRow] = []

        # Process point_data fields
        if mesh.point_data is not None:
            for key, tensor in _extract_leaf_tensors(mesh.point_data):
                field_key = f"point_data/{key}"
                rows.extend(_compute_component_stats(field_key, tensor))

        # Process cell_data fields
        if mesh.cell_data is not None:
            for key, tensor in _extract_leaf_tensors(mesh.cell_data):
                field_key = f"cell_data/{key}"
                rows.extend(_compute_component_stats(field_key, tensor))

        return rows

    def _compute_domain_mesh_stats(self, domain_mesh: object) -> list[_StatsRow]:
        """Compute statistics for all fields in a DomainMesh.

        Aggregates statistics from interior and all boundary sub-meshes.

        Parameters
        ----------
        domain_mesh : DomainMesh
            The input domain mesh.

        Returns
        -------
        list[_StatsRow]
            List of stats rows (one per field/component).
        """
        rows: list[_StatsRow] = []

        interior = domain_mesh.interior  # ty: ignore[unresolved-attribute]
        if interior.point_data is not None:
            for key, tensor in _extract_leaf_tensors(interior.point_data):
                field_key = f"interior/point_data/{key}"
                rows.extend(_compute_component_stats(field_key, tensor))

        if interior.cell_data is not None:
            for key, tensor in _extract_leaf_tensors(interior.cell_data):
                field_key = f"interior/cell_data/{key}"
                rows.extend(_compute_component_stats(field_key, tensor))

        boundaries = domain_mesh.boundaries  # ty: ignore[unresolved-attribute]
        if boundaries is not None:
            for bnd_name in boundaries.keys():  # noqa: SIM118 - TensorDict needs .keys()
                boundary = boundaries[bnd_name]
                if boundary.point_data is not None:
                    for key, tensor in _extract_leaf_tensors(boundary.point_data):
                        field_key = f"boundary.{bnd_name}/point_data/{key}"
                        rows.extend(_compute_component_stats(field_key, tensor))
                if boundary.cell_data is not None:
                    for key, tensor in _extract_leaf_tensors(boundary.cell_data):
                        field_key = f"boundary.{bnd_name}/cell_data/{key}"
                        rows.extend(_compute_component_stats(field_key, tensor))

        return rows


def merge_welford_stats(parquet_paths: list[str]) -> pa.Table:
    """Merge Welford statistics from multiple Parquet files.

    Uses the parallel Welford algorithm (Chan et al., 1979) to combine
    per-file statistics into exact aggregate statistics without re-reading
    the raw tensor data.

    Parameters
    ----------
    parquet_paths : list[str]
        Paths to per-file statistics Parquet files.

    Returns
    -------
    pa.Table
        Aggregated statistics table with the same schema.

    Examples
    --------
    >>> paths = ["mesh_0_stats.parquet", "mesh_1_stats.parquet"]
    >>> merged = merge_welford_stats(paths)
    >>> pq.write_table(merged, "aggregate_stats.parquet")
    """
    # Group rows by (field_key, component)
    groups: dict[tuple[str, int], list[_StatsRow]] = {}

    for path in parquet_paths:
        table = pq.read_table(path)
        for i in range(table.num_rows):
            row = _StatsRow(**{col: table[col][i].as_py() for col in table.column_names})
            key = (row["field_key"], row["component"])
            if key not in groups:
                groups[key] = []
            groups[key].append(row)

    # Merge each group using parallel Welford algorithm
    merged_rows: list[_StatsRow] = []

    for (field_key, component), rows in groups.items():
        merged = _merge_welford_group(rows)
        merged["field_key"] = field_key
        merged["component"] = component
        merged_rows.append(merged)

    # Build output table
    columns: dict[str, list[object]] = {field.name: [] for field in _STATS_SCHEMA}

    for row in merged_rows:
        for col_name in columns:
            columns[col_name].append(row.get(col_name))  # type: ignore[arg-type]

    return pa.table(columns, schema=_STATS_SCHEMA)


def _merge_welford_group(rows: list[_StatsRow]) -> _StatsRow:
    """Merge Welford statistics for a single field/component group.

    Parameters
    ----------
    rows : list[_StatsRow]
        Per-file statistics rows for the same field/component.

    Returns
    -------
    _StatsRow
        Merged statistics.
    """
    if len(rows) == 1:
        return _StatsRow(**rows[0])

    # Initialize with first row
    n_total: float = float(rows[0]["welford_n"])
    mean_total: float = float(rows[0]["welford_mean"])
    m2_total: float = float(rows[0]["welford_m2"])
    m3_total: float = float(rows[0]["welford_m3"])
    m4_total: float = float(rows[0]["welford_m4"])
    abs_sum_total: float = float(rows[0]["welford_abs_sum"])
    min_total: float = float(rows[0]["min"])
    max_total: float = float(rows[0]["max"])
    n_spatial_total: int = int(rows[0]["n_spatial"])
    n_components: int = int(rows[0]["n_components"])

    # Merge remaining rows using Chan's parallel algorithm
    for row in rows[1:]:
        n_b: float = float(row["welford_n"])
        mean_b: float = float(row["welford_mean"])
        m2_b: float = float(row["welford_m2"])
        m3_b: float = float(row["welford_m3"])
        m4_b: float = float(row["welford_m4"])

        n_ab = n_total + n_b
        delta = mean_b - mean_total
        delta2 = delta * delta
        delta3 = delta * delta2
        delta4 = delta2 * delta2

        # Update mean
        mean_ab = (n_total * mean_total + n_b * mean_b) / n_ab

        # Update M2 (for variance)
        m2_ab = m2_total + m2_b + delta2 * n_total * n_b / n_ab

        # Update M3 (for skewness) - using Chan's formula
        m3_ab = (
            m3_total
            + m3_b
            + delta3 * n_total * n_b * (n_total - n_b) / (n_ab * n_ab)
            + 3 * delta * (n_total * m2_b - n_b * m2_total) / n_ab
        )

        # Update M4 (for kurtosis) - using Chan's formula
        m4_ab = (
            m4_total
            + m4_b
            + delta4 * n_total * n_b * (n_total * n_total - n_total * n_b + n_b * n_b) / (n_ab * n_ab * n_ab)
            + 6 * delta2 * (n_total * n_total * m2_b + n_b * n_b * m2_total) / (n_ab * n_ab)
            + 4 * delta * (n_total * m3_b - n_b * m3_total) / n_ab
        )

        n_total = n_ab
        mean_total = mean_ab
        m2_total = m2_ab
        m3_total = m3_ab
        m4_total = m4_ab
        abs_sum_total += float(row["welford_abs_sum"])
        min_total = min(min_total, float(row["min"]))
        max_total = max(max_total, float(row["max"]))
        n_spatial_total += int(row["n_spatial"])

    # Compute final statistics from Welford state
    var_total = m2_total / n_total if n_total > 0 else 0.0
    std_total = math.sqrt(var_total)

    if std_total > 0:
        skewness_total = (m3_total / n_total) / (std_total**3)
        kurtosis_total = (m4_total / n_total) / (std_total**4) - 3.0
    else:
        skewness_total = 0.0
        kurtosis_total = 0.0

    abs_mean_total = abs_sum_total / n_total if n_total > 0 else 0.0

    return _StatsRow(
        n_spatial=n_spatial_total,
        n_components=n_components,
        mean=mean_total,
        std=std_total,
        var=var_total,
        min=min_total,
        max=max_total,
        median=float("nan"),  # Cannot compute median from Welford state
        abs_mean=abs_mean_total,
        abs_max=float("nan"),  # Cannot compute abs_max from Welford state
        skewness=skewness_total,
        kurtosis=kurtosis_total,
        welford_n=int(n_total),
        welford_mean=mean_total,
        welford_m2=m2_total,
        welford_m3=m3_total,
        welford_m4=m4_total,
        welford_abs_sum=abs_sum_total,
    )
