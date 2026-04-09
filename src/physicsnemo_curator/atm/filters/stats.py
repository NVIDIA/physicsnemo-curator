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

"""Comprehensive statistics filter for atomic/molecular pipelines.

Computes per-field statistics (mean, std, min, max, skewness, kurtosis, etc.)
for each :class:`~nvalchemi.data.AtomicData` flowing through, accumulates
results using numerically stable Welford accumulators, and writes them to a
Parquet file.  The Welford state is preserved to enable cross-worker
aggregation without re-reading raw data.

The item is yielded unchanged (pass-through).
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

    from nvalchemi.data import AtomicData


class _StatsRow(TypedDict, total=False):
    """Typed dictionary for a single statistics row.

    All numeric fields use ``float`` or ``int`` to avoid ``object``
    arithmetic errors in the type checker.
    """

    field_key: str
    level: str
    component: int
    n_values: int
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


# Schema for the statistics Parquet output.
_STATS_SCHEMA = pa.schema(
    [
        # Identity
        ("field_key", pa.string()),
        ("level", pa.string()),  # "node", "edge", or "system"
        ("component", pa.int32()),  # -1 for scalar, 0/1/2/... for vector
        # Shape info
        ("n_values", pa.int64()),  # number of values seen
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
        ("welford_m2", pa.float64()),
        ("welford_m3", pa.float64()),
        ("welford_m4", pa.float64()),
        ("welford_abs_sum", pa.float64()),
    ]
)

# AtomicData tensor fields grouped by their semantic level.
# Node-level fields have shape [n_nodes, ...].
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

# Edge-level fields have shape [n_edges, ...].
_EDGE_FIELDS: tuple[str, ...] = (
    "edge_index",
    "shifts",
    "unit_shifts",
    "edge_embeddings",
)

# System-level fields have shape [B, ...] or scalar.
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


def _compute_component_stats(field_key: str, level: str, tensor: torch.Tensor) -> list[_StatsRow]:
    """Compute statistics for each component of a tensor.

    Parameters
    ----------
    field_key : str
        The field name/key.
    level : str
        Semantic level (``"node"``, ``"edge"``, or ``"system"``).
    tensor : torch.Tensor
        The tensor to compute statistics for.

    Returns
    -------
    list[_StatsRow]
        One stats dict per component (or one for scalars).
    """
    if tensor.numel() == 0:
        return []

    # Flatten to 2-D: (n_values, n_components)
    tensor = tensor.to(torch.float64)

    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0).unsqueeze(1)
    elif tensor.ndim == 1:
        tensor = tensor.unsqueeze(1)
    elif tensor.ndim > 2:
        # Collapse all trailing dims into a single component axis.
        tensor = tensor.reshape(tensor.shape[0], -1)

    n_values = tensor.shape[0]
    n_components = tensor.shape[1]
    is_scalar = n_components == 1
    rows: list[_StatsRow] = []

    for comp_idx in range(n_components):
        comp_data = tensor[:, comp_idx].flatten()
        n = comp_data.numel()

        mean_val: float = comp_data.mean().item()
        var_val: float = comp_data.var(unbiased=False).item()
        std_val = math.sqrt(var_val)
        min_val: float = comp_data.min().item()
        max_val: float = comp_data.max().item()
        median_val: float = comp_data.median().item()
        abs_data = comp_data.abs()
        abs_mean_val: float = abs_data.mean().item()
        abs_max_val: float = abs_data.max().item()

        # Higher-order moments via population formulas.
        centered = comp_data - mean_val
        m2: float = (centered**2).sum().item()
        m3: float = (centered**3).sum().item()
        m4: float = (centered**4).sum().item()

        skewness_val = (m3 / n) / (std_val**3) if std_val > 0 else 0.0
        kurtosis_val = (m4 / n) / (std_val**4) - 3.0 if std_val > 0 else 0.0

        rows.append(
            _StatsRow(
                field_key=field_key,
                level=level,
                component=-1 if is_scalar else comp_idx,
                n_values=n_values,
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


def _extract_atomic_fields(data: AtomicData) -> list[tuple[str, str, torch.Tensor]]:
    """Extract tensor fields from an AtomicData object.

    Parameters
    ----------
    data : AtomicData
        The atomic data object.

    Returns
    -------
    list[tuple[str, str, torch.Tensor]]
        Triples of ``(field_name, level, tensor)``.
    """
    fields: list[tuple[str, str, torch.Tensor]] = []

    for name in _NODE_FIELDS:
        val = getattr(data, name, None)
        if val is not None and isinstance(val, torch.Tensor):
            fields.append((name, "node", val))

    for name in _EDGE_FIELDS:
        val = getattr(data, name, None)
        if val is not None and isinstance(val, torch.Tensor):
            fields.append((name, "edge", val))

    for name in _SYSTEM_FIELDS:
        val = getattr(data, name, None)
        if val is not None and isinstance(val, torch.Tensor):
            fields.append((name, "system", val))

    # Also pick up extra_data fields (dynamically set via Pydantic extra="allow").
    extra_data: dict[str, object] | None = getattr(data, "extra_data", None)
    if extra_data:
        for name, val in extra_data.items():
            if isinstance(val, torch.Tensor):
                fields.append((f"extra/{name}", "extra", val))

    return fields


class AtomicStatsFilter(Filter["AtomicData"]):
    """Compute comprehensive per-field statistics for atomic data.

    For each incoming :class:`~nvalchemi.data.AtomicData` the filter
    computes statistics for every tensor field (positions, forces, energies,
    etc.) at the node, edge, and system level.  Statistics include mean,
    std, variance, min, max, median, skewness, kurtosis, and absolute value
    metrics.

    The Welford accumulator state is also stored to enable exact cross-worker
    aggregation without re-reading raw tensor data.

    The item is yielded unchanged so downstream filters and sinks receive
    the full data.

    Parameters
    ----------
    output : str
        File path for the output Parquet file.

    Examples
    --------
    >>> filt = AtomicStatsFilter(output="stats.parquet")  # doctest: +SKIP
    >>> pipeline = source.filter(filt).write(sink)  # doctest: +SKIP
    >>> for i in range(len(pipeline)):
    ...     pipeline[i]
    >>> filt.flush()  # doctest: +SKIP
    """

    name: ClassVar[str] = "Atomic Statistics"
    description: ClassVar[str] = (
        "Compute per-field statistics (mean, std, skewness, kurtosis, etc.) for AtomicData and save to Parquet"
    )

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the atomic stats filter.

        Returns
        -------
        list[Param]
            The ``output`` parameter.
        """
        return [
            Param(name="output", description="Output Parquet file path for statistics", type=str),
        ]

    def __init__(self, output: str) -> None:
        self._output_path = pathlib.Path(output)
        self._rows: list[_StatsRow] = []
        self._last_artifacts: list[str] = []

    def __call__(self, items: Generator[AtomicData]) -> Generator[AtomicData]:
        """Compute statistics for each item and yield it unchanged.

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
            rows = self._compute_stats(data)
            self._rows.extend(rows)
            yield data

    def flush(self) -> str | None:
        """Write accumulated statistics to the Parquet file.

        Returns
        -------
        str or None
            The path of the written Parquet file, or ``None`` if there
            are no rows to write.
        """
        if not self._rows:
            return None

        columns: dict[str, list[object]] = {field.name: [] for field in _STATS_SCHEMA}

        for row in self._rows:
            for col_name in columns:
                columns[col_name].append(row.get(col_name))  # type: ignore[arg-type]

        table = pa.table(columns, schema=_STATS_SCHEMA)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, str(self._output_path))
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
        re-reading the raw tensor data.

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
        """
        if not parquet_paths:
            msg = "parquet_paths must be a non-empty list."
            raise ValueError(msg)

        merged = merge_welford_stats(parquet_paths)

        out_path = pathlib.Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(merged, str(out_path))
        return str(out_path)

    def _compute_stats(self, data: AtomicData) -> list[_StatsRow]:
        """Compute statistics for all tensor fields in an AtomicData.

        Parameters
        ----------
        data : AtomicData
            The input atomic data.

        Returns
        -------
        list[_StatsRow]
            List of stats rows (one per field/component).
        """
        rows: list[_StatsRow] = []

        for field_name, level, tensor in _extract_atomic_fields(data):
            rows.extend(_compute_component_stats(field_name, level, tensor))

        return rows

    # -- Properties -----------------------------------------------------------

    @property
    def output_path(self) -> pathlib.Path:
        """Return the output Parquet path."""
        return self._output_path


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
    >>> paths = ["shard_0_stats.parquet", "shard_1_stats.parquet"]
    >>> merged = merge_welford_stats(paths)  # doctest: +SKIP
    """
    groups: dict[tuple[str, int], list[_StatsRow]] = {}

    for path in parquet_paths:
        table = pq.read_table(path)
        for i in range(table.num_rows):
            row = _StatsRow(**{col: table[col][i].as_py() for col in table.column_names})
            key = (row["field_key"], row["component"])
            if key not in groups:
                groups[key] = []
            groups[key].append(row)

    merged_rows: list[_StatsRow] = []

    for (_field_key, _component), rows in groups.items():
        merged = _merge_welford_group(rows)
        merged["field_key"] = _field_key
        merged["component"] = _component
        merged_rows.append(merged)

    columns: dict[str, list[object]] = {field.name: [] for field in _STATS_SCHEMA}

    for row in merged_rows:
        for col_name in columns:
            columns[col_name].append(row.get(col_name))  # type: ignore[arg-type]

    return pa.table(columns, schema=_STATS_SCHEMA)


def _merge_welford_group(rows: list[_StatsRow]) -> _StatsRow:
    """Merge Welford statistics for a single field/component group.

    Implements the parallel combination rules from Chan, Golub & LeVeque
    (1979) for M2, M3, and M4.

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

    n_total: float = float(rows[0]["welford_n"])
    mean_total: float = float(rows[0]["welford_mean"])
    m2_total: float = float(rows[0]["welford_m2"])
    m3_total: float = float(rows[0]["welford_m3"])
    m4_total: float = float(rows[0]["welford_m4"])
    abs_sum_total: float = float(rows[0]["welford_abs_sum"])
    min_total: float = float(rows[0]["min"])
    max_total: float = float(rows[0]["max"])
    n_values_total: int = int(rows[0]["n_values"])
    n_components: int = int(rows[0]["n_components"])
    level: str = str(rows[0]["level"])

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

        mean_ab = (n_total * mean_total + n_b * mean_b) / n_ab
        m2_ab = m2_total + m2_b + delta2 * n_total * n_b / n_ab
        m3_ab = (
            m3_total
            + m3_b
            + delta3 * n_total * n_b * (n_total - n_b) / (n_ab * n_ab)
            + 3 * delta * (n_total * m2_b - n_b * m2_total) / n_ab
        )
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
        n_values_total += int(row["n_values"])

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
        level=level,
        n_values=n_values_total,
        n_components=n_components,
        mean=mean_total,
        std=std_total,
        var=var_total,
        min=min_total,
        max=max_total,
        median=float("nan"),
        abs_mean=abs_mean_total,
        abs_max=float("nan"),
        skewness=skewness_total,
        kurtosis=kurtosis_total,
        welford_n=int(n_total),
        welford_mean=mean_total,
        welford_m2=m2_total,
        welford_m3=m3_total,
        welford_m4=m4_total,
        welford_abs_sum=abs_sum_total,
    )
