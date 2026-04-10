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

"""Mesh quality filter for data integrity and cell geometry assessment.

Computes mesh quality metrics in a single pass and writes a Parquet report.
Two categories of checks are supported:

* **NaN/Inf detection** — counts non-finite values in every tensor field of
  ``point_data`` and ``cell_data``.
* **Cell geometry** — computes aspect ratio, equiangle skewness, and minimum
  interior angle for each cell (triangles and quads only).

The mesh is yielded unchanged (pass-through).  For parallel backends the
:meth:`merge` static method concatenates per-worker Parquet files.
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


# ---------------------------------------------------------------------------
# Row type for the quality report
# ---------------------------------------------------------------------------


class _QualityRow(TypedDict, total=False):
    """Typed dictionary for a single quality report row."""

    # Mesh identity (one row per mesh)
    mesh_index: int
    n_points: int
    n_cells: int

    # NaN/Inf counts
    nan_point_data_total: int
    inf_point_data_total: int
    nan_cell_data_total: int
    inf_cell_data_total: int
    nan_field_details: str  # JSON: {"field_name": {"nan": N, "inf": N}, ...}

    # Cell geometry (per-mesh aggregates)
    geom_min_aspect_ratio: float
    geom_max_aspect_ratio: float
    geom_mean_aspect_ratio: float
    geom_min_skewness: float
    geom_max_skewness: float
    geom_mean_skewness: float
    geom_min_angle_deg: float
    geom_max_angle_deg: float
    geom_mean_min_angle_deg: float
    geom_n_degenerate_cells: int


# Parquet schema for the quality report.
_QUALITY_SCHEMA = pa.schema(
    [
        ("mesh_index", pa.int64()),
        ("n_points", pa.int64()),
        ("n_cells", pa.int64()),
        # NaN/Inf
        ("nan_point_data_total", pa.int64()),
        ("inf_point_data_total", pa.int64()),
        ("nan_cell_data_total", pa.int64()),
        ("inf_cell_data_total", pa.int64()),
        ("nan_field_details", pa.string()),
        # Cell geometry
        ("geom_min_aspect_ratio", pa.float64()),
        ("geom_max_aspect_ratio", pa.float64()),
        ("geom_mean_aspect_ratio", pa.float64()),
        ("geom_min_skewness", pa.float64()),
        ("geom_max_skewness", pa.float64()),
        ("geom_mean_skewness", pa.float64()),
        ("geom_min_angle_deg", pa.float64()),
        ("geom_max_angle_deg", pa.float64()),
        ("geom_mean_min_angle_deg", pa.float64()),
        ("geom_n_degenerate_cells", pa.int64()),
    ]
)


# ---------------------------------------------------------------------------
# Geometry helpers (pure torch, no mesh dependency)
# ---------------------------------------------------------------------------


def _triangle_angles(pts: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """Compute all interior angles of triangular cells.

    Parameters
    ----------
    pts : torch.Tensor
        Vertex coordinates, shape ``(n_points, D)``.
    cells : torch.Tensor
        Triangle connectivity, shape ``(n_cells, 3)``.

    Returns
    -------
    torch.Tensor
        Angles in radians, shape ``(n_cells, 3)``.
    """
    v0 = pts[cells[:, 0]]
    v1 = pts[cells[:, 1]]
    v2 = pts[cells[:, 2]]

    e01 = v1 - v0
    e02 = v2 - v0
    e12 = v2 - v1

    def _angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        cos = (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-30)
        return torch.acos(cos.clamp(-1.0, 1.0))

    a0 = _angle(e01, e02)
    a1 = _angle(-e01, e12)
    a2 = _angle(-e02, -e12)
    return torch.stack([a0, a1, a2], dim=-1)


def _quad_angles(pts: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """Compute all interior angles of quadrilateral cells.

    Parameters
    ----------
    pts : torch.Tensor
        Vertex coordinates, shape ``(n_points, D)``.
    cells : torch.Tensor
        Quad connectivity, shape ``(n_cells, 4)``, vertices in order.

    Returns
    -------
    torch.Tensor
        Angles in radians, shape ``(n_cells, 4)``.
    """
    v = [pts[cells[:, i]] for i in range(4)]

    def _angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        cos = (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-30)
        return torch.acos(cos.clamp(-1.0, 1.0))

    # Edges around the quad: 0→1, 1→2, 2→3, 3→0
    angles = [
        _angle(v[1] - v[0], v[3] - v[0]),
        _angle(v[0] - v[1], v[2] - v[1]),
        _angle(v[1] - v[2], v[3] - v[2]),
        _angle(v[2] - v[3], v[0] - v[3]),
    ]
    return torch.stack(angles, dim=-1)


def _aspect_ratios(pts: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """Compute aspect ratio for triangular cells.

    The aspect ratio is ``longest_edge / shortest_edge``.  A perfect
    equilateral triangle has aspect ratio 1.0.

    Parameters
    ----------
    pts : torch.Tensor
        Vertex coordinates, shape ``(n_points, D)``.
    cells : torch.Tensor
        Triangle connectivity, shape ``(n_cells, 3)``.

    Returns
    -------
    torch.Tensor
        Aspect ratios, shape ``(n_cells,)``.
    """
    v0 = pts[cells[:, 0]]
    v1 = pts[cells[:, 1]]
    v2 = pts[cells[:, 2]]

    e0 = (v1 - v0).norm(dim=-1)
    e1 = (v2 - v1).norm(dim=-1)
    e2 = (v0 - v2).norm(dim=-1)

    edges = torch.stack([e0, e1, e2], dim=-1)
    longest = edges.max(dim=-1).values
    shortest = edges.min(dim=-1).values
    return longest / (shortest + 1e-30)


def _equiangle_skewness(angles: torch.Tensor, ideal_angle: float) -> torch.Tensor:
    """Compute equiangle skewness from interior angles.

    Skewness is defined as ``max((theta_max - theta_ideal), (theta_ideal - theta_min)) / (pi - theta_ideal)``
    where ``theta_ideal`` is 60° for triangles and 90° for quads.

    Parameters
    ----------
    angles : torch.Tensor
        Interior angles in radians, shape ``(n_cells, n_verts)``.
    ideal_angle : float
        Ideal angle in radians (π/3 for tri, π/2 for quad).

    Returns
    -------
    torch.Tensor
        Skewness values in ``[0, 1]``, shape ``(n_cells,)``.
        0 = perfect, 1 = degenerate.
    """
    theta_max = angles.max(dim=-1).values
    theta_min = angles.min(dim=-1).values
    denom = math.pi - ideal_angle
    skew = torch.maximum(theta_max - ideal_angle, ideal_angle - theta_min) / denom
    return skew.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# MeshQualityFilter
# ---------------------------------------------------------------------------


class MeshQualityFilter(Filter["Mesh"]):
    """Assess mesh data integrity and cell geometry quality.

    This filter inspects every mesh flowing through the pipeline and
    produces a quality report as a Parquet file.  Two categories of
    checks are supported:

    * **NaN/Inf detection** (``check_nan=True``) — counts non-finite values
      in every tensor field of ``point_data`` and ``cell_data``.
    * **Cell geometry** (``check_geometry=True``) — computes aspect ratio,
      equiangle skewness, and minimum interior angle for each cell.
      Currently supports triangles (3-node) and quads (4-node).

    The mesh is yielded unchanged (pass-through) so downstream filters
    and sinks receive the full data.

    Parameters
    ----------
    output : str
        File path for the output Parquet quality report.
    check_nan : bool
        Enable NaN/Inf detection in field data.  Default is ``True``.
    check_geometry : bool
        Enable cell geometry quality metrics.  Default is ``True``.

    Examples
    --------
    >>> filt = MeshQualityFilter(output="quality.parquet")
    >>> pipeline = source.filter(filt).write(sink)
    >>> for i in range(len(pipeline)):
    ...     pipeline[i]
    >>> filt.flush()  # write quality report
    """

    name: ClassVar[str] = "Mesh Quality"
    description: ClassVar[str] = (
        "Assess data integrity (NaN/Inf) and cell geometry quality (aspect ratio, skewness, angles)"
    )

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the quality filter.

        Returns
        -------
        list[Param]
            The ``output``, ``check_nan``, and ``check_geometry`` parameters.
        """
        return [
            Param(name="output", description="Output Parquet file path for quality report", type=str),
            Param(
                name="check_nan",
                description="Enable NaN/Inf detection in field data",
                type=bool,
                default=True,
            ),
            Param(
                name="check_geometry",
                description="Enable cell geometry quality metrics",
                type=bool,
                default=True,
            ),
        ]

    def __init__(
        self,
        output: str,
        check_nan: bool = True,
        check_geometry: bool = True,
    ) -> None:
        self._output_path = pathlib.Path(output)
        self._check_nan = check_nan
        self._check_geometry = check_geometry
        self._rows: list[_QualityRow] = []
        self._mesh_counter: int = 0
        self._last_artifacts: list[str] = []

    def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
        """Compute quality metrics for each mesh and yield it unchanged.

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
            row = self._assess(mesh)
            self._rows.append(row)
            yield mesh

    def flush(self) -> str | None:
        """Write accumulated quality report to the Parquet file.

        Returns
        -------
        str or None
            The path of the written Parquet file, or ``None`` if there are
            no rows to write.
        """
        if not self._rows:
            return None

        columns: dict[str, list[object]] = {field.name: [] for field in _QUALITY_SCHEMA}
        for row in self._rows:
            for col_name in columns:
                columns[col_name].append(row.get(col_name))  # type: ignore[arg-type]

        table = pa.table(columns, schema=_QUALITY_SCHEMA)
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
        """Merge quality-report Parquet files produced by parallel workers.

        Unlike statistics files that require Welford aggregation, quality
        reports are simple row-per-mesh tables that can be concatenated.

        Parameters
        ----------
        parquet_paths : list[str]
            Paths to per-worker quality Parquet files.
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
        >>> paths = ["worker_0/quality.parquet", "worker_1/quality.parquet"]
        >>> MeshQualityFilter.merge(paths, output="merged.parquet")  # doctest: +SKIP
        'merged.parquet'
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assess(self, mesh: Mesh) -> _QualityRow:
        """Run all enabled quality checks on a single mesh.

        Parameters
        ----------
        mesh : Mesh
            The input mesh.

        Returns
        -------
        _QualityRow
            Quality metrics for this mesh.
        """
        import json

        row: _QualityRow = {
            "mesh_index": self._mesh_counter,
            "n_points": mesh.n_points,
            "n_cells": mesh.n_cells if mesh.cells is not None else 0,
        }
        self._mesh_counter += 1

        if self._check_nan:
            self._check_nan_inf(mesh, row)

        if self._check_geometry and mesh.cells is not None and mesh.n_cells > 0:
            self._check_cell_geometry(mesh, row)

        # Ensure nan_field_details is JSON string
        if "nan_field_details" not in row:
            row["nan_field_details"] = "{}"
        elif not isinstance(row["nan_field_details"], str):
            row["nan_field_details"] = json.dumps(row["nan_field_details"])

        return row

    @staticmethod
    def _check_nan_inf(mesh: Mesh, row: _QualityRow) -> None:
        """Count NaN and Inf values across all field data.

        Parameters
        ----------
        mesh : Mesh
            The input mesh.
        row : _QualityRow
            Row dict to populate with NaN/Inf counts.
        """
        import json

        field_details: dict[str, dict[str, int]] = {}
        total_nan_point = 0
        total_inf_point = 0
        total_nan_cell = 0
        total_inf_cell = 0

        for prefix, td in [("point_data", mesh.point_data), ("cell_data", mesh.cell_data)]:
            if td is None:
                continue
            for key in td.keys():  # noqa: SIM118
                tensor = td[key]
                if not isinstance(tensor, torch.Tensor):
                    continue
                if not tensor.is_floating_point():
                    continue

                n_nan = int(torch.isnan(tensor).sum().item())
                n_inf = int(torch.isinf(tensor).sum().item())

                if n_nan > 0 or n_inf > 0:
                    field_details[f"{prefix}/{key}"] = {"nan": n_nan, "inf": n_inf}

                if prefix == "point_data":
                    total_nan_point += n_nan
                    total_inf_point += n_inf
                else:
                    total_nan_cell += n_nan
                    total_inf_cell += n_inf

        row["nan_point_data_total"] = total_nan_point
        row["inf_point_data_total"] = total_inf_point
        row["nan_cell_data_total"] = total_nan_cell
        row["inf_cell_data_total"] = total_inf_cell
        row["nan_field_details"] = json.dumps(field_details)

    @staticmethod
    def _check_cell_geometry(mesh: Mesh, row: _QualityRow) -> None:
        """Compute cell geometry quality metrics.

        Parameters
        ----------
        mesh : Mesh
            The input mesh (must have ``cells`` and ``points``).
        row : _QualityRow
            Row dict to populate with geometry metrics.
        """
        pts = mesh.points.float()
        cells = mesh.cells
        n_verts = cells.shape[-1]

        # Compute angles and aspect ratios based on cell type
        if n_verts == 3:
            # Triangular cells
            angles = _triangle_angles(pts, cells)
            aspect = _aspect_ratios(pts, cells)
            ideal_angle = math.pi / 3.0
        elif n_verts == 4:
            # Quad cells
            angles = _quad_angles(pts, cells)
            # For quads, use diagonal ratio as aspect ratio
            v0 = pts[cells[:, 0]]
            v1 = pts[cells[:, 1]]
            v2 = pts[cells[:, 2]]
            v3 = pts[cells[:, 3]]
            d1 = (v2 - v0).norm(dim=-1)
            d2 = (v3 - v1).norm(dim=-1)
            aspect = torch.maximum(d1, d2) / (torch.minimum(d1, d2) + 1e-30)
            ideal_angle = math.pi / 2.0
        else:
            # Unsupported cell type — skip geometry checks
            return

        skewness = _equiangle_skewness(angles, ideal_angle)
        min_angles_per_cell = angles.min(dim=-1).values
        max_angles_per_cell = angles.max(dim=-1).values

        # Detect degenerate cells (any angle < 1° or > 179°)
        degenerate = (min_angles_per_cell < math.radians(1.0)) | (max_angles_per_cell > math.radians(179.0))

        # Convert to degrees for reporting
        min_angle_deg = torch.rad2deg(min_angles_per_cell)

        row["geom_min_aspect_ratio"] = float(aspect.min().item())
        row["geom_max_aspect_ratio"] = float(aspect.max().item())
        row["geom_mean_aspect_ratio"] = float(aspect.mean().item())
        row["geom_min_skewness"] = float(skewness.min().item())
        row["geom_max_skewness"] = float(skewness.max().item())
        row["geom_mean_skewness"] = float(skewness.mean().item())
        row["geom_min_angle_deg"] = float(min_angle_deg.min().item())
        row["geom_max_angle_deg"] = float(torch.rad2deg(max_angles_per_cell).max().item())
        row["geom_mean_min_angle_deg"] = float(min_angle_deg.mean().item())
        row["geom_n_degenerate_cells"] = int(degenerate.sum().item())
