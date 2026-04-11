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
Four categories of checks are supported:

* **NaN/Inf detection** — counts non-finite values in every tensor field of
  ``point_data`` and ``cell_data``.
* **Cell geometry** — computes aspect ratio, equiangle skewness, and minimum
  interior angle for each cell (triangles and tetrahedra).
* **Cell volume** — computes volume/area statistics (min, max, mean, std,
  ratio) and counts zero-volume (degenerate) cells.
* **Scaled Jacobian** — computes the scaled Jacobian quality metric for each
  cell.  Values range from −1 (inverted) through 0 (degenerate) to +1
  (ideal).  Counts inverted and poor-quality cells.

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

    # Cell volume / area statistics
    vol_min: float
    vol_max: float
    vol_mean: float
    vol_std: float
    vol_ratio: float  # max / min (clamped to avoid division by zero)
    vol_n_zero: int  # cells with volume < epsilon

    # Scaled Jacobian quality
    jac_min: float
    jac_max: float
    jac_mean: float
    jac_n_inverted: int  # cells with negative Jacobian
    jac_n_poor: int  # cells with |Jacobian| < 0.2


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
        # Cell volume / area
        ("vol_min", pa.float64()),
        ("vol_max", pa.float64()),
        ("vol_mean", pa.float64()),
        ("vol_std", pa.float64()),
        ("vol_ratio", pa.float64()),
        ("vol_n_zero", pa.int64()),
        # Scaled Jacobian
        ("jac_min", pa.float64()),
        ("jac_max", pa.float64()),
        ("jac_mean", pa.float64()),
        ("jac_n_inverted", pa.int64()),
        ("jac_n_poor", pa.int64()),
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


def _tetrahedron_dihedral_angles(pts: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """Compute the six dihedral angles of tetrahedral cells.

    Each tetrahedron has six edges, each of which defines a dihedral angle
    between its two adjacent faces.  The angle is computed as the arc-cosine
    of the dot product of the outward face normals (negated, since adjacent
    faces share an edge and their outward normals point away from each other).

    Parameters
    ----------
    pts : torch.Tensor
        Vertex coordinates, shape ``(n_points, 3)``.
    cells : torch.Tensor
        Tetrahedron connectivity, shape ``(n_cells, 4)``.

    Returns
    -------
    torch.Tensor
        Dihedral angles in radians, shape ``(n_cells, 6)``.
        Edge order: (0-1, 0-2, 0-3, 1-2, 1-3, 2-3).
    """
    v0 = pts[cells[:, 0]]
    v1 = pts[cells[:, 1]]
    v2 = pts[cells[:, 2]]
    v3 = pts[cells[:, 3]]

    # Face normals (unnormalized).  Each face is opposite one vertex.
    # We orient every normal to point *away from* the opposite vertex,
    # so that adjacent outward normals meet at the supplement of the
    # dihedral angle.  The dihedral angle is then π − acos(cos).
    n0 = torch.cross(v2 - v1, v3 - v1, dim=-1)  # face opposite v0 (v1,v2,v3)
    n1 = torch.cross(v2 - v0, v3 - v0, dim=-1)  # face opposite v1 (v0,v2,v3)
    n2 = torch.cross(v1 - v0, v3 - v0, dim=-1)  # face opposite v2 (v0,v1,v3)
    n3 = torch.cross(v1 - v0, v2 - v0, dim=-1)  # face opposite v3 (v0,v1,v2)

    # Orient each face normal to point away from the opposite vertex.
    # dot(n_i, centroid_of_face_i − v_i) should be positive.
    centroid_0 = (v1 + v2 + v3) / 3.0
    centroid_1 = (v0 + v2 + v3) / 3.0
    centroid_2 = (v0 + v1 + v3) / 3.0
    centroid_3 = (v0 + v1 + v2) / 3.0

    # Flip normals that point toward the opposite vertex instead of away
    sign0 = ((centroid_0 - v0) * n0).sum(dim=-1).sign().unsqueeze(-1)
    sign1 = ((centroid_1 - v1) * n1).sum(dim=-1).sign().unsqueeze(-1)
    sign2 = ((centroid_2 - v2) * n2).sum(dim=-1).sign().unsqueeze(-1)
    sign3 = ((centroid_3 - v3) * n3).sum(dim=-1).sign().unsqueeze(-1)

    n0 = n0 * sign0
    n1 = n1 * sign1
    n2 = n2 * sign2
    n3 = n3 * sign3

    eps = 1e-30

    def _dihedral(na: torch.Tensor, nb: torch.Tensor) -> torch.Tensor:
        """Dihedral angle between two faces sharing an edge.

        With outward-oriented normals, the angle between them is the
        supplement of the internal dihedral angle, so we return
        ``π − acos(cos)``.
        """
        cos = (na * nb).sum(dim=-1) / (na.norm(dim=-1) * nb.norm(dim=-1) + eps)
        return math.pi - torch.acos(cos.clamp(-1.0, 1.0))

    # Six edges → six dihedral angles.
    # Edge (i,j) is shared by faces opposite the other two vertices.
    return torch.stack(
        [
            _dihedral(n2, n3),  # edge 0-1: faces opposite v2, v3
            _dihedral(n1, n3),  # edge 0-2: faces opposite v1, v3
            _dihedral(n1, n2),  # edge 0-3: faces opposite v1, v2
            _dihedral(n0, n3),  # edge 1-2: faces opposite v0, v3
            _dihedral(n0, n2),  # edge 1-3: faces opposite v0, v2
            _dihedral(n0, n1),  # edge 2-3: faces opposite v0, v1
        ],
        dim=-1,
    )


def _tet_aspect_ratios(pts: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """Compute edge-length aspect ratios for tetrahedral cells.

    The aspect ratio is ``longest_edge / shortest_edge`` across all six
    edges.  A regular tetrahedron has aspect ratio 1.0.

    Parameters
    ----------
    pts : torch.Tensor
        Vertex coordinates, shape ``(n_points, 3)``.
    cells : torch.Tensor
        Tetrahedron connectivity, shape ``(n_cells, 4)``.

    Returns
    -------
    torch.Tensor
        Aspect ratios, shape ``(n_cells,)``.
    """
    v0 = pts[cells[:, 0]]
    v1 = pts[cells[:, 1]]
    v2 = pts[cells[:, 2]]
    v3 = pts[cells[:, 3]]

    edges = torch.stack(
        [
            (v1 - v0).norm(dim=-1),
            (v2 - v0).norm(dim=-1),
            (v3 - v0).norm(dim=-1),
            (v2 - v1).norm(dim=-1),
            (v3 - v1).norm(dim=-1),
            (v3 - v2).norm(dim=-1),
        ],
        dim=-1,
    )

    longest = edges.max(dim=-1).values
    shortest = edges.min(dim=-1).values
    return longest / (shortest + 1e-30)


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


def _scaled_jacobian(pts: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """Compute the scaled Jacobian for each simplex cell.

    The scaled Jacobian is defined as::

        J_scaled = det(J) / prod(||e_i||)

    where ``J = [e_1, e_2, ...]`` is the edge-vector matrix from vertex 0
    and ``e_i = v_i - v_0``.  Values:

    * +1 = ideal (equilateral simplex)
    * 0 = degenerate (zero-volume)
    * negative = inverted element

    For **triangles in 3-D** the signed area is used (cross-product norm
    with sign), giving values in ``[0, 1]`` (triangles cannot be "inverted"
    in 3-D).  For **triangles in 2-D** the 2-D cross product gives a signed
    value.

    For **tetrahedra** the scalar triple product gives a signed volume.

    For **general simplices** the Gram determinant is used (always ≥ 0).

    Parameters
    ----------
    pts : torch.Tensor
        Vertex coordinates, shape ``(n_points, n_spatial_dims)``.
    cells : torch.Tensor
        Cell connectivity, shape ``(n_cells, n_verts_per_cell)``.

    Returns
    -------
    torch.Tensor
        Scaled Jacobian values, shape ``(n_cells,)``.
    """
    eps = 1e-30
    n_verts = cells.shape[-1]
    n_manifold = n_verts - 1

    # Edge vectors from vertex 0: (n_cells, n_manifold, n_spatial)
    rel = pts[cells[:, 1:]] - pts[cells[:, :1]]
    # Product of edge norms: (n_cells,)
    norms = rel.norm(dim=-1)  # (n_cells, n_manifold)
    norm_prod = norms.prod(dim=-1) + eps

    n_spatial = pts.shape[-1]

    if n_manifold == 2 and n_spatial == 2:
        # 2-D triangles: signed area via 2-D cross product
        det = rel[:, 0, 0] * rel[:, 1, 1] - rel[:, 0, 1] * rel[:, 1, 0]
        return det / norm_prod

    if n_manifold == 2 and n_spatial >= 3:
        # 3-D+ triangles: area via cross product magnitude (always >= 0)
        cross = torch.cross(rel[:, 0], rel[:, 1], dim=-1)
        area = cross.norm(dim=-1)
        return area / norm_prod

    if n_manifold == 3 and n_spatial == 3:
        # Tetrahedra in 3-D: signed volume via scalar triple product
        cross = torch.cross(rel[:, 1], rel[:, 2], dim=-1)
        det = (rel[:, 0] * cross).sum(dim=-1)
        return det / norm_prod

    # General fallback: Gram determinant (unsigned)
    gram = torch.matmul(rel, rel.transpose(-2, -1))
    det_gram = gram.det()
    signed_vol = det_gram.abs().sqrt()
    return signed_vol / norm_prod


# ---------------------------------------------------------------------------
# MeshQualityFilter
# ---------------------------------------------------------------------------


class MeshQualityFilter(Filter["Mesh"]):
    """Assess mesh data integrity and cell geometry quality.

    This filter inspects every mesh flowing through the pipeline and
    produces a quality report as a Parquet file.  Four categories of
    checks are supported:

    * **NaN/Inf detection** (``check_nan=True``) — counts non-finite values
      in every tensor field of ``point_data`` and ``cell_data``.
    * **Cell geometry** (``check_geometry=True``) — computes aspect ratio,
      equiangle skewness, and minimum interior angle for each cell.
      Supports triangles (3-node) and tetrahedra (4-node).
    * **Cell volume** (``check_volume=True``) — computes volume/area
      statistics and counts zero-volume (degenerate) cells.
    * **Scaled Jacobian** (``check_jacobian=True``) — computes the FEM
      scaled Jacobian metric per cell.  Values range from −1 (inverted)
      to +1 (ideal).

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
    check_volume : bool
        Enable cell volume/area statistics.  Default is ``True``.
    check_jacobian : bool
        Enable scaled Jacobian quality metrics.  Default is ``True``.

    Examples
    --------
    >>> filt = MeshQualityFilter(output="quality.parquet")
    >>> pipeline = source.filter(filt).write(sink)
    >>> for i in range(len(pipeline)):
    ...     pipeline[i]
    >>> filt.flush()  # write quality report
    """

    name: ClassVar[str] = "Mesh Quality"
    description: ClassVar[str] = "Assess data integrity (NaN/Inf), cell geometry, volume, and Jacobian quality"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the quality filter.

        Returns
        -------
        list[Param]
            The ``output``, ``check_nan``, ``check_geometry``,
            ``check_volume``, and ``check_jacobian`` parameters.
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
            Param(
                name="check_volume",
                description="Enable cell volume/area statistics",
                type=bool,
                default=True,
            ),
            Param(
                name="check_jacobian",
                description="Enable scaled Jacobian quality metrics",
                type=bool,
                default=True,
            ),
        ]

    def __init__(
        self,
        output: str,
        check_nan: bool = True,
        check_geometry: bool = True,
        check_volume: bool = True,
        check_jacobian: bool = True,
    ) -> None:
        self._output_path = pathlib.Path(output)
        self._check_nan = check_nan
        self._check_geometry = check_geometry
        self._check_volume = check_volume
        self._check_jacobian = check_jacobian
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

        has_cells = mesh.cells is not None and mesh.n_cells > 0

        if self._check_nan:
            self._check_nan_inf(mesh, row)

        if self._check_geometry and has_cells:
            self._check_cell_geometry(mesh, row)

        if self._check_volume and has_cells:
            self._check_cell_volume(mesh, row)

        if self._check_jacobian and has_cells:
            self._check_cell_jacobian(mesh, row)

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
            # Tetrahedral cells (Mesh stores simplices only — 4-node = tet)
            angles = _tetrahedron_dihedral_angles(pts, cells)
            aspect = _tet_aspect_ratios(pts, cells)
            ideal_angle = math.acos(1.0 / 3.0)  # ~70.53° for regular tet
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

    @staticmethod
    def _check_cell_volume(mesh: Mesh, row: _QualityRow) -> None:
        """Compute cell volume / area statistics.

        Uses the ``mesh.cell_areas`` cached property which handles simplices
        of any manifold dimension (edges, triangles, tetrahedra, etc.).

        Parameters
        ----------
        mesh : Mesh
            The input mesh (must have ``cells`` and ``points``).
        row : _QualityRow
            Row dict to populate with volume metrics.
        """
        volumes = mesh.cell_areas  # (n_cells,)
        vol_min = float(volumes.min().item())
        vol_max = float(volumes.max().item())

        row["vol_min"] = vol_min
        row["vol_max"] = vol_max
        row["vol_mean"] = float(volumes.mean().item())
        row["vol_std"] = float(volumes.std().item()) if volumes.numel() > 1 else 0.0
        row["vol_ratio"] = vol_max / (vol_min + 1e-30)
        row["vol_n_zero"] = int((volumes < 1e-30).sum().item())

    @staticmethod
    def _check_cell_jacobian(mesh: Mesh, row: _QualityRow) -> None:
        """Compute scaled Jacobian quality metrics.

        Parameters
        ----------
        mesh : Mesh
            The input mesh (must have ``cells`` and ``points``).
        row : _QualityRow
            Row dict to populate with Jacobian metrics.
        """
        pts = mesh.points.float()
        cells = mesh.cells
        jac = _scaled_jacobian(pts, cells)

        row["jac_min"] = float(jac.min().item())
        row["jac_max"] = float(jac.max().item())
        row["jac_mean"] = float(jac.mean().item())
        row["jac_n_inverted"] = int((jac < 0.0).sum().item())
        row["jac_n_poor"] = int((jac.abs() < 0.2).sum().item())
