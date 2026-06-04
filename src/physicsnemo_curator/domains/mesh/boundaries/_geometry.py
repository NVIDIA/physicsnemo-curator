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

"""Dataset-agnostic boundary-synthesis geometry toolkit.

These helpers synthesize CFD-domain outer boundaries (inlet / outlet / walls /
symmetry planes) for :class:`~physicsnemo.mesh.domain_mesh.DomainMesh` samples
that ship with only a geometry surface (``vehicle``).  They are the reusable
primitives behind the :mod:`...boundaries.generators` strategies and cover the
two common archetypes:

* **box wind tunnel** — axis-aligned box-face patches with inward normals
  (:func:`box_face_patch`, :func:`box_tunnel_boundaries`).
* **hemispherical open-road domain** — a triangulated hemisphere split into
  inlet/outlet by freestream direction (:func:`build_hemisphere`,
  :func:`split_by_freestream`) plus a constrained-Delaunay symmetry disk with
  the vehicle silhouette as holes (:func:`silhouette_loops`,
  :func:`constrained_delaunay_disk`).

All synthesized boundaries use inward-pointing cell normals (into the fluid).
"""

from __future__ import annotations

import logging
import warnings
from typing import Literal

import numpy as np
import torch
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.primitives.planar import structured_grid
from physicsnemo.mesh.projections import embed

logger = logging.getLogger(__name__)

#: ``((x_min, x_max), (y_min, y_max), (z_min, z_max))`` axis-aligned bounds.
Bounds3D = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]

#: Box face label.
Face = Literal["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]


# ---------------------------------------------------------------------------
# Per-sample geometry inference
# ---------------------------------------------------------------------------


def z_floor_from_boundary(mesh: Mesh) -> float:
    """Infer the ground-plane height from a boundary mesh's minimum z.

    Parameters
    ----------
    mesh : Mesh
        Boundary mesh (e.g. the vehicle surface).

    Returns
    -------
    float
        The minimum z-coordinate (tire / contact-patch height).
    """
    return float(mesh.points[:, 2].min())


def radius_from_interior(interior: Mesh) -> float:
    """Infer a hemisphere radius from the interior point-cloud bounding box.

    Parameters
    ----------
    interior : Mesh
        Interior point cloud (roughly fills the hemisphere).

    Returns
    -------
    float
        The max of ``|x|``, ``y``, ``|z|`` over the interior points.
    """
    p = interior.points
    return max(float(p[:, 0].abs().max()), float(p[:, 1].max()), float(p[:, 2].abs().max()))


# ---------------------------------------------------------------------------
# Box wind-tunnel boundaries
# ---------------------------------------------------------------------------


def box_face_patch(
    *,
    bounds: Bounds3D,
    face: Face,
    n_per_side: tuple[int, int] = (20, 20),
) -> Mesh:
    """Build one triangulated rectangular face of an axis-aligned box.

    Cell winding is chosen so that normals point inward (into the box
    interior).

    Parameters
    ----------
    bounds : Bounds3D
        ``((x_min, x_max), (y_min, y_max), (z_min, z_max))`` of the box.
    face : Face
        Which face to construct (e.g. ``"z_min"`` is the floor).
    n_per_side : tuple[int, int]
        Grid resolution along the two in-plane axes (in xyz order).

    Returns
    -------
    Mesh
        A ``Mesh[2, 3]`` surface patch with inward normals.
    """
    axis_name, side = face.split("_")
    axis_idx = "xyz".index(axis_name)
    constant_value = bounds[axis_idx][0 if side == "min" else 1]

    # Inward normal: at the min face the fluid lies at higher coordinate, so
    # inward is +axis; at the max face inward is -axis.
    inward_sign = +1 if side == "min" else -1

    in_plane_axes = [i for i in range(3) if i != axis_idx]
    a_idx, b_idx = in_plane_axes
    a_min, a_max = bounds[a_idx]
    b_min, b_max = bounds[b_idx]
    n_a, n_b = n_per_side

    grid_2d = structured_grid.load(x_min=a_min, x_max=a_max, y_min=b_min, y_max=b_max, n_x=n_a, n_y=n_b)
    embedded = embed(grid_2d, target_n_spatial_dims=3, insert_at=axis_idx)
    offset = [0.0, 0.0, 0.0]
    offset[axis_idx] = constant_value
    embedded = embedded.translate(offset)

    # Probe the first cell's natural normal and flip winding if it does not
    # already match the inward direction on the constant axis.
    p0 = embedded.points[embedded.cells[0, 0]]
    p1 = embedded.points[embedded.cells[0, 1]]
    p2 = embedded.points[embedded.cells[0, 2]]
    natural_normal = torch.linalg.cross(p1 - p0, p2 - p0)
    natural_sign = float(torch.sign(natural_normal[axis_idx]))

    cells = embedded.cells
    if natural_sign * inward_sign < 0.0:
        cells = cells[:, [0, 2, 1]]

    return Mesh(points=embedded.points, cells=cells)


def box_tunnel_boundaries(
    *,
    bounds: Bounds3D,
    x_bl: float,
    n_per_side: tuple[int, int] = (20, 20),
) -> dict[str, Mesh]:
    """Build rectangular wind-tunnel boundaries: inlet/outlet/slip/no_slip.

    ``slip`` merges the two sidewalls, the ceiling, and the forward floor
    (upstream of *x_bl*); ``no_slip`` is the rear floor.  All normals point
    inward.

    Parameters
    ----------
    bounds : Bounds3D
        CFD-domain bounds; ``z_min`` is the floor, ``z_max`` the ceiling.
    x_bl : float
        Streamwise coordinate where the floor transitions from slip
        (upstream) to no-slip (downstream).
    n_per_side : tuple[int, int]
        Per-face triangulation resolution.

    Returns
    -------
    dict[str, Mesh]
        ``{"inlet", "outlet", "slip", "no_slip"}``.

    Raises
    ------
    ValueError
        If *x_bl* lies outside the streamwise extent.
    """
    if not (bounds[0][0] <= x_bl <= bounds[0][1]):
        msg = f"x_bl={x_bl!r} must lie inside x range {bounds[0]!r}."
        raise ValueError(msg)

    inlet = box_face_patch(bounds=bounds, face="x_min", n_per_side=n_per_side)
    outlet = box_face_patch(bounds=bounds, face="x_max", n_per_side=n_per_side)
    ceiling = box_face_patch(bounds=bounds, face="z_max", n_per_side=n_per_side)
    side_y_min = box_face_patch(bounds=bounds, face="y_min", n_per_side=n_per_side)
    side_y_max = box_face_patch(bounds=bounds, face="y_max", n_per_side=n_per_side)

    floor_slip_bounds: Bounds3D = ((bounds[0][0], x_bl), bounds[1], bounds[2])
    floor_noslip_bounds: Bounds3D = ((x_bl, bounds[0][1]), bounds[1], bounds[2])
    floor_slip = box_face_patch(bounds=floor_slip_bounds, face="z_min", n_per_side=n_per_side)
    no_slip = box_face_patch(bounds=floor_noslip_bounds, face="z_min", n_per_side=n_per_side)

    slip = Mesh.merge([floor_slip, ceiling, side_y_min, side_y_max])

    return {"inlet": inlet, "outlet": outlet, "slip": slip, "no_slip": no_slip}


# ---------------------------------------------------------------------------
# Hemisphere open-road boundaries
# ---------------------------------------------------------------------------


def build_hemisphere(
    radius: float,
    *,
    n_theta: int = 64,
    n_phi: int = 32,
) -> tuple[Mesh, torch.Tensor]:
    """Build a triangulated hemisphere ``y >= 0`` with the equator at ``y = 0``.

    Cell winding is chosen so normals point inward (toward the origin).

    Parameters
    ----------
    radius : float
        Hemisphere radius.
    n_theta : int
        Azimuthal divisions around the y-axis (``>= 3``).
    n_phi : int
        Polar divisions from north pole to equator (``>= 2``).

    Returns
    -------
    tuple[Mesh, torch.Tensor]
        ``(mesh, equator_indices)`` — the hemisphere ``Mesh[2, 3]`` and the
        ``(n_theta,)`` point indices on the equator (the last ring), whose
        coordinates the symmetry disk reuses for a watertight seam.

    Raises
    ------
    ValueError
        If *n_theta* < 3 or *n_phi* < 2.
    """
    if n_theta < 3:
        msg = f"need n_theta >= 3, got {n_theta=}"
        raise ValueError(msg)
    if n_phi < 2:
        msg = f"need n_phi >= 2, got {n_phi=}"
        raise ValueError(msg)

    phi = torch.linspace(0.0, torch.pi / 2, n_phi)
    theta = torch.linspace(0.0, 2.0 * torch.pi, n_theta + 1)[:-1]

    pole = torch.tensor([[0.0, radius, 0.0]])
    rings: list[torch.Tensor] = []
    for r in range(n_phi - 1):
        p = phi[r + 1].item()
        ring_y = 0.0 if r == n_phi - 2 else radius * np.cos(p)
        ring_xz_radius = radius * np.sin(p)
        rx = ring_xz_radius * torch.cos(theta)
        rz = ring_xz_radius * torch.sin(theta)
        rings.append(torch.stack([rx, torch.full_like(rx, ring_y), rz], dim=-1))

    points = torch.cat([pole, *rings], dim=0).float()

    cell_rows: list[list[int]] = []
    for i in range(n_theta):
        i_next = (i + 1) % n_theta
        cell_rows.append([0, 1 + i, 1 + i_next])
    for r in range(n_phi - 2):
        ring_r = 1 + r * n_theta
        ring_r1 = 1 + (r + 1) * n_theta
        for i in range(n_theta):
            i_next = (i + 1) % n_theta
            v00 = ring_r + i
            v01 = ring_r + i_next
            v10 = ring_r1 + i
            v11 = ring_r1 + i_next
            cell_rows.append([v00, v10, v01])
            cell_rows.append([v01, v10, v11])

    cells = torch.tensor(cell_rows, dtype=torch.int64)

    # Verify (and enforce) inward winding across every cell.
    p0_all = points[cells[:, 0]]
    p1_all = points[cells[:, 1]]
    p2_all = points[cells[:, 2]]
    centroids_all = (p0_all + p1_all + p2_all) / 3.0
    normals_all = torch.linalg.cross(p1_all - p0_all, p2_all - p0_all)
    dots = (centroids_all * normals_all).sum(dim=-1)
    if bool((dots > 0).all()):
        cells = cells[:, [0, 2, 1]]
        dots = -dots
    if not bool((dots < 0).all().item()):
        n_outward = int((dots > 0).sum().item())
        msg = f"hemisphere triangulation has mixed winding: {n_outward}/{cells.shape[0]} cells point outward."
        raise RuntimeError(msg)

    mesh = Mesh(points=points, cells=cells)
    equator_indices = torch.arange(points.shape[0] - n_theta, points.shape[0])
    return mesh, equator_indices


def split_by_freestream(hemisphere: Mesh, u_inf: torch.Tensor) -> tuple[Mesh, Mesh]:
    """Split a hemisphere into inlet (upstream) and outlet (downstream).

    Each cell is classified by the sign of ``dot(centroid, U_inf)``:
    ``<= 0`` -> inlet, ``> 0`` -> outlet.

    Parameters
    ----------
    hemisphere : Mesh
        Hemisphere mesh from :func:`build_hemisphere`.
    u_inf : torch.Tensor
        ``(3,)`` freestream velocity vector (direction only).

    Returns
    -------
    tuple[Mesh, Mesh]
        ``(inlet, outlet)`` meshes sharing exact seam vertex coordinates.

    Raises
    ------
    ValueError
        If *u_inf* is not shape ``(3,)``.
    """
    if tuple(u_inf.shape) != (3,):
        msg = f"u_inf must be shape (3,), got {tuple(u_inf.shape)}"
        raise ValueError(msg)

    centroids = hemisphere.cell_centroids
    direction = u_inf / u_inf.norm()
    dots = centroids @ direction.to(centroids.dtype)

    inlet_idx = (dots <= 0).nonzero(as_tuple=True)[0]
    outlet_idx = (dots > 0).nonzero(as_tuple=True)[0]

    inlet = hemisphere.slice_cells(inlet_idx).clean(
        merge_points=False, remove_duplicate_cells=False, remove_unused_points=True
    )
    outlet = hemisphere.slice_cells(outlet_idx).clean(
        merge_points=False, remove_duplicate_cells=False, remove_unused_points=True
    )
    return inlet, outlet


# ---------------------------------------------------------------------------
# Vehicle silhouette extraction (open edges on a symmetry plane)
# ---------------------------------------------------------------------------


def silhouette_loops(vehicle: Mesh, *, y_eps: float = 1e-3) -> list[torch.Tensor]:
    """Extract closed loops of vehicle vertex indices on the ``y = 0`` plane.

    An "open" edge (shared by exactly one triangle) whose endpoints both lie
    on the symmetry plane traces the silhouette of a half-geometry mesh.

    Parameters
    ----------
    vehicle : Mesh
        Half-geometry surface mesh (points with ``y >= 0``).
    y_eps : float
        Tolerance for "vertex lies on the symmetry plane".

    Returns
    -------
    list[torch.Tensor]
        One 1-D tensor of vertex indices per closed loop (loop order, no
        repeated start).  Empty if there are no open edges on the plane.
    """
    cells = vehicle.cells
    on_plane = vehicle.points[:, 1].abs() < y_eps

    cell_on_plane = on_plane[cells]
    n_on_plane_per_cell = cell_on_plane.sum(dim=1)

    relevant = n_on_plane_per_cell >= 2
    rel_cells = cells[relevant]
    rel_on_plane = cell_on_plane[relevant]

    edge_pairs = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.int64)
    edge_endpoints = rel_cells[:, edge_pairs]
    edge_both_on = rel_on_plane[:, edge_pairs].all(dim=-1)

    candidate_edges = edge_endpoints[edge_both_on]
    if candidate_edges.numel() == 0:
        return []

    candidate_edges_sorted = candidate_edges.sort(dim=-1).values
    unique_edges, inverse = torch.unique(candidate_edges_sorted, dim=0, return_inverse=True)
    counts = torch.bincount(inverse)
    boundary_edges = unique_edges[counts == 1]

    if boundary_edges.shape[0] == 0:
        return []

    return walk_closed_loops(boundary_edges)


def walk_closed_loops(edges: torch.Tensor) -> list[torch.Tensor]:
    """Group an undirected edge set into closed loops via half-edge traversal.

    Parameters
    ----------
    edges : torch.Tensor
        ``(K, 2)`` vertex index pairs.

    Returns
    -------
    list[torch.Tensor]
        One 1-D long tensor per closed loop (arbitrary start, no repeated end).
    """
    edges_np = edges.cpu().numpy()

    adjacency: dict[int, list[int]] = {}
    for a, b in edges_np:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))

    bad_degree = {v: len(ns) for v, ns in adjacency.items() if len(ns) != 2}
    if bad_degree:
        warnings.warn(
            f"silhouette: {len(bad_degree)} vertices with non-2 degree "
            f"(sample: {dict(list(bad_degree.items())[:5])}); loop extraction may be incomplete.",
            stacklevel=2,
        )

    visited: set[int] = set()
    loops: list[list[int]] = []
    for start in adjacency:
        if start in visited or len(adjacency[start]) != 2:
            continue
        loop = [start]
        visited.add(start)
        prev = -1
        current = start
        while True:
            options = [n for n in adjacency[current] if n != prev]
            if not options:
                break
            next_v = options[0]
            if next_v == start:
                break
            loop.append(next_v)
            visited.add(next_v)
            prev = current
            current = next_v
        if len(loop) >= 3:
            loops.append(loop)

    return [torch.tensor(loop, dtype=torch.int64) for loop in loops]


# ---------------------------------------------------------------------------
# 2D geometry helpers
# ---------------------------------------------------------------------------


def polygon_area(loop_xz: torch.Tensor) -> float:
    """Return the absolute area of a closed 2D polygon (shoelace formula).

    Parameters
    ----------
    loop_xz : torch.Tensor
        ``(n, 2)`` vertex coordinates.

    Returns
    -------
    float
        Absolute polygon area.
    """
    x = loop_xz[:, 0]
    z = loop_xz[:, 1]
    return float(0.5 * torch.abs((x * z.roll(-1) - x.roll(-1) * z).sum()))


def point_in_polygon(points_xz: torch.Tensor, polygon_xz: torch.Tensor) -> torch.Tensor:
    """Vectorised point-in-polygon test via horizontal ray casting.

    Parameters
    ----------
    points_xz : torch.Tensor
        ``(N, 2)`` query points.
    polygon_xz : torch.Tensor
        ``(M, 2)`` polygon vertices in loop order (implicitly closed).

    Returns
    -------
    torch.Tensor
        ``(N,)`` bool tensor; ``True`` where a point is inside the polygon.
    """
    qx = points_xz[:, 0:1]
    qz = points_xz[:, 1:2]
    a = polygon_xz
    b = polygon_xz.roll(-1, dims=0)
    az = a[:, 1].unsqueeze(0)
    bz = b[:, 1].unsqueeze(0)
    ax = a[:, 0].unsqueeze(0)
    bx = b[:, 0].unsqueeze(0)

    crosses = (az > qz) != (bz > qz)
    denom = bz - az
    safe_denom = torch.where(denom != 0.0, denom, torch.ones_like(denom))
    t = (qz - az) / safe_denom
    inter_x = ax + t * (bx - ax)

    n_crossings = (crosses & (inter_x > qx)).sum(dim=-1)
    return (n_crossings % 2) == 1


def constrained_delaunay_disk(
    *,
    equator_xz: torch.Tensor,
    silhouette_xzs: list[torch.Tensor],
) -> Mesh:
    """Triangulate the symmetry-plane disk with silhouettes as holes.

    Uses ``vtkDelaunay2D`` (via PyVista) with the equator + silhouette loops
    as edge constraints.  The output sits in the ``y = 0`` plane with ``+y``
    (inward) normals; output point coordinates are bit-identical to the inputs
    so they dedupe cleanly against the hemisphere equator and vehicle
    silhouette.

    Parameters
    ----------
    equator_xz : torch.Tensor
        ``(n_eq, 2)`` ``(x, z)`` of the hemisphere equator vertices.
    silhouette_xzs : list[torch.Tensor]
        List of ``(n_i, 2)`` ``(x, z)`` silhouette loops (may be empty).

    Returns
    -------
    Mesh
        Disk ``Mesh[2, 3]`` at ``y = 0`` with inward (``+y``) normals.

    Raises
    ------
    ValueError
        If the equator has fewer than 3 vertices.
    RuntimeError
        If the triangulation returns no / non-triangle cells.
    """
    import pyvista as pv

    if equator_xz.shape[0] < 3:
        msg = f"equator must have >=3 vertices, got {equator_xz.shape[0]}"
        raise ValueError(msg)

    blocks = [equator_xz, *list(silhouette_xzs)]
    offsets = np.cumsum([0] + [b.shape[0] for b in blocks]).tolist()
    all_xz = torch.cat(blocks, dim=0).cpu().numpy()

    points_in_xy = np.column_stack([all_xz[:, 0], all_xz[:, 1], np.zeros(len(all_xz))]).astype(np.float64)

    line_records: list[int] = []
    for lo, hi in zip(offsets[:-1], offsets[1:], strict=True):
        n = hi - lo
        for i in range(n):
            line_records.extend([2, lo + i, lo + (i + 1) % n])

    edge_source = pv.PolyData(points_in_xy, lines=np.asarray(line_records))
    point_set = pv.PolyData(points_in_xy)
    triangulated = point_set.delaunay_2d(edge_source=edge_source, alpha=0.0, tol=1e-8)

    faces = np.asarray(triangulated.faces)
    if faces.size == 0:
        msg = "vtkDelaunay2D returned zero triangles; check that the equator encloses every silhouette."
        raise RuntimeError(msg)
    faces = faces.reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        msg = f"vtkDelaunay2D returned non-triangle cells (sizes: {np.unique(faces[:, 0])})"
        raise RuntimeError(msg)
    out_cells = torch.from_numpy(faces[:, 1:].astype(np.int64))

    # Drop spurious triangles that vtkDelaunay2D places inside silhouette holes.
    if silhouette_xzs:
        cell_centroids_xz = torch.from_numpy(all_xz[out_cells.numpy()].mean(axis=1))
        keep_mask = torch.ones(out_cells.shape[0], dtype=torch.bool)
        for sil in silhouette_xzs:
            keep_mask &= ~point_in_polygon(cell_centroids_xz, sil)
        out_cells = out_cells[keep_mask]

    out_points = torch.from_numpy(
        np.column_stack([all_xz[:, 0], np.zeros(len(all_xz)), all_xz[:, 1]])
    ).float()

    # Re-orient each cell to +y (inward); vtkDelaunay2D winding is not
    # guaranteed consistent under our axis swap.
    p0_all = out_points[out_cells[:, 0]]
    p1_all = out_points[out_cells[:, 1]]
    p2_all = out_points[out_cells[:, 2]]
    ny_all = torch.linalg.cross(p1_all - p0_all, p2_all - p0_all)[:, 1]
    flip_mask = ny_all < 0.0
    if bool(flip_mask.any()):
        fixed = out_cells.clone()
        fixed[flip_mask] = out_cells[flip_mask][:, [0, 2, 1]]
        out_cells = fixed

    return Mesh(points=out_points, cells=out_cells)
