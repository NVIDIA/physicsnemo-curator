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

"""Shared, dataset-agnostic VTK -> physicsnemo Mesh conversion helpers.

This module centralises the VTK reading and conversion mechanics that are
otherwise duplicated between
:class:`~physicsnemo_curator.domains.mesh.sources.vtk.VTKSource` and
:class:`~physicsnemo_curator.domains.mesh.sources.drivaerml.DrivAerMLSource`.

It supports two backends:

* **pyvista** — full-featured reading via :func:`physicsnemo.mesh.io.from_pyvista`
  with optional reader-level data-array filtering (``vtkXML*Reader`` array
  selection) to avoid materialising unwanted fields.
* **rust** — the native ``physicsnemo_curator._lib.vtk`` reader (ASCII/binary
  VTU/VTP only) for fast I/O, building a :class:`~physicsnemo.mesh.Mesh`
  directly from raw NumPy arrays.

The conversion supports both point sources (mesh ``vertices`` or
``cell_centroids``), polygon tessellation for mixed surface cells, and an
optional float64 -> float32 downcast.
"""

from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from physicsnemo.mesh import Mesh

if TYPE_CHECKING:
    import pyvista as pv

logger = logging.getLogger(__name__)

#: VTK reading backend identifiers.
Backend = Literal["pyvista", "rust"]

#: File extensions the Rust reader can parse (VTK XML formats).
_RUST_EXTENSIONS: frozenset[str] = frozenset({".vtu", ".vtp"})

#: VTK XML reader class names per extension (for reader-level array filtering).
_XML_READERS: dict[str, str] = {
    ".vtu": "vtkXMLUnstructuredGridReader",
    ".vtp": "vtkXMLPolyDataReader",
}


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def can_use_rust(
    path: str | pathlib.Path,
    point_source: str,
    manifold_dim: int | Literal["auto"],
) -> bool:
    """Return ``True`` if the Rust reader can handle this file + config.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the VTK file.
    point_source : str
        Point-source mode (``"vertices"`` or ``"cell_centroids"``).
    manifold_dim : int or {"auto"}
        Target manifold dimension.

    Returns
    -------
    bool
        ``True`` when the file extension is a Rust-supported VTK XML format
        and the requested conversion is expressible without PyVista.
    """
    suffix = pathlib.Path(path).suffix.lower()
    if suffix not in _RUST_EXTENSIONS:
        return False
    # The dual-graph (cell_centroids + manifold_dim=1) construction requires
    # face-adjacency that the Rust reader does not expose; defer to PyVista.
    return not (point_source == "cell_centroids" and manifold_dim == 1)


# ---------------------------------------------------------------------------
# Cell / connectivity helpers
# ---------------------------------------------------------------------------


def build_cells_from_rust(rust_data: Any) -> torch.Tensor | None:
    """Build a uniform ``(n_cells, nodes_per_cell)`` cell tensor.

    Returns ``None`` when the mesh has no cells or contains mixed cell
    types (non-uniform vertex counts), which cannot be packed into a
    rectangular tensor.

    Parameters
    ----------
    rust_data : VtkMeshData
        Parsed VTK data from the Rust reader.

    Returns
    -------
    torch.Tensor or None
        Cell connectivity of shape ``(n_cells, nodes_per_cell)`` (dtype
        ``int64``), or ``None`` if unavailable / mixed.
    """
    connectivity = rust_data.cells
    offsets = rust_data.cell_offsets
    n_cells = rust_data.n_cells

    if connectivity is None or offsets is None or connectivity.size == 0 or n_cells == 0:
        return None

    if offsets.size > 1:
        nodes_per_cell = int(offsets[1] - offsets[0])
    elif connectivity.size > 0:
        nodes_per_cell = connectivity.size // n_cells
    else:
        return None

    if nodes_per_cell <= 0 or connectivity.size != n_cells * nodes_per_cell:
        # Mixed cell types — cannot form a uniform tensor.
        return None

    return torch.from_numpy(np.ascontiguousarray(connectivity)).reshape(n_cells, nodes_per_cell).to(torch.int64)


def _offset_starts_ends(offsets: np.ndarray, n_cells: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalise VTK cell offsets into per-cell ``(starts, ends)`` arrays.

    Accepts either the VTK convention where ``offsets`` holds the cumulative
    *end* index of each cell (length ``n_cells``) or the ``n_cells + 1``
    convention with a leading ``0``.

    Parameters
    ----------
    offsets : numpy.ndarray
        Cell offset array.
    n_cells : int
        Number of cells.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(starts, ends)`` index arrays, each of length ``n_cells``.
    """
    offs = np.asarray(offsets, dtype=np.int64)
    if offs.size == n_cells:
        starts = np.empty(n_cells, dtype=np.int64)
        starts[0] = 0
        starts[1:] = offs[:-1]
        ends = offs
    else:
        starts = offs[:-1]
        ends = offs[1:]
    return starts, ends


def compute_centroids_from_rust(rust_data: Any) -> torch.Tensor:
    """Compute per-cell centroids from Rust connectivity.

    Uses a fast vectorised path for uniform cell types and a scatter-add
    path for mixed polyhedral / polygonal meshes.

    Parameters
    ----------
    rust_data : VtkMeshData
        Parsed VTK data from the Rust reader.  Must have cells.

    Returns
    -------
    torch.Tensor
        Centroid coordinates of shape ``(n_cells, 3)``.

    Raises
    ------
    ValueError
        If the mesh has no cell connectivity.
    """
    n_cells = rust_data.n_cells
    connectivity = rust_data.cells
    offsets = rust_data.cell_offsets

    if connectivity is None or connectivity.size == 0 or n_cells == 0:
        raise ValueError("point_source='cell_centroids' requires cells, but the file has none.")

    points = torch.from_numpy(np.ascontiguousarray(rust_data.points))

    starts, ends = _offset_starts_ends(offsets, n_cells)
    nodes_per_cell = ends - starts
    nodes_first = int(nodes_per_cell[0]) if nodes_per_cell.size > 0 else 0

    if nodes_first > 0 and connectivity.size == n_cells * nodes_first:
        # Uniform cell type — fast vectorised gather + mean.
        conn = torch.from_numpy(np.ascontiguousarray(connectivity)).reshape(n_cells, nodes_first).to(torch.int64)
        return points[conn].mean(dim=1)

    # Mixed cell types — scatter-add each node's coords into its cell, then
    # divide by the per-cell node count.
    conn_t = torch.from_numpy(np.ascontiguousarray(connectivity)).to(torch.int64)
    all_pts = points[conn_t]  # (total_nodes, 3)
    cell_ids = np.repeat(np.arange(n_cells, dtype=np.int64), nodes_per_cell)
    cell_ids_t = torch.from_numpy(cell_ids).unsqueeze(1).expand(-1, points.shape[1])
    centroids = torch.zeros(n_cells, points.shape[1], dtype=points.dtype)
    centroids.scatter_add_(0, cell_ids_t, all_pts)
    npc_t = torch.from_numpy(nodes_per_cell.astype(np.float64)).to(points.dtype)
    centroids /= npc_t.unsqueeze(1).clamp_min(1)
    return centroids


def tessellate_polygons(
    connectivity: Any,
    offsets: Any,
    n_cells: int,
) -> torch.Tensor:
    """Fan-tessellate mixed polygons into triangles.

    A polygon with *k* vertices produces *k - 2* triangles via fan
    triangulation from its first vertex.

    Parameters
    ----------
    connectivity : numpy.ndarray
        Flat connectivity array (vertex indices for all cells).
    offsets : numpy.ndarray
        Per-cell offsets into *connectivity* (VTK convention).
    n_cells : int
        Number of polygons.

    Returns
    -------
    torch.Tensor
        Triangle cells of shape ``(n_triangles, 3)`` with dtype ``int64``.
    """
    conn = np.asarray(connectivity, dtype=np.int64)
    starts, ends = _offset_starts_ends(offsets, n_cells)

    n_verts = ends - starts
    n_tris_per_cell = n_verts - 2
    total_tris = int(n_tris_per_cell.sum())

    cell_of_tri = np.repeat(np.arange(n_cells, dtype=np.int64), n_tris_per_cell)
    local_j = np.arange(total_tris, dtype=np.int64)
    cum_tris = np.zeros(n_cells + 1, dtype=np.int64)
    np.cumsum(n_tris_per_cell, out=cum_tris[1:])
    local_j -= cum_tris[cell_of_tri]

    poly_starts = starts[cell_of_tri]

    triangles = np.empty((total_tris, 3), dtype=np.int64)
    triangles[:, 0] = conn[poly_starts]
    triangles[:, 1] = conn[poly_starts + local_j + 1]
    triangles[:, 2] = conn[poly_starts + local_j + 2]

    return torch.from_numpy(triangles)


def expand_cell_data_for_tessellation(
    cell_data_dict: dict[str, torch.Tensor],
    offsets: Any,
    n_cells: int,
) -> dict[str, torch.Tensor]:
    """Repeat cell-data entries to match tessellated triangles.

    When a polygon with *k* vertices is split into *k - 2* triangles, its
    cell-data value is repeated *k - 2* times.

    Parameters
    ----------
    cell_data_dict : dict[str, torch.Tensor]
        Mapping of field name to tensor of shape ``(n_cells, ...)``.
    offsets : numpy.ndarray
        Cell offsets (same convention as :func:`tessellate_polygons`).
    n_cells : int
        Number of original polygons.

    Returns
    -------
    dict[str, torch.Tensor]
        Expanded cell data with shape ``(n_triangles, ...)``.
    """
    starts, ends = _offset_starts_ends(offsets, n_cells)
    n_tris_per_cell = (ends - starts - 2).astype(np.int64)
    repeat_counts = torch.from_numpy(n_tris_per_cell)
    return {name: torch.repeat_interleave(tensor, repeat_counts, dim=0) for name, tensor in cell_data_dict.items()}


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------


def downcast_fp32(mesh: Mesh) -> Mesh:
    """Downcast float64 tensors in a Mesh to float32 in place.

    Parameters
    ----------
    mesh : Mesh
        Input mesh (modified in place).

    Returns
    -------
    Mesh
        The same mesh with float64 arrays converted to float32.
    """
    if mesh.points is not None and mesh.points.dtype == torch.float64:
        mesh.points = mesh.points.float()

    for store in (mesh.point_data, mesh.cell_data):
        if store is None:
            continue
        for key in list(store.keys()):  # noqa: SIM118 - TensorDict needs .keys()
            tensor = store[key]
            if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float64:
                store[key] = tensor.float()

    return mesh


# ---------------------------------------------------------------------------
# Rust VtkMeshData -> Mesh
# ---------------------------------------------------------------------------


def _arrays_to_dict(arrays: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Convert a ``{name: ndarray}`` mapping to ``{name: tensor}``."""
    return {name: torch.from_numpy(np.ascontiguousarray(data)) for name, data in arrays.items()}


def mesh_from_rust_data(
    rust_data: Any,
    *,
    manifold_dim: int | Literal["auto"] = "auto",
    point_source: str = "vertices",
    fp32: bool = False,
    tessellate_surface: bool = False,
    cell_data_only: bool = False,
) -> Mesh:
    """Build a :class:`~physicsnemo.mesh.Mesh` from Rust ``VtkMeshData``.

    Parameters
    ----------
    rust_data : VtkMeshData
        Parsed VTK data from ``physicsnemo_curator._lib.vtk.read_vtk``.
    manifold_dim : int or {"auto"}
        Target manifold dimension (only used for ``point_source="vertices"``).
        ``"auto"`` resolves to 2 when cells are present, else 0.
    point_source : {"vertices", "cell_centroids"}
        ``"vertices"`` keeps mesh vertices and ``point_data``; ``"cell_centroids"``
        builds a point cloud from cell centroids and maps ``cell_data`` to
        ``point_data``.
    fp32 : bool
        If ``True``, downcast float64 arrays to float32.
    tessellate_surface : bool
        If ``True`` and the mesh has mixed-type surface cells, fan-tessellate
        polygons into triangles (and expand cell data accordingly).
    cell_data_only : bool
        If ``True`` (surface meshes), drop ``point_data`` and keep only
        ``cell_data``.

    Returns
    -------
    Mesh
        The converted mesh.
    """
    if point_source == "cell_centroids":
        centroids = compute_centroids_from_rust(rust_data)
        point_data = _arrays_to_dict(rust_data.cell_data)
        mesh = Mesh(points=centroids, cells=None, point_data=point_data or None)
        return downcast_fp32(mesh) if fp32 else mesh

    points = torch.from_numpy(np.ascontiguousarray(rust_data.points))
    point_data = None if cell_data_only else _arrays_to_dict(rust_data.point_data)

    resolved_dim = manifold_dim
    if resolved_dim == "auto":
        resolved_dim = 2 if (rust_data.cells is not None and rust_data.n_cells > 0) else 0

    has_cells = rust_data.cells is not None and rust_data.n_cells > 0
    if resolved_dim == 0 or not has_cells:
        mesh = Mesh(points=points, cells=None, point_data=point_data or None)
        return downcast_fp32(mesh) if fp32 else mesh

    cells = build_cells_from_rust(rust_data)
    cell_data = _arrays_to_dict(rust_data.cell_data)

    if cells is None:
        # Mixed cell types.
        if tessellate_surface:
            cells = tessellate_polygons(rust_data.cells, rust_data.cell_offsets, rust_data.n_cells)
            cell_data = expand_cell_data_for_tessellation(cell_data, rust_data.cell_offsets, rust_data.n_cells)
        else:
            logger.warning(
                "Mesh has mixed cell types and tessellate_surface=False; dropping cells/cell_data.",
            )
            cell_data = {}

    mesh = Mesh(
        points=points,
        cells=cells,
        point_data=point_data or None,
        cell_data=cell_data or None,
    )
    return downcast_fp32(mesh) if fp32 else mesh


# ---------------------------------------------------------------------------
# PyVista reading
# ---------------------------------------------------------------------------


def read_pyvista_dataset(
    path: str | pathlib.Path,
    *,
    include_arrays: list[str] | None = None,
    exclude_arrays: list[str] | None = None,
) -> pv.DataSet:
    """Read a VTK file with PyVista, optionally filtering data arrays.

    When *include_arrays* or *exclude_arrays* is given and the file is a VTK
    XML format (``.vtu`` / ``.vtp``), array selection is applied at the
    reader level so unwanted fields are never materialised.  Otherwise this
    is a plain :func:`pyvista.read`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the VTK file.
    include_arrays : list[str] or None
        If given, keep only these named point/cell data arrays.
    exclude_arrays : list[str] or None
        If given, drop these named point/cell data arrays.

    Returns
    -------
    pyvista.DataSet
        The loaded dataset.
    """
    import pyvista as pv

    path = str(path)
    suffix = pathlib.Path(path).suffix.lower()
    reader_name = _XML_READERS.get(suffix)

    if (include_arrays is None and exclude_arrays is None) or reader_name is None:
        return pv.read(path)

    import vtk as _vtk

    reader = getattr(_vtk, reader_name)()
    reader.SetFileName(path)
    reader.UpdateInformation()

    for selection in (reader.GetPointDataArraySelection(), reader.GetCellDataArraySelection()):
        if include_arrays is not None:
            selection.DisableAllArrays()
            for key in include_arrays:
                selection.EnableArray(key)
        elif exclude_arrays is not None:
            for key in exclude_arrays:
                selection.DisableArray(key)

    reader.Update()
    return pv.wrap(reader.GetOutput())


def mesh_from_pyvista(
    path: str | pathlib.Path,
    *,
    manifold_dim: int | Literal["auto"] = "auto",
    point_source: str = "vertices",
    warn_on_lost_data: bool = True,
    include_arrays: list[str] | None = None,
    exclude_arrays: list[str] | None = None,
    fp32: bool = False,
    cell_data_only: bool = False,
) -> Mesh:
    """Read a VTK file via PyVista and convert to a Mesh.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the VTK file.
    manifold_dim : int or {"auto"}
        Target manifold dimension passed to ``from_pyvista``.
    point_source : {"vertices", "cell_centroids"}
        Point-source mode passed to ``from_pyvista``.
    warn_on_lost_data : bool
        Forwarded to ``from_pyvista``.
    include_arrays, exclude_arrays : list[str] or None
        Reader-level data-array filters (see :func:`read_pyvista_dataset`).
    fp32 : bool
        If ``True``, downcast float64 arrays to float32.
    cell_data_only : bool
        If ``True``, drop ``point_data`` after conversion (surface meshes).

    Returns
    -------
    Mesh
        The converted mesh.
    """
    from physicsnemo.mesh.io import from_pyvista

    data = read_pyvista_dataset(path, include_arrays=include_arrays, exclude_arrays=exclude_arrays)
    mesh = from_pyvista(
        data,
        manifold_dim=manifold_dim,
        point_source=point_source,
        warn_on_lost_data=warn_on_lost_data,
    )
    if cell_data_only and mesh.point_data is not None:
        for key in list(mesh.point_data.keys()):  # noqa: SIM118 - TensorDict needs .keys()
            del mesh.point_data[key]
    return downcast_fp32(mesh) if fp32 else mesh


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def read_vtk_mesh(
    path: str | pathlib.Path,
    *,
    backend: Backend = "pyvista",
    manifold_dim: int | Literal["auto"] = "auto",
    point_source: str = "vertices",
    include_arrays: list[str] | None = None,
    exclude_arrays: list[str] | None = None,
    warn_on_lost_data: bool = True,
    fp32: bool = False,
    tessellate_surface: bool = False,
    cell_data_only: bool = False,
) -> Mesh:
    """Read a VTK file into a Mesh using the requested backend.

    When ``backend="rust"`` and the file/config are Rust-compatible (see
    :func:`can_use_rust`), the native reader is used with a transparent
    fallback to PyVista on any failure.  Otherwise PyVista is used directly.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the VTK file.
    backend : {"pyvista", "rust"}
        Preferred reading backend.
    manifold_dim : int or {"auto"}
        Target manifold dimension.
    point_source : {"vertices", "cell_centroids"}
        Point-source mode.
    include_arrays, exclude_arrays : list[str] or None
        Data-array include/exclude filters (applied at the reader level on
        both backends).
    warn_on_lost_data : bool
        Forwarded to the PyVista conversion.
    fp32 : bool
        If ``True``, downcast float64 arrays to float32.
    tessellate_surface : bool
        If ``True``, fan-tessellate mixed surface polygons (Rust path).
    cell_data_only : bool
        If ``True``, keep only cell data (surface meshes).

    Returns
    -------
    Mesh
        The converted mesh.
    """
    path = str(path)

    if backend == "rust" and can_use_rust(path, point_source, manifold_dim):
        try:
            from physicsnemo_curator._lib import vtk as _rust_vtk

            skip_cells = point_source == "vertices" and manifold_dim == 0
            skip_point_data = cell_data_only or point_source == "cell_centroids"
            rust_data = _rust_vtk.read_vtk(
                path,
                include_arrays=include_arrays,
                exclude_arrays=exclude_arrays,
                skip_cells=skip_cells,
                skip_point_data=skip_point_data,
            )
            return mesh_from_rust_data(
                rust_data,
                manifold_dim=manifold_dim,
                point_source=point_source,
                fp32=fp32,
                tessellate_surface=tessellate_surface,
                cell_data_only=cell_data_only,
            )
        except Exception as exc:  # noqa: BLE001 - any Rust failure falls back to PyVista
            logger.warning("Rust VTK reader failed for %s (%s); falling back to PyVista.", path, exc)

    return mesh_from_pyvista(
        path,
        manifold_dim=manifold_dim,
        point_source=point_source,
        warn_on_lost_data=warn_on_lost_data,
        include_arrays=include_arrays,
        exclude_arrays=exclude_arrays,
        fp32=fp32,
        cell_data_only=cell_data_only,
    )
