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

"""Original Nastran connectivity plus explicit simplicial projections by dimension."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

# Nastran node order agrees with these VTK cells except CHEXA20, whose
# midside edge ordering is explicitly remapped below.
_CELL_TYPES: dict[str, dict[int, tuple[int, int]]] = {
    "CROD": {2: (3, 1)},
    "CONROD": {2: (3, 1)},
    "CTUBE": {2: (3, 1)},
    "CBAR": {2: (3, 1)},
    "CBEAM": {2: (3, 1)},
    "CTRIA3": {3: (5, 2)},
    "CTRIA6": {6: (22, 2)},
    "CQUAD4": {4: (9, 2)},
    "CQUADR": {4: (9, 2)},
    "CQUAD8": {8: (23, 2)},
    "CSHEAR": {4: (9, 2)},
    "CTETRA": {4: (10, 3), 10: (24, 3)},
    "CHEXA": {8: (12, 3), 20: (25, 3)},
    "CPENTA": {6: (13, 3), 15: (26, 3)},
    "CPYRAM": {5: (14, 3), 13: (27, 3)},
}


def _element_material(element: Any) -> tuple[int, Any | None]:
    """Find the primary material without assuming all properties are PSOLID/MAT1."""
    prop = getattr(element, "pid_ref", None)
    if prop is None:
        return int(getattr(element, "mid", 0) or 0), getattr(element, "mid_ref", None)
    for name in ("mid_ref", "mid1_ref"):
        material = getattr(prop, name, None)
        if material is not None:
            return int(material.mid), material
    mids = getattr(prop, "mids_ref", None)
    if mids:
        return int(mids[0].mid), mids[0]
    return 0, None


def _original_geometry(model: Any) -> tuple[torch.Tensor, TensorDict, TensorDict, TensorDict]:
    """Retain original element IDs/node order, numeric card properties, and GRID frames."""
    node_ids = sorted(model.nodes)
    if not node_ids or not model.elements:
        raise ValueError("Nastran deck must contain GRID nodes and structural elements")
    index = {nid: i for i, nid in enumerate(node_ids)}
    points = np.array([model.nodes[nid].get_position() for nid in node_ids], dtype=np.float64)
    permanent = np.zeros((len(node_ids), 6), dtype=bool)
    basis = np.full((len(node_ids), 3, 3), np.nan)
    for i, nid in enumerate(node_ids):
        node = model.nodes[nid]
        for component in str(node.ps):
            if component in "123456":
                permanent[i, int(component) - 1] = True
        coord = model.coords[node.Cd()]
        if coord.type in {"CORD1R", "CORD2R"}:
            basis[i] = coord.beta()
    point_data = TensorDict(
        {
            "node_id": torch.tensor(node_ids),
            "input_coordinate_id": torch.tensor([model.nodes[n].Cp() for n in node_ids]),
            "output_coordinate_id": torch.tensor([model.nodes[n].Cd() for n in node_ids]),
            "output_coordinate_basis": torch.from_numpy(basis),
            "permanent_spc": torch.from_numpy(permanent),
        },
        batch_size=[len(node_ids)],
    )
    connectivity, nastran_connectivity, offsets, types, dimensions = [], [], [0], [], []
    eids, pids, mids, materials = [], [], [], []
    scalar_properties: list[dict[str, float]] = []
    for eid, element in sorted(model.elements.items()):
        nids = list(element.node_ids)
        info = _CELL_TYPES.get(element.type, {}).get(len(nids))
        if info is None or any(nid is None for nid in nids):
            raise NotImplementedError(f"Element {eid}: unsupported {element.type} topology with {len(nids)} nodes")
        vtk_type, dimension = info
        vtk_ids = nids
        if element.type == "CHEXA" and len(nids) == 20:
            vtk_ids = [*nids[:12], *nids[16:20], *nids[12:16]]
        nastran_connectivity.extend(nids)
        connectivity.extend(index[nid] for nid in vtk_ids)
        offsets.append(len(connectivity))
        types.append(vtk_type)
        dimensions.append(dimension)
        eids.append(eid)
        pids.append(int(getattr(element, "pid", 0) or 0))
        mid, material = _element_material(element)
        mids.append(mid)
        materials.append(material)
        prop = getattr(element, "pid_ref", None)
        scalars = {}
        for prefix, card in (("material", material), ("property", prop)):
            if card is not None:
                for name, value in vars(card).items():
                    if not name.startswith("_") and isinstance(value, float | np.floating):
                        scalars[prefix + "_" + name] = float(value)
        scalar_properties.append(scalars)
    cell_data = TensorDict(
        {"element_id": torch.tensor(eids), "property_id": torch.tensor(pids), "material_id": torch.tensor(mids)},
        batch_size=[len(eids)],
    )
    for name in sorted({key for row in scalar_properties for key in row}):
        cell_data[name] = torch.tensor([row.get(name, np.nan) for row in scalar_properties], dtype=torch.float64)
    # Keep the original public MAT1 aliases when a material supports them.
    for field, method in (("young_modulus", "E"), ("poisson_ratio", "Nu"), ("density", "Rho")):
        cell_data[field] = torch.tensor(
            [
                float(getattr(material, method)())
                if material is not None and callable(getattr(material, method, None))
                else np.nan
                for material in materials
            ],
            dtype=torch.float64,
        )
    geometry = TensorDict(
        {
            "point_node_ids": torch.tensor(node_ids),
            "connectivity": torch.tensor(connectivity),
            "nastran_node_ids": torch.tensor(nastran_connectivity),
            "offsets": torch.tensor(offsets),
            "vtk_cell_type": torch.tensor(types, dtype=torch.uint8),
            "dimension": torch.tensor(dimensions),
            "element_id": torch.tensor(eids),
        },
        batch_size=[],
    )
    return torch.from_numpy(points), point_data, geometry, cell_data


def _geometry_groups(points: torch.Tensor, geometry: TensorDict, cell_data: TensorDict) -> list[dict[str, Any]]:
    """Build separate native meshes for line/surface/volume groups, retaining their original cells."""
    import pyvista as pv

    groups = []
    for dimension in sorted(geometry["dimension"].unique().tolist()):
        selected = (geometry["dimension"] == dimension).nonzero().flatten()
        flat, original_nodes, offsets = [], [], [0]
        for row in selected.tolist():
            start, end = geometry["offsets"][row : row + 2].tolist()
            flat.extend(geometry["connectivity"][start:end].tolist())
            original_nodes.extend(geometry["nastran_node_ids"][start:end].tolist())
            offsets.append(len(flat))
        raw = TensorDict(
            {
                "point_node_ids": geometry["point_node_ids"],
                "connectivity": torch.tensor(flat),
                "nastran_node_ids": torch.tensor(original_nodes),
                "offsets": torch.tensor(offsets),
                "vtk_cell_type": geometry["vtk_cell_type"][selected],
                "dimension": geometry["dimension"][selected],
                "element_id": geometry["element_id"][selected],
            },
            batch_size=[],
        )
        vtk_cells = []
        for start, end in zip(offsets[:-1], offsets[1:], strict=True):
            vtk_cells.extend([end - start, *flat[start:end]])
        grid = pv.UnstructuredGrid(np.array(vtk_cells), raw["vtk_cell_type"].numpy(), points.numpy())
        grid.cell_data["parent_row"] = np.arange(len(selected))
        grid.point_data["original_row"] = np.arange(len(points))
        tet_rows = np.flatnonzero(np.isin(raw["vtk_cell_type"].numpy(), [10, 24]))
        parts = []
        if len(tet_rows):
            tet10 = np.array(
                [
                    [0, 4, 6, 7],
                    [4, 1, 5, 8],
                    [6, 5, 2, 9],
                    [7, 8, 9, 3],
                    [4, 5, 6, 9],
                    [4, 5, 8, 9],
                    [4, 7, 8, 9],
                    [4, 6, 7, 9],
                ]
            )
            children, parents = [], []
            for row in tet_rows:
                start, end = offsets[row : row + 2]
                ids = np.array(flat[start:end])
                child = ids[tet10] if len(ids) == 10 else ids.reshape(1, 4)
                children.extend(child)
                parents.extend([row] * len(child))
            child_cells = np.array(children, dtype=np.int64)
            packed = np.column_stack((np.full(len(child_cells), 4), child_cells)).ravel()
            triangulated = pv.UnstructuredGrid(packed, np.full(len(child_cells), 10, dtype=np.uint8), points.numpy())
            triangulated.point_data["original_row"] = np.arange(len(points))
            triangulated.cell_data["parent_row"] = np.array(parents)
            parts.append(triangulated)
        other_rows = np.flatnonzero(~np.isin(raw["vtk_cell_type"].numpy(), [10, 24]))
        if len(other_rows):
            parts.append(grid.extract_cells(other_rows).triangulate())
        child_batches, parent_batches = [], []
        for triangulated in parts:
            expected_type = {1: 3, 2: 5, 3: 10}[dimension]
            if np.any(triangulated.celltypes != expected_type):
                raise NotImplementedError(f"Cannot create a simplicial projection for dimension {dimension}")
            # Reject added interpolation points instead of inventing nodal values.
            rows = triangulated.point_data["original_row"].astype(np.int64)
            if not np.array_equal(triangulated.points, points.numpy()[rows]):
                raise NotImplementedError(
                    "Tessellation introduced interpolation points; original results cannot be guessed"
                )
            child_batches.append(rows[triangulated.cell_connectivity.reshape(-1, dimension + 1)])
            parent_batches.append(triangulated.cell_data["parent_row"].astype(np.int64))
        cells = np.concatenate(child_batches)
        parent = torch.from_numpy(np.concatenate(parent_batches))
        if set(parent.tolist()) != set(range(len(selected))):
            raise ValueError("Tessellation dropped original Nastran elements")
        if dimension == 3:
            vertices = points.numpy()[cells]
            volumes = np.linalg.det(vertices[:, 1:] - vertices[:, :1])
            negative = volumes < 0
            cells[negative, :2] = cells[negative, 1::-1]
            if np.any(volumes == 0):
                raise ValueError("Nastran mesh contains degenerate tetrahedra")
        groups.append(
            {
                "dimension": dimension,
                "geometry": raw,
                "element_data": cell_data[selected].clone(),
                "cells": torch.from_numpy(cells.copy()),
                "parent_row": parent,
            }
        )
    return groups
