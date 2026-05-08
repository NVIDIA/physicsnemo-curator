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

"""VTU writer sink for PhysicsNeMo Mesh objects.

Writes :class:`physicsnemo.mesh.Mesh` objects to VTU (VTK UnstructuredGrid)
files, producing output compatible with the PhysicsNeMo drop_test recipe
and other VTK-based workflows.

The sink stores per-timestep displacement fields as point arrays
(``displacement_tNNNN``), with mesh points set to reference coordinates
(t=0).  This format enables efficient storage and downstream reconstruction
of position trajectories.
"""

from __future__ import annotations

import pathlib
import re
import time
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from physicsnemo_curator.core.base import Param, Sink
from physicsnemo_curator.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

    from physicsnemo.mesh import Mesh

    from physicsnemo_curator.core.base import Source


class MeshVTUSink(Sink["Mesh"]):
    """Write :class:`~physicsnemo.mesh.Mesh` objects to VTU files.

    Each mesh is written to a separate ``.vtu`` file.  The sink stores:

    * **Points**: Reference coordinates (t=0 positions)
    * **Cells**: Element connectivity with VTK cell types
    * **Point arrays**: ``thickness`` plus ``displacement_tNNNN`` for each
      timestep, and optional stress fields (``Von_Mises_tNNNN``)
    * **Cell arrays**: Optional per-element fields

    This format matches the output expected by the PhysicsNeMo drop_test
    recipe for physics-informed machine learning.

    Parameters
    ----------
    output_dir : str
        Directory where VTU files will be written.
    naming_template : str | None
        Format string for output names.  Placeholders: ``{index}``,
        ``{seq}``, ``{run_id}``.  Default: ``mesh_{index:04d}``.
    flip_triangle_normals : bool
        Whether to reverse triangle vertex order for VTK normal convention.
        Default is True.

    Examples
    --------
    >>> sink = MeshVTUSink(output_dir="./output/")  # doctest: +SKIP
    >>> paths = sink(mesh_generator, index=0)  # doctest: +SKIP
    >>> paths  # doctest: +SKIP
    ['./output/mesh_0000.vtu']
    """

    name: ClassVar[str] = "Mesh VTU Writer"
    description: ClassVar[str] = "Write Mesh objects to VTK UnstructuredGrid files"

    # VTK cell type mapping (nodes per cell -> VTK type code)
    _VTK_CELL_TYPES: ClassVar[dict[int, int]] = {
        3: 5,  # VTK_TRIANGLE
        4: 10,  # VTK_TETRA
        5: 14,  # VTK_PYRAMID
        6: 13,  # VTK_WEDGE
        8: 12,  # VTK_HEXAHEDRON
    }

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the VTU sink.

        Returns
        -------
        list[Param]
            Descriptors for configuration parameters.
        """
        return [
            Param(name="output_dir", description="Output directory for VTU files", type=str),
            Param(
                name="naming_template",
                description="Format string for output names ({index}, {seq}, {run_id})",
                type=str,
                default=None,
            ),
            Param(
                name="flip_triangle_normals",
                description="Reverse triangle vertex order for VTK normal convention",
                type=bool,
                default=True,
            ),
        ]

    def __init__(
        self,
        output_dir: str,
        naming_template: str | None = None,
        flip_triangle_normals: bool = True,
    ) -> None:
        """Initialize the VTU sink.

        Parameters
        ----------
        output_dir : str
            Directory where VTU files will be written.
        naming_template : str | None
            Format string for output names.
        flip_triangle_normals : bool
            Whether to reverse triangle vertex order.
        """
        self._log = get_logger(self)
        self._output_dir = pathlib.Path(output_dir)
        self._naming_template = naming_template or "mesh_{index:04d}"
        self._flip_triangle_normals = flip_triangle_normals
        self._source: Source[Mesh] | None = None

    def set_source(self, source: Source[Mesh]) -> None:
        """Set the source for resolving naming placeholders.

        Parameters
        ----------
        source : Source[Mesh]
            The pipeline source.
        """
        self._source = source

    def __call__(self, items: Iterator[Mesh], index: int) -> list[str]:
        """Write meshes to VTU files.

        Parameters
        ----------
        items : Iterator[Mesh]
            Stream of Mesh objects to write.
        index : int
            Pipeline source index.

        Returns
        -------
        list[str]
            Paths of the written VTU files.
        """
        t0 = time.perf_counter()
        self._log.info("Writing index %d to VTU", index)

        paths: list[str] = []

        # Resolve run_id if source supports it
        run_id: int | str = ""
        if self._source is not None and hasattr(self._source, "run_id"):
            run_id = self._source.run_id(index)  # ty: ignore[call-non-callable]

        for seq, mesh in enumerate(items):
            output_name = self._naming_template.format(index=index, seq=seq, run_id=run_id)
            if not output_name.endswith(".vtu"):
                output_name = f"{output_name}.vtu"

            output_path = self._output_dir / output_name
            self._write_mesh(mesh, output_path)
            paths.append(str(output_path))
            self._log.debug("Wrote mesh to %s", output_path)

        self._log.info("Write complete: %d files (%.2fs)", len(paths), time.perf_counter() - t0)
        return paths

    def _write_mesh(self, mesh: Mesh, output_path: pathlib.Path) -> None:
        """Write a single mesh to a VTU file.

        Parameters
        ----------
        mesh : Mesh
            The mesh to write.
        output_path : pathlib.Path
            Path to the output VTU file.
        """
        import pyvista as pv

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract points (reference coordinates)
        points = mesh.points.numpy().astype(np.float32)

        # Build cells array and cell types for pyvista
        cells_array, cell_types = self._build_cells(mesh)

        # Create UnstructuredGrid
        grid: Any = pv.UnstructuredGrid(cells_array, cell_types, points)

        # Add displacement fields as point arrays
        disp_keys = self._get_displacement_keys(mesh)
        for key in disp_keys:
            disp_tensor = mesh.point_data.get(key)
            if disp_tensor is None:
                continue
            disp = disp_tensor.numpy().astype(np.float32)
            # Convert to 4-digit format for consistency with PR #56
            idx_match = re.match(r"displacement_t(\d+)", key)
            if idx_match:
                idx = int(idx_match.group(1))
                vtu_key = f"displacement_t{idx:04d}"
            else:
                vtu_key = key
            grid.point_data[vtu_key] = disp

        # Add thickness
        n_points = mesh.n_points
        if mesh.point_data is not None and "thickness" in mesh.point_data.keys():  # noqa: SIM118
            thickness = mesh.point_data.get("thickness").numpy().astype(np.float32)
        else:
            thickness = np.zeros(n_points, dtype=np.float32)
        grid.point_data["thickness"] = thickness

        # Add stress fields if present (Von_Mises_tNNNN)
        if mesh.point_data is not None:
            stress_pattern = re.compile(r"^stress_vm_t(\d+)$")
            for key in mesh.point_data.keys():  # noqa: SIM118
                key_str = str(key)
                match = stress_pattern.match(key_str)
                if match:
                    idx = int(match.group(1))
                    data = mesh.point_data.get(key).numpy().astype(np.float32)
                    grid.point_data[f"Von_Mises_t{idx:04d}"] = data

            # Add stress_voigt fields if present
            voigt_pattern = re.compile(r"^stress_voigt_t(\d+)$")
            for key in mesh.point_data.keys():  # noqa: SIM118
                key_str = str(key)
                match = voigt_pattern.match(key_str)
                if match:
                    idx = int(match.group(1))
                    data = mesh.point_data.get(key).numpy().astype(np.float32)
                    grid.point_data[f"stress_voigt_t{idx:04d}"] = data

        # Add other point data (excluding displacement_t*, thickness, stress_vm_t*, stress_voigt_t*)
        if mesh.point_data is not None:
            skip_patterns = [r"^displacement_t\d+$", r"^stress_vm_t\d+$", r"^stress_voigt_t\d+$", r"^thickness$"]
            for key in mesh.point_data.keys():  # noqa: SIM118
                key_str = str(key)
                if any(re.match(p, key_str) for p in skip_patterns):
                    continue
                data = mesh.point_data.get(key).numpy().astype(np.float32)
                grid.point_data[key_str] = data

        # Add cell data (skip internal metadata fields)
        if mesh.cell_data is not None:
            for key in mesh.cell_data.keys():  # noqa: SIM118
                key_str = str(key)
                if key_str == "vtk_cell_type":
                    continue
                data = mesh.cell_data.get(key).numpy().astype(np.float32)
                grid.cell_data[key_str] = data

        # Write to file (atomic: write to temp .vtu, then rename)
        temp_path = output_path.parent / f".{output_path.stem}_temp.vtu"
        grid.save(str(temp_path))
        temp_path.rename(output_path)

    def _build_cells(self, mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
        """Build pyvista-compatible cells array and cell types.

        Handles both uniform-size cells and mixed-size cells.  Mixed cells
        are stored in global_data as flat connectivity + offsets (because the
        Mesh class requires uniform cell sizes).  When cell_data contains a
        'vtk_cell_type' field, it is used directly for per-cell VTK type codes.
        Otherwise, the cell type is inferred from the number of nodes per cell.

        Parameters
        ----------
        mesh : Mesh
            The mesh with cells tensor.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Cells array (VTK format) and cell types array.
        """
        # Check for empty cells (point-cloud sentinel from Mesh constructor)
        if mesh.cells.shape[0] == 0:
            # Check for mixed connectivity in global_data
            if mesh.global_data is not None and "mixed_connectivity" in mesh.global_data:
                return self._build_mixed_cells_from_global_data(mesh)
            return np.array([], dtype=np.int64), np.array([], dtype=np.uint8)

        cells_tensor = mesh.cells.numpy()
        n_cells = cells_tensor.shape[0]
        max_nodes_per_cell = cells_tensor.shape[1]

        # Check if we have mixed cells (indicated by -1 padding)
        has_padding = np.any(cells_tensor < 0)

        if not has_padding:
            # Uniform cell type — original path
            nodes_per_cell = max_nodes_per_cell
            vtk_type = self._VTK_CELL_TYPES.get(nodes_per_cell)
            if vtk_type is None:
                self._log.warning("Unsupported cell type with %d nodes, skipping cells", nodes_per_cell)
                return np.array([], dtype=np.int64), np.array([], dtype=np.uint8)

            # Handle triangle normal flipping
            if nodes_per_cell == 3 and self._flip_triangle_normals:
                cells_tensor = cells_tensor[:, [0, 2, 1]]

            # Build VTK cells array
            cells_list = []
            for i in range(n_cells):
                cells_list.append(nodes_per_cell)
                cells_list.extend(cells_tensor[i].tolist())

            cells_array = np.array(cells_list, dtype=np.int64)
            cell_types = np.full(n_cells, vtk_type, dtype=np.uint8)

            return cells_array, cell_types

        # Mixed cell types — use per-cell node counts from non-negative entries
        # Try to use explicit vtk_cell_type from cell_data if available
        has_explicit_types = (
            mesh.cell_data is not None and "vtk_cell_type" in mesh.cell_data  # noqa: SIM118
        )

        explicit_types = mesh.cell_data.get("vtk_cell_type").numpy().astype(np.uint8) if has_explicit_types else None

        cells_list = []
        cell_types_list = []

        for i in range(n_cells):
            row = cells_tensor[i]
            # Count valid (non-negative) node indices
            valid_mask = row >= 0
            actual_nodes = int(valid_mask.sum())
            valid_nodes = row[valid_mask]

            # Handle triangle normal flipping
            if actual_nodes == 3 and self._flip_triangle_normals:
                valid_nodes = valid_nodes[[0, 2, 1]]

            cells_list.append(actual_nodes)
            cells_list.extend(valid_nodes.tolist())

            # Determine VTK cell type
            if explicit_types is not None:
                cell_types_list.append(explicit_types[i])
            else:
                vtk_type = self._VTK_CELL_TYPES.get(actual_nodes, 0)
                cell_types_list.append(vtk_type)

        cells_array = np.array(cells_list, dtype=np.int64)
        cell_types = np.array(cell_types_list, dtype=np.uint8)

        return cells_array, cell_types

    def _build_mixed_cells_from_global_data(self, mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
        """Build cells from mixed connectivity stored in global_data.

        Parameters
        ----------
        mesh : Mesh
            Mesh with mixed connectivity in global_data.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Cells array (VTK format) and cell types array.
        """
        connectivity: np.ndarray = mesh.global_data["mixed_connectivity"].numpy()  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        offsets: np.ndarray = mesh.global_data["mixed_offsets"].numpy()  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        mixed_cell_types: np.ndarray = mesh.global_data["mixed_cell_types"].numpy()  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        mixed_cell_types = mixed_cell_types.astype(np.uint8)

        n_cells = len(offsets) - 1
        if n_cells == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.uint8)

        cells_list: list[int] = []
        for i in range(n_cells):
            start = offsets[i]
            end = offsets[i + 1]
            cell_nodes = connectivity[start:end]
            n_nodes = len(cell_nodes)

            # Handle triangle normal flipping
            if n_nodes == 3 and self._flip_triangle_normals:
                cell_nodes = cell_nodes[[0, 2, 1]]

            cells_list.append(n_nodes)
            cells_list.extend(cell_nodes.tolist())

        cells_array = np.array(cells_list, dtype=np.int64)
        return cells_array, mixed_cell_types

    def _get_displacement_keys(self, mesh: Mesh) -> list[str]:
        """Extract sorted displacement field keys from mesh point_data.

        Parameters
        ----------
        mesh : Mesh
            The mesh to inspect.

        Returns
        -------
        list[str]
            Sorted displacement keys (e.g., ["displacement_t000", ...]).
        """
        if mesh.point_data is None:
            return []

        pattern = re.compile(r"^displacement_t(\d+)$")
        keys = []
        for key in mesh.point_data.keys():  # noqa: SIM118
            key_str = str(key)
            if pattern.match(key_str):
                keys.append(key_str)

        return sorted(keys)

    @property
    def output_dir(self) -> pathlib.Path:
        """Return the output directory path."""
        return self._output_dir
