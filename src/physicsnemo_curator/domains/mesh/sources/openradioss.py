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

"""OpenRadioss VTK data source for drop test mesh pipelines.

Reads per-timestep VTK files produced by OpenRadioss ``anim_to_vtk``
converter and yields :class:`~physicsnemo.mesh.Mesh` objects for use
in curator pipelines.

Each source index maps to one simulation run directory containing
multiple VTK files (one per timestep).  The resulting mesh carries:

* ``points`` --- *(N, 3)* reference coordinates at t=0
* ``cells`` --- *(E, nodes_per_cell)* element connectivity
* ``point_data`` --- ``thickness`` *(N,)* (zeros for solid elements)
  plus ``displacement_t{idx:03d}`` *(N, 3)* for each timestep
* ``cell_data`` (optional) --- ``stress_vm_t{idx:03d}`` *(E,)* and
  other per-element fields for each timestep
* ``global_data`` --- ``num_timesteps`` scalar

Examples
--------
>>> source = OpenRadiossSource(input_dir="/data/drop_test_runs")  # doctest: +SKIP
>>> len(source)  # doctest: +SKIP
50
>>> mesh = next(source[0])  # doctest: +SKIP
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from tensordict import TensorDict

from physicsnemo_curator.core.base import REQUIRED, Param, Source
from physicsnemo_curator.core.logging import flush_logs, get_logger

if TYPE_CHECKING:
    from collections.abc import Generator

    from physicsnemo.mesh import Mesh

logger = logging.getLogger(__name__)


def _von_mises_from_voigt(sig: np.ndarray) -> np.ndarray:
    """Compute von Mises stress from Voigt components.

    Voigt ordering: ``[sigma_x, sigma_y, sigma_z, tau_xy, tau_yz, tau_zx]``.

    Parameters
    ----------
    sig : np.ndarray
        Shape ``(..., 6)``.

    Returns
    -------
    np.ndarray
        Shape ``(...)`` — scalar von Mises stress.
    """
    sx, sy, sz = sig[..., 0], sig[..., 1], sig[..., 2]
    txy, tyz, tzx = sig[..., 3], sig[..., 4], sig[..., 5]
    j2 = 0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3.0 * (txy**2 + tyz**2 + tzx**2)
    return np.sqrt(np.maximum(j2, 0.0))


class OpenRadiossSource(Source["Mesh"]):
    """Read drop test simulation meshes from OpenRadioss VTK files.

    Scans ``input_dir`` for subdirectories containing VTK files matching
    ``vtk_glob``.  Each subdirectory is one simulation run.  The source
    reads node coordinates from each timestep file, computes displacements
    relative to t=0, and optionally extracts stress/strain fields.

    Parameters
    ----------
    input_dir : str
        Root directory containing run subdirectories, each with VTK files.
    vtk_glob : str
        Glob pattern for VTK files within each run directory.  Default is
        ``"*.vtk"`` which matches all VTK files.
    read_stress : bool
        If ``True``, read nodal stress from point data (GPS_SIGXX, etc.)
        and compute Von Mises stress.  Default is ``False``.
    read_velocity : bool
        If ``True``, read velocity fields from point data.  Default is
        ``False``.
    read_acceleration : bool
        If ``True``, read acceleration fields from point data.  Default
        is ``False``.
    read_temperature : bool
        If ``True``, read temperature fields from point data.  Default
        is ``False``.

    Examples
    --------
    >>> source = OpenRadiossSource(
    ...     input_dir="/data/drop_test_runs",
    ...     vtk_glob="Cell_Phone_Drop*.vtk",
    ...     read_stress=True,
    ... )  # doctest: +SKIP
    >>> len(source)  # doctest: +SKIP
    50
    """

    name: ClassVar[str] = "OpenRadioss VTK"
    description: ClassVar[str] = (
        "OpenRadioss per-timestep VTK reader --- solid/shell meshes with "
        "multi-timestep displacement and optional stress/velocity fields"
    )

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the OpenRadioss source.

        Returns
        -------
        list[Param]
            Parameter list for CLI configuration.
        """
        return [
            Param(
                name="input_dir",
                description="Root directory containing run subdirectories with VTK files",
                type=str,
                default=REQUIRED,
            ),
            Param(
                name="vtk_glob",
                description="Glob pattern for VTK files within each run",
                type=str,
                default="*.vtk",
            ),
            Param(
                name="read_stress",
                description="Read nodal stress from point data (GPS_SIG* fields)",
                type=bool,
                default=False,
            ),
            Param(
                name="read_velocity",
                description="Read velocity fields from point data",
                type=bool,
                default=False,
            ),
            Param(
                name="read_acceleration",
                description="Read acceleration fields from point data",
                type=bool,
                default=False,
            ),
            Param(
                name="read_temperature",
                description="Read temperature fields from point data",
                type=bool,
                default=False,
            ),
            Param(
                name="cell_type",
                description=(
                    "VTK cell type to extract: 'tet' (10), 'tri' (5), "
                    "'all' (most common), or 'mixed' (all types, padded)"
                ),
                type=str,
                default="tet",
            ),
        ]

    # VTK cell type codes
    _VTK_TET = 10
    _VTK_TRI = 5
    _VTK_QUAD = 9
    _VTK_HEX = 12
    _VTK_WEDGE = 13
    _VTK_PYRAMID = 14

    def __init__(
        self,
        input_dir: str,
        vtk_glob: str = "*.vtk",
        read_stress: bool = False,
        read_velocity: bool = False,
        read_acceleration: bool = False,
        read_temperature: bool = False,
        cell_type: str = "tet",
    ) -> None:
        """Initialize the OpenRadioss source.

        Parameters
        ----------
        input_dir : str
            Root directory containing run subdirectories.
        vtk_glob : str
            Glob pattern for VTK files.
        read_stress : bool
            Read von Mises stress from element data.
        read_velocity : bool
            Read velocity fields from point data.
        read_acceleration : bool
            Read acceleration fields from point data.
        read_temperature : bool
            Read temperature fields from point data.
        cell_type : str
            VTK cell type to extract: 'tet', 'tri', 'all', or 'mixed'.
        """
        self._log = get_logger(self)
        self._flush_logs = flush_logs

        self._input_dir = Path(input_dir)
        self._vtk_glob = vtk_glob
        self._read_stress = read_stress
        self._read_velocity = read_velocity
        self._read_acceleration = read_acceleration
        self._read_temperature = read_temperature
        self._cell_type = cell_type

        # Validate cell_type
        if cell_type not in ("tet", "tri", "all", "mixed"):
            msg = f"cell_type must be 'tet', 'tri', 'all', or 'mixed', got: {cell_type}"
            raise ValueError(msg)

        if not self._input_dir.is_dir():
            msg = f"input_dir does not exist or is not a directory: {self._input_dir}"
            raise FileNotFoundError(msg)

        # Eagerly discover run directories (lightweight).
        self._run_dirs = self._discover_runs()
        logger.info("OpenRadiossSource: discovered %d runs in %s", len(self._run_dirs), self._input_dir)

    def __len__(self) -> int:
        """Return the number of simulation runs discovered.

        Returns
        -------
        int
            Number of runs.
        """
        return len(self._run_dirs)

    def __getitem__(self, index: int) -> Generator[Mesh]:
        """Read the mesh for the *index*-th simulation run.

        Parameters
        ----------
        index : int
            Zero-based run index (supports negative indexing).

        Yields
        ------
        Mesh
            Mesh with displacement fields per timestep.
        """
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            msg = f"Index {index} out of range for source with {len(self)} items."
            raise IndexError(msg)

        run_dir = self._run_dirs[index]
        yield self._read_run(run_dir, index)

    def _discover_runs(self) -> list[Path]:
        """Find subdirectories containing VTK files matching the glob.

        Returns
        -------
        list[Path]
            Sorted list of run directories.
        """
        runs = set()
        for vtk_file in self._input_dir.rglob(self._vtk_glob):
            if vtk_file.is_file() and vtk_file.parent != self._input_dir:
                runs.add(vtk_file.parent)

        return sorted(runs)

    def _read_run(self, run_dir: Path, index: int) -> Mesh:
        """Read a single run directory into a Mesh.

        Parameters
        ----------
        run_dir : Path
            Path to the run directory.
        index : int
            Source index for logging.

        Returns
        -------
        Mesh
            The constructed mesh with all fields.
        """
        import pyvista as pv
        from physicsnemo.mesh import Mesh

        t0 = time.perf_counter()
        self._log.info("Reading run %d: %s", index, run_dir.name)
        self._flush_logs()

        # Find and sort VTK files
        vtk_files = sorted(run_dir.glob(self._vtk_glob))
        if not vtk_files:
            msg = f"No VTK files matching '{self._vtk_glob}' found in {run_dir}"
            raise FileNotFoundError(msg)

        n_timesteps = len(vtk_files)
        self._log.debug("Found %d VTK files", n_timesteps)

        # Read first file to get mesh structure
        grid0 = pv.read(vtk_files[0])
        n_points = grid0.n_points

        # Extract reference coordinates (t=0)
        coords = np.array(grid0.points, dtype=np.float32)  # (N, 3)

        # Extract cell connectivity (filtered by cell_type)
        cells_array, cell_types, cell_mask = self._extract_connectivity(grid0)
        n_cells = cells_array.shape[0]  # Number of filtered cells

        # Compute referenced nodes for point pruning (done after reading all data)
        # In mixed mode, -1 is the padding sentinel and should be excluded
        valid_indices = cells_array[cells_array >= 0]
        referenced_nodes = np.unique(valid_indices)
        n_points_pruned = len(referenced_nodes)

        if n_points_pruned < n_points:
            self._log.info(
                "Will prune unreferenced points: %d -> %d (%.1f%% reduction)",
                n_points,
                n_points_pruned,
                100.0 * (1.0 - n_points_pruned / n_points),
            )

        # Initialize arrays for all timesteps (at full original size for reading)
        positions = np.zeros((n_timesteps, n_points, 3), dtype=np.float32)
        positions[0] = coords

        # Optional field arrays
        velocities = np.zeros((n_timesteps, n_points, 3), dtype=np.float32) if self._read_velocity else None
        accelerations = np.zeros((n_timesteps, n_points, 3), dtype=np.float32) if self._read_acceleration else None
        temperatures = np.zeros((n_timesteps, n_points), dtype=np.float32) if self._read_temperature else None
        # Nodal stress: Voigt tensor (N, 6) and Von Mises scalar (N,) per timestep
        stress_voigt = np.zeros((n_timesteps, n_points, 6), dtype=np.float32) if self._read_stress else None
        stress_vm = np.zeros((n_timesteps, n_points), dtype=np.float32) if self._read_stress else None

        # Extract fields from first timestep
        self._extract_fields(grid0, 0, velocities, accelerations, temperatures, stress_voigt, stress_vm)

        # Read remaining timesteps
        for t, vtk_file in enumerate(vtk_files[1:], start=1):
            grid = pv.read(vtk_file)

            # Validate consistency
            if grid.n_points != n_points:
                msg = f"Point count mismatch at timestep {t}: expected {n_points}, got {grid.n_points}"
                raise ValueError(msg)

            positions[t] = np.array(grid.points, dtype=np.float32)
            self._extract_fields(grid, t, velocities, accelerations, temperatures, stress_voigt, stress_vm)

        # Compute displacements relative to t=0
        displacements = positions - positions[0:1]  # (T, N, 3)

        # Prune unreferenced points: slice all point arrays to only referenced nodes
        if n_points_pruned < n_points:
            # Build old-to-new index mapping for cell connectivity
            old_to_new = np.full(n_points, -1, dtype=np.int64)
            old_to_new[referenced_nodes] = np.arange(n_points_pruned, dtype=np.int64)

            # Remap cell connectivity (preserve -1 sentinels for mixed mode padding)
            mask_valid = cells_array >= 0
            cells_array[mask_valid] = old_to_new[cells_array[mask_valid]]

            # Slice point arrays to referenced nodes only
            coords = coords[referenced_nodes]
            displacements = displacements[:, referenced_nodes]
            if velocities is not None:
                velocities = velocities[:, referenced_nodes]
            if accelerations is not None:
                accelerations = accelerations[:, referenced_nodes]
            if temperatures is not None:
                temperatures = temperatures[:, referenced_nodes]
            if stress_voigt is not None:
                stress_voigt = stress_voigt[:, referenced_nodes]
            if stress_vm is not None:
                stress_vm = stress_vm[:, referenced_nodes]

            n_points = n_points_pruned

        # Build point_data
        pd_dict: dict[str, torch.Tensor] = {}

        # Thickness is zero for solid elements
        pd_dict["thickness"] = torch.zeros(n_points, dtype=torch.float32)

        # Displacement fields per timestep
        for t in range(n_timesteps):
            pd_dict[f"displacement_t{t:03d}"] = torch.from_numpy(displacements[t])

        # Optional velocity fields
        if velocities is not None:
            for t in range(n_timesteps):
                pd_dict[f"velocity_t{t:03d}"] = torch.from_numpy(velocities[t])

        # Optional acceleration fields
        if accelerations is not None:
            for t in range(n_timesteps):
                pd_dict[f"acceleration_t{t:03d}"] = torch.from_numpy(accelerations[t])

        # Optional temperature fields
        if temperatures is not None:
            for t in range(n_timesteps):
                pd_dict[f"temperature_t{t:03d}"] = torch.from_numpy(temperatures[t])

        # Optional nodal stress fields (Voigt tensor and Von Mises)
        if stress_voigt is not None:
            for t in range(n_timesteps):
                pd_dict[f"stress_voigt_t{t:03d}"] = torch.from_numpy(stress_voigt[t])
        if stress_vm is not None:
            for t in range(n_timesteps):
                pd_dict[f"stress_vm_t{t:03d}"] = torch.from_numpy(stress_vm[t])

        point_data = TensorDict(pd_dict, batch_size=[n_points])  # ty: ignore[invalid-argument-type]

        # Build cell_data — not used for mixed mode (cell types are in global_data)
        cd: TensorDict | None = None

        # Build global_data
        gd_dict: dict[str, torch.Tensor] = {
            "num_timesteps": torch.tensor([n_timesteps], dtype=torch.int64),
        }

        # For mixed mode, store connectivity as flat array + offsets in global_data
        # because Mesh.cells requires uniform node count (infers manifold dim from shape).
        if self._cell_type == "mixed":
            # Convert padded (E, max_nodes) to flat connectivity + offsets
            flat_conn_parts: list[np.ndarray] = []
            offsets = np.zeros(n_cells + 1, dtype=np.int64)
            for i in range(n_cells):
                row = cells_array[i]
                valid = row[row >= 0]
                flat_conn_parts.append(valid)
                offsets[i + 1] = offsets[i] + len(valid)
            flat_connectivity = np.concatenate(flat_conn_parts) if flat_conn_parts else np.array([], dtype=np.int64)

            gd_dict["mixed_connectivity"] = torch.from_numpy(flat_connectivity)
            gd_dict["mixed_offsets"] = torch.from_numpy(offsets)
            gd_dict["mixed_cell_types"] = torch.from_numpy(cell_types.astype(np.int64))

        global_data = TensorDict(gd_dict, batch_size=[])  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

        # Build mesh
        points_tensor = torch.from_numpy(coords)

        # For mixed mode, pass cells=None because Mesh uses cells.shape[-1] to infer
        # manifold dimensionality and cannot handle padded cells.
        cells_for_mesh = None if self._cell_type == "mixed" else torch.from_numpy(cells_array)

        mesh = Mesh(
            points=points_tensor,
            cells=cells_for_mesh,
            point_data=point_data,
            cell_data=cd,
            global_data=global_data,
        )

        elapsed = time.perf_counter() - t0
        self._log.info(
            "Read complete: %d points, %d cells, %d timesteps (%.2fs)",
            n_points,
            n_cells,
            n_timesteps,
            elapsed,
        )

        return mesh

    def _extract_connectivity(self, grid: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract cell connectivity from a PyVista grid.

        Filters cells to only include the configured cell type (tet, tri, all,
        or mixed).  For 'tet' and 'tri', only cells with exactly 4 or 3 nodes
        are kept.  For 'mixed', ALL cells are kept and padded to the maximum
        node count with -1 sentinel values.

        Parameters
        ----------
        grid : pv.UnstructuredGrid
            The VTK grid.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Connectivity array (E, nodes_per_cell), cell types, and cell mask.
        """
        # Get cell connectivity - PyVista stores as flat array with counts
        cells = grid.cells
        cell_types = np.array(grid.celltypes)
        n_cells = grid.n_cells

        # Determine which cells to keep based on cell_type setting
        if self._cell_type == "tet":
            # Keep only tetrahedra (VTK type 10, 4 nodes)
            keep_mask = cell_types == self._VTK_TET
            nodes_per_cell = 4
        elif self._cell_type == "tri":
            # Keep only triangles (VTK type 5, 3 nodes)
            keep_mask = cell_types == self._VTK_TRI
            nodes_per_cell = 3
        elif self._cell_type == "mixed":
            # Keep ALL cells, pad to max node count with -1 sentinel
            keep_mask = np.ones(n_cells, dtype=bool)
            # Determine max nodes per cell across all cell types
            type_to_nodes = {
                self._VTK_TRI: 3,
                self._VTK_TET: 4,
                self._VTK_QUAD: 4,
                self._VTK_PYRAMID: 5,
                self._VTK_WEDGE: 6,
                self._VTK_HEX: 8,
            }
            max_nodes = max(type_to_nodes.get(ct, 4) for ct in np.unique(cell_types))
            nodes_per_cell = max_nodes
        else:
            # 'all' mode - but we still need uniform cell sizes
            # Find the most common cell type and use that
            from collections import Counter

            type_counts = Counter(cell_types)
            most_common_type = type_counts.most_common(1)[0][0]
            keep_mask = cell_types == most_common_type

            # Determine nodes per cell from VTK type
            type_to_nodes = {
                self._VTK_TRI: 3,
                self._VTK_TET: 4,
                self._VTK_QUAD: 4,
                self._VTK_PYRAMID: 5,  # Will fail Mesh validation
                self._VTK_WEDGE: 6,  # Will fail Mesh validation
                self._VTK_HEX: 8,  # Will fail Mesh validation
            }
            nodes_per_cell = type_to_nodes.get(most_common_type, 4)
            if nodes_per_cell > 4:
                self._log.warning(
                    "Most common cell type %d has %d nodes, which exceeds Mesh limit of 4. "
                    "Use cell_type='tet' to filter to tetrahedra only.",
                    most_common_type,
                    nodes_per_cell,
                )

        n_keep = int(keep_mask.sum())
        if n_keep == 0:
            msg = f"No cells of type '{self._cell_type}' found in mesh"
            raise ValueError(msg)

        self._log.debug(
            "Filtering cells: keeping %d of %d (type=%s)",
            n_keep,
            n_cells,
            self._cell_type,
        )

        # Parse and filter connectivity
        # For 'mixed' mode, pad shorter cells with -1
        connectivity = np.full((n_keep, nodes_per_cell), -1, dtype=np.int64)
        filtered_types = np.zeros(n_keep, dtype=np.uint8)

        offset = 0
        out_idx = 0
        for i in range(n_cells):
            n_nodes = cells[offset]
            if keep_mask[i]:
                # Extract actual node count (may be less than nodes_per_cell)
                actual_nodes = min(n_nodes, nodes_per_cell)
                connectivity[out_idx, :actual_nodes] = cells[offset + 1 : offset + 1 + actual_nodes]
                filtered_types[out_idx] = cell_types[i]
                out_idx += 1
            offset += n_nodes + 1

        return connectivity, filtered_types, keep_mask

    def _extract_fields(
        self,
        grid: Any,
        t: int,
        velocities: np.ndarray | None,
        accelerations: np.ndarray | None,
        temperatures: np.ndarray | None,
        stress_voigt: np.ndarray | None,
        stress_vm: np.ndarray | None,
    ) -> None:
        """Extract field data from a VTK grid for a single timestep.

        Parameters
        ----------
        grid : pv.UnstructuredGrid
            The VTK grid.
        t : int
            Timestep index.
        velocities : np.ndarray | None
            Array to fill with velocity data.
        accelerations : np.ndarray | None
            Array to fill with acceleration data.
        temperatures : np.ndarray | None
            Array to fill with temperature data.
        stress_voigt : np.ndarray | None
            Array to fill with Voigt stress tensor (N, 6).
        stress_vm : np.ndarray | None
            Array to fill with von Mises stress (N,).
        """
        # Velocity (point data)
        if velocities is not None:
            for name in ["velocity", "Velocity", "VELOCITY", "VEL"]:
                if name in grid.point_data:
                    velocities[t] = np.array(grid.point_data[name], dtype=np.float32)
                    break

        # Acceleration (point data)
        if accelerations is not None:
            for name in ["acceleration", "Acceleration", "ACCELERATION", "ACC"]:
                if name in grid.point_data:
                    accelerations[t] = np.array(grid.point_data[name], dtype=np.float32)
                    break

        # Temperature (point data)
        if temperatures is not None:
            for name in ["temperature", "Temperature", "TEMPERATURE", "TEMP"]:
                if name in grid.point_data:
                    temperatures[t] = np.array(grid.point_data[name], dtype=np.float32)
                    break

        # Nodal stress (point data) - read GPS_SIG* components from OpenRadioss
        if stress_voigt is not None and stress_vm is not None:
            # Try to read stress Voigt components from point_data
            # OpenRadioss exports: GPS_SIGXX, GPS_SIGYY, GPS_SIGZZ, GPS_SIGXY, GPS_SIGYZ/GPS_SIGZY, GPS_SIGXZ
            pd = grid.point_data
            keys_lower = {k.lower(): k for k in pd}

            # Component order: [xx, yy, zz, xy, yz, xz] (Voigt notation)
            component_candidates = [
                ("gps_sigxx",),  # xx
                ("gps_sigyy",),  # yy
                ("gps_sigzz",),  # zz
                ("gps_sigxy",),  # xy
                ("gps_sigyz", "gps_sigzy"),  # yz (OpenRadioss may use SIGZY)
                ("gps_sigxz",),  # xz
            ]

            found_keys = []
            for candidates in component_candidates:
                key = next((keys_lower[c] for c in candidates if c in keys_lower), None)
                found_keys.append(key)

            if all(k is not None for k in found_keys):
                # Stack into Voigt tensor (N, 6)
                components = [np.array(pd[k], dtype=np.float32) for k in found_keys]
                voigt = np.stack(components, axis=1)  # (N, 6)
                stress_voigt[t] = voigt
                stress_vm[t] = _von_mises_from_voigt(voigt)
            elif t == 0:
                # Log warning only on first timestep
                self._log.warning(
                    "Nodal stress not found in point_data. Expected GPS_SIGXX, GPS_SIGYY, "
                    "GPS_SIGZZ, GPS_SIGXY, GPS_SIGYZ, GPS_SIGXZ. Found: %s",
                    list(pd.keys()),
                )

    def run_id(self, index: int) -> str:
        """Return the run identifier for a given index.

        Parameters
        ----------
        index : int
            Source index.

        Returns
        -------
        str
            Run directory name.
        """
        if index < 0:
            index += len(self)
        return self._run_dirs[index].name
