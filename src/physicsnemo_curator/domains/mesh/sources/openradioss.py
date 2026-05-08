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
        If ``True``, read von Mises stress from element data.  Default
        is ``False``.
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
                description="Read von Mises stress from element data",
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
        ]

    def __init__(
        self,
        input_dir: str,
        vtk_glob: str = "*.vtk",
        read_stress: bool = False,
        read_velocity: bool = False,
        read_acceleration: bool = False,
        read_temperature: bool = False,
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
        """
        self._log = get_logger(self)
        self._flush_logs = flush_logs

        self._input_dir = Path(input_dir)
        self._vtk_glob = vtk_glob
        self._read_stress = read_stress
        self._read_velocity = read_velocity
        self._read_acceleration = read_acceleration
        self._read_temperature = read_temperature

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
        n_cells = grid0.n_cells

        # Extract reference coordinates (t=0)
        coords = np.array(grid0.points, dtype=np.float32)  # (N, 3)

        # Extract cell connectivity
        cells_array, cell_types = self._extract_connectivity(grid0)

        # Initialize arrays for all timesteps
        positions = np.zeros((n_timesteps, n_points, 3), dtype=np.float32)
        positions[0] = coords

        # Optional field arrays
        velocities = np.zeros((n_timesteps, n_points, 3), dtype=np.float32) if self._read_velocity else None
        accelerations = np.zeros((n_timesteps, n_points, 3), dtype=np.float32) if self._read_acceleration else None
        temperatures = np.zeros((n_timesteps, n_points), dtype=np.float32) if self._read_temperature else None
        stress_vm = np.zeros((n_timesteps, n_cells), dtype=np.float32) if self._read_stress else None

        # Extract fields from first timestep
        self._extract_fields(grid0, 0, velocities, accelerations, temperatures, stress_vm)

        # Read remaining timesteps
        for t, vtk_file in enumerate(vtk_files[1:], start=1):
            grid = pv.read(vtk_file)

            # Validate consistency
            if grid.n_points != n_points:
                msg = f"Point count mismatch at timestep {t}: expected {n_points}, got {grid.n_points}"
                raise ValueError(msg)

            positions[t] = np.array(grid.points, dtype=np.float32)
            self._extract_fields(grid, t, velocities, accelerations, temperatures, stress_vm)

        # Compute displacements relative to t=0
        displacements = positions - positions[0:1]  # (T, N, 3)

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

        point_data = TensorDict(pd_dict, batch_size=[n_points])  # ty: ignore[invalid-argument-type]

        # Build cell_data
        cell_data = None
        if stress_vm is not None:
            cd_dict: dict[str, torch.Tensor] = {}
            for t in range(n_timesteps):
                cd_dict[f"stress_vm_t{t:03d}"] = torch.from_numpy(stress_vm[t])
            cell_data = TensorDict(cd_dict, batch_size=[n_cells])  # ty: ignore[invalid-argument-type]

        # Build global_data
        global_data = TensorDict(
            {"num_timesteps": torch.tensor([n_timesteps], dtype=torch.int64)},
            batch_size=[],
        )

        # Build mesh
        points_tensor = torch.from_numpy(coords)
        cells_tensor = torch.from_numpy(cells_array)

        mesh = Mesh(
            points=points_tensor,
            cells=cells_tensor,
            point_data=point_data,
            cell_data=cell_data,
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

    def _extract_connectivity(self, grid: Any) -> tuple[np.ndarray, np.ndarray]:
        """Extract cell connectivity from a PyVista grid.

        Parameters
        ----------
        grid : pv.UnstructuredGrid
            The VTK grid.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Connectivity array (E, max_nodes_per_cell) and cell types.
        """
        # Get cell connectivity - PyVista stores as flat array with counts
        cells = grid.cells
        cell_types = np.array(grid.celltypes)
        n_cells = grid.n_cells

        # Parse connectivity - find max nodes per cell
        offset = 0
        cell_node_counts = []
        for _ in range(n_cells):
            n_nodes = cells[offset]
            cell_node_counts.append(n_nodes)
            offset += n_nodes + 1

        max_nodes = max(cell_node_counts)

        # Build padded connectivity array
        connectivity = np.full((n_cells, max_nodes), -1, dtype=np.int64)
        offset = 0
        for i in range(n_cells):
            n_nodes = cells[offset]
            connectivity[i, :n_nodes] = cells[offset + 1 : offset + 1 + n_nodes]
            offset += n_nodes + 1

        return connectivity, cell_types

    def _extract_fields(
        self,
        grid: Any,
        t: int,
        velocities: np.ndarray | None,
        accelerations: np.ndarray | None,
        temperatures: np.ndarray | None,
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
        stress_vm : np.ndarray | None
            Array to fill with von Mises stress data.
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

        # Stress (cell data) - compute von Mises if Voigt components available
        if stress_vm is not None:
            # Try direct von Mises first
            for name in ["stress_vm", "von_mises", "VonMises", "VONMISES"]:
                if name in grid.cell_data:
                    stress_vm[t] = np.array(grid.cell_data[name], dtype=np.float32)
                    return

            # Try Voigt stress tensor
            for name in ["stress", "Stress", "STRESS"]:
                if name in grid.cell_data:
                    stress_voigt = np.array(grid.cell_data[name], dtype=np.float32)
                    if stress_voigt.shape[-1] == 6:
                        stress_vm[t] = _von_mises_from_voigt(stress_voigt)
                        return

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
