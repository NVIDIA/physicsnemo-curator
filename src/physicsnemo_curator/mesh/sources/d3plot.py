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

"""LS-DYNA d3plot data source for crash simulation mesh pipelines.

Reads LS-DYNA ``d3plot`` binary result files (and optional ``.k`` keyword
files for part thickness) and yields :class:`~physicsnemo.mesh.Mesh` objects
for use in curator pipelines.

Each source index maps to one simulation run directory containing a
``d3plot`` file.  The resulting mesh carries:

* ``points`` --- *(N, 3)* reference coordinates at t=0
* ``cells`` --- *(E, nodes_per_cell)* shell element connectivity
* ``point_data`` --- ``thickness`` *(N,)* plus ``displacement_t{idx:03d}``
  *(N, 3)* for each timestep
* ``cell_data`` (optional) --- ``stress_vm_t{idx:03d}`` *(E,)* and
  ``effective_plastic_strain_t{idx:03d}`` *(E,)* for each timestep
* ``global_data`` --- ``num_timesteps`` scalar

Examples
--------
>>> source = D3PlotSource(input_dir="/data/crash_sims")  # doctest: +SKIP
>>> len(source)  # doctest: +SKIP
50
>>> mesh = next(source[0])  # doctest: +SKIP
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np
import torch
from tensordict import TensorDict

from physicsnemo_curator.core.base import REQUIRED, Param, Source

if TYPE_CHECKING:
    from collections.abc import Generator

    from physicsnemo.mesh import Mesh

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rust backend helpers
# ---------------------------------------------------------------------------

_RUST_AVAILABLE: bool = False
try:
    from physicsnemo_curator._lib import d3plot as _rust_d3plot  # type: ignore[attr-defined]

    _RUST_AVAILABLE = True
except ImportError:
    _rust_d3plot = None  # type: ignore[assignment]


def _parse_k_file_rust(k_file_path: Path) -> dict[int, float]:
    """Parse a ``.k`` file using the Rust backend.

    Parameters
    ----------
    k_file_path : Path
        Path to the ``.k`` file.

    Returns
    -------
    dict[int, float]
        Mapping from part ID to thickness value.
    """
    return _rust_d3plot.parse_k_file(str(k_file_path))  # ty: ignore[unresolved-attribute]


def _compute_node_thickness_rust(
    mesh_connectivity: np.ndarray,
    part_ids: np.ndarray,
    part_thickness_map: dict[int, float],
    actual_part_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Compute per-node thickness using the Rust backend.

    Parameters
    ----------
    mesh_connectivity : np.ndarray
        Element connectivity, shape ``(E, nodes_per_cell)``.
    part_ids : np.ndarray
        Part index per element, shape ``(E,)``.
    part_thickness_map : dict[int, float]
        Mapping from actual part ID to thickness.
    actual_part_ids : np.ndarray | None
        Actual part IDs array for index-to-ID translation.

    Returns
    -------
    np.ndarray
        Per-node thickness, shape ``(N,)``.
    """
    conn = np.ascontiguousarray(mesh_connectivity, dtype=np.int64)
    pids = np.ascontiguousarray(part_ids, dtype=np.int64)
    apids = np.ascontiguousarray(actual_part_ids, dtype=np.int64) if actual_part_ids is not None else None
    return np.asarray(_rust_d3plot.compute_node_thickness(conn, pids, part_thickness_map, apids))  # ty: ignore[unresolved-attribute]


def _von_mises_from_voigt_rust(sig: np.ndarray) -> np.ndarray:
    """Compute von Mises stress from Voigt components using Rust.

    Parameters
    ----------
    sig : np.ndarray
        Shape ``(..., 6)``.

    Returns
    -------
    np.ndarray
        Shape ``(...)`` — scalar von Mises stress.
    """
    original_shape = sig.shape[:-1]
    n_total = int(np.prod(original_shape)) if original_shape else 1
    flat = np.ascontiguousarray(sig.reshape(-1), dtype=np.float64)
    result = np.asarray(_rust_d3plot.von_mises_from_voigt(flat, n_total))  # ty: ignore[unresolved-attribute]
    return result.reshape(original_shape)


def _find_k_file(run_dir: Path) -> Path | None:
    """Find the first ``.k`` keyword file in a run directory.

    Parameters
    ----------
    run_dir : Path
        Directory to search.

    Returns
    -------
    Path | None
        Path to the ``.k`` file, or ``None`` if not found.
    """
    k_files = list(run_dir.glob("*.k"))
    return k_files[0] if k_files else None


def _parse_k_file(k_file_path: Path) -> dict[int, float]:
    """Parse an LS-DYNA ``.k`` keyword file for part thickness.

    Extracts ``*PART`` definitions (part ID -> section ID) and
    ``*SECTION_SHELL`` definitions (section ID -> thickness).

    Parameters
    ----------
    k_file_path : Path
        Path to the ``.k`` file.

    Returns
    -------
    dict[int, float]
        Mapping from part ID to thickness value.
    """
    part_to_section: dict[int, int] = {}
    section_thickness: dict[int, float] = {}

    with k_file_path.open() as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("$")]

    i = 0
    while i < len(lines):
        line = lines[i]
        if "*PART" in line.upper():
            # i+1 = part name (skip), i+2 = part_id section_id material_id ...
            if i + 2 < len(lines):
                tokens = lines[i + 2].split()
                if len(tokens) >= 2:
                    try:
                        part_id = int(tokens[0])
                        section_id = int(tokens[1])
                        part_to_section[part_id] = section_id
                    except ValueError:
                        pass
            i += 3
        elif "*SECTION_SHELL" in line.upper():
            i += 1  # skip keyword line
            while i < len(lines) and not lines[i].startswith("*"):
                if lines[i][0].isdigit():
                    header_tokens = lines[i].split()
                    thickness_line = lines[i + 1] if i + 1 < len(lines) else ""
                    section_id = None
                    if header_tokens:
                        with contextlib.suppress(ValueError):
                            section_id = int(header_tokens[0])

                    thickness_values: list[float] = []
                    for tok in thickness_line.split():
                        try:
                            thickness_values.append(float(tok))
                        except ValueError:
                            thickness_values.append(0.0)

                    non_zero = [t for t in thickness_values if t > 0.0]
                    thickness = (
                        sum(non_zero) / len(non_zero)
                        if non_zero
                        else (sum(thickness_values) / len(thickness_values) if thickness_values else 0.0)
                    )
                    if section_id is not None:
                        section_thickness[section_id] = thickness
                    i += 2
                else:
                    i += 1
        else:
            i += 1

    return {pid: section_thickness.get(sid, 0.0) for pid, sid in part_to_section.items()}


def _compute_node_thickness(
    mesh_connectivity: np.ndarray,
    part_ids: np.ndarray,
    part_thickness_map: dict[int, float],
    actual_part_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Compute per-node thickness averaged from incident elements.

    Parameters
    ----------
    mesh_connectivity : np.ndarray
        Element connectivity, shape ``(E, nodes_per_cell)``.
    part_ids : np.ndarray
        Part index per element, shape ``(E,)``.
    part_thickness_map : dict[int, float]
        Mapping from actual part ID to thickness.
    actual_part_ids : np.ndarray | None
        Actual part IDs array for index-to-ID translation.

    Returns
    -------
    np.ndarray
        Per-node thickness, shape ``(N,)``.
    """
    if actual_part_ids is not None:
        part_index_to_id = {i: int(pid) for i, pid in enumerate(actual_part_ids) if i > 0}
    else:
        sorted_pids = sorted(part_thickness_map.keys())
        part_index_to_id = dict(enumerate(sorted_pids, 1))

    element_thickness = np.zeros(len(part_ids), dtype=np.float64)
    for i, part_index in enumerate(part_ids):
        actual_id = part_index_to_id.get(int(part_index))
        if actual_id is not None:
            element_thickness[i] = part_thickness_map.get(actual_id, 0.0)

    max_node = int(mesh_connectivity.max()) + 1
    node_thickness = np.zeros(max_node, dtype=np.float64)
    node_count = np.zeros(max_node, dtype=np.float64)

    for elem_idx in range(len(mesh_connectivity)):
        t = element_thickness[elem_idx]
        for node_idx in mesh_connectivity[elem_idx]:
            node_thickness[node_idx] += t
            node_count[node_idx] += 1

    mask = node_count > 0
    node_thickness[mask] /= node_count[mask]
    return node_thickness


def _reduce_shell_layers_scalar(x: np.ndarray) -> np.ndarray:
    """Average through-thickness layers for scalar fields.

    Parameters
    ----------
    x : np.ndarray
        Shape ``(T, E, 2)``.

    Returns
    -------
    np.ndarray
        Shape ``(T, E)``.
    """
    return np.nanmean(x, axis=2)


def _reduce_shell_layers_stress_voigt(sig: np.ndarray) -> np.ndarray:
    """Average through-thickness layers for stress in Voigt form.

    Parameters
    ----------
    sig : np.ndarray
        Shape ``(T, E, 2, 6)``.

    Returns
    -------
    np.ndarray
        Shape ``(T, E, 6)``.
    """
    return np.nanmean(sig, axis=2)


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


class D3PlotSource(Source["Mesh"]):
    """Read crash simulation meshes from LS-DYNA d3plot files.

    Scans ``input_dir`` for subdirectories containing a ``d3plot`` file.
    Each subdirectory is one simulation run.  The source reads node
    coordinates, shell connectivity, displacements over timesteps, and
    optionally stress / strain fields and per-node thickness from a
    ``.k`` keyword file.

    Parameters
    ----------
    input_dir : str
        Root directory containing run subdirectories, each with a ``d3plot``.
    read_stress : bool
        If ``True``, read element shell stress and effective plastic strain.
    read_k_file : bool
        If ``True``, look for a ``.k`` file in each run directory to extract
        per-node thickness.
    backend : {"python", "rust"}
        Computation backend for k-file parsing, node thickness, and von
        Mises stress.  ``"rust"`` uses the native Rust extension for
        faster processing of large meshes.  Defaults to ``"rust"`` when
        available, otherwise ``"python"``.

    Examples
    --------
    >>> source = D3PlotSource(input_dir="/data/crash_sims")  # doctest: +SKIP
    >>> len(source)  # doctest: +SKIP
    50

    Note
    ----
    Requires the ``lasso-python`` package (``pip install lasso-python``).
    """

    name: ClassVar[str] = "LS-DYNA D3Plot"
    description: ClassVar[str] = (
        "LS-DYNA d3plot crash simulation reader --- shell meshes with "
        "multi-timestep displacement and optional stress/strain fields"
    )

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the D3Plot source.

        Returns
        -------
        list[Param]
            Parameter list for CLI configuration.
        """
        return [
            Param(
                name="input_dir",
                description="Root directory containing run subdirectories with d3plot files",
                type=str,
                default=REQUIRED,
            ),
            Param(
                name="read_stress",
                description="Read element shell stress and effective plastic strain",
                type=bool,
                default=False,
            ),
            Param(
                name="read_k_file",
                description="Read .k keyword files for per-node thickness",
                type=bool,
                default=True,
            ),
            Param(
                name="backend",
                description="Computation backend: 'python' or 'rust' (default auto-selects)",
                type=str,
                default="rust" if _RUST_AVAILABLE else "python",
            ),
        ]

    def __init__(
        self,
        input_dir: str,
        read_stress: bool = False,
        read_k_file: bool = True,
        backend: Literal["python", "rust"] | None = None,
    ) -> None:
        self._input_dir = Path(input_dir)
        self._read_stress = read_stress
        self._read_k_file = read_k_file

        # Resolve backend.
        if backend is None:
            backend = "rust" if _RUST_AVAILABLE else "python"
        if backend == "rust" and not _RUST_AVAILABLE:
            logger.warning("Rust d3plot backend not available, falling back to Python")
            backend = "python"
        self._backend: Literal["python", "rust"] = backend

        if not self._input_dir.is_dir():
            msg = f"input_dir does not exist or is not a directory: {self._input_dir}"
            raise FileNotFoundError(msg)

        # Eagerly discover run directories (lightweight).
        self._run_dirs = self._discover_runs()
        logger.info("D3PlotSource: discovered %d runs in %s", len(self._run_dirs), self._input_dir)

    # -- Source interface -----------------------------------------------------

    def __len__(self) -> int:
        """Return the number of simulation runs discovered."""
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
            Shell mesh with displacement fields per timestep.
        """
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            msg = f"Index {index} out of range for source with {len(self)} items."
            raise IndexError(msg)

        run_dir = self._run_dirs[index]
        d3plot_path = str(run_dir / "d3plot")

        yield self._read_run(run_dir, d3plot_path)

    # -- Internal helpers ----------------------------------------------------

    def _discover_runs(self) -> list[Path]:
        """Find subdirectories containing a d3plot file.

        Returns
        -------
        list[Path]
            Sorted list of run directories.
        """
        runs = sorted(d.parent for d in self._input_dir.rglob("d3plot") if d.is_file())
        return runs

    def _read_run(self, run_dir: Path, d3plot_path: str) -> Mesh:
        """Read a single run directory into a Mesh.

        Parameters
        ----------
        run_dir : Path
            Path to the run directory.
        d3plot_path : str
            Path to the d3plot file.

        Returns
        -------
        Mesh
            The constructed mesh with all fields.
        """
        from lasso.dyna import ArrayType, D3plot
        from physicsnemo.mesh import Mesh

        dp = D3plot(d3plot_path)

        # Reference coordinates (N, 3) at t=0.
        coords = dp.arrays[ArrayType.node_coordinates]  # (N, 3)
        # Displacements (T, N, 3) relative to reference.
        displacements = dp.arrays[ArrayType.node_displacement]  # (T, N, 3)
        # Shell connectivity.
        connectivity = dp.arrays[ArrayType.element_shell_node_indexes]  # (E, nodes_per_cell)
        # Part IDs per element.
        part_ids = dp.arrays[ArrayType.element_shell_part_indexes]  # (E,)
        # Actual part IDs if available.
        actual_part_ids = dp.arrays.get(ArrayType.part_ids)

        n_timesteps = displacements.shape[0]
        n_points = coords.shape[0]

        # Build points tensor (reference configuration).
        points = torch.from_numpy(coords.astype(np.float64))

        # Build cells tensor.
        cells = torch.from_numpy(connectivity.astype(np.int64))

        # Build point_data: thickness + displacement per timestep.
        pd_dict: dict[str, torch.Tensor] = {}

        # Thickness from .k file.
        if self._read_k_file:
            k_file = _find_k_file(run_dir)
            if k_file is not None:
                if self._backend == "rust":
                    part_thickness_map = _parse_k_file_rust(k_file)
                    node_thickness = _compute_node_thickness_rust(
                        connectivity, part_ids, part_thickness_map, actual_part_ids
                    )
                else:
                    part_thickness_map = _parse_k_file(k_file)
                    node_thickness = _compute_node_thickness(
                        connectivity, part_ids, part_thickness_map, actual_part_ids
                    )
                # Ensure thickness covers all nodes (some may not appear in connectivity).
                if len(node_thickness) < n_points:
                    padded = np.zeros(n_points, dtype=node_thickness.dtype)
                    padded[: len(node_thickness)] = node_thickness
                    node_thickness = padded
                pd_dict["thickness"] = torch.from_numpy(node_thickness[:n_points]).float()
            else:
                pd_dict["thickness"] = torch.zeros(n_points, dtype=torch.float32)
                logger.debug("No .k file found in %s, using zero thickness", run_dir)
        else:
            pd_dict["thickness"] = torch.zeros(n_points, dtype=torch.float32)

        # Displacement fields per timestep.
        for t in range(n_timesteps):
            key = f"displacement_t{t:03d}"
            pd_dict[key] = torch.from_numpy(displacements[t].astype(np.float64))

        point_data = TensorDict(pd_dict, batch_size=[n_points])

        # Build cell_data (optional stress/strain).
        cell_data = None
        n_cells = connectivity.shape[0]
        if self._read_stress:
            cd_dict: dict[str, torch.Tensor] = {}
            raw_stress = dp.arrays.get(ArrayType.element_shell_stress)  # (T, E, 2, 6) or None
            raw_epsp = dp.arrays.get(ArrayType.element_shell_effective_plastic_strain)  # (T, E, 2) or None

            if raw_stress is not None:
                stress_voigt = _reduce_shell_layers_stress_voigt(raw_stress)  # (T, E, 6)
                if self._backend == "rust":
                    stress_vm = _von_mises_from_voigt_rust(stress_voigt)  # (T, E)
                else:
                    stress_vm = _von_mises_from_voigt(stress_voigt)  # (T, E)
                for t in range(n_timesteps):
                    cd_dict[f"stress_vm_t{t:03d}"] = torch.from_numpy(stress_vm[t].astype(np.float64))

            if raw_epsp is not None:
                epsp = _reduce_shell_layers_scalar(raw_epsp)  # (T, E)
                for t in range(n_timesteps):
                    cd_dict[f"effective_plastic_strain_t{t:03d}"] = torch.from_numpy(epsp[t].astype(np.float64))

            if cd_dict:
                cell_data = TensorDict(cd_dict, batch_size=[n_cells])

        # Build global_data.
        global_data = TensorDict(
            {"num_timesteps": torch.tensor([n_timesteps], dtype=torch.int64)},
            batch_size=[],
        )

        mesh = Mesh(
            points=points,
            cells=cells,
            point_data=point_data,
            cell_data=cell_data,
            global_data=global_data,
        )

        logger.info(
            "D3PlotSource: read %s — %d points, %d cells, %d timesteps",
            run_dir.name,
            n_points,
            n_cells,
            n_timesteps,
        )

        return mesh
