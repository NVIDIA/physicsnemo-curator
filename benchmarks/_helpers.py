# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shared synthetic data helpers for ASV benchmarks."""

from __future__ import annotations

import shutil
import tempfile
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


def create_temp_dir() -> str:
    """Create and return a temporary directory path."""
    return tempfile.mkdtemp()


def cleanup_temp_dir(path: str) -> None:
    """Remove a temporary directory and all contents."""
    shutil.rmtree(path, ignore_errors=True)


def write_synthetic_vtu(
    path: Path,
    n_points: int,
    n_cells: int,
    *,
    seed: int = 42,
) -> None:
    """Write a synthetic VTU file with random points, triangles, and point data.

    Parameters
    ----------
    path : Path
        Output file path (should end in ``.vtu``).
    n_points : int
        Number of mesh vertices.
    n_cells : int
        Number of triangle cells.
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    points = rng.random((n_points, 3)).flatten()
    points_str = " ".join(f"{x:.6f}" for x in points)

    connectivity = rng.integers(0, n_points, size=n_cells * 3)
    connectivity_str = " ".join(str(x) for x in connectivity)

    offsets = [3 * (i + 1) for i in range(n_cells)]
    offsets_str = " ".join(str(x) for x in offsets)

    types_str = " ".join(["5"] * n_cells)

    temperature = rng.random(n_points)
    temp_str = " ".join(f"{x:.6f}" for x in temperature)

    pressure = rng.random(n_points)
    pressure_str = " ".join(f"{x:.6f}" for x in pressure)

    xml = f"""\
<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1">
  <UnstructuredGrid>
    <Piece NumberOfPoints="{n_points}" NumberOfCells="{n_cells}">
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">
          {points_str}
        </DataArray>
      </Points>
      <Cells>
        <DataArray Name="connectivity" type="Int64" format="ascii">
          {connectivity_str}
        </DataArray>
        <DataArray Name="offsets" type="Int64" format="ascii">
          {offsets_str}
        </DataArray>
        <DataArray Name="types" type="UInt8" format="ascii">
          {types_str}
        </DataArray>
      </Cells>
      <PointData>
        <DataArray Name="Temperature" type="Float64" \
NumberOfComponents="1" format="ascii">
          {temp_str}
        </DataArray>
        <DataArray Name="Pressure" type="Float64" \
NumberOfComponents="1" format="ascii">
          {pressure_str}
        </DataArray>
      </PointData>
    </Piece>
  </UnstructuredGrid>
</VTKFile>"""
    path.write_text(xml)


def write_synthetic_k_file(path: Path, n_parts: int) -> None:
    """Write a synthetic LS-DYNA ``.k`` file.

    Parameters
    ----------
    path : Path
        Output file path.
    n_parts : int
        Number of ``*PART`` / ``*SECTION_SHELL`` entries.
    """
    lines = ["$", "*KEYWORD"]
    for i in range(1, n_parts + 1):
        lines.append("*PART")
        lines.append(f"Part_{i}")
        lines.append(f"       {i}       {i}       {i}")
    lines.append("*SECTION_SHELL")
    for i in range(1, n_parts + 1):
        lines.append(f"       {i}")
        t = float(i) * 0.5
        lines.append(f"     {t:.1f}     {t:.1f}     {t:.1f}     {t:.1f}")
    lines.append("*END")
    path.write_text("\n".join(lines))


def create_synthetic_aselmdb(
    db_path: Path,
    n_rows: int,
    *,
    with_calc: bool = False,
    seed: int = 42,
) -> None:
    """Create an ``.aselmdb`` file with synthetic water-like molecules.

    Parameters
    ----------
    db_path : Path
        Output file path.
    n_rows : int
        Number of rows to insert.
    with_calc : bool
        Attach energy and forces via SinglePointCalculator.
    seed : int
        Random seed for reproducibility.
    """
    from ase import Atoms

    from physicsnemo_curator.domains.atm.sources.aselmdb import (
        _atoms_to_row_dict,
        _write_aselmdb,
    )

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for i in range(n_rows):
        positions = rng.random((3, 3)) * 10.0
        atoms = Atoms("H2O", positions=positions, cell=[10.0, 10.0, 10.0], pbc=True)
        if with_calc:
            from ase.calculators.singlepoint import SinglePointCalculator

            forces = rng.random((3, 3)) - 0.5
            calc = SinglePointCalculator(atoms, energy=-100.0 * i, forces=forces)
            atoms.calc = calc
        rows.append(_atoms_to_row_dict(atoms, row_id=i + 1, key_value_pairs={"row_index": i}))
    _write_aselmdb(db_path, rows)


def make_connectivity(
    n_elements: int,
    n_nodes: int,
    nodes_per_cell: int = 4,
    *,
    seed: int = 42,
) -> np.ndarray:
    """Create a random element connectivity array.

    Parameters
    ----------
    n_elements : int
        Number of elements.
    n_nodes : int
        Number of nodes.
    nodes_per_cell : int
        Nodes per element.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Shape ``(n_elements, nodes_per_cell)``, dtype int64.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_nodes, size=(n_elements, nodes_per_cell), dtype=np.int64)


def make_synthetic_dataarray(
    n_timesteps: int = 1,
    variables: list[str] | None = None,
    n_lat: int = 36,
    n_lon: int = 72,
    *,
    seed: int = 42,
) -> object:
    """Create a synthetic xarray DataArray mimicking ERA5 output.

    Parameters
    ----------
    n_timesteps : int
        Number of time steps.
    variables : list[str] | None
        Variable names. Defaults to ``["t2m", "u10m"]``.
    n_lat : int
        Number of latitude grid points.
    n_lon : int
        Number of longitude grid points.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    xr.DataArray
        Dims ``(time, variable, lat, lon)``.
    """
    from datetime import datetime

    import xarray as xr

    variables = variables or ["t2m", "u10m"]
    lats = np.linspace(90, -90, n_lat)
    lons = np.linspace(0, 350, n_lon)
    rng = np.random.default_rng(seed)

    data = rng.standard_normal((n_timesteps, len(variables), n_lat, n_lon))
    times = [np.datetime64(datetime(2020, 1, 1, h)) for h in range(n_timesteps)]
    return xr.DataArray(
        data=data,
        dims=["time", "variable", "lat", "lon"],
        coords={
            "time": times,
            "variable": variables,
            "lat": lats,
            "lon": lons,
        },
    )
