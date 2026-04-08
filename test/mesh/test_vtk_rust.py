# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Consistency tests comparing Rust VTK reader against PyVista."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import pyvista as pv

from physicsnemo.curator._lib import vtk

if TYPE_CHECKING:
    import pathlib


@pytest.fixture
def simple_vtu_file(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a simple VTU file with known data."""
    path = tmp_path / "simple.vtu"
    xml = """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1">
  <UnstructuredGrid>
    <Piece NumberOfPoints="4" NumberOfCells="1">
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">
          0.0 0.0 0.0  1.0 0.0 0.0  1.0 1.0 0.0  0.0 1.0 0.0
        </DataArray>
      </Points>
      <Cells>
        <DataArray Name="connectivity" type="Int64" format="ascii">
          0 1 2 3
        </DataArray>
        <DataArray Name="offsets" type="Int64" format="ascii">
          4
        </DataArray>
        <DataArray Name="types" type="UInt8" format="ascii">
          9
        </DataArray>
      </Cells>
      <PointData>
        <DataArray Name="Temperature" type="Float64" NumberOfComponents="1" format="ascii">
          100.0 200.0 300.0 400.0
        </DataArray>
        <DataArray Name="Velocity" type="Float64" NumberOfComponents="3" format="ascii">
          1.0 0.0 0.0  0.0 1.0 0.0  0.0 0.0 1.0  1.0 1.0 1.0
        </DataArray>
      </PointData>
    </Piece>
  </UnstructuredGrid>
</VTKFile>"""
    path.write_text(xml)
    return path


@pytest.fixture
def simple_vtp_file(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a simple VTP (PolyData) file."""
    path = tmp_path / "simple.vtp"
    xml = """<?xml version="1.0"?>
<VTKFile type="PolyData" version="0.1">
  <PolyData>
    <Piece NumberOfPoints="3" NumberOfPolys="1">
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">
          0.0 0.0 0.0  1.0 0.0 0.0  0.5 1.0 0.0
        </DataArray>
      </Points>
      <Polys>
        <DataArray Name="connectivity" type="Int64" format="ascii">
          0 1 2
        </DataArray>
        <DataArray Name="offsets" type="Int64" format="ascii">
          3
        </DataArray>
      </Polys>
    </Piece>
  </PolyData>
</VTKFile>"""
    path.write_text(xml)
    return path


class TestRustVTKReader:
    """Tests for the Rust VTK reader."""

    def test_read_vtu_basic(self, simple_vtu_file: pathlib.Path) -> None:
        """Test reading a basic VTU file."""
        mesh = vtk.read_vtk(str(simple_vtu_file))

        assert mesh.n_points == 4
        assert mesh.n_cells == 1
        assert mesh.format == "vtu"

    def test_read_vtu_points(self, simple_vtu_file: pathlib.Path) -> None:
        """Test that points match expected values."""
        mesh = vtk.read_vtk(str(simple_vtu_file))
        points = mesh.points()

        expected = np.array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0], dtype=np.float64)
        np.testing.assert_array_almost_equal(points, expected)

    def test_read_vtu_connectivity(self, simple_vtu_file: pathlib.Path) -> None:
        """Test that cell connectivity is correct."""
        mesh = vtk.read_vtk(str(simple_vtu_file))
        connectivity = mesh.connectivity()
        offsets = mesh.offsets()
        types = mesh.types()

        np.testing.assert_array_equal(connectivity, [0, 1, 2, 3])
        np.testing.assert_array_equal(offsets, [4])
        np.testing.assert_array_equal(types, [9])  # VTK_QUAD

    def test_read_vtu_point_data(self, simple_vtu_file: pathlib.Path) -> None:
        """Test that point data arrays are read correctly."""
        mesh = vtk.read_vtk(str(simple_vtu_file))
        point_data = mesh.point_data()

        assert "Temperature" in point_data
        temp_data, temp_components = point_data["Temperature"]
        assert temp_components == 1
        np.testing.assert_array_almost_equal(temp_data, [100.0, 200.0, 300.0, 400.0])

        assert "Velocity" in point_data
        vel_data, vel_components = point_data["Velocity"]
        assert vel_components == 3
        expected_vel = [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1]
        np.testing.assert_array_almost_equal(vel_data, expected_vel)

    def test_read_vtp_basic(self, simple_vtp_file: pathlib.Path) -> None:
        """Test reading a VTP PolyData file."""
        mesh = vtk.read_vtk(str(simple_vtp_file))

        assert mesh.n_points == 3
        assert mesh.n_cells == 1
        assert mesh.format == "vtp"

    def test_read_parallel(self, simple_vtu_file: pathlib.Path, simple_vtp_file: pathlib.Path) -> None:
        """Test parallel reading of multiple files."""
        paths = [str(simple_vtu_file), str(simple_vtp_file)]
        meshes = vtk.read_vtk_parallel(paths)

        assert len(meshes) == 2
        assert meshes[0].n_points == 4
        assert meshes[1].n_points == 3


class TestRustVsPyVistaConsistency:
    """Tests comparing Rust reader output against PyVista."""

    def test_points_match_pyvista(self, simple_vtu_file: pathlib.Path) -> None:
        """Verify point coordinates match between Rust and PyVista."""
        rust_mesh = vtk.read_vtk(str(simple_vtu_file))
        pv_mesh = pv.read(str(simple_vtu_file))

        rust_points = rust_mesh.points().reshape(-1, 3)
        pv_points = np.array(pv_mesh.points)

        np.testing.assert_array_almost_equal(rust_points, pv_points)

    def test_n_points_match_pyvista(self, simple_vtu_file: pathlib.Path) -> None:
        """Verify point count matches."""
        rust_mesh = vtk.read_vtk(str(simple_vtu_file))
        pv_mesh = pv.read(str(simple_vtu_file))

        assert rust_mesh.n_points == pv_mesh.n_points

    def test_n_cells_match_pyvista(self, simple_vtu_file: pathlib.Path) -> None:
        """Verify cell count matches."""
        rust_mesh = vtk.read_vtk(str(simple_vtu_file))
        pv_mesh = pv.read(str(simple_vtu_file))

        assert rust_mesh.n_cells == pv_mesh.n_cells

    def test_point_data_values_match_pyvista(self, simple_vtu_file: pathlib.Path) -> None:
        """Verify point data values match between Rust and PyVista."""
        rust_mesh = vtk.read_vtk(str(simple_vtu_file))
        pv_mesh = pv.read(str(simple_vtu_file))

        rust_point_data = rust_mesh.point_data()

        # Check Temperature
        rust_temp, _ = rust_point_data["Temperature"]
        pv_temp = pv_mesh.point_data["Temperature"]
        np.testing.assert_array_almost_equal(rust_temp, pv_temp)

        # Check Velocity
        rust_vel, num_comp = rust_point_data["Velocity"]
        pv_vel = pv_mesh.point_data["Velocity"].flatten()
        assert num_comp == 3
        np.testing.assert_array_almost_equal(rust_vel, pv_vel)


class TestErrorHandling:
    """Tests for error handling."""

    def test_file_not_found(self) -> None:
        """Test that missing file raises IOError."""
        with pytest.raises(IOError):
            vtk.read_vtk("/nonexistent/path.vtu")

    def test_invalid_extension(self, tmp_path: pathlib.Path) -> None:
        """Test that invalid extension raises IOError."""
        path = tmp_path / "test.xyz"
        path.write_text("invalid")
        with pytest.raises(IOError):
            vtk.read_vtk(str(path))
