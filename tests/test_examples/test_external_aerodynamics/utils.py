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

"""Utility functions for External Aerodynamics tests."""

import vtk


def create_mock_stl(path):
    """Create a simple STL file."""
    points = vtk.vtkPoints()
    points.InsertNextPoint(0, 0, 0)
    points.InsertNextPoint(1, 0, 0)
    points.InsertNextPoint(0, 1, 0)

    triangle = vtk.vtkTriangle()
    triangle.GetPointIds().SetId(0, 0)
    triangle.GetPointIds().SetId(1, 1)
    triangle.GetPointIds().SetId(2, 2)

    triangles = vtk.vtkCellArray()
    triangles.InsertNextCell(triangle)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(triangles)

    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(polydata)
    writer.Write()


def create_mock_surface_vtk(path):
    """Create a simple surface VTK file."""
    polydata = vtk.vtkPolyData()

    # Points
    points = vtk.vtkPoints()
    points.InsertNextPoint(0, 0, 0)
    points.InsertNextPoint(1, 0, 0)
    points.InsertNextPoint(0, 1, 0)
    polydata.SetPoints(points)

    # Create a cell (triangle)
    triangle = vtk.vtkTriangle()
    triangle.GetPointIds().SetId(0, 0)
    triangle.GetPointIds().SetId(1, 1)
    triangle.GetPointIds().SetId(2, 2)

    triangles = vtk.vtkCellArray()
    triangles.InsertNextCell(triangle)
    polydata.SetPolys(triangles)

    # Surface data - add to CellData instead of PointData
    pressure = vtk.vtkFloatArray()
    pressure.SetName("pMeanTrim")
    pressure.SetNumberOfComponents(1)
    pressure.InsertNextValue(1.0)  # One value per cell

    shear = vtk.vtkFloatArray()
    shear.SetName("wallShearStressMeanTrim")
    shear.SetNumberOfComponents(3)
    shear.InsertNextTuple3(1, 0, 0)  # One vector per cell

    polydata.GetCellData().AddArray(pressure)  # Changed from PointData to CellData
    polydata.GetCellData().AddArray(shear)  # Changed from PointData to CellData

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(polydata)
    writer.Write()


def create_mock_volume_vtk(path):
    """Create a simple volume VTK file."""
    grid = vtk.vtkUnstructuredGrid()

    # Points
    points = vtk.vtkPoints()
    points.InsertNextPoint(0, 0, 0)
    points.InsertNextPoint(1, 0, 0)
    points.InsertNextPoint(0, 1, 0)
    points.InsertNextPoint(0, 0, 1)
    grid.SetPoints(points)

    # Volume data
    velocity = vtk.vtkFloatArray()
    velocity.SetName("UMeanTrim")
    velocity.SetNumberOfComponents(3)
    for _ in range(4):
        velocity.InsertNextTuple3(1, 0, 0)

    pressure = vtk.vtkFloatArray()
    pressure.SetName("pMeanTrim")
    pressure.SetNumberOfComponents(1)
    for i in range(4):
        pressure.InsertNextValue(float(i))

    grid.GetPointData().AddArray(velocity)
    grid.GetPointData().AddArray(pressure)

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(grid)
    writer.Write()
