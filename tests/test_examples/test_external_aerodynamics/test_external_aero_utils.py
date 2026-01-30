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

import numpy as np
import pytest
import vtk
from external_aero_utils import (
    get_fields,
    get_node_to_elem,
    get_vertices,
    get_volume_data,
    to_float32,
)


@pytest.fixture
def mock_polydata():
    """Create a VTK polydata object with test data."""
    polydata = vtk.vtkPolyData()

    # Create points
    points = vtk.vtkPoints()
    points.InsertNextPoint(1, 2, 3)
    points.InsertNextPoint(4, 5, 6)
    polydata.SetPoints(points)

    # Create point data arrays
    umean = vtk.vtkFloatArray()
    umean.SetName("UMean")
    umean.SetNumberOfComponents(3)
    umean.InsertNextTuple3(1, 0, 0)
    umean.InsertNextTuple3(0, 1, 0)

    pmean = vtk.vtkFloatArray()
    pmean.SetName("pMean")
    pmean.SetNumberOfComponents(1)
    pmean.InsertNextValue(1.0)
    pmean.InsertNextValue(2.0)

    point_data = polydata.GetPointData()
    point_data.AddArray(umean)
    point_data.AddArray(pmean)

    return polydata


def test_to_float32():
    assert to_float32(1) == np.ones(1, dtype=np.float32)
    assert to_float32([1]) == np.ones(1, dtype=np.float32)
    assert to_float32(np.ones(1)) == np.ones(1, dtype=np.float32)
    assert to_float32(None) is None


def test_get_node_to_elem(mock_polydata):
    """Test node to element conversion."""
    cell_data = get_node_to_elem(mock_polydata)
    assert isinstance(cell_data, vtk.vtkDataSet)


def test_get_fields(mock_polydata):
    """Test field extraction from VTK data."""
    point_data = mock_polydata.GetPointData()
    variables = ["UMean", "pMean"]

    fields = get_fields(point_data, variables)
    assert len(fields) == len(variables)
    assert isinstance(fields[0], np.ndarray)
    assert fields[0].shape == (2, 3)  # UMean is vector
    assert fields[1].shape == (2, 1)  # pMean is scalar


def test_get_fields_missing_array(mock_polydata):
    """Test error handling for missing arrays."""
    point_data = mock_polydata.GetPointData()

    with pytest.raises(ValueError, match="Failed to get array"):
        get_fields(point_data, ["nonexistent"])


def test_get_vertices(mock_polydata):
    """Test vertex extraction."""
    vertices = get_vertices(mock_polydata)
    assert isinstance(vertices, np.ndarray)
    assert vertices.shape == (2, 3)  # Two points, 3D coordinates
    np.testing.assert_array_equal(vertices[0], [1, 2, 3])
    np.testing.assert_array_equal(vertices[1], [4, 5, 6])


def test_get_volume_data(mock_polydata):
    """Test volume data extraction."""
    variables = ["UMean", "pMean"]
    vertices, fields = get_volume_data(mock_polydata, variables)

    assert isinstance(vertices, np.ndarray)
    assert vertices.shape == (2, 3)
    assert len(fields) == len(variables)
    assert fields[0].shape == (2, 3)  # UMean
    assert fields[1].shape == (2, 1)  # pMean
