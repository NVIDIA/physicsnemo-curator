# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""
Utilities for processing DoMINO data.
"""

import warnings
from typing import Optional, TypeAlias

import numpy as np
import pyvista as pv
import vtk
from vtk.util import numpy_support

OptionalNDArray: TypeAlias = Optional[np.ndarray]


def to_float32(array: OptionalNDArray) -> OptionalNDArray:
    """Convert array to float32 if not None.

    Args:
        array: Input array or None

    Returns:
        Array converted to float32 or None if input was None
    """
    return np.float32(array) if array is not None else None


def get_node_to_elem(polydata):
    """Function to convert node to elem"""
    c2p = vtk.vtkPointDataToCellData()
    c2p.SetInputData(polydata)
    c2p.Update()
    cell_data = c2p.GetOutput()
    return cell_data


def get_fields(data, variables):
    """Function to get fields from VTP/VTU"""
    fields = []
    for array_name in variables:
        array = data.GetArray(array_name)
        if array is None:
            raise ValueError(
                f"Failed to get array {array_name} from the unstructured grid."
            )
        array_data = numpy_support.vtk_to_numpy(array).reshape(
            array.GetNumberOfTuples(), array.GetNumberOfComponents()
        )
        fields.append(array_data)
    return fields


def get_vertices(polydata):
    """Function to get vertices"""
    points = polydata.GetPoints()
    vertices = numpy_support.vtk_to_numpy(points.GetData())
    return vertices


def get_volume_data(polydata, variables):
    """Function to get volume data"""
    vertices = get_vertices(polydata)
    point_data = polydata.GetPointData()

    fields = get_fields(point_data, variables)

    return vertices, fields


def decimate_mesh(
    mesh: pv.PolyData,
    algo: str,
    reduction: float,
    kwargs: dict,
) -> pv.PolyData:
    """Decimate mesh using pyvista."""

    # Need point_data to interpolate target mesh node values.
    mesh = mesh.cell_data_to_point_data()

    # Decimation algos require tri-mesh.
    mesh = mesh.triangulate()
    match algo:
        case "decimate_pro":
            mesh = mesh.decimate_pro(reduction, **kwargs)
        case "decimate":
            if mesh.n_points > 400_000:
                warnings.warn("decimate algo may hang on meshes of size more than 400K")
            mesh = mesh.decimate(
                reduction,
                attribute_error=True,
                scalars=True,
                vectors=True,
                **kwargs,
            )
        case _:
            raise ValueError(f"Unsupported decimation algo {algo}")

    # Compute cell data.
    return mesh.point_data_to_cell_data()
