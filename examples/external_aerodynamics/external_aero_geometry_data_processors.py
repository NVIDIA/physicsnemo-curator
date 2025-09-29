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


import numpy as np

from examples.external_aerodynamics.external_aero_utils import to_float32
from examples.external_aerodynamics.schemas import (
    ExternalAerodynamicsExtractedDataInMemory,
)


def default_geometry_processing_for_external_aerodynamics(
    data: ExternalAerodynamicsExtractedDataInMemory,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Default geometry processing for External Aerodynamics."""

    data.stl_vertices = data.stl_polydata.points
    data.stl_faces = (
        np.array(data.stl_polydata.faces).reshape((-1, 4))[:, 1:].astype(np.int32)
    ).flatten()  # Assuming triangular elements
    data.stl_areas = data.stl_polydata.compute_cell_sizes(
        length=False, area=True, volume=False
    )
    data.stl_areas = np.array(data.stl_areas.cell_data["Area"])
    data.stl_centers = np.array(data.stl_polydata.cell_centers().points)

    # Update metadata
    bounds = data.stl_polydata.bounds
    data.metadata.x_bound = bounds[0:2]  # xmin, xmax
    data.metadata.y_bound = bounds[2:4]  # ymin, ymax
    data.metadata.z_bound = bounds[4:6]  # zmin, zmax
    data.metadata.num_points = len(data.stl_polydata.points)
    data.metadata.num_faces = len(data.stl_faces)

    return data


def update_geometry_data_to_float32(
    data: ExternalAerodynamicsExtractedDataInMemory,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Update geometry data to float32."""

    data.stl_coordinates = to_float32(data.stl_vertices)
    data.stl_centers = to_float32(data.stl_centers)
    data.stl_areas = to_float32(data.stl_areas)
    # data.stl_faces will be left as is (np.int32)

    return data
