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

import warnings

import numpy as np

from examples.external_aerodynamics.constants import PhysicsConstants
from examples.external_aerodynamics.external_aero_utils import to_float32
from examples.external_aerodynamics.schemas import (
    ExternalAerodynamicsExtractedDataInMemory,
)


def default_surface_processing_for_external_aerodynamics(
    data: ExternalAerodynamicsExtractedDataInMemory,
    surface_variables: list[str],
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Default surface processing for External Aerodynamics."""

    try:
        cell_data = (data.surface_polydata.cell_data[k] for k in surface_variables)
        data.surface_fields = np.concatenate(
            [d if d.ndim > 1 else d[:, np.newaxis] for d in cell_data], axis=-1
        )
        data.surface_coordinates = np.array(data.surface_polydata.cell_centers().points)
        data.surface_normals = np.array(data.surface_polydata.cell_normals)
        data.surface_areas = data.surface_polydata.compute_cell_sizes(
            length=False, area=True, volume=False
        )
        data.surface_areas = np.array(data.surface_areas.cell_data["Area"])

    except Exception as e:
        raise ValueError(
            f"Error in default_surface_processing_for_external_aerodynamics: {e}"
        )

    return data


def normalize_surface_normals(
    data: ExternalAerodynamicsExtractedDataInMemory,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Normalize surface normals."""

    # Normalize cell normals
    data.surface_normals = (
        data.surface_normals
        / np.linalg.norm(data.surface_normals, axis=1)[:, np.newaxis]
    )

    return data


def non_dimensionalize_surface_fields(
    data: ExternalAerodynamicsExtractedDataInMemory,
    air_density: float = PhysicsConstants.AIR_DENSITY,
    stream_velocity: float = PhysicsConstants.STREAM_VELOCITY,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Non-dimensionalize surface fields."""

    # Non-dimensionalize surface fields
    data.surface_fields = data.surface_fields / (air_density * stream_velocity**2.0)

    return data


def update_surface_data_to_float32(
    data: ExternalAerodynamicsExtractedDataInMemory,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Update surface data to float32."""

    # Update processed surface data
    data.surface_mesh_centers = to_float32(data.surface_coordinates)
    data.surface_normals = to_float32(data.surface_normals)
    data.surface_areas = to_float32(data.surface_areas)
    data.surface_fields = to_float32(data.surface_fields)

    return data


def decimate_mesh(
    data: ExternalAerodynamicsExtractedDataInMemory,
    algo: str = None,
    reduction: float = 0.0,
    **kwargs,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Decimate mesh using pyvista."""

    if not algo or reduction <= 0:
        warnings.warn("Decimation algo or reduction is not set or is <= 0")
        return data

    mesh = data.surface_polydata

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
    data.surface_polydata = mesh.point_data_to_cell_data()

    # Update metadata
    data.metadata.decimation_algo = algo
    data.metadata.decimation_reduction = reduction

    return data
