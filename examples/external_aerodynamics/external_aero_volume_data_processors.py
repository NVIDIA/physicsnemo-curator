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

from examples.external_aerodynamics.constants import PhysicsConstants
from examples.external_aerodynamics.external_aero_utils import (
    get_volume_data,
    to_float32,
)
from examples.external_aerodynamics.schemas import (
    ExternalAerodynamicsExtractedDataInMemory,
)

import logging
from examples.external_aerodynamics.constants import PhysicsConstants
from examples.external_aerodynamics.external_aero_utils import to_float32
from examples.external_aerodynamics.schemas import (
    ExternalAerodynamicsExtractedDataInMemory,
)

# Add these lines:
logging.basicConfig(
    format="%(asctime)s - Process %(process)d - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)  # ← ADD THIS


def default_volume_processing_for_external_aerodynamics(
    data: ExternalAerodynamicsExtractedDataInMemory,
    volume_variables: list[str],
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Default volume processing for External Aerodynamics."""
    data.volume_mesh_centers, data.volume_fields = get_volume_data(
        data.volume_unstructured_grid, volume_variables
    )
    data.volume_fields = np.concatenate(data.volume_fields, axis=-1)
    return data


def non_dimensionalize_volume_fields(
    data: ExternalAerodynamicsExtractedDataInMemory,
    air_density: float = None,  # do not use physics constants here
    stream_velocity: float = None,  # do not use physics constants here
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Non-dimensionalize volume fields."""

     # Prefer metadata values (from params.json), fall back to config parameters
    rho = data.metadata.air_density if data.metadata.air_density is not None else air_density
    V = data.metadata.stream_velocity if data.metadata.stream_velocity is not None else stream_velocity

    logger.info(f"Volume, using air density: {rho} kg/m^3")
    logger.info(f"Volume, using stream velocity: {V} m/s")

    if rho <= 0:
        logger.error(f"Air density must be > 0: {rho}")
        return data
    if V <= 0:
        logger.error(f"Stream velocity must be > 0: {V}")
        return data

    stl_vertices = data.stl_polydata.points
    length_scale = np.amax(np.amax(stl_vertices, 0) - np.amin(stl_vertices, 0))

    num_fields = data.volume_fields.shape[1]

    # Normalize velocity (columns 0-2)
    if num_fields >= 3:
        data.volume_fields[:, :3] = data.volume_fields[:, :3] / V

    # Normalize pressure (column 3)
    if num_fields >= 4:
        data.volume_fields[:, 3:4] = data.volume_fields[:, 3:4] / (rho * V**2.0)

    # Normalize turbulent viscosity (column 4+) if present
    if num_fields > 4:
        data.volume_fields[:, 4:] = data.volume_fields[:, 4:] / (V * length_scale)
        logger.info(f"Normalized {num_fields - 4} turbulent viscosity field(s)")
    else:
        logger.info("No turbulent viscosity field found, skipping normalization")
   
    return data


def update_volume_data_to_float32(
    data: ExternalAerodynamicsExtractedDataInMemory,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Update volume data to float32."""
    data.volume_mesh_centers = to_float32(data.volume_mesh_centers)
    data.volume_fields = to_float32(data.volume_fields)
    return data
