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
    air_density: float = PhysicsConstants.AIR_DENSITY,
    stream_velocity: float = PhysicsConstants.STREAM_VELOCITY,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Non-dimensionalize volume fields."""

    stl_vertices = data.stl_polydata.points
    length_scale = np.amax(np.amax(stl_vertices, 0) - np.amin(stl_vertices, 0))
    data.volume_fields[:, :3] = data.volume_fields[:, :3] / stream_velocity
    data.volume_fields[:, 3:4] = data.volume_fields[:, 3:4] / (
        air_density * stream_velocity**2.0
    )
    data.volume_fields[:, 4:] = data.volume_fields[:, 4:] / (
        stream_velocity * length_scale
    )
    return data


def update_volume_data_to_float32(
    data: ExternalAerodynamicsExtractedDataInMemory,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Update volume data to float32."""
    data.volume_mesh_centers = to_float32(data.volume_mesh_centers)
    data.volume_fields = to_float32(data.volume_fields)
    return data


def shuffle_volume_data(
    data: ExternalAerodynamicsExtractedDataInMemory,
    seed: int = 42,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """
    Shuffle volume data.

    This is useful because instead of randomly accessing the data upon read,
    we can shuffle the data during preprocessing, and do sequential reads.
    """

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(data.volume_mesh_centers))
    data.volume_mesh_centers = data.volume_mesh_centers[indices]
    data.volume_fields = data.volume_fields[indices]

    return data
