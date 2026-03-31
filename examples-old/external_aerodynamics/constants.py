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

"""
This module contains constants and enums for external aerodynamics datasets.
"""

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class PhysicsConstantsCarAerodynamics:
    """Physical constants used in the simulation in DriveAerML or AhmedML"""

    AIR_DENSITY: float = 1.205  # kg/m³
    STREAM_VELOCITY: float = 30.00  # m/s


@dataclass(frozen=True)
class PhysicsConstantsHLPW:
    """Physical constants used in the simulation for HLPW dataset."""

    PREF: float = 176.352  # HLPW reference pressure
    UREF: float = 2679.505  # HLPW reference velocity
    TREF: float = 518.67  # HLPW reference temperature


class ModelType(str, Enum):
    """Types of models that can be processed."""

    SURFACE = "surface"
    VOLUME = "volume"
    COMBINED = "combined"


class DatasetKind(str, Enum):
    """Types of datasets that can be processed."""

    DRIVESIM = "drivesim"
    DRIVAERML = "drivaerml"
    AHMEDML = "ahmedml"
    HLPW = "hlpw"


@dataclass(frozen=True)
class DefaultVariables:
    """Default variables to extract from the simulation."""

    SURFACE: tuple[str, ...] = ("pMean", "wallShearStress")
    VOLUME: tuple[str, ...] = ("UMean", "pMean")


def get_physics_constants(kind: DatasetKind) -> dict[str, float]:
    """Get physics constants dict based on dataset kind. Add a branch
    to the if-elif pipeline below to populate metadata with values
    used for non-dimensionalization.

    Args:
        kind: The dataset kind (from config etl.common.kind)

    Returns:
        Dictionary of physics constant names to values.

    Raises:
        ValueError: If dataset kind is unknown.
    """
    if kind in (DatasetKind.DRIVAERML, DatasetKind.AHMEDML, DatasetKind.DRIVESIM):
        c = PhysicsConstantsCarAerodynamics()
        return {"air_density": c.AIR_DENSITY, "stream_velocity": c.STREAM_VELOCITY}
    elif kind == DatasetKind.HLPW:
        c = PhysicsConstantsHLPW()
        return {"pref": c.PREF, "uref": c.UREF, "tref": c.TREF}
    else:
        raise ValueError(f"Unknown dataset kind: {kind}")
