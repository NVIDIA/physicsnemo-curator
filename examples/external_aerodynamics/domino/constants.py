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
This module contains constants and enums for the DoMINO dataset.
"""

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class PhysicsConstants:
    """Physical constants used in the simulation."""

    AIR_DENSITY: float = 1.205  # kg/m³
    STREAM_VELOCITY: float = 30.00  # m/s


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


@dataclass(frozen=True)
class DefaultVariables:
    """Default variables to extract from the simulation."""

    SURFACE: tuple[str, ...] = ("pMean", "wallShearStress")
    VOLUME: tuple[str, ...] = ("UMean", "pMean")
