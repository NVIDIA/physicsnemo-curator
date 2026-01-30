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

import dataclasses

import pytest
from constants import (
    DatasetKind,
    ModelType,
    PhysicsConstantsCarAerodynamics,
    PhysicsConstantsHLPW,
    get_physics_constants,
)


def test_physics_constants_car_aerodynamics_immutability():
    """Test that PhysicsConstantsCarAerodynamics cannot be modified."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        PhysicsConstantsCarAerodynamics().AIR_DENSITY = 2.0


def test_physics_constants_hlpw_immutability():
    """Test that PhysicsConstantsHLPW cannot be modified."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        PhysicsConstantsHLPW().PREF = 200.0


def test_get_physics_constants_drivaerml():
    """Test get_physics_constants returns correct dict for car aero datasets."""
    result = get_physics_constants(DatasetKind.DRIVAERML)
    assert "air_density" in result
    assert "stream_velocity" in result
    assert result["air_density"] == PhysicsConstantsCarAerodynamics.AIR_DENSITY
    assert result["stream_velocity"] == PhysicsConstantsCarAerodynamics.STREAM_VELOCITY


def test_get_physics_constants_hlpw():
    """Test get_physics_constants returns correct dict for HLPW dataset."""
    result = get_physics_constants(DatasetKind.HLPW)
    assert "pref" in result
    assert "uref" in result
    assert "tref" in result
    assert result["pref"] == PhysicsConstantsHLPW.PREF
    assert result["uref"] == PhysicsConstantsHLPW.UREF
    assert result["tref"] == PhysicsConstantsHLPW.TREF


def test_model_type_validation():
    """Test that ModelType only accepts valid values."""
    assert ModelType.SURFACE == "surface"
    with pytest.raises(ValueError):
        ModelType("invalid_type")


def test_dataset_kind_validation():
    """Test that DatasetKind only accepts valid values."""
    assert DatasetKind.DRIVESIM == "drivesim"
    with pytest.raises(ValueError):
        DatasetKind("invalid_dataset")
