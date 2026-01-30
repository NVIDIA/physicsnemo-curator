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

from pathlib import Path

import pytest
from paths import (
    DrivAerMLPaths,
    DriveSimPaths,
    VTKPaths,
)


def test_vtk_paths_enum():
    """Test VTKPaths enum values."""
    assert VTKPaths.FOAM_DIR == "VTK/simpleFoam_steady_3000"
    assert VTKPaths.INTERNAL == "internal.vtu"
    assert VTKPaths.BOUNDARY == "boundary"


class TestDriveSimPaths:
    """Test DriveSimPaths class."""

    @pytest.fixture
    def car_dir(self):
        return Path("/test/car")

    def test_geometry_path(self, car_dir):
        """Test geometry path construction."""
        path = DriveSimPaths.geometry_path(car_dir)
        assert path == car_dir / DriveSimPaths.GEOMETRY_FILE
        assert path.name == "body.stl"

    def test_volume_path(self, car_dir):
        """Test volume path construction."""
        path = DriveSimPaths.volume_path(car_dir)
        assert path == car_dir / VTKPaths.FOAM_DIR.value / VTKPaths.INTERNAL.value

    def test_surface_path(self, car_dir):
        """Test surface path construction."""
        path = DriveSimPaths.surface_path(car_dir)
        assert (
            path
            == car_dir
            / VTKPaths.FOAM_DIR.value
            / VTKPaths.BOUNDARY.value
            / DriveSimPaths.SURFACE_FILE
        )


class TestDrivAerMLPaths:
    """Test DrivAerMLPaths class."""

    @pytest.fixture
    def car_dir(self):
        return Path("/test/run_123")

    def test_get_index(self, car_dir):
        """Test index extraction from directory name."""
        assert DrivAerMLPaths._get_index(car_dir) == "123"

    def test_geometry_path(self, car_dir):
        """Test geometry path construction."""
        path = DrivAerMLPaths.geometry_path(car_dir)
        assert path == car_dir / "drivaer_123.stl"

    def test_volume_path(self, car_dir):
        """Test volume path construction."""
        path = DrivAerMLPaths.volume_path(car_dir)
        assert path == car_dir / "volume_123.vtu"

    def test_surface_path(self, car_dir):
        """Test surface path construction."""
        path = DrivAerMLPaths.surface_path(car_dir)
        assert path == car_dir / "boundary_123.vtp"

    def test_invalid_directory_name(self):
        """Test handling of invalid directory names."""
        invalid_dir = Path("/test/invalid")
        with pytest.raises(ValueError):
            DrivAerMLPaths._get_index(invalid_dir)
