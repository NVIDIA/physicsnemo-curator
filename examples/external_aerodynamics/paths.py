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
This module contains the path definitions for the External Aerodynamics dataset.
"""

from enum import Enum
from pathlib import Path

from .constants import DatasetKind


class VTKPaths(str, Enum):
    """Common VTK directory and file patterns."""

    FOAM_DIR = "VTK/simpleFoam_steady_3000"
    INTERNAL = "internal.vtu"
    BOUNDARY = "boundary"


class DriveSimPaths:
    """Utility class for handling DriveSim dataset file paths.

    This class provides static methods to construct paths for different components
    of the DriveSim dataset (geometry, volume, and surface data).
    """

    GEOMETRY_FILE = "body.stl"
    SURFACE_FILE = "aero_suv.vtp"

    @staticmethod
    def geometry_path(car_dir: Path) -> Path:
        """Get path to the STL geometry file.

        Args:
            car_dir: Base directory for the car data

        Returns:
            Path: Path to the STL file containing car geometry
        """
        return car_dir / DriveSimPaths.GEOMETRY_FILE

    @staticmethod
    def volume_path(car_dir: Path) -> Path:
        """Get path to the volume data file.

        Args:
            car_dir: Base directory for the car data

        Returns:
            Path: Path to the VTU file containing volume data
        """
        return car_dir / VTKPaths.FOAM_DIR.value / VTKPaths.INTERNAL.value

    @staticmethod
    def surface_path(car_dir: Path) -> Path:
        """Get path to the surface data file.

        Args:
            car_dir: Base directory for the car data

        Returns:
            Path: Path to the VTP file containing surface data
        """
        return (
            car_dir
            / VTKPaths.FOAM_DIR.value
            / VTKPaths.BOUNDARY.value
            / DriveSimPaths.SURFACE_FILE
        )


class OpenFoamDatasetPaths:
    """Utility base class for handling OpenFOAM-produced datasets file paths.

    This class provides static methods to construct paths for different components
    of the OpenFOAM dataset such as volume and surface data which are common
    across OpenFOAM datasets.
    """

    @staticmethod
    def _get_index(car_dir: Path) -> str:
        name = car_dir.name
        if not name.startswith("run_"):
            raise ValueError(f"Directory name must start with 'run_', got: {name}")
        return name.removeprefix("run_")

    @staticmethod
    def volume_path(car_dir: Path) -> Path:
        index = OpenFoamDatasetPaths._get_index(car_dir)
        return car_dir / f"volume_{index}.vtu"

    @staticmethod
    def surface_path(car_dir: Path) -> Path:
        index = OpenFoamDatasetPaths._get_index(car_dir)
        return car_dir / f"boundary_{index}.vtp"


class DrivAerMLPaths(OpenFoamDatasetPaths):
    """Utility class for handling DrivAerML dataset file paths.

    This class provides static methods to construct paths for different components
    of the DrivAerML dataset (geometry, volume, and surface data).
    """

    @staticmethod
    def geometry_path(car_dir: Path) -> Path:
        """Returns geometry path."""

        index = DrivAerMLPaths._get_index(car_dir)
        return car_dir / f"drivaer_{index}.stl"


class AhmedMLPaths(OpenFoamDatasetPaths):
    """Utility class for handling AhmedML dataset file paths.

    This class provides static methods to construct paths for different components
    of the AhmedML dataset (geometry, volume, and surface data).
    """

    @staticmethod
    def geometry_path(car_dir: Path) -> Path:
        """Returns geometry path."""

        index = AhmedMLPaths._get_index(car_dir)
        return car_dir / f"ahmed_{index}.stl"


def get_path_getter(kind: DatasetKind):
    """Returns path getter for a given dataset type."""

    match kind:
        case DatasetKind.AHMEDML:
            return AhmedMLPaths
        case DatasetKind.DRIVAERML:
            return DrivAerMLPaths
        case DatasetKind.DRIVESIM:
            return DriveSimPaths
