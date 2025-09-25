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

from dataclasses import dataclass
from typing import Optional

import numcodecs
import numpy as np
import pyvista as pv
import vtk

from .constants import ModelType


@dataclass
class ExternalAerodynamicsMetadata:
    """Metadata for External Aerodynamics simulation data.

    Version history:
    - 1.0: Initial version with expected metadata fields.
    """

    # Simulation identifiers
    filename: str
    dataset_type: ModelType

    # Physical parameters
    stream_velocity: Optional[float] = None
    air_density: Optional[float] = None

    # Geometry bounds
    x_bound: Optional[tuple[float, float]] = None  # xmin, xmax
    y_bound: Optional[tuple[float, float]] = None  # ymin, ymax
    z_bound: Optional[tuple[float, float]] = None  # zmin, zmax

    # Mesh statistics
    num_points: Optional[int] = None
    num_faces: Optional[int] = None

    # Processing parameters
    decimation_reduction: Optional[float] = None
    decimation_algo: Optional[str] = None


@dataclass
class ExternalAerodynamicsExtractedDataInMemory:
    """Container for External Aerodynamics data and metadata extracted from the simulation.

    Version history:
    - 1.0: Initial version with expected data fields.
    """

    # Metadata
    metadata: ExternalAerodynamicsMetadata

    # Raw data
    stl_polydata: Optional[pv.PolyData] = None
    surface_polydata: Optional[pv.PolyData] = None
    volume_unstructured_grid: Optional[vtk.vtkUnstructuredGrid] = None

    # Processed geometry data
    stl_coordinates: Optional[np.ndarray] = None
    stl_centers: Optional[np.ndarray] = None
    stl_faces: Optional[np.ndarray] = None
    stl_areas: Optional[np.ndarray] = None

    # Processed surface data
    surface_mesh_centers: Optional[np.ndarray] = None
    surface_normals: Optional[np.ndarray] = None
    surface_areas: Optional[np.ndarray] = None
    surface_fields: Optional[np.ndarray] = None

    # Processed volume data
    volume_mesh_centers: Optional[np.ndarray] = None
    volume_fields: Optional[np.ndarray] = None


@dataclass(frozen=True)
class PreparedZarrArrayInfo:
    """Information for preparing an array for Zarr storage.

    Version history:
    - 1.0: Initial version with compression and chunking info
    """

    data: np.ndarray
    chunks: tuple[int, ...]
    compressor: numcodecs.abc.Codec


@dataclass(frozen=True)
class ExternalAerodynamicsZarrDataInMemory:
    """Container for External Aerodynamics data prepared for Zarr storage.

    Version history:
    - 1.0: Initial version with prepared arrays for Zarr storage
    """

    # Metadata
    metadata: ExternalAerodynamicsMetadata

    # Geometry data
    stl_coordinates: PreparedZarrArrayInfo
    stl_centers: PreparedZarrArrayInfo
    stl_faces: PreparedZarrArrayInfo
    stl_areas: PreparedZarrArrayInfo

    # Surface data
    surface_mesh_centers: Optional[PreparedZarrArrayInfo] = None
    surface_normals: Optional[PreparedZarrArrayInfo] = None
    surface_areas: Optional[PreparedZarrArrayInfo] = None
    surface_fields: Optional[PreparedZarrArrayInfo] = None

    # Volume data
    volume_mesh_centers: Optional[PreparedZarrArrayInfo] = None
    volume_fields: Optional[PreparedZarrArrayInfo] = None


@dataclass(frozen=True)
class ExternalAerodynamicsNumpyMetadata:
    """Minimal metadata for legacy NumPy storage format.

    Note: For full metadata support, use Zarr storage format instead.
    """

    filename: str
    stream_velocity: float
    air_density: float


@dataclass(frozen=True)
class ExternalAerodynamicsNumpyDataInMemory:
    """Container for External Aerodynamics data prepared for NumPy storage.

    Version history:
    - 1.0: Legacy version with basic arrays and minimal metadata.
        For full feature support (including complete metadata), use Zarr format.
    """

    # Basic metadata (legacy support)
    metadata: ExternalAerodynamicsNumpyMetadata

    # Geometry data
    stl_coordinates: np.ndarray
    stl_centers: np.ndarray
    stl_faces: np.ndarray
    stl_areas: np.ndarray

    # Surface data
    surface_mesh_centers: Optional[np.ndarray] = None
    surface_normals: Optional[np.ndarray] = None
    surface_areas: Optional[np.ndarray] = None
    surface_fields: Optional[np.ndarray] = None

    # Volume data
    volume_mesh_centers: Optional[np.ndarray] = None
    volume_fields: Optional[np.ndarray] = None
