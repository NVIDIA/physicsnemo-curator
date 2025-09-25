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
from typing import Any, Optional

import numpy as np
import pyvista as pv
import vtk
from numcodecs import Blosc

from physicsnemo_curator.etl.data_transformations import DataTransformation
from physicsnemo_curator.etl.processing_config import ProcessingConfig

from .constants import PhysicsConstants
from .external_aero_utils import decimate_mesh, get_volume_data, to_float32
from .schemas import (
    ExternalAerodynamicsExtractedDataInMemory,
    ExternalAerodynamicsNumpyDataInMemory,
    ExternalAerodynamicsNumpyMetadata,
    ExternalAerodynamicsZarrDataInMemory,
    PreparedZarrArrayInfo,
)


class ExternalAerodynamicsNumpyTransformation(DataTransformation):
    """Transforms External Aerodynamics data for NumPy storage format (legacy support)."""

    def __init__(self, cfg: ProcessingConfig):
        super().__init__(cfg)

    def transform(
        self, data: ExternalAerodynamicsExtractedDataInMemory
    ) -> ExternalAerodynamicsNumpyDataInMemory:
        """Transform data for NumPy storage format.

        Note: This is a legacy format with minimal metadata support.
        For full metadata support, use Zarr storage format instead.

        Args:
            data: External Aerodynamics extracted data in memory

        Returns:
            Data formatted for NumPy storage with basic metadata
        """
        # Create minimal metadata
        numpy_metadata = ExternalAerodynamicsNumpyMetadata(
            filename=data.metadata.filename,
            stream_velocity=data.metadata.stream_velocity,
            air_density=data.metadata.air_density,
        )

        return ExternalAerodynamicsNumpyDataInMemory(
            stl_coordinates=to_float32(data.stl_coordinates),
            stl_centers=to_float32(data.stl_centers),
            stl_faces=to_float32(data.stl_faces),
            stl_areas=to_float32(data.stl_areas),
            metadata=numpy_metadata,
            surface_mesh_centers=to_float32(data.surface_mesh_centers),
            surface_normals=to_float32(data.surface_normals),
            surface_areas=to_float32(data.surface_areas),
            surface_fields=to_float32(data.surface_fields),
            volume_mesh_centers=to_float32(data.volume_mesh_centers),
            volume_fields=to_float32(data.volume_fields),
        )


class ExternalAerodynamicsPreprocessingTransformation(DataTransformation):
    """General preprocessing of data for External Aerodynamics model."""

    DECIMATION_ALGOS: tuple[str, ...] = ("decimate_pro", "decimate")

    def __init__(
        self,
        cfg: ProcessingConfig,
        surface_variables: Optional[dict[str, str]] = None,
        volume_variables: Optional[dict[str, str]] = None,
        decimation: Optional[dict[str, Any]] = None,
    ):
        super().__init__(cfg)

        self.surface_variables = surface_variables
        self.volume_variables = volume_variables

        self.decimation_algo = None
        self.target_reduction = None
        if decimation is not None:
            self.decimation_algo = decimation.get("algo")
            if self.decimation_algo not in self.DECIMATION_ALGOS:
                raise ValueError(
                    f"Unsupported decimation algo {self.decimation_algo}, must be one of {', '.join(self.DECIMATION_ALGOS)}"
                )
            self.target_reduction = decimation.get("reduction", 0.0)
            if not 0 <= self.target_reduction < 1.0:
                raise ValueError(
                    f"Expected value in [0, 1), got {self.target_reduction}"
                )
            # Copy decimation dict, excluding 'algo' and 'reduction'
            self.decimation_kwargs = {
                k: v for k, v in decimation.items() if k not in ("algo", "reduction")
            }

        self.constants = PhysicsConstants()

    def transform(
        self, data: ExternalAerodynamicsExtractedDataInMemory
    ) -> ExternalAerodynamicsExtractedDataInMemory:
        """Transform data for preprocessing."""

        # Process STL data
        mesh_stl = data.stl_polydata
        stl_vertices = mesh_stl.points
        stl_faces = (
            np.array(mesh_stl.faces).reshape((-1, 4))[:, 1:].astype(np.int32)
        )  # Assuming triangular elements
        mesh_indices_flattened = stl_faces.flatten()
        stl_sizes = mesh_stl.compute_cell_sizes(length=False, area=True, volume=False)
        stl_sizes = np.array(stl_sizes.cell_data["Area"])
        stl_centers = np.array(mesh_stl.cell_centers().points)

        # Delete raw STL data to save memory
        data.stl_polydata = None

        # Update processed STL data
        data.stl_coordinates = to_float32(stl_vertices)
        data.stl_centers = to_float32(stl_centers)
        data.stl_faces = mesh_indices_flattened
        data.stl_areas = to_float32(stl_sizes)

        # Update metadata
        bounds = mesh_stl.bounds
        data.metadata.x_bound = bounds[0:2]  # xmin, xmax
        data.metadata.y_bound = bounds[2:4]  # ymin, ymax
        data.metadata.z_bound = bounds[4:6]  # zmin, zmax
        data.metadata.num_points = len(mesh_stl.points)
        data.metadata.num_faces = len(mesh_indices_flattened)
        data.metadata.stream_velocity = self.constants.STREAM_VELOCITY
        data.metadata.air_density = self.constants.AIR_DENSITY

        # Load volume data if needed
        if data.volume_unstructured_grid is not None:
            # Process volume data
            length_scale = np.amax(np.amax(stl_vertices, 0) - np.amin(stl_vertices, 0))
            volume_coordinates, volume_fields = self._process_volume_data(
                data.volume_unstructured_grid, length_scale
            )

            # Delete raw volume data to save memory
            data.volume_unstructured_grid = None

            # Update processed volume data
            data.volume_mesh_centers = to_float32(volume_coordinates)
            data.volume_fields = to_float32(volume_fields)

        if data.surface_polydata is not None:

            # Process surface data
            (
                surface_coordinates,
                surface_normals,
                surface_sizes,
                surface_fields,
            ) = self._process_surface_data(data.surface_polydata)

            # Delete raw surface data to save memory
            data.surface_polydata = None

            # Update processed surface data
            data.surface_mesh_centers = to_float32(surface_coordinates)
            data.surface_normals = to_float32(surface_normals)
            data.surface_areas = to_float32(surface_sizes)
            data.surface_fields = to_float32(surface_fields)

            # Update metadata
            data.metadata.decimation_algo = self.decimation_algo
            data.metadata.decimation_reduction = self.target_reduction

        return data

    def _process_volume_data(
        self, unstructured_grid: vtk.vtkUnstructuredGrid, length_scale: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Process volume mesh data."""

        volume_coordinates, volume_fields = get_volume_data(
            unstructured_grid, self.volume_variables
        )
        volume_fields = np.concatenate(volume_fields, axis=-1)

        # Non-dimensionalize volume fields
        volume_fields[:, :3] = volume_fields[:, :3] / self.constants.STREAM_VELOCITY
        volume_fields[:, 3:4] = volume_fields[:, 3:4] / (
            self.constants.AIR_DENSITY * self.constants.STREAM_VELOCITY**2.0
        )
        volume_fields[:, 4:] = volume_fields[:, 4:] / (
            self.constants.STREAM_VELOCITY * length_scale
        )

        return volume_coordinates, volume_fields

    def _process_surface_data(
        self,
        mesh: pv.PolyData,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Process surface mesh data."""

        # Decimate mesh if needed.
        if self.decimation_algo is not None and self.target_reduction > 0:
            mesh = decimate_mesh(
                mesh,
                self.decimation_algo,
                self.target_reduction,
                self.decimation_kwargs,
            )

        cell_data = (mesh.cell_data[k] for k in self.surface_variables)
        surface_fields = np.concatenate(
            [d if d.ndim > 1 else d[:, np.newaxis] for d in cell_data], axis=-1
        )
        surface_coordinates = np.array(mesh.cell_centers().points)
        surface_normals = np.array(mesh.cell_normals)
        surface_sizes = mesh.compute_cell_sizes(length=False, area=True, volume=False)
        surface_sizes = np.array(surface_sizes.cell_data["Area"])

        # Normalize cell normals
        surface_normals = (
            surface_normals / np.linalg.norm(surface_normals, axis=1)[:, np.newaxis]
        )

        # Non-dimensionalize surface fields
        surface_fields = surface_fields / (
            self.constants.AIR_DENSITY * self.constants.STREAM_VELOCITY**2.0
        )

        return surface_coordinates, surface_normals, surface_sizes, surface_fields


class ExternalAerodynamicsZarrTransformation(DataTransformation):
    """Transforms External Aerodynamics data for Zarr storage format."""

    def __init__(
        self,
        cfg: ProcessingConfig,
        compression_method: str = "zstd",
        compression_level: int = 5,
        chunk_size_mb: float = 1.0,  # Default 1MB chunk size
    ):
        super().__init__(cfg)
        self.compressor = Blosc(
            cname=compression_method,
            clevel=compression_level,
            shuffle=Blosc.SHUFFLE,
        )
        self.chunk_size_mb = chunk_size_mb

        # Warn if chunk size might be problematic
        if chunk_size_mb < 1.0:
            warnings.warn(
                f"Chunk size of {chunk_size_mb}MB might be too small. "
                "This could lead to poor performance due to overhead.",
                UserWarning,
            )
        elif chunk_size_mb > 50.0:
            warnings.warn(
                f"Chunk size of {chunk_size_mb}MB might be too large. "
                "This could lead to memory issues and poor random access performance.",
                UserWarning,
            )

    def _prepare_array(self, array: np.ndarray) -> PreparedZarrArrayInfo:
        """Prepare array for Zarr storage with compression and chunking."""
        if array is None:
            return None

        # Calculate chunk size based on configured size in MB
        target_chunk_size = int(self.chunk_size_mb * 1024 * 1024)  # Convert MB to bytes
        item_size = array.itemsize
        shape = array.shape

        if len(shape) == 1:
            chunk_size = min(shape[0], target_chunk_size // item_size)
            chunks = (chunk_size,)
        else:
            # For 2D arrays, try to keep rows together
            chunk_rows = min(
                shape[0], max(1, target_chunk_size // (item_size * shape[1]))
            )
            chunks = (chunk_rows, shape[1])

        return PreparedZarrArrayInfo(
            data=np.float32(array),
            chunks=chunks,
            compressor=self.compressor,
        )

    def transform(
        self, data: ExternalAerodynamicsExtractedDataInMemory
    ) -> ExternalAerodynamicsZarrDataInMemory:
        """Transform data for Zarr storage format.

        Organizes data into hierarchical groups and applies compression settings.

        Args:
            data: Dictionary containing External Aerodynamics data

        Returns:
            Dictionary with data formatted for Zarr storage, including:
                - Data organized into groups (stl, surface, volume)
                - Compression settings
                - Chunking configurations
        """
        return ExternalAerodynamicsZarrDataInMemory(
            stl_coordinates=self._prepare_array(data.stl_coordinates),
            stl_centers=self._prepare_array(data.stl_centers),
            stl_faces=self._prepare_array(data.stl_faces),
            stl_areas=self._prepare_array(data.stl_areas),
            metadata=data.metadata,
            surface_mesh_centers=self._prepare_array(data.surface_mesh_centers),
            surface_normals=self._prepare_array(data.surface_normals),
            surface_areas=self._prepare_array(data.surface_areas),
            surface_fields=self._prepare_array(data.surface_fields),
            volume_mesh_centers=self._prepare_array(data.volume_mesh_centers),
            volume_fields=self._prepare_array(data.volume_fields),
        )
