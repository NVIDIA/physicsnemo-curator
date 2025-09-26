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
from typing import Any, Callable, Optional

import numpy as np
from numcodecs import Blosc

from examples.external_aerodynamics.external_aero_surface_data_processors import (
    default_surface_processing_for_external_aerodynamics,
)
from examples.external_aerodynamics.external_aero_volume_data_processors import (
    default_volume_processing_for_external_aerodynamics,
)
from physicsnemo_curator.etl.data_transformations import DataTransformation
from physicsnemo_curator.etl.processing_config import ProcessingConfig
from physicsnemo_curator.utils.utils import setup_logger

from .constants import PhysicsConstants
from .external_aero_utils import (
    to_float32,
)
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


class ExternalAerodynamicsSTLTransformation(DataTransformation):
    """Transforms STL data for External Aerodynamics model."""

    def __init__(self, cfg: ProcessingConfig):
        super().__init__(cfg)
        self.constants = PhysicsConstants()

    def transform(
        self, data: ExternalAerodynamicsExtractedDataInMemory
    ) -> ExternalAerodynamicsExtractedDataInMemory:
        """Transform STL data for External Aerodynamics model."""

        # Process STL data
        stl_vertices = data.stl_polydata.points
        stl_faces = (
            np.array(data.stl_polydata.faces).reshape((-1, 4))[:, 1:].astype(np.int32)
        )  # Assuming triangular elements
        mesh_indices_flattened = stl_faces.flatten()
        stl_sizes = data.stl_polydata.compute_cell_sizes(
            length=False, area=True, volume=False
        )
        stl_sizes = np.array(stl_sizes.cell_data["Area"])
        stl_centers = np.array(data.stl_polydata.cell_centers().points)

        # Update processed STL data
        data.stl_coordinates = to_float32(stl_vertices)
        data.stl_centers = to_float32(stl_centers)
        data.stl_faces = mesh_indices_flattened
        data.stl_areas = to_float32(stl_sizes)

        # Update metadata
        bounds = data.stl_polydata.bounds
        data.metadata.x_bound = bounds[0:2]  # xmin, xmax
        data.metadata.y_bound = bounds[2:4]  # ymin, ymax
        data.metadata.z_bound = bounds[4:6]  # zmin, zmax
        data.metadata.num_points = len(data.stl_polydata.points)
        data.metadata.num_faces = len(mesh_indices_flattened)
        data.metadata.stream_velocity = self.constants.STREAM_VELOCITY
        data.metadata.air_density = self.constants.AIR_DENSITY

        # Delete raw STL data to save memory
        data.stl_polydata = None

        return data


class ExternalAerodynamicsSurfaceTransformation(DataTransformation):
    """Transforms surface data for External Aerodynamics model."""

    def __init__(
        self,
        cfg: ProcessingConfig,
        surface_variables: Optional[dict[str, str]] = None,
        surface_processors: Optional[tuple[Callable, ...]] = None,
    ):
        super().__init__(cfg)
        self.logger = setup_logger()
        self.logger.info(
            f"Initializing ExternalAerodynamicsSurfaceTransformation with surface_variables: {surface_variables} and surface_processors: {surface_processors}"
        )
        self.surface_variables = surface_variables
        self.surface_processors = surface_processors
        self.constants = PhysicsConstants()

    def transform(
        self, data: ExternalAerodynamicsExtractedDataInMemory
    ) -> ExternalAerodynamicsExtractedDataInMemory:
        """Transform surface data for External Aerodynamics model."""

        if data.surface_polydata is not None:

            data = default_surface_processing_for_external_aerodynamics(
                data, self.surface_variables
            )

            if self.surface_processors is not None:
                for processor in self.surface_processors:
                    self.logger.info(
                        f"Processing surface data with processor: {processor}"
                    )
                    data = processor(data)

            # Delete raw surface data to save memory
            data.surface_polydata = None

        return data


class ExternalAerodynamicsVolumeTransformation(DataTransformation):
    """Transforms volume data for External Aerodynamics model."""

    def __init__(
        self,
        cfg: ProcessingConfig,
        volume_variables: Optional[dict[str, str]] = None,
        volume_processors: Optional[tuple[tuple[Callable, dict[str, Any]], ...]] = None,
    ):
        super().__init__(cfg)
        self.volume_variables = volume_variables
        self.volume_processors = volume_processors
        self.constants = PhysicsConstants()

    def transform(
        self, data: ExternalAerodynamicsExtractedDataInMemory
    ) -> ExternalAerodynamicsExtractedDataInMemory:

        # Load volume data if needed
        if data.volume_unstructured_grid is not None:

            data = default_volume_processing_for_external_aerodynamics(
                data, self.volume_variables
            )

            for processor, kwargs in self.volume_processors:
                data = processor(data, **kwargs)

            # Delete raw volume data to save memory
            data.volume_unstructured_grid = None

        return data


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
