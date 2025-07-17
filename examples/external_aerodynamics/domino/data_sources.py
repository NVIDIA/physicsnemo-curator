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

import shutil
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyvista as pv
import vtk
import zarr

from physicsnemo_curator.etl.data_sources import DataSource
from physicsnemo_curator.etl.processing_config import ProcessingConfig

from .constants import DatasetKind, ModelType, PhysicsConstants
from .domino_utils import get_volume_data, to_float32
from .paths import get_path_getter
from .schemas import (
    DoMINOExtractedDataInMemory,
    DoMINOMetadata,
    DoMINONumpyDataInMemory,
    DoMINOZarrDataInMemory,
)


class DoMINODataSource(DataSource):
    """Data source for reading and writing DoMINO simulation data."""

    DECIMATION_ALGOS: tuple[str, ...] = ("decimate_pro", "decimate")

    def __init__(
        self,
        cfg: ProcessingConfig,
        input_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        kind: DatasetKind | str = DatasetKind.DRIVAERML,
        surface_variables: Optional[dict[str, str]] = None,
        volume_variables: Optional[dict[str, str]] = None,
        model_type: Optional[ModelType | str] = None,
        serialization_method: str = "numpy",
        overwrite_existing: bool = True,
        decimation: Optional[dict[str, Any]] = None,
    ):
        super().__init__(cfg)

        self.input_dir = Path(input_dir) if input_dir else None
        self.output_dir = Path(output_dir) if output_dir else None
        self.kind = DatasetKind(kind.lower()) if isinstance(kind, str) else kind
        self.surface_variables = surface_variables
        self.volume_variables = volume_variables
        self.model_type = (
            ModelType(model_type.lower()) if isinstance(model_type, str) else None
        )
        self.serialization_method = serialization_method
        self.overwrite_existing = overwrite_existing

        self.decimation_algo = None
        self.target_reduction = None
        if decimation is not None:
            self.decimation_algo = decimation.pop("algo")
            if self.decimation_algo not in self.DECIMATION_ALGOS:
                raise ValueError(
                    f"Unsupported decimation algo {self.decimation_algo}, must be one of {', '.join(self.DECIMATION_ALGOS)}"
                )
            self.target_reduction = decimation.pop("reduction", 0.0)
            if not 0 <= self.target_reduction < 1.0:
                raise ValueError(
                    f"Expected value in [0, 1), got {self.target_reduction}"
                )
            self.decimation_kwargs = dict(decimation)

        # Validate directories based on read/write usage
        if self.input_dir and not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")
        if self.output_dir and not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.path_getter = get_path_getter(kind)
        self.constants = PhysicsConstants()

    def get_file_list(self) -> list[str]:
        """Get list of simulation directories to process."""
        return sorted(d.name for d in self.input_dir.iterdir() if d.is_dir())

    def read_file(self, dirname: str) -> DoMINOExtractedDataInMemory:
        """Read DoMINO simulation data from a directory.

        Args:
            dirname: Name of the simulation directory

        Returns:
            DoMINOExtractedDataInMemory containing processed simulation data,
            and metadata (DoMINOMetadata).

        Raises:
            FileNotFoundError: STL file is not found.
            FileNotFoundError: Model type is volume/combined and volume data file is not found.
            FileNotFoundError: Model type is surface/combined and surface data file is not found.
        """
        car_dir = self.input_dir / dirname

        # Load STL geometry
        stl_path = self.path_getter.geometry_path(car_dir)
        if not stl_path.exists():
            raise FileNotFoundError(f"STL file not found: {stl_path}")

        reader = pv.get_reader(str(stl_path))
        mesh_stl = reader.read()

        bounds = mesh_stl.bounds
        stl_vertices = mesh_stl.points
        stl_faces = np.array(mesh_stl.faces).reshape((-1, 4))[
            :, 1:
        ]  # Assuming triangular elements
        mesh_indices_flattened = stl_faces.flatten()
        stl_sizes = mesh_stl.compute_cell_sizes(length=False, area=True, volume=False)
        stl_sizes = np.array(stl_sizes.cell_data["Area"])
        stl_centers = np.array(mesh_stl.cell_centers().points)

        length_scale = np.amax(np.amax(stl_vertices, 0) - np.amin(stl_vertices, 0))

        # Initialize volume and surface data
        volume_fields = None
        volume_coordinates = None
        surface_fields = None
        surface_coordinates = None
        surface_normals = None
        surface_sizes = None

        # Load volume data if needed
        if self.model_type in [ModelType.VOLUME, ModelType.COMBINED]:
            volume_path = self.path_getter.volume_path(car_dir)
            if not volume_path.exists():
                raise FileNotFoundError(f"Volume data file not found: {volume_path}")

            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(str(volume_path))
            reader.Update()
            polydata = reader.GetOutput()

            # Process volume data
            volume_coordinates, volume_fields = self._process_volume_data(
                polydata, length_scale
            )

        # Load surface data if needed
        if self.model_type in [ModelType.SURFACE, ModelType.COMBINED]:
            surface_path = self.path_getter.surface_path(car_dir)
            if not surface_path.exists():
                raise FileNotFoundError(f"Surface data file not found: {surface_path}")

            # Process surface data
            (
                surface_coordinates,
                surface_normals,
                surface_sizes,
                surface_fields,
            ) = self._process_surface_data(surface_path)

        metadata = DoMINOMetadata(
            filename=dirname,
            stream_velocity=self.constants.STREAM_VELOCITY,
            air_density=self.constants.AIR_DENSITY,
            x_bound=bounds[0:2],  # xmin, xmax
            y_bound=bounds[2:4],  # ymin, ymax
            z_bound=bounds[4:6],  # zmin, zmax
            dataset_type=self.model_type,  # surface, volume, combined
            num_points=len(mesh_stl.points),
            num_faces=len(mesh_indices_flattened),
            decimation_reduction=self.target_reduction,
            decimation_algo=self.decimation_algo,
        )

        return DoMINOExtractedDataInMemory(
            stl_coordinates=to_float32(stl_vertices),
            stl_centers=to_float32(stl_centers),
            stl_faces=to_float32(mesh_indices_flattened),
            stl_areas=to_float32(stl_sizes),
            surface_mesh_centers=to_float32(surface_coordinates),
            surface_normals=to_float32(surface_normals),
            surface_areas=to_float32(surface_sizes),
            volume_fields=to_float32(volume_fields),
            volume_mesh_centers=to_float32(volume_coordinates),
            surface_fields=to_float32(surface_fields),
            metadata=metadata,
        )

    def _process_volume_data(self, polydata, length_scale):
        """Process volume mesh data."""

        volume_coordinates, volume_fields = get_volume_data(
            polydata, self.volume_variables
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

    def _process_surface_data(self, filename: Path):
        """Process surface mesh data."""

        mesh = pv.read(filename)

        # Decimate mesh if needed.
        mesh = self._decimate_mesh(mesh)

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

    def _decimate_mesh(self, mesh: pv.PolyData):
        """Decimate source mesh."""

        if not (self.decimation_algo is not None and self.target_reduction > 0):
            return mesh

        # Need point_data to interpolate target mesh node values.
        mesh = mesh.cell_data_to_point_data()
        # Decimation algos require tri-mesh.
        mesh = mesh.triangulate()
        match self.decimation_algo:
            case "decimate_pro":
                mesh = mesh.decimate_pro(
                    self.target_reduction, **self.decimation_kwargs
                )
            case "decimate":
                if mesh.n_points > 400_000:
                    warnings.warn(
                        "decimate algo may hang on meshes of size more than 400K"
                    )
                mesh = mesh.decimate(
                    self.target_reduction,
                    attribute_error=True,
                    scalars=True,
                    vectors=True,
                    **self.decimation_kwargs,
                )
            case _:
                raise ValueError(f"Unsupported decimation algo {self.algo}")
        # Compute cell data.
        return mesh.point_data_to_cell_data()

    def write(
        self,
        data: DoMINONumpyDataInMemory | DoMINOZarrDataInMemory,
        filename: str,
    ) -> None:
        """Write transformed data to storage.

        Args:
            data: Transformed data to write (either NumPy or Zarr format)
            filename: Name of the simulation case
        """
        if self.serialization_method == "numpy":
            if not isinstance(data, DoMINONumpyDataInMemory):
                raise TypeError(
                    "Expected DoMINONumpyDataInMemory for numpy serialization"
                )
            self._write_numpy(data, filename)
        elif self.serialization_method == "zarr":
            if not isinstance(data, DoMINOZarrDataInMemory):
                raise TypeError(
                    "Expected DoMINOZarrDataInMemory for zarr serialization"
                )
            self._write_zarr(data, filename)
        else:
            raise ValueError(
                f"Unsupported serialization method: {self.serialization_method}"
            )

    def _write_numpy(self, data: DoMINONumpyDataInMemory, filename: str) -> None:
        """Write data in NumPy format (legacy support).

        Note: This format supports only basic metadata. For full metadata support,
        use Zarr format instead.
        """
        output_file = self.output_dir / f"{filename}.npz"

        # Convert to dict for numpy storage
        save_dict = {
            # Arrays
            "stl_coordinates": data.stl_coordinates,
            "stl_centers": data.stl_centers,
            "stl_faces": data.stl_faces,
            "stl_areas": data.stl_areas,
            # Basic metadata
            "filename": data.metadata.filename,
            "stream_velocity": data.metadata.stream_velocity,
            "air_density": data.metadata.air_density,
        }

        # Add optional arrays if present
        for field in [
            "surface_mesh_centers",
            "surface_normals",
            "surface_areas",
            "surface_fields",
            "volume_mesh_centers",
            "volume_fields",
        ]:
            value = getattr(data, field)
            if value is not None:
                save_dict[field] = value

        np.savez(output_file, **save_dict)

    def _write_zarr(self, data: DoMINOZarrDataInMemory, filename: str) -> None:
        """Write data in Zarr format with full metadata support."""
        store_path = self.output_dir / f"{filename}.zarr"

        # Check if store exists
        if store_path.exists():
            self.logger.warning(f"Overwriting existing data for {filename}")
            shutil.rmtree(store_path)

        # Create store
        zarr_store = zarr.DirectoryStore(store_path)
        root = zarr.group(store=zarr_store)

        # Write metadata as attributes
        root.attrs.update(asdict(data.metadata))

        # Write required arrays
        for field in ["stl_coordinates", "stl_centers", "stl_faces", "stl_areas"]:
            array_info = getattr(data, field)
            root.create_dataset(
                field,
                data=array_info.data,
                chunks=array_info.chunks,
                compressor=array_info.compressor,
            )

        # Write optional arrays if present
        for field in [
            "surface_mesh_centers",
            "surface_normals",
            "surface_areas",
            "surface_fields",
            "volume_mesh_centers",
            "volume_fields",
        ]:
            array_info = getattr(data, field)
            if array_info is not None:
                root.create_dataset(
                    field,
                    data=array_info.data,
                    chunks=array_info.chunks,
                    compressor=array_info.compressor,
                )

    def should_skip(self, filename: str) -> bool:
        """Checks whether the file should be skipped."""
        if self.overwrite_existing:
            return False

        match self.serialization_method:
            case "numpy":
                # Skip if the file already exists.
                return (self.output_dir / f"{filename}.npz").exists()
            case "zarr":
                # Skip if the file already exists.
                return (self.output_dir / f"{filename}.zarr").exists()
            case _:
                raise ValueError(
                    f"Unsupported serialization method: {self.serialization_method}"
                )
