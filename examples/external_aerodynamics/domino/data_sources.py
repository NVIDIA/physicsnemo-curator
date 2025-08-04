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
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pyvista as pv
import vtk
import zarr

from physicsnemo_curator.etl.data_sources import DataSource
from physicsnemo_curator.etl.processing_config import ProcessingConfig

from .constants import DatasetKind, ModelType
from .paths import get_path_getter
from .schemas import (
    DoMINOExtractedDataInMemory,
    DoMINOMetadata,
    DoMINONumpyDataInMemory,
    DoMINOZarrDataInMemory,
)


class DoMINODataSource(DataSource):
    """Data source for reading and writing DoMINO simulation data."""

    def __init__(
        self,
        cfg: ProcessingConfig,
        input_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        kind: DatasetKind | str = DatasetKind.DRIVAERML,
        model_type: Optional[ModelType | str] = None,
        serialization_method: str = "numpy",
        overwrite_existing: bool = True,
    ):
        super().__init__(cfg)

        self.input_dir = Path(input_dir) if input_dir else None
        self.output_dir = Path(output_dir) if output_dir else None
        self.kind = DatasetKind(kind.lower()) if isinstance(kind, str) else kind
        self.model_type = (
            ModelType(model_type.lower()) if isinstance(model_type, str) else None
        )
        self.serialization_method = serialization_method
        self.overwrite_existing = overwrite_existing

        # Validate directories based on read/write usage
        if self.input_dir and not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")
        if self.output_dir and not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.path_getter = get_path_getter(kind)

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
        stl_polydata = reader.read()

        # Initialize volume and surface data
        surface_polydata = None
        volume_unstructured_grid = None

        # Load volume data if needed
        if self.model_type in [ModelType.VOLUME, ModelType.COMBINED]:
            volume_path = self.path_getter.volume_path(car_dir)
            if not volume_path.exists():
                raise FileNotFoundError(f"Volume data file not found: {volume_path}")

            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(str(volume_path))
            reader.Update()
            volume_unstructured_grid = reader.GetOutput()

        # Load surface data if needed
        if self.model_type in [ModelType.SURFACE, ModelType.COMBINED]:
            surface_path = self.path_getter.surface_path(car_dir)
            if not surface_path.exists():
                raise FileNotFoundError(f"Surface data file not found: {surface_path}")

            surface_polydata = pv.read(surface_path)

        metadata = DoMINOMetadata(
            filename=dirname,
            dataset_type=self.model_type,  # surface, volume, combined
        )

        return DoMINOExtractedDataInMemory(
            stl_polydata=stl_polydata,
            surface_polydata=surface_polydata,
            volume_unstructured_grid=volume_unstructured_grid,
            metadata=metadata,
        )

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
