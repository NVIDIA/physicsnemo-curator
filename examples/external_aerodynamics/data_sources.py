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
    ExternalAerodynamicsExtractedDataInMemory,
    ExternalAerodynamicsMetadata,
    ExternalAerodynamicsNumpyDataInMemory,
    ExternalAerodynamicsZarrDataInMemory,
)


class ExternalAerodynamicsDataSource(DataSource):
    """Data source for reading and writing External Aerodynamics simulation data."""

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

    def read_file(self, dirname: str) -> ExternalAerodynamicsExtractedDataInMemory:
        """Read External Aerodynamics simulation data from a directory.

        Args:
            dirname: Name of the simulation directory

        Returns:
            ExternalAerodynamicsExtractedDataInMemory containing processed simulation data,
            and metadata (ExternalAerodynamicsMetadata).

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

            # TODO (@saikrishnanc): Use pyvista to read the volume data.
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

        metadata = ExternalAerodynamicsMetadata(
            filename=dirname,
            dataset_type=self.model_type,  # surface, volume, combined
        )

        return ExternalAerodynamicsExtractedDataInMemory(
            stl_polydata=stl_polydata,
            surface_polydata=surface_polydata,
            volume_unstructured_grid=volume_unstructured_grid,
            metadata=metadata,
        )

    def _get_output_path(self, filename: str) -> Path:
        """Get the final output path for a given filename.

        Args:
            filename: Name of the simulation case

        Returns:
            Path to the output file/directory
        """
        if self.serialization_method == "numpy":
            return self.output_dir / f"{filename}.npz"
        elif self.serialization_method == "zarr":
            return self.output_dir / f"{filename}.zarr"
        else:
            raise ValueError(
                f"Unsupported serialization method: {self.serialization_method}"
            )

    def _write_impl_temp_file(
        self,
        data: (
            ExternalAerodynamicsNumpyDataInMemory | ExternalAerodynamicsZarrDataInMemory
        ),
        output_path: Path,
    ) -> None:
        """Write transformed data to the specified output path.

        Args:
            data: Transformed data to write (either NumPy or Zarr format)
            output_path: Path where data should be written (may be temporary)
        """
        if self.serialization_method == "numpy":
            if not isinstance(data, ExternalAerodynamicsNumpyDataInMemory):
                raise TypeError(
                    "Expected ExternalAerodynamicsNumpyDataInMemory for numpy serialization"
                )
            self._write_numpy(data, output_path)
        elif self.serialization_method == "zarr":
            if not isinstance(data, ExternalAerodynamicsZarrDataInMemory):
                raise TypeError(
                    "Expected ExternalAerodynamicsZarrDataInMemory for zarr serialization"
                )
            self._write_zarr(data, output_path)
        else:
            raise ValueError(
                f"Unsupported serialization method: {self.serialization_method}"
            )

    def _write_numpy(
        self, data: ExternalAerodynamicsNumpyDataInMemory, output_path: Path
    ) -> None:
        """Write data in NumPy format (legacy support).

        Note: This format supports only basic metadata. For full metadata support,
        use Zarr format instead.
        """
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

        np.savez(output_path, **save_dict)

    def _write_zarr(
        self, data: ExternalAerodynamicsZarrDataInMemory, output_path: Path
    ) -> None:
        """Write data in Zarr format with full metadata support.

        Args:
            data: Data to write in Zarr format
            output_path: Path where the .zarr directory should be written
        """
        # Clean up if output path already exists
        if output_path.exists():
            self.logger.warning(f"Overwriting existing data for {output_path}")
            shutil.rmtree(output_path)

        # Create store
        zarr_store = zarr.DirectoryStore(output_path)
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
        """Checks whether the file should be skipped.

        Args:
            filename: Name of the file to check

        Returns:
            True if processing should be skipped, False otherwise
        """
        if self.overwrite_existing:
            return False

        output_path = self._get_output_path(filename)
        if output_path.exists():
            self.logger.info(f"Skipping {filename} - File already exists")
            return True
        return False

    def cleanup_temp_files(self) -> None:
        """Clean up orphaned temporary files from interrupted runs."""
        if not self.output_dir or not self.output_dir.exists():
            return

        # Find all temp files/directories for this serialization method
        if self.serialization_method == "numpy":
            pattern = "*_temp.npz"
        elif self.serialization_method == "zarr":
            pattern = "*_temp.zarr"
        else:
            return

        for temp_file in self.output_dir.glob(pattern):
            self.logger.warning(f"Removing orphaned temp file: {temp_file}")
            if temp_file.is_dir():
                shutil.rmtree(temp_file)
            else:
                temp_file.unlink()
