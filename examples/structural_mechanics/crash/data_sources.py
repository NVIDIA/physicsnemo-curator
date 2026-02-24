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

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pyvista as pv
import zarr
from crash_data_processors import (
    compute_node_thickness,
    find_k_file,
    load_d3plot_data,
    load_d3plot_element_fields,
    parse_k_file,
)
from schemas import CrashExtractedDataInMemory, CrashMetadata
from zarr.storage import LocalStore

from physicsnemo_curator.etl.data_sources import DataSource
from physicsnemo_curator.etl.processing_config import ProcessingConfig


class CrashD3PlotDataSource(DataSource):
    """Data source for reading LS-DYNA d3plot simulation files."""

    def __init__(
        self,
        cfg: ProcessingConfig,
        input_dir: str,
    ):
        super().__init__(cfg)
        self.input_dir = Path(input_dir)
        self.logger = logging.getLogger(__name__)

        if not self.input_dir.exists():
            self.logger.error(f"Input directory does not exist: {self.input_dir}")
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")

    def get_file_list(self) -> List[str]:
        """Find all run folders containing d3plot files."""
        run_folders = []
        for item in self.input_dir.iterdir():
            if item.is_dir():
                d3plot_path = item / "d3plot"
                if d3plot_path.exists():
                    run_folders.append(item.name)

        self.logger.info(f"Found {len(run_folders)} d3plot runs to process")
        return sorted(run_folders)

    def read_file(self, run_id: str) -> CrashExtractedDataInMemory:
        """Read d3plot and .k file for one simulation run.

        Returns CrashExtractedDataInMemory object.
        """
        run_dir = self.input_dir / run_id
        d3plot_path = run_dir / "d3plot"

        self.logger.info(f"Reading {d3plot_path}")

        coords, pos_raw, mesh_connectivity, part_ids, actual_part_ids = (
            load_d3plot_data(str(d3plot_path))
        )

        # Optional element-level fields (stress, effective plastic strain)
        element_shell_stress, element_shell_effective_plastic_strain = (
            load_d3plot_element_fields(str(d3plot_path))
        )

        # Parse .k file for thickness
        k_file_path = find_k_file(run_dir=run_dir)
        node_thickness = np.zeros(len(coords))
        if k_file_path:
            self.logger.info(f"Parsing thickness from {k_file_path}")
            part_thickness_map = parse_k_file(k_file_path=k_file_path)
            node_thickness = compute_node_thickness(
                mesh_connectivity=mesh_connectivity,
                part_ids=part_ids,
                part_thickness_map=part_thickness_map,
                actual_part_ids=actual_part_ids,
            )
        else:
            self.logger.warning(
                f"No .k file found in {run_dir}, defaulting thickness=0"
            )

        return CrashExtractedDataInMemory(
            metadata=CrashMetadata(
                filename=run_id,
            ),
            pos_raw=pos_raw,
            mesh_connectivity=mesh_connectivity,
            node_thickness=node_thickness,
            element_shell_stress=element_shell_stress,
            element_shell_effective_plastic_strain=element_shell_effective_plastic_strain,
        )

    def _get_output_path(self, filename: str) -> Path:
        """Not implemented - this source only reads."""
        raise NotImplementedError("CrashD3PlotDataSource only supports reading")

    def _write_impl_temp_file(self, data: Dict[str, Any], output_path: Path) -> None:
        """Not implemented - this source only reads."""
        raise NotImplementedError("CrashD3PlotDataSource only supports reading")

    def should_skip(self, filename: str) -> bool:
        """Never skip for reading."""
        return False

    def write(self, data: Any, filename: str) -> None:
        """Not implemented - this source only reads."""
        raise NotImplementedError("CrashD3PlotDataSource only supports reading")


class CrashVTPDataSource(DataSource):
    """Data source for writing crash simulation VTP files.

    Outputs one VTP file per run containing all timesteps as displacement fields.
    """

    def __init__(
        self,
        cfg: ProcessingConfig,
        output_dir: str,
        overwrite_existing: bool = True,
        time_step: float = 0.005,
    ):
        """Initialize the VTP data source.

        Args:
            cfg: Processing configuration
            output_dir: Directory to write VTP files
            overwrite_existing: Whether to overwrite existing files
            time_step: Time step between frames (default: 0.005s)
        """
        super().__init__(cfg)
        self.output_dir = Path(output_dir)
        self.overwrite_existing = overwrite_existing
        self.time_step = time_step

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_file_list(self) -> List[str]:
        """Not implemented - this sink only writes."""
        raise NotImplementedError("CrashVTPDataSource only supports writing")

    def read_file(self, filename: str) -> Dict[str, Any]:
        """Not implemented - this sink only writes."""
        raise NotImplementedError("CrashVTPDataSource only supports writing")

    def _get_output_path(self, filename: str) -> Path:
        """Get the output file path for a given run.

        Args:
            filename: Run ID (e.g., "Run100")

        Returns:
            Path to the VTP file (e.g., Run100.vtp)
        """
        return self.output_dir / f"{filename}.vtp"

    def _get_temporary_output_path(self, final_path: Path) -> Path:
        """Get temporary path for a final output path.

        Args:
            final_path: The final destination path

        Returns:
            Temporary path with _temp suffix before the extension
        """
        vtp_path = final_path.with_suffix(".vtp")
        return vtp_path.with_name(f"{vtp_path.name}_temp.vtp")

    def _write_impl_temp_file(
        self, data: CrashExtractedDataInMemory, output_path: Path
    ) -> None:
        """Write a single VTP file containing all timesteps as displacement fields.

        This method receives a TEMPORARY path from the base class (e.g., Run100.vtp_temp).
        After writing completes, the base class atomically renames it to the final path.

        The VTP format expected by the reader:
        - poly.points = reference coordinates (t=0)
        - poly.point_data['displacement_t{time}'] = displacement at each timestep
        - poly.point_data['thickness'] = node thickness values

        Args:
            data: Transformed crash data containing filtered_pos_raw,
                  filtered_mesh_connectivity, filtered_node_thickness, and edges
            output_path: TEMPORARY file path where VTP should be written.
                        Base class will rename this to final path after writing completes.
        """
        self.logger.info(f"Writing VTP file to temporary location: {output_path.name}")

        n_timesteps = data.filtered_pos_raw.shape[0]

        # Use the first timestep as reference coordinates
        reference_coords = data.filtered_pos_raw[0, :, :]

        # Build mesh connectivity for PyVista
        faces = []
        for cell in data.filtered_mesh_connectivity:
            if len(cell) == 3:  # Triangle
                faces.extend([3, *cell])
            elif len(cell) == 4:  # Quad
                faces.extend([4, *cell])
            elif len(cell) > 4:  # Higher order elements - skip
                continue

        faces = np.array(faces)
        mesh = pv.PolyData(reference_coords, faces)

        # Add thickness as point data (must match number of points in mesh)
        n_points = len(reference_coords)
        if len(data.filtered_node_thickness) != n_points:
            self.logger.error(
                f"Thickness array size ({len(data.filtered_node_thickness)}) "
                f"does not match number of points ({n_points})"
            )
            return

        mesh.point_data["thickness"] = data.filtered_node_thickness

        # Add displacement fields for each timestep
        for t in range(n_timesteps):
            # Compute displacement from reference position
            displacement = data.filtered_pos_raw[t, :, :] - reference_coords

            # Format time value with 3 decimal places (e.g., t0.000, t0.005, t0.010)
            time_value = t * self.time_step
            time_str = f"t{time_value:.3f}"
            field_name = f"displacement_{time_str}"

            mesh.point_data[field_name] = displacement

            # Optional: add per-timestep cell scalar fields
            n_cells = int(mesh.n_cells)
            if getattr(data, "filtered_element_effective_plastic_strain", None) is not None:
                eps_elem_t = data.filtered_element_effective_plastic_strain[t, :]
                if eps_elem_t.shape[0] == n_cells:
                    mesh.cell_data[f"cell_effective_plastic_strain_{time_str}"] = eps_elem_t
            if getattr(data, "filtered_element_stress_vm", None) is not None:
                vm_elem_t = data.filtered_element_stress_vm[t, :]
                if vm_elem_t.shape[0] == n_cells:
                    mesh.cell_data[f"cell_stress_vm_{time_str}"] = vm_elem_t

        # Save the single VTP file
        mesh.save(output_path)

        self.logger.info(
            f"Wrote VTP file with {n_timesteps} timesteps for {output_path.stem}"
        )

    def should_skip(self, run_id: str) -> bool:
        """Check if output already exists.

        Args:
            run_id: Run ID to check

        Returns:
            True if processing should be skipped, False otherwise
        """
        if self.overwrite_existing:
            return False

        output_path = self._get_output_path(run_id)
        if output_path.exists():
            self.logger.info(f"Skipping {run_id} - VTP file already exists")
            return True
        return False

    def cleanup_temp_files(self) -> None:
        """Clean up orphaned temporary VTP files from interrupted runs."""
        if not self.output_dir or not self.output_dir.exists():
            return

        # Find all temp VTP files (*._temp.vtp)
        for temp_file in self.output_dir.glob("*_temp.vtp"):
            self.logger.warning(f"Removing orphaned temp VTP file: {temp_file}")
            temp_file.unlink()


class CrashZarrDataSource(DataSource):
    """Data source for writing crash simulation data to Zarr format."""

    def __init__(
        self,
        cfg: ProcessingConfig,
        output_dir: str,
        overwrite_existing: bool = True,
        compression_level: int = 3,
        compression_method: str = "zstd",
        chunk_size_mb: float = 1.0,
    ):
        """Initialize the Zarr data source.

        Args:
            cfg: Processing configuration
            output_dir: Directory to write Zarr stores
            overwrite_existing: Whether to overwrite existing files
            compression_level: Compression level (1-9, higher = more compression)
            compression_method: Compression method
            chunk_size_mb: Target chunk size in MB (default: 1.0)
        """
        super().__init__(cfg)
        self.output_dir = Path(output_dir)
        self.overwrite_existing = overwrite_existing
        self.compression_level = compression_level
        self.compression_method = compression_method
        self.chunk_size_mb = chunk_size_mb

        # Set up compressor
        self.compressor = zarr.codecs.BloscCodec(
            cname=compression_method,
            clevel=compression_level,
            shuffle=zarr.codecs.BloscShuffle.shuffle,
        )

        # Warn if chunk size might be problematic
        if chunk_size_mb < 0.1:
            warnings.warn(
                f"Chunk size of {chunk_size_mb}MB is very small. "
                "This could lead to poor performance due to overhead.",
                UserWarning,
            )
        elif chunk_size_mb > 100.0:
            warnings.warn(
                f"Chunk size of {chunk_size_mb}MB is very large. "
                "This could lead to memory issues and poor random access performance.",
                UserWarning,
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_file_list(self) -> List[str]:
        """Not implemented - this sink only writes."""
        raise NotImplementedError("CrashZarrDataSource only supports writing")

    def read_file(self, filename: str) -> Dict[str, Any]:
        """Not implemented - this sink only writes."""
        raise NotImplementedError("CrashZarrDataSource only supports writing")

    def _calculate_chunks(self, array: np.ndarray) -> tuple:
        """Calculate optimal chunk sizes based on target chunk size in MB.

        Args:
            array: Array to calculate chunks for

        Returns:
            Tuple of chunk dimensions
        """
        target_chunk_size = int(self.chunk_size_mb * 1024 * 1024)  # Convert MB to bytes
        item_size = array.itemsize
        shape = array.shape

        if len(shape) == 1:
            # 1D array: chunk along the single dimension
            chunk_size = min(shape[0], target_chunk_size // item_size)
            return (max(1, chunk_size),)
        elif len(shape) == 2:
            # 2D array: try to keep rows together
            chunk_rows = min(
                shape[0], max(1, target_chunk_size // (item_size * shape[1]))
            )
            return (max(1, chunk_rows), shape[1])
        elif len(shape) == 3:
            # 3D array (e.g., mesh_pos with shape [T, N, 3]):
            # Try to balance between timesteps and nodes
            # Keep the last dimension (3 for coordinates) intact
            elements_per_slice = shape[1] * shape[2]
            chunk_timesteps = max(
                1, min(shape[0], target_chunk_size // (item_size * elements_per_slice))
            )

            # If we can fit multiple timesteps, reduce node chunks
            if chunk_timesteps >= shape[0]:
                # All timesteps fit, chunk along nodes
                chunk_nodes = min(
                    shape[1],
                    max(1, target_chunk_size // (item_size * shape[0] * shape[2])),
                )
                return (shape[0], max(1, chunk_nodes), shape[2])
            else:
                # Chunk along timesteps, keep reasonable node chunks
                remaining_size = target_chunk_size // (
                    item_size * chunk_timesteps * shape[2]
                )
                chunk_nodes = min(shape[1], max(1, remaining_size))
                return (chunk_timesteps, max(1, chunk_nodes), shape[2])
        else:
            # For higher-dimensional arrays, use simple heuristic
            # Chunk the first dimension, keep others intact
            chunk_first = max(
                1, min(shape[0], target_chunk_size // (item_size * np.prod(shape[1:])))
            )
            return (chunk_first,) + shape[1:]

    def _get_output_path(self, filename: str) -> Path:
        """Get the output path for the Zarr store.

        Args:
            filename: Run ID (e.g., "run_001")

        Returns:
            Path to the Zarr store (e.g., run_001.zarr)
        """
        return self.output_dir / f"{filename}.zarr"

    def _write_impl_temp_file(
        self, data: CrashExtractedDataInMemory, output_path: Path
    ) -> None:
        """Write crash data to a Zarr store.

        This method receives a TEMPORARY path from the base class (e.g., run_001.zarr_temp).
        After writing completes, the base class atomically renames it to the final path.

        Args:
            data: Transformed crash data containing:
                - filtered_pos_raw: (timesteps, nodes, 3) temporal positions
                - filtered_mesh_connectivity: list of cell connectivities
                - filtered_node_thickness: (nodes,) thickness values
                - edges: (num_edges, 2) edge connectivity
            output_path: TEMPORARY directory path for the Zarr store.
                        Base class will rename this to final path after writing completes.
        """
        self.logger.info(
            f"Creating Zarr store at temporary location: {output_path.name}"
        )

        # Create Zarr store
        zarr_store = LocalStore(output_path)
        root = zarr.group(store=zarr_store)

        # Write metadata as root attributes
        root.attrs["filename"] = data.metadata.filename
        root.attrs["num_timesteps"] = data.filtered_pos_raw.shape[0]
        root.attrs["num_nodes"] = data.filtered_pos_raw.shape[1]
        root.attrs["num_edges"] = len(data.edges)
        root.attrs["compression"] = self.compression_method
        root.attrs["compression_level"] = self.compression_level
        root.attrs["chunk_size_mb"] = self.chunk_size_mb

        # Convert data to appropriate dtypes
        num_timesteps, num_nodes, _ = data.filtered_pos_raw.shape
        mesh_pos_data = data.filtered_pos_raw.astype(np.float32)
        thickness_data = data.filtered_node_thickness.astype(np.float32)
        edges_array = np.array(list(data.edges), dtype=np.int64)

        # Calculate optimal chunks for each array
        mesh_pos_chunks = self._calculate_chunks(mesh_pos_data)
        thickness_chunks = self._calculate_chunks(thickness_data)
        edges_chunks = self._calculate_chunks(edges_array)

        # Write temporal position data
        root.create_array(
            name="mesh_pos",
            data=mesh_pos_data,
            chunks=mesh_pos_chunks,
            compressors=(self.compressor,),
        )

        # Write node thickness (static per node)
        root.create_array(
            name="thickness",
            data=thickness_data,
            chunks=thickness_chunks,
            compressors=(self.compressor,),
        )

        # Write edges connectivity
        root.create_array(
            name="edges",
            data=edges_array,
            chunks=edges_chunks,
            compressors=(self.compressor,),
        )

        # Optionally write processed element/node fields if available
        def _maybe_write(name: str, arr: np.ndarray) -> None:
            if arr is None:
                return
            arr32 = arr.astype(np.float32)
            chunks = self._calculate_chunks(arr32)
            root.create_array(
                name=name,
                data=arr32,
                chunks=chunks,
                compressors=(self.compressor,),
            )

        _maybe_write("element_stress_voigt", data.filtered_element_stress_voigt)
        _maybe_write("element_stress_vm", data.filtered_element_stress_vm)
        _maybe_write(
            "element_effective_plastic_strain",
            data.filtered_element_effective_plastic_strain,
        )
        # Node-level stress/strain intentionally not written (cell-only per request)

        # Add some statistics as metadata
        root.attrs["thickness_min"] = float(np.min(data.filtered_node_thickness))
        root.attrs["thickness_max"] = float(np.max(data.filtered_node_thickness))
        root.attrs["thickness_mean"] = float(np.mean(data.filtered_node_thickness))

        self.logger.info(
            f"Successfully created Zarr store with {num_timesteps} timesteps, "
            f"{num_nodes} nodes, {len(edges_array)} edges"
        )

    def should_skip(self, run_id: str) -> bool:
        """Check if output already exists.

        Args:
            run_id: Run ID to check

        Returns:
            True if processing should be skipped, False otherwise
        """
        if self.overwrite_existing:
            return False

        output_path = self._get_output_path(run_id)
        if output_path.exists():
            self.logger.info(f"Skipping {run_id} - Zarr store already exists")
            return True
        return False

    def cleanup_temp_files(self) -> None:
        """Clean up orphaned temporary Zarr stores from interrupted runs."""
        if not self.output_dir or not self.output_dir.exists():
            return

        # Find all temp Zarr stores (*.zarr_temp)
        for temp_store in self.output_dir.glob("*.zarr_temp"):
            self.logger.warning(f"Removing orphaned temp Zarr store: {temp_store}")
            import shutil

            shutil.rmtree(temp_store)
