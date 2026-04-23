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
from schemas import DropTestExtractedDataInMemory, DropTestMetadata
from zarr.storage import LocalStore

from drop_test_data_processors import von_mises_from_voigt
from physicsnemo_curator.etl.data_sources import DataSource
from physicsnemo_curator.etl.processing_config import ProcessingConfig


def _tet_signed_volume(pts: np.ndarray, v0: int, v1: int, v2: int, v3: int) -> float:
    """Signed volume of tet (v0,v1,v2,v3). Positive = right-handed (VTK convention)."""
    a = pts[v1] - pts[v0]
    b = pts[v2] - pts[v0]
    c = pts[v3] - pts[v0]
    return float(np.dot(np.cross(a, b), c))


def _ensure_tet_orientation(cell: List[int], pts: np.ndarray) -> List[int]:
    """Ensure tet has positive volume (VTK convention). Swap v2,v3 if inverted."""
    vol = _tet_signed_volume(pts, cell[0], cell[1], cell[2], cell[3])
    if vol < 0:
        return [cell[0], cell[1], cell[3], cell[2]]
    return list(cell)


class DropTestVTKDataSource(DataSource):
    """Data source for reading VTK files from OpenRadioss anim_to_vtk."""

    def __init__(
        self,
        cfg: ProcessingConfig,
        input_dir: str,
        vtk_glob: str = "Cell_Phone_DropA*.vtk",
    ):
        super().__init__(cfg)
        self.input_dir = Path(input_dir)
        self.vtk_glob = vtk_glob
        self.logger = logging.getLogger(__name__)
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")

    def get_file_list(self) -> List[str]:
        run_folders = []
        for item in self.input_dir.iterdir():
            if item.is_dir() and item.name.startswith("run"):
                vtks = list(item.glob(self.vtk_glob))
                if vtks:
                    run_folders.append(item.name)
        self.logger.info(f"Found {len(run_folders)} VTK runs to process")
        return sorted(run_folders)

    def read_file(self, run_id: str) -> DropTestExtractedDataInMemory:
        run_dir = self.input_dir / run_id
        vtk_files = sorted(run_dir.glob(self.vtk_glob))
        if not vtk_files:
            raise FileNotFoundError(f"No VTK files matching {self.vtk_glob} in {run_dir}")

        self.logger.info(f"Reading {len(vtk_files)} VTK files from {run_dir}")

        pos_list = []
        vel_list = []
        acc_list = []
        temp_list = []
        resf_list = []
        stress_list = []
        epsp_list = []
        mesh_connectivity = None

        # Map VTK point_data/cell_data names to our expected fields (case-insensitive)
        def _find_key(data_dict, candidates):
            if data_dict is None:
                return None
            keys_lower = {k.lower(): k for k in data_dict.keys()}
            for c in candidates:
                if c in keys_lower:
                    return keys_lower[c]
            return None

        for vp in vtk_files:
            mesh = pv.read(str(vp))
            pts = np.array(mesh.points, dtype=np.float64)
            pos_list.append(pts)

            # Point data (anim_to_vtk: Acceleration, Displacement, Velocity)
            pd = mesh.point_data
            vk = _find_key(pd, ["velocity", "vel", "node_velocity"])
            if vk is not None:
                vel_list.append(np.asarray(pd[vk], dtype=np.float64))
            ak = _find_key(pd, ["acceleration", "acc", "node_acceleration"])
            if ak is not None:
                acc_list.append(np.asarray(pd[ak], dtype=np.float64))
            tk = _find_key(pd, ["temperature", "temp", "node_temperature"])
            if tk is not None:
                arr = np.asarray(pd[tk], dtype=np.float64)
                if arr.ndim == 1:
                    arr = arr[:, np.newaxis]
                temp_list.append(arr)
            rk = _find_key(pd, ["residual_forces", "residual", "node_residual_forces"])
            if rk is not None:
                resf_list.append(np.asarray(pd[rk], dtype=np.float64))

            # Stress: GPS_SIGXX, etc. are point data in anim_to_vtk; keep as point data
            cd = mesh.cell_data
            if pd is not None:
                comps = [
                    ("gps_sigxx",), ("gps_sigyy",), ("gps_sigzz",),
                    ("gps_sigxy",), ("gps_sigyz", "gps_sigzy"), ("gps_sigxz",),
                ]
                keys_lower = {k.lower(): k for k in pd.keys()}
                found = [next((keys_lower[v] for v in c if v in keys_lower), None) for c in comps]
                if all(f is not None for f in found):
                    parts = [np.asarray(pd[f], dtype=np.float64).flatten() for f in found]
                    stress_list.append(np.stack(parts, axis=1))  # (N, 6)
                elif vp == vtk_files[0]:
                    self.logger.warning(
                        f"Stress not found in point_data: {list(pd.keys())}. "
                        f"Need GPS_SIGXX, GPS_SIGYY, GPS_SIGZZ, GPS_SIGXY, GPS_SIGYZ, GPS_SIGXZ."
                    )
            ek = _find_key(cd, ["effective_plastic_strain", "plastic_strain", "epsp"])
            if ek is not None:
                epsp_list.append(np.asarray(cd[ek], dtype=np.float64))

            if mesh_connectivity is None:
                cells = mesh.cells
                cell_list = []
                i = 0
                while i < len(cells):
                    n = int(cells[i])
                    ids = cells[i + 1 : i + 1 + n].tolist()
                    if len(ids) >= 3:
                        cell_list.append(ids)
                    i += 1 + n
                mesh_connectivity = cell_list

        pos_raw = np.stack(pos_list, axis=0)
        node_velocity = np.stack(vel_list, axis=0) if len(vel_list) == len(pos_list) else None
        node_acceleration = np.stack(acc_list, axis=0) if len(acc_list) == len(pos_list) else None
        node_temperature = (
            np.stack(temp_list, axis=0).squeeze()
            if len(temp_list) == len(pos_list)
            else None
        )
        node_residual_forces = (
            np.stack(resf_list, axis=0) if len(resf_list) == len(pos_list) else None
        )

        # Element fields (cell data from VTK)
        element_solid_effective_plastic_strain = None
        if len(epsp_list) == len(pos_list) and epsp_list:
            stacked = np.stack(epsp_list, axis=0)
            if stacked.ndim == 2:
                element_solid_effective_plastic_strain = stacked[:, :, np.newaxis]
            elif stacked.ndim == 3:
                element_solid_effective_plastic_strain = stacked

        # Node stress (point data from VTK: GPS_SIGXX, etc.)
        node_stress_voigt = None
        node_stress_vm = None
        if len(stress_list) == len(pos_list) and stress_list:
            stacked = np.stack(stress_list, axis=0)  # (T, N, 6)
            node_stress_voigt = stacked
            node_stress_vm = von_mises_from_voigt(stacked)

        return DropTestExtractedDataInMemory(
            metadata=DropTestMetadata(filename=run_id),
            pos_raw=pos_raw,
            node_velocity=node_velocity,
            node_acceleration=node_acceleration,
            node_temperature=node_temperature,
            node_residual_forces=node_residual_forces,
            node_stress_voigt=node_stress_voigt,
            node_stress_vm=node_stress_vm,
            mesh_connectivity=mesh_connectivity,
            element_solid_stress=None,
            element_solid_stress_vm=None,
            element_solid_effective_plastic_strain=element_solid_effective_plastic_strain,
            element_solid_strain=None,
            element_solid_plastic_strain_tensor=None,
        )

    def _get_output_path(self, filename: str) -> Path:
        raise NotImplementedError("DropTestVTKDataSource only supports reading")

    def _write_impl_temp_file(self, data: Dict[str, Any], output_path: Path) -> None:
        raise NotImplementedError("DropTestVTKDataSource only supports reading")

    def should_skip(self, filename: str) -> bool:
        return False

    def write(self, data: Any, filename: str) -> None:
        raise NotImplementedError("DropTestVTKDataSource only supports reading")


class DropTestVTUDataSource(DataSource):
    """Data source for writing drop test (solid) simulation VTU files."""

    def __init__(
        self,
        cfg: ProcessingConfig,
        output_dir: str,
        overwrite_existing: bool = True,
        time_step: float = 0.001,
        flip_triangle_normals: bool = True,
    ):
        super().__init__(cfg)
        self.output_dir = Path(output_dir)
        self.overwrite_existing = overwrite_existing
        self.time_step = time_step
        self.flip_triangle_normals = flip_triangle_normals
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_file_list(self) -> List[str]:
        raise NotImplementedError("DropTestVTUDataSource only supports writing")

    def read_file(self, filename: str) -> Dict[str, Any]:
        raise NotImplementedError("DropTestVTUDataSource only supports writing")

    def _get_output_path(self, filename: str) -> Path:
        return self.output_dir / f"{filename}.vtu"

    def _get_temporary_output_path(self, final_path: Path) -> Path:
        vtu_path = final_path.with_suffix(".vtu")
        return vtu_path.with_name(f"{vtu_path.name}_temp.vtu")

    def _write_impl_temp_file(
        self, data: DropTestExtractedDataInMemory, output_path: Path
    ) -> None:
        """Write VTU file with all available nodal and element fields."""
        self.logger.info(f"Writing VTU file to temporary location: {output_path.name}")

        n_timesteps = data.filtered_pos_raw.shape[0]
        reference_coords = data.filtered_pos_raw[0, :, :]

        # Build UnstructuredGrid for solid elements (hex=8, tet=4, tri=3, wedge=6, pyramid=5)
        cells = []
        cell_types = []
        added_cell_indices = []  # track for element data alignment
        for e_idx, cell in enumerate(data.filtered_mesh_connectivity):
            n = len(cell)
            if n == 3:
                # Optionally reverse vertex order to fix normals (LS-DYNA vs VTK convention)
                tri = [cell[0], cell[2], cell[1]] if self.flip_triangle_normals else cell
                cells.extend([3, *tri])
                cell_types.append(pv.CellType.TRIANGLE)
                added_cell_indices.append(e_idx)
            elif n == 4:
                # Fix tet orientation: VTK expects positive volume (right-hand rule)
                tet = _ensure_tet_orientation(cell, reference_coords)
                cells.extend([4, *tet])
                cell_types.append(pv.CellType.TETRA)
                added_cell_indices.append(e_idx)
            elif n == 5:
                cells.extend([5, *cell])
                cell_types.append(pv.CellType.PYRAMID)
                added_cell_indices.append(e_idx)
            elif n == 6:
                cells.extend([6, *cell])
                cell_types.append(pv.CellType.WEDGE)
                added_cell_indices.append(e_idx)
            elif n == 8:
                cells.extend([8, *cell])
                cell_types.append(pv.CellType.HEXAHEDRON)
                added_cell_indices.append(e_idx)
            # Skip 7-node cells (no standard VTK type; would need tet decomposition)

        cells = np.array(cells)
        added_cell_indices = np.array(added_cell_indices)
        cell_types = np.array(cell_types)
        mesh = pv.UnstructuredGrid(cells, cell_types, reference_coords)

        n_points = len(reference_coords)
        mesh.point_data["thickness"] = data.filtered_node_thickness

        for t in range(n_timesteps):
            displacement = data.filtered_pos_raw[t, :, :] - reference_coords
            time_str = f"t{t:04d}"  # Unique per timestep (avoids collision with small time_step)
            mesh.point_data[f"displacement_{time_str}"] = displacement

            if getattr(data, "filtered_node_velocity", None) is not None:
                mesh.point_data[f"velocity_{time_str}"] = data.filtered_node_velocity[
                    t, :, :
                ]
            if getattr(data, "filtered_node_acceleration", None) is not None:
                mesh.point_data[f"acceleration_{time_str}"] = (
                    data.filtered_node_acceleration[t, :, :]
                )
            if getattr(data, "filtered_node_temperature", None) is not None:
                mesh.point_data[f"temperature_{time_str}"] = (
                    data.filtered_node_temperature[t, :]
                )
            if getattr(data, "filtered_node_residual_forces", None) is not None:
                mesh.point_data[f"residual_forces_{time_str}"] = (
                    data.filtered_node_residual_forces[t, :, :]
                )

            n_cells = int(mesh.n_cells)
            if getattr(data, "filtered_element_effective_plastic_strain", None) is not None:
                eps_t = data.filtered_element_effective_plastic_strain[t, :]
                if len(added_cell_indices) > 0 and eps_t.shape[0] > max(added_cell_indices):
                    mesh.cell_data[f"cell_effective_plastic_strain_{time_str}"] = eps_t[
                        added_cell_indices
                    ]
            if getattr(data, "filtered_node_stress_vm", None) is not None:
                mesh.point_data[f"Von_Mises_{time_str}"] = data.filtered_node_stress_vm[
                    t, :
                ]
            if getattr(data, "filtered_node_stress_voigt", None) is not None:
                mesh.point_data[f"stress_voigt_{time_str}"] = (
                    data.filtered_node_stress_voigt[t, :, :]
                )

        mesh.save(output_path)
        self.logger.info(
            f"Wrote VTU file with {n_timesteps} timesteps for {output_path.stem}"
        )

    def should_skip(self, run_id: str) -> bool:
        if self.overwrite_existing:
            return False
        if self._get_output_path(run_id).exists():
            self.logger.info(f"Skipping {run_id} - VTU file already exists")
            return True
        return False

    def cleanup_temp_files(self) -> None:
        if not self.output_dir or not self.output_dir.exists():
            return
        for temp_file in self.output_dir.glob("*_temp.vtu"):
            self.logger.warning(f"Removing orphaned temp VTU file: {temp_file}")
            temp_file.unlink()


class DropTestZarrDataSource(DataSource):
    """Data source for writing drop test (solid) simulation data to Zarr format."""

    def __init__(
        self,
        cfg: ProcessingConfig,
        output_dir: str,
        overwrite_existing: bool = True,
        compression_level: int = 3,
        compression_method: str = "zstd",
        chunk_size_mb: float = 1.0,
    ):
        super().__init__(cfg)
        self.output_dir = Path(output_dir)
        self.overwrite_existing = overwrite_existing
        self.compression_level = compression_level
        self.compression_method = compression_method
        self.chunk_size_mb = chunk_size_mb
        self.compressor = zarr.codecs.BloscCodec(
            cname=compression_method,
            clevel=compression_level,
            shuffle=zarr.codecs.BloscShuffle.shuffle,
        )
        if chunk_size_mb < 0.1:
            warnings.warn(
                f"Chunk size of {chunk_size_mb}MB is very small.",
                UserWarning,
            )
        elif chunk_size_mb > 100.0:
            warnings.warn(
                f"Chunk size of {chunk_size_mb}MB is very large.",
                UserWarning,
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_file_list(self) -> List[str]:
        raise NotImplementedError("DropTestZarrDataSource only supports writing")

    def read_file(self, filename: str) -> Dict[str, Any]:
        raise NotImplementedError("DropTestZarrDataSource only supports writing")

    def _calculate_chunks(self, array: np.ndarray) -> tuple:
        target_chunk_size = int(self.chunk_size_mb * 1024 * 1024)
        item_size = array.itemsize
        shape = array.shape
        if len(shape) == 1:
            chunk_size = min(shape[0], target_chunk_size // item_size)
            return (max(1, chunk_size),)
        elif len(shape) == 2:
            chunk_rows = min(
                shape[0], max(1, target_chunk_size // (item_size * shape[1]))
            )
            return (max(1, chunk_rows), shape[1])
        elif len(shape) == 3:
            elements_per_slice = shape[1] * shape[2]
            chunk_timesteps = max(
                1, min(shape[0], target_chunk_size // (item_size * elements_per_slice))
            )
            if chunk_timesteps >= shape[0]:
                chunk_nodes = min(
                    shape[1],
                    max(1, target_chunk_size // (item_size * shape[0] * shape[2])),
                )
                return (shape[0], max(1, chunk_nodes), shape[2])
            else:
                remaining_size = target_chunk_size // (
                    item_size * chunk_timesteps * shape[2]
                )
                chunk_nodes = min(shape[1], max(1, remaining_size))
                return (chunk_timesteps, max(1, chunk_nodes), shape[2])
        else:
            chunk_first = max(
                1, min(shape[0], target_chunk_size // (item_size * np.prod(shape[1:])))
            )
            return (chunk_first,) + shape[1:]

    def _get_output_path(self, filename: str) -> Path:
        return self.output_dir / f"{filename}.zarr"

    def _write_impl_temp_file(
        self, data: DropTestExtractedDataInMemory, output_path: Path
    ) -> None:
        self.logger.info(
            f"Creating Zarr store at temporary location: {output_path.name}"
        )

        zarr_store = LocalStore(output_path)
        root = zarr.group(store=zarr_store)

        root.attrs["filename"] = data.metadata.filename
        root.attrs["num_timesteps"] = data.filtered_pos_raw.shape[0]
        root.attrs["num_nodes"] = data.filtered_pos_raw.shape[1]
        root.attrs["num_edges"] = len(data.edges)

        num_timesteps, num_nodes, _ = data.filtered_pos_raw.shape
        mesh_pos_data = data.filtered_pos_raw.astype(np.float32)
        thickness_data = data.filtered_node_thickness.astype(np.float32)
        edges_array = np.array(list(data.edges), dtype=np.int64)

        def _write_array(name: str, arr: np.ndarray) -> None:
            if arr is None:
                return
            arr32 = arr.astype(np.float32) if np.issubdtype(arr.dtype, np.floating) else arr
            chunks = self._calculate_chunks(arr32)
            root.create_array(
                name=name,
                data=arr32,
                chunks=chunks,
                compressors=(self.compressor,),
            )

        root.create_array(
            name="mesh_pos",
            data=mesh_pos_data,
            chunks=self._calculate_chunks(mesh_pos_data),
            compressors=(self.compressor,),
        )
        root.create_array(
            name="thickness",
            data=thickness_data,
            chunks=self._calculate_chunks(thickness_data),
            compressors=(self.compressor,),
        )
        root.create_array(
            name="edges",
            data=edges_array,
            chunks=self._calculate_chunks(edges_array),
            compressors=(self.compressor,),
        )

        _write_array("node_velocity", data.filtered_node_velocity)
        _write_array("node_acceleration", data.filtered_node_acceleration)
        _write_array("node_temperature", data.filtered_node_temperature)
        _write_array("node_residual_forces", data.filtered_node_residual_forces)
        _write_array("node_stress_voigt", data.filtered_node_stress_voigt)
        _write_array("node_stress_vm", data.filtered_node_stress_vm)
        _write_array(
            "element_effective_plastic_strain",
            data.filtered_element_effective_plastic_strain,
        )
        _write_array("element_strain_voigt", data.filtered_element_strain_voigt)
        _write_array(
            "element_plastic_strain_voigt",
            data.filtered_element_plastic_strain_voigt,
        )

        root.attrs["thickness_min"] = float(np.min(data.filtered_node_thickness))
        root.attrs["thickness_max"] = float(np.max(data.filtered_node_thickness))
        root.attrs["thickness_mean"] = float(np.mean(data.filtered_node_thickness))

        self.logger.info(
            f"Successfully created Zarr store with {num_timesteps} timesteps, "
            f"{num_nodes} nodes, {len(edges_array)} edges"
        )

    def should_skip(self, run_id: str) -> bool:
        if self.overwrite_existing:
            return False
        if self._get_output_path(run_id).exists():
            self.logger.info(f"Skipping {run_id} - Zarr store already exists")
            return True
        return False

    def cleanup_temp_files(self) -> None:
        if not self.output_dir or not self.output_dir.exists():
            return
        import shutil

        for temp_store in self.output_dir.glob("*.zarr_temp"):
            self.logger.warning(f"Removing orphaned temp Zarr store: {temp_store}")
            shutil.rmtree(temp_store)
