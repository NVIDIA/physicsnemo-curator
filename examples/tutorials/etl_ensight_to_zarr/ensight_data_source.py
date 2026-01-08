# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
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
from typing import Any, Dict, List

import numpy as np
import pyvista as pv
import vtk

from physicsnemo_curator.etl.data_sources import DataSource
from physicsnemo_curator.etl.processing_config import ProcessingConfig


class EnSightDataSource(DataSource):
    """Drop-in replacement for CGNSDataSource to read EnSight Gold .case files."""

    def __init__(self, cfg: ProcessingConfig, input_dir: str):
        super().__init__(cfg)
        self.input_dir = Path(input_dir)
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory {self.input_dir} does not exist")

    def get_file_list(self) -> List[str]:
        """Get list of .case files to process."""
        case_files = list(self.input_dir.glob("*.case"))
        filenames = [f.stem for f in case_files]
        self.logger.info(f"Found {len(filenames)} EnSight files to process")
        return sorted(filenames)

    def _read_ensight(self, filepath: Path) -> pv.MultiBlock:
        """Read an EnSight Gold file and return a MultiBlock dataset."""
        try:
            reader = vtk.vtkEnSightGoldBinaryReader()
            reader.SetCaseFileName(str(filepath))
            reader.ReadAllVariablesOn()
            reader.Update()
            return pv.wrap(reader.GetOutput())
        except Exception as e:
            self.logger.error(f"Error reading EnSight file {filepath}: {e}")
            return None

    def _flatten_multiblock(self, dataset: pv.MultiBlock) -> List[pv.DataSet]:
        """Recursively flatten MultiBlock into a list of non-empty blocks."""
        blocks = []
        for block in dataset:
            if isinstance(block, pv.MultiBlock):
                blocks.extend(self._flatten_multiblock(block))
            elif block and block.n_points > 0 and block.n_cells > 0:
                blocks.append(block)
        return blocks

    def _extract_surface_mesh(self, dataset: pv.MultiBlock) -> pv.PolyData:
        """Extract a clean surface mesh handling inconsistent variables across blocks."""
        blocks = self._flatten_multiblock(dataset)

        if not blocks:
            raise RuntimeError("No valid blocks found in EnSight dataset")

        # Separate blocks into full vs partial (having >1 point data field)
        blocks_full, blocks_partial = [], []
        for block in blocks:
            if len(block.point_data.keys()) > 1:
                blocks_full.append(block)
            else:
                blocks_partial.append(block)

        surface_full = pv.merge([b.extract_surface() for b in blocks_full]) if blocks_full else None
        surface_partial = pv.merge([b.extract_surface() for b in blocks_partial]) if blocks_partial else None

        if surface_full is None and surface_partial is None:
            raise RuntimeError("No valid surfaces could be extracted")

        if surface_full is None:
            return surface_partial.triangulate()
        if surface_partial is None:
            return surface_full.triangulate()

        # Fill missing variables in partial blocks
        full_arrays = surface_full.point_data
        partial_arrays = surface_partial.point_data
        missing = set(full_arrays.keys()) - set(partial_arrays.keys())
        for name in missing:
            src = full_arrays[name]
            placeholder = np.zeros((surface_partial.n_points, src.shape[1])) if src.ndim > 1 else np.zeros(surface_partial.n_points)
            surface_partial.point_data[name] = placeholder
            self.logger.debug(f"Added placeholder for missing variable '{name}'")

        # Merge surfaces
        final_mesh = surface_full.merge(surface_partial)
        return final_mesh.triangulate()

    def _read_mesh(self, filename: str) -> pv.PolyData:
        """Read a single EnSight file and return the triangulated surface mesh."""
        filepath = self.input_dir / f"{filename}.case"
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        self.logger.warning(f"Reading {filepath}")
        dataset = self._read_ensight(filepath)
        if dataset is None:
            raise ValueError(f"Failed to read EnSight file: {filepath}")

        surface_mesh = self._extract_surface_mesh(dataset)

        # Convert cell data to point data if present
        if surface_mesh.cell_data:
            surface_mesh = surface_mesh.cell_data_to_point_data()

        return surface_mesh

    def read_file(self, filename: str) -> Dict[str, Any]:
        """Read one EnSight file and extract all data in the same format as CGNSDataSource."""
        surface_mesh = self._read_mesh(filename)

        data = {
            "coordinates": np.array(surface_mesh.points),
            "faces": np.array(surface_mesh.faces).reshape(-1, 4)[:, 1:],  # triangulated
            "metadata": {
                "n_points": surface_mesh.n_points,
                "n_cells": surface_mesh.n_cells,
                "bounds": surface_mesh.bounds,
            },
            "filename": filename
        }

        for field_name in surface_mesh.point_data.keys():
            data[field_name] = np.array(surface_mesh.point_data[field_name])
            self.logger.info(f"Extracted point data field: {field_name}")

        self.logger.warning(f"Loaded data with {surface_mesh.n_points} points and {surface_mesh.n_cells} cells")
        return data

    def _get_output_path(self, filename: str) -> Path:
        raise NotImplementedError("EnSightDataSource only supports reading")

    def _write_impl_temp_file(self, data: Dict[str, Any], output_path: Path) -> None:
        raise NotImplementedError("EnSightDataSource only supports reading")

    def should_skip(self, filename: str) -> bool:
        return False
