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
from typing import Callable, Optional

import numpy as np
from drop_test_data_processors import (
    build_edges_from_mesh_connectivity,
    compute_element_fields,
    compute_node_type,
)
from schemas import DropTestExtractedDataInMemory

from physicsnemo_curator.etl.data_transformations import DataTransformation
from physicsnemo_curator.etl.processing_config import ProcessingConfig


class DropTestDataTransformation(DataTransformation):
    """Transform drop test (solid) simulation data: filter nodes, build edges."""

    def __init__(
        self,
        cfg: ProcessingConfig,
        wall_threshold: float = 1.0,
        drop_test_processors: Optional[tuple[Callable, ...]] = None,
    ):
        super().__init__(cfg)
        self.logger = logging.getLogger(__name__)
        self.wall_threshold = wall_threshold
        self.drop_test_processors = drop_test_processors

    def transform(
        self,
        data: DropTestExtractedDataInMemory,
    ) -> DropTestExtractedDataInMemory:
        """Transform raw simulation data for solid elements.

        Steps:
        1. Identify wall nodes (low displacement)
        2. Filter arrays (positions, velocity, acceleration, etc.)
        3. Remap mesh connectivity
        4. Compact to contiguous indices
        5. Filter element fields to kept elements
        6. Build edges
        """
        self.logger.info(f"Transforming {data.metadata.filename}")

        node_type = compute_node_type(data.pos_raw, threshold=self.wall_threshold)
        keep_nodes = sorted(np.where(node_type == 0)[0])
        node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_nodes)}

        n_total = data.pos_raw.shape[1]
        n_wall = n_total - len(keep_nodes)
        n_kept = len(keep_nodes)
        self.logger.info(
            f"Filtered {n_wall} wall nodes; kept {n_kept} structure nodes"
        )

        filtered_pos_raw = data.pos_raw[:, keep_nodes, :]
        filtered_node_thickness = np.zeros(len(keep_nodes), dtype=np.float32)

        def _filter_nodal(arr, axis_node=1):
            if arr is None:
                return None
            return np.take(arr, keep_nodes, axis=axis_node)

        filtered_node_velocity = _filter_nodal(data.node_velocity)
        filtered_node_acceleration = _filter_nodal(data.node_acceleration)
        filtered_node_temperature = (
            _filter_nodal(data.node_temperature)
            if data.node_temperature is not None
            else None
        )
        filtered_node_residual_forces = (
            _filter_nodal(data.node_residual_forces)
            if data.node_residual_forces is not None
            else None
        )
        filtered_node_stress_voigt = (
            _filter_nodal(data.node_stress_voigt)
            if getattr(data, "node_stress_voigt", None) is not None
            else None
        )
        filtered_node_stress_vm = (
            _filter_nodal(data.node_stress_vm)
            if getattr(data, "node_stress_vm", None) is not None
            else None
        )

        filtered_mesh_connectivity = []
        kept_element_indices = []
        for e_idx, cell in enumerate(data.mesh_connectivity):
            filtered_cell = [node_map[n] for n in cell if n in node_map]
            if len(filtered_cell) >= 3:
                filtered_mesh_connectivity.append(filtered_cell)
                kept_element_indices.append(e_idx)

        used = np.unique(
            np.array([i for cell in filtered_mesh_connectivity for i in cell])
        )
        if used.size == 0:
            raise ValueError("No cells left after filtering")

        def _filter_by_keep(arr, keep_list):
            if arr is None:
                return None
            if arr.ndim == 2:
                return arr[:, keep_list]
            if arr.ndim == 3:
                return arr[:, keep_list, :]
            return arr[:, keep_list]

        num_kept = filtered_pos_raw.shape[1]
        if (used.min() != 0) or (used.max() != num_kept - 1) or (used.size != num_kept):
            keep2 = used.tolist()
            remap2 = {old_idx: new_idx for new_idx, old_idx in enumerate(keep2)}
            filtered_pos_raw = filtered_pos_raw[:, keep2, :]
            filtered_node_thickness = filtered_node_thickness[keep2]
            filtered_node_velocity = _filter_by_keep(filtered_node_velocity, keep2)
            filtered_node_acceleration = _filter_by_keep(
                filtered_node_acceleration, keep2
            )
            filtered_node_temperature = _filter_by_keep(
                filtered_node_temperature, keep2
            )
            filtered_node_residual_forces = _filter_by_keep(
                filtered_node_residual_forces, keep2
            )
            filtered_node_stress_voigt = _filter_by_keep(
                filtered_node_stress_voigt, keep2
            )
            filtered_node_stress_vm = _filter_by_keep(
                filtered_node_stress_vm, keep2
            )
            filtered_mesh_connectivity = [
                [remap2[n] for n in cell] for cell in filtered_mesh_connectivity
            ]
            num_kept = filtered_pos_raw.shape[1]

        filtered_element_stress_voigt = None
        filtered_element_stress_vm = None
        filtered_element_effective_plastic_strain = None
        filtered_element_strain_voigt = None
        filtered_element_plastic_strain_voigt = None

        has_epsp = data.element_solid_effective_plastic_strain is not None
        has_stress = data.element_solid_stress is not None
        has_stress_vm = getattr(data, "element_solid_stress_vm", None) is not None
        has_strain = data.element_solid_strain is not None
        has_plastic_strain = data.element_solid_plastic_strain_tensor is not None

        if has_epsp or has_stress or has_stress_vm or has_strain or has_plastic_strain:
            epsp_kept = (
                data.element_solid_effective_plastic_strain[
                    :, kept_element_indices, :
                ]
                if has_epsp
                else None
            )
            stress_kept = (
                data.element_solid_stress[:, kept_element_indices, :, :]
                if has_stress
                else None
            )
            strain_kept = (
                data.element_solid_strain[:, kept_element_indices, :, :]
                if has_strain
                else None
            )
            plastic_strain_kept = (
                data.element_solid_plastic_strain_tensor[
                    :, kept_element_indices, :, :
                ]
                if has_plastic_strain
                else None
            )
            fields = compute_element_fields(
                element_solid_stress=stress_kept,
                element_solid_effective_plastic_strain=epsp_kept,
                element_solid_strain=strain_kept,
                element_solid_plastic_strain_tensor=plastic_strain_kept,
                compute_von_mises=True,
            )
            filtered_element_effective_plastic_strain = fields["element"][
                "effective_plastic_strain"
            ]
            filtered_element_stress_voigt = fields["element"]["stress_voigt"]
            filtered_element_stress_vm = fields["element"]["stress_vm"]
            if filtered_element_stress_vm is None and has_stress_vm:
                vm_raw = data.element_solid_stress_vm[:, kept_element_indices]
                filtered_element_stress_vm = vm_raw
            filtered_element_strain_voigt = fields["element"]["strain_voigt"]
            filtered_element_plastic_strain_voigt = fields["element"][
                "plastic_strain_voigt"
            ]

        edges = build_edges_from_mesh_connectivity(filtered_mesh_connectivity)

        self.logger.info(
            f"Processed: {filtered_pos_raw.shape[1]} nodes, "
            f"{len(filtered_mesh_connectivity)} cells, {len(edges)} edges"
        )

        data.pos_raw = None
        data.mesh_connectivity = None
        data.node_velocity = None
        data.node_acceleration = None
        data.node_temperature = None
        data.node_residual_forces = None
        data.node_stress_voigt = None
        data.node_stress_vm = None
        data.element_solid_stress = None
        data.element_solid_effective_plastic_strain = None
        data.element_solid_strain = None
        data.element_solid_plastic_strain_tensor = None

        return DropTestExtractedDataInMemory(
            metadata=data.metadata,
            filtered_pos_raw=filtered_pos_raw,
            filtered_node_velocity=filtered_node_velocity,
            filtered_node_acceleration=filtered_node_acceleration,
            filtered_node_temperature=filtered_node_temperature,
            filtered_node_residual_forces=filtered_node_residual_forces,
            filtered_node_stress_voigt=filtered_node_stress_voigt,
            filtered_node_stress_vm=filtered_node_stress_vm,
            filtered_mesh_connectivity=filtered_mesh_connectivity,
            filtered_node_thickness=filtered_node_thickness,
            edges=edges,
            filtered_element_stress_voigt=filtered_element_stress_voigt,
            filtered_element_stress_vm=filtered_element_stress_vm,
            filtered_element_effective_plastic_strain=filtered_element_effective_plastic_strain,
            filtered_element_strain_voigt=filtered_element_strain_voigt,
            filtered_element_plastic_strain_voigt=filtered_element_plastic_strain_voigt,
        )
