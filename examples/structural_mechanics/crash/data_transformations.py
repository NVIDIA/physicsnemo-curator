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
from crash_data_processors import (
    build_edges_from_mesh_connectivity,
    compute_element_and_node_fields,
    compute_node_type,
)
from schemas import CrashExtractedDataInMemory

from physicsnemo_curator.etl.data_transformations import DataTransformation
from physicsnemo_curator.etl.processing_config import ProcessingConfig


class CrashDataTransformation(DataTransformation):
    """Transform crash simulation data: filter nodes, compute thickness, build edges."""

    def __init__(
        self,
        cfg: ProcessingConfig,
        wall_threshold: float = 1.0,
        crash_processors: Optional[tuple[Callable, ...]] = None,
    ):
        super().__init__(cfg)
        self.logger = logging.getLogger(__name__)
        self.wall_threshold = wall_threshold
        self.crash_processors = crash_processors

    def transform(
        self,
        data: CrashExtractedDataInMemory,
    ) -> CrashExtractedDataInMemory:
        """Transform raw d3plot data into VTP format.

        Steps:
        1. Identify wall nodes (low displacement)
        2. Filter arrays
        3. Remap mesh connectivity
        4. Compact to only nodes that are actually used by any cell
        5. If not contiguous 0..(num_kept-1), compact and reindex everything
        6. Build edges

        Returns dict with:
        - filtered_pos_raw: (timesteps, filtered_nodes, 3)
        - filtered_mesh_connectivity: remapped connectivity
        - node_thickness: per-node thickness values
        - edges: edge connectivity
        """

        self.logger.info(f"Transforming {data.metadata.filename}")

        # Step 1: Identify wall nodes
        node_type = compute_node_type(data.pos_raw, threshold=self.wall_threshold)
        keep_nodes = sorted(np.where(node_type == 0)[0])  # keep structure
        node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_nodes)}

        self.logger.info(
            f"Filtered {data.pos_raw.shape[1] - len(keep_nodes)} wall nodes; "
            f"kept {len(keep_nodes)} structure nodes"
        )

        # Step 2: Filter arrays
        filtered_pos_raw = data.pos_raw[:, keep_nodes, :]
        filtered_node_thickness = data.node_thickness[keep_nodes]

        # Step 3: Remap mesh connectivity
        filtered_mesh_connectivity = []
        kept_element_indices = []
        for e_idx, cell in enumerate(data.mesh_connectivity):
            filtered_cell = [node_map[n] for n in cell if n in node_map]
            if len(filtered_cell) >= 3:
                filtered_mesh_connectivity.append(filtered_cell)
                kept_element_indices.append(e_idx)

        # Step 4: Compact to contiguous indices (reuse your logic)
        used = np.unique(
            np.array([i for cell in filtered_mesh_connectivity for i in cell])
        )
        if used.size == 0:
            raise ValueError("No cells left after filtering")

        # Step 5: If not contiguous 0..(num_kept-1), compact and reindex everything
        num_kept = filtered_pos_raw.shape[1]
        if (used.min() != 0) or (used.max() != num_kept - 1) or (used.size != num_kept):
            keep2 = used.tolist()
            remap2 = {old_idx: new_idx for new_idx, old_idx in enumerate(keep2)}
            filtered_pos_raw = filtered_pos_raw[:, keep2, :]
            filtered_node_thickness = filtered_node_thickness[keep2]
            filtered_mesh_connectivity = [
                [remap2[n] for n in cell] for cell in filtered_mesh_connectivity
            ]
            num_kept = filtered_pos_raw.shape[1]

        # Optional: filter element fields to kept elements
        filtered_element_stress_voigt = None
        filtered_element_stress_vm = None
        filtered_element_effective_plastic_strain = None

        has_epsp = data.element_shell_effective_plastic_strain is not None
        has_stress = data.element_shell_stress is not None
        if has_epsp or has_stress:
            epsp_kept = (
                data.element_shell_effective_plastic_strain[:, kept_element_indices, :]
                if has_epsp
                else None
            )
            stress_kept = (
                data.element_shell_stress[:, kept_element_indices, :, :]
                if has_stress
                else None
            )
            # Compute reduced element/node fields on filtered mesh
            fields = compute_element_and_node_fields(
                filtered_mesh_connectivity,
                element_shell_stress=stress_kept,
                element_shell_effective_plastic_strain=epsp_kept,
                compute_von_mises=True,
            )
            # Unpack
            filtered_element_effective_plastic_strain = fields["element"][
                "effective_plastic_strain"
            ]
            filtered_element_stress_voigt = fields["element"]["stress_voigt"]
            filtered_element_stress_vm = fields["element"]["stress_vm"]

        # Step 6: Build edges
        edges = build_edges_from_mesh_connectivity(filtered_mesh_connectivity)

        self.logger.info(
            f"Processed: {filtered_pos_raw.shape[1]} nodes, "
            f"{len(filtered_mesh_connectivity)} cells, {len(edges)} edges"
        )

        # Delete raw data to save memory
        data.pos_raw = None
        data.mesh_connectivity = None
        data.node_thickness = None
        data.element_shell_stress = None
        data.element_shell_effective_plastic_strain = None

        return CrashExtractedDataInMemory(
            metadata=data.metadata,
            filtered_pos_raw=filtered_pos_raw,
            filtered_mesh_connectivity=filtered_mesh_connectivity,
            filtered_node_thickness=filtered_node_thickness,
            edges=edges,
            filtered_element_stress_voigt=filtered_element_stress_voigt,
            filtered_element_stress_vm=filtered_element_stress_vm,
            filtered_element_effective_plastic_strain=filtered_element_effective_plastic_strain,
        )
