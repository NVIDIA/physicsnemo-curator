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

"""Ansys RST data source for structural and thermal simulation meshes.

Reads Ansys ``.rst`` result files via `ansys-dpf-core
<https://dpf.docs.pyansys.com/>`__ and yields :class:`~physicsnemo.mesh.Mesh`
objects for use in curator pipelines.

Each source index maps to one ``.rst`` file in the input directory.  The
resulting mesh carries:

* ``points`` --- *(N, 3)* node coordinates
* ``cells`` --- *(E, nodes_per_cell)* element connectivity
* ``point_data`` --- discovered nodal result fields (temperature,
  displacement, etc.)
* ``cell_data`` --- discovered elemental result fields (stress, strain,
  heat_flux, etc.) if present
* ``global_data`` --- scalar metadata (``num_nodes``, ``num_elements``)

The source auto-discovers available results in the ``.rst`` file and
extracts all of them by default, or a user-specified subset.

Examples
--------
>>> source = AnsysRSTSource(input_dir="/data/ansys_sims")  # doctest: +SKIP
>>> len(source)  # doctest: +SKIP
5
>>> mesh = next(source[0])  # doctest: +SKIP
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from tensordict import TensorDict

from physicsnemo.curator.core.base import REQUIRED, Param, Source

if TYPE_CHECKING:
    from collections.abc import Generator

    from physicsnemo.mesh import Mesh  # noqa: F821

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result-type helpers
# ---------------------------------------------------------------------------

# Map of DPF result operator names to human-readable field names and their
# expected data layout ("nodal" = per-node, "elemental" = per-element).
_KNOWN_RESULT_TYPES: dict[str, dict[str, str]] = {
    "temperature": {"field_name": "temperature", "location": "nodal"},
    "displacement": {"field_name": "displacement", "location": "nodal"},
    "heat_flux": {"field_name": "heat_flux", "location": "elemental"},
    "stress": {"field_name": "stress", "location": "elemental"},
    "elastic_strain": {"field_name": "elastic_strain", "location": "elemental"},
    "structural_temperature": {"field_name": "structural_temperature", "location": "nodal"},
    "velocity": {"field_name": "velocity", "location": "nodal"},
    "acceleration": {"field_name": "acceleration", "location": "nodal"},
}


def _extract_result_field(
    model: Any,
    result_name: str,
) -> tuple[np.ndarray, str] | None:
    """Extract a single result field from a DPF model.

    Parameters
    ----------
    model : Any
        A ``dpf.Model`` instance.
    result_name : str
        Name of the result operator (e.g. ``"temperature"``).

    Returns
    -------
    tuple[np.ndarray, str] | None
        ``(data_array, location)`` where *location* is ``"nodal"`` or
        ``"elemental"``, or ``None`` if the result is not available.
    """
    try:
        result_op = getattr(model.results, result_name)()
        fields_container = result_op.outputs.fields_container()
        if len(fields_container) == 0:
            return None
        data = np.array(fields_container[0].data, dtype=np.float64)
        location = _KNOWN_RESULT_TYPES.get(result_name, {}).get("location", "nodal")
        return data, location
    except Exception:  # noqa: BLE001
        logger.debug("Result '%s' not available in model", result_name)
        return None


def _discover_available_results(model: Any) -> list[str]:
    """Discover which result types are available in the DPF model.

    Parameters
    ----------
    model : Any
        A ``dpf.Model`` instance.

    Returns
    -------
    list[str]
        Sorted list of available result operator names.
    """
    available: list[str] = []
    for result_name in _KNOWN_RESULT_TYPES:
        try:
            result_op = getattr(model.results, result_name)()
            fc = result_op.outputs.fields_container()
            if len(fc) > 0 and len(fc[0].data) > 0:
                available.append(result_name)
        except Exception:  # noqa: BLE001
            continue
    return sorted(available)


def _extract_connectivity(meshed_region: Any) -> np.ndarray:
    """Extract element connectivity from a DPF meshed region.

    Pads ragged connectivity to a uniform width using ``-1`` fill.

    Parameters
    ----------
    meshed_region : Any
        A ``dpf.MeshedRegion`` instance.

    Returns
    -------
    np.ndarray
        Element connectivity, shape ``(E, max_nodes_per_element)``, int64.
    """
    elements = meshed_region.elements
    n_elements = elements.n_elements

    # Collect element connectivity lists.
    conn_lists: list[list[int]] = []
    max_nodes = 0
    for i in range(n_elements):
        elem = elements.element_by_index(i)
        node_ids = list(elem.node_ids)
        conn_lists.append(node_ids)
        if len(node_ids) > max_nodes:
            max_nodes = len(node_ids)

    # Build node-ID-to-index map for 0-based indexing.
    nodes = meshed_region.nodes
    node_id_to_index: dict[int, int] = {}
    for i in range(nodes.n_nodes):
        node_id_to_index[nodes.node_by_index(i).id] = i

    # Pad and remap to 0-based indices.
    connectivity = np.full((n_elements, max_nodes), -1, dtype=np.int64)
    for i, nids in enumerate(conn_lists):
        for j, nid in enumerate(nids):
            connectivity[i, j] = node_id_to_index.get(nid, -1)

    return connectivity


# ---------------------------------------------------------------------------
# Source class
# ---------------------------------------------------------------------------


class AnsysRSTSource(Source["Mesh"]):  # noqa: F821
    """Read simulation meshes from Ansys ``.rst`` result files.

    Scans ``input_dir`` for files matching ``*.rst`` and reads each one
    via `ansys-dpf-core <https://dpf.docs.pyansys.com/>`__.  The source
    auto-discovers available result types (temperature, displacement,
    stress, etc.) and populates ``point_data`` and ``cell_data``
    accordingly.

    Parameters
    ----------
    input_dir : str
        Directory containing ``.rst`` files.
    result_types : list[str] | None
        Explicit list of result types to extract (e.g.
        ``["temperature", "heat_flux"]``).  If ``None`` or empty, all
        available results are extracted.

    Note
    ----
    Requires the ``ansys-dpf-core`` package
    (``pip install ansys-dpf-core``).  Reading real ``.rst`` files also
    requires an Ansys installation (2021 R1+) with a valid license.

    Examples
    --------
    >>> source = AnsysRSTSource(input_dir="/data/ansys_thermal")  # doctest: +SKIP
    >>> len(source)  # doctest: +SKIP
    5
    >>> mesh = next(source[0])  # doctest: +SKIP
    """

    name: ClassVar[str] = "Ansys RST"
    description: ClassVar[str] = (
        "Ansys .rst result file reader --- auto-discovers nodal and "
        "elemental fields (temperature, displacement, stress, etc.)"
    )

    @classmethod
    def params(cls) -> list[Param]:
        """Return configurable parameters for this source.

        Returns
        -------
        list[Param]
            Parameter descriptors for CLI configuration.
        """
        return [
            Param(
                name="input_dir",
                description="Directory containing .rst result files",
                type=str,
                default=REQUIRED,
            ),
            Param(
                name="result_types",
                description=(
                    "Comma-separated list of result types to extract "
                    "(e.g. 'temperature,heat_flux'). Empty = auto-discover all."
                ),
                type=str,
                default="",
            ),
        ]

    def __init__(
        self,
        input_dir: str,
        result_types: list[str] | None = None,
    ) -> None:
        self._input_dir = Path(input_dir)
        self._result_types = result_types or []

        if not self._input_dir.is_dir():
            msg = f"input_dir does not exist or is not a directory: {self._input_dir}"
            raise FileNotFoundError(msg)

        # Eagerly discover .rst files (lightweight).
        self._rst_files = sorted(self._input_dir.glob("*.rst"))
        logger.info(
            "AnsysRSTSource: discovered %d .rst files in %s",
            len(self._rst_files),
            self._input_dir,
        )

    # -- Source interface -----------------------------------------------------

    def __len__(self) -> int:
        """Return the number of ``.rst`` files discovered."""
        return len(self._rst_files)

    def __getitem__(self, index: int) -> Generator[Mesh]:
        """Read the mesh from the *index*-th ``.rst`` file.

        Parameters
        ----------
        index : int
            Zero-based file index (supports negative indexing).

        Yields
        ------
        Mesh
            Mesh with nodal/elemental result fields.

        Raises
        ------
        IndexError
            If *index* is out of range.
        """
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            msg = f"Index {index} out of range for source with {len(self)} items."
            raise IndexError(msg)

        rst_path = self._rst_files[index]
        yield self._read_rst(rst_path)

    # -- Internal helpers ----------------------------------------------------

    def _read_rst(self, rst_path: Path) -> Mesh:
        """Read a single ``.rst`` file into a Mesh.

        Parameters
        ----------
        rst_path : Path
            Path to the ``.rst`` file.

        Returns
        -------
        Mesh
            The constructed mesh with discovered fields.
        """
        from ansys.dpf import core as dpf
        from physicsnemo.mesh import Mesh

        model = dpf.Model(str(rst_path))
        meshed_region = model.metadata.meshed_region

        # -- Coordinates (N, 3) ----------------------------------------------
        nodes = meshed_region.nodes
        n_nodes = nodes.n_nodes
        coords = np.array(nodes.coordinates_field.data, dtype=np.float64).reshape(n_nodes, 3)
        points = torch.from_numpy(coords)

        # -- Connectivity (E, max_nodes_per_cell) ----------------------------
        connectivity = _extract_connectivity(meshed_region)
        n_elements = connectivity.shape[0]
        cells = torch.from_numpy(connectivity)

        # -- Determine which results to extract ------------------------------
        if self._result_types:
            requested = self._result_types
        else:
            requested = _discover_available_results(model)
            logger.info(
                "AnsysRSTSource: auto-discovered results in %s: %s",
                rst_path.name,
                requested,
            )

        # -- Extract result fields -------------------------------------------
        pd_dict: dict[str, torch.Tensor] = {}
        cd_dict: dict[str, torch.Tensor] = {}

        for result_name in requested:
            result = _extract_result_field(model, result_name)
            if result is None:
                logger.debug("Skipping unavailable result '%s' in %s", result_name, rst_path.name)
                continue

            data_array, location = result
            field_name = _KNOWN_RESULT_TYPES.get(result_name, {}).get("field_name", result_name)
            tensor = torch.from_numpy(data_array)

            if location == "nodal":
                # Ensure shape is (n_nodes,) or (n_nodes, D).
                valid = (tensor.ndim == 1 and tensor.shape[0] == n_nodes) or (
                    tensor.ndim == 2 and tensor.shape[0] == n_nodes
                )
                if valid:
                    pd_dict[field_name] = tensor
                else:
                    logger.warning(
                        "Unexpected shape %s for nodal result '%s' (n_nodes=%d), skipping",
                        tensor.shape,
                        field_name,
                        n_nodes,
                    )
            elif location == "elemental":
                valid = (tensor.ndim == 1 and tensor.shape[0] == n_elements) or (
                    tensor.ndim == 2 and tensor.shape[0] == n_elements
                )
                if valid:
                    cd_dict[field_name] = tensor
                else:
                    logger.warning(
                        "Unexpected shape %s for elemental result '%s' (n_elements=%d), skipping",
                        tensor.shape,
                        field_name,
                        n_elements,
                    )

        # -- Build TensorDicts -----------------------------------------------
        point_data = TensorDict(pd_dict, batch_size=[n_nodes]) if pd_dict else None
        cell_data = TensorDict(cd_dict, batch_size=[n_elements]) if cd_dict else None

        global_data = TensorDict(
            {
                "num_nodes": torch.tensor([n_nodes], dtype=torch.int64),
                "num_elements": torch.tensor([n_elements], dtype=torch.int64),
            },
            batch_size=[],
        )

        mesh = Mesh(
            points=points,
            cells=cells,
            point_data=point_data,
            cell_data=cell_data,
            global_data=global_data,
        )

        logger.info(
            "AnsysRSTSource: read %s — %d nodes, %d elements, point_data=%s, cell_data=%s",
            rst_path.name,
            n_nodes,
            n_elements,
            list(pd_dict.keys()),
            list(cd_dict.keys()),
        )

        return mesh
