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

"""Wall-node filter for crash simulation mesh pipelines.

Removes wall (non-deforming) nodes from crash simulation meshes by
analysing per-timestep displacement fields.  Nodes whose maximum
displacement variation is below a threshold are classified as *wall*
and dropped.  Connectivity is remapped and degenerate cells (fewer
than 3 surviving nodes) are removed.

This is a **stateful** filter — the mesh is modified and a new
:class:`~physicsnemo.mesh.Mesh` is yielded.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch
from tensordict import TensorDict

from physicsnemo_curator.core.base import Filter, Param

if TYPE_CHECKING:
    from collections.abc import Generator

    from physicsnemo.mesh import Mesh

logger = logging.getLogger(__name__)


def _identify_displacement_keys(point_data: TensorDict) -> list[str]:
    """Return sorted ``displacement_t*`` keys from point data.

    Parameters
    ----------
    point_data : TensorDict
        The mesh's point data.

    Returns
    -------
    list[str]
        Sorted displacement field keys (e.g. ``["displacement_t000", ...]``).
    """
    return sorted(
        str(k)
        for k in point_data.keys()  # noqa: SIM118
        if isinstance(k, str) and k.startswith("displacement_t")
    )


def _compute_keep_mask(
    point_data: TensorDict,
    disp_keys: list[str],
    threshold: float,
) -> np.ndarray:
    """Compute a boolean mask of nodes to keep (structural, not wall).

    A node is "wall" if the maximum absolute displacement variation
    (relative to the first timestep) across all timesteps and spatial
    dimensions is below ``threshold``.

    Parameters
    ----------
    point_data : TensorDict
        The mesh's point data containing displacement fields.
    disp_keys : list[str]
        Sorted displacement field keys.
    threshold : float
        Minimum displacement variation to keep a node.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(N,)`` — ``True`` for nodes to keep.
    """
    # Stack displacements: (T, N, 3).
    disp_stack = torch.stack([point_data[k] for k in disp_keys], dim=0)  # ty: ignore[invalid-argument-type]
    # Reference is first timestep.
    variation = (disp_stack - disp_stack[0:1]).abs()  # (T, N, 3)
    max_variation = variation.amax(dim=(0, 2))  # (N,)
    keep_mask = max_variation.numpy() >= threshold
    return keep_mask


def _remap_connectivity(
    cells: torch.Tensor,
    keep_indices: np.ndarray,
) -> tuple[torch.Tensor, np.ndarray]:
    """Remap cell connectivity after node removal.

    Parameters
    ----------
    cells : torch.Tensor
        Original cells tensor, shape ``(E, nodes_per_cell)``.
    keep_indices : np.ndarray
        Sorted array of kept node indices.

    Returns
    -------
    tuple[torch.Tensor, np.ndarray]
        - New cells tensor with remapped node indices.
        - Boolean mask of shape ``(E,)`` — ``True`` for kept cells.
    """
    n_original = int(cells.max().item()) + 1
    # Build old-to-new index map.  Unmapped nodes get -1.
    old_to_new = np.full(n_original, -1, dtype=np.int64)
    old_to_new[keep_indices] = np.arange(len(keep_indices), dtype=np.int64)

    cells_np = cells.numpy()
    remapped = old_to_new[cells_np]  # (E, nodes_per_cell)

    # A cell is valid if all its nodes survive.
    cell_valid = np.all(remapped >= 0, axis=1)

    new_cells = torch.from_numpy(remapped[cell_valid])
    return new_cells, cell_valid


def _filter_tensordict(
    td: TensorDict | None,
    mask: np.ndarray,
    new_size: int,
) -> TensorDict | None:
    """Filter a TensorDict along its batch dimension.

    Parameters
    ----------
    td : TensorDict | None
        TensorDict to filter.
    mask : np.ndarray
        Boolean or integer index array for the batch dimension.
    new_size : int
        New batch size after filtering.

    Returns
    -------
    TensorDict | None
        Filtered TensorDict, or ``None`` if input is ``None``.
    """
    if td is None:
        return None

    idx = torch.from_numpy(np.where(mask)[0] if mask.dtype == np.bool_ else mask).long()
    filtered: dict[str, torch.Tensor] = {}
    for key in td.keys():  # noqa: SIM118
        filtered[key] = td[key][idx]  # ty: ignore[invalid-assignment]
    return TensorDict(filtered, batch_size=[new_size])  # ty: ignore[invalid-argument-type]


class WallNodeFilter(Filter["Mesh"]):
    """Remove wall (non-deforming) nodes from crash simulation meshes.

    Analyses ``displacement_t*`` fields in the mesh's ``point_data`` to
    identify nodes with negligible deformation.  Nodes whose maximum
    displacement variation (relative to the first timestep) is below
    ``threshold`` are removed.  Connectivity is remapped and cells that
    lose all their nodes are dropped.

    This filter **creates a new mesh** (it does not modify the input
    in place).

    Parameters
    ----------
    threshold : float
        Minimum displacement variation to retain a node.  Nodes below
        this value are classified as wall nodes and removed.

    Examples
    --------
    >>> filt = WallNodeFilter(threshold=1.0)
    >>> pipeline = source.filter(filt).write(sink)

    Note
    ----
    This filter requires that the mesh ``point_data`` contains fields
    named ``displacement_t000``, ``displacement_t001``, etc. (as produced
    by :class:`~physicsnemo_curator.domains.mesh.sources.d3plot.D3PlotSource`).
    """

    name: ClassVar[str] = "Wall Node Filter"
    description: ClassVar[str] = (
        "Remove non-deforming wall nodes from crash simulation meshes based on displacement variation threshold"
    )

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the wall-node filter.

        Returns
        -------
        list[Param]
            The ``threshold`` parameter.
        """
        return [
            Param(
                name="threshold",
                description="Minimum displacement variation to keep a node",
                type=float,
                default=1.0,
            ),
        ]

    def __init__(self, threshold: float = 1.0) -> None:
        if threshold < 0:
            msg = f"threshold must be non-negative, got {threshold}"
            raise ValueError(msg)
        self._threshold = threshold

    def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
        """Filter wall nodes from each incoming mesh.

        Parameters
        ----------
        items : Generator[Mesh]
            Stream of incoming meshes.

        Yields
        ------
        Mesh
            New mesh with wall nodes removed.
        """
        from physicsnemo.mesh import Mesh

        for mesh in items:
            if mesh.point_data is None:
                yield mesh
                continue

            disp_keys = _identify_displacement_keys(mesh.point_data)
            if not disp_keys:
                logger.warning("WallNodeFilter: no displacement_t* fields found, passing mesh through unchanged")
                yield mesh
                continue

            # Identify which nodes to keep.
            keep_mask = _compute_keep_mask(mesh.point_data, disp_keys, self._threshold)
            keep_indices = np.where(keep_mask)[0]
            n_original = mesh.n_points
            n_kept = len(keep_indices)

            if n_kept == 0:
                logger.warning(
                    "WallNodeFilter: all %d nodes classified as wall (threshold=%.2f), skipping mesh",
                    n_original,
                    self._threshold,
                )
                continue

            if n_kept == n_original:
                logger.debug("WallNodeFilter: all nodes kept (threshold=%.2f)", self._threshold)
                yield mesh
                continue

            # Filter points.
            new_points = mesh.points[torch.from_numpy(keep_indices).long()]

            # Remap connectivity.
            new_cells, cell_mask = _remap_connectivity(mesh.cells, keep_indices)

            if len(new_cells) == 0:
                logger.warning(
                    "WallNodeFilter: all cells degenerate after filtering, skipping mesh",
                )
                continue

            # Filter point_data.
            new_point_data = _filter_tensordict(mesh.point_data, keep_mask, n_kept)

            # Filter cell_data.
            new_cell_data = _filter_tensordict(mesh.cell_data, cell_mask, int(cell_mask.sum()))

            new_mesh = Mesh(
                points=new_points,
                cells=new_cells,
                point_data=new_point_data,
                cell_data=new_cell_data,
                global_data=mesh.global_data,
            )

            logger.info(
                "WallNodeFilter: kept %d/%d nodes (%.1f%%), %d/%d cells",
                n_kept,
                n_original,
                100.0 * n_kept / n_original,
                len(new_cells),
                mesh.n_cells,
            )

            yield new_mesh
