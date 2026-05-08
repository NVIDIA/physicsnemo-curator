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

"""Edge computation filter for mesh pipelines.

Computes unique edge connectivity from cell connectivity and stores
the result in ``global_data["edges"]``.  This is typically used before
sinks that require edge information, such as
:class:`~physicsnemo_curator.domains.mesh.sinks.mesh_zarr.MeshZarrSink`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from physicsnemo_curator.core.base import Filter, Param

if TYPE_CHECKING:
    from collections.abc import Generator

    from physicsnemo.mesh import Mesh

logger = logging.getLogger(__name__)


class EdgeComputeFilter(Filter["Mesh"]):
    """Compute edge connectivity from mesh cells.

    Uses the mesh's :meth:`to_edge_graph` method to extract unique
    edges from cell connectivity.  The resulting edge tensor of shape
    ``(E, 2)`` is stored in ``global_data["edges"]``.

    This filter should be placed **after** any filters that modify
    mesh connectivity (e.g., :class:`WallNodeFilter`) to ensure edges
    are computed from the final mesh state.

    Parameters
    ----------
    None
        This filter has no configurable parameters.

    Examples
    --------
    >>> from physicsnemo_curator.domains.mesh.filters.edge_compute import EdgeComputeFilter
    >>> filter = EdgeComputeFilter()  # doctest: +SKIP
    >>> filtered_meshes = filter(mesh_generator)  # doctest: +SKIP
    """

    name: ClassVar[str] = "Edge Compute Filter"
    description: ClassVar[str] = "Compute unique edge connectivity from cell connectivity"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the filter.

        Returns
        -------
        list[Param]
            Empty list — this filter has no parameters.
        """
        return []

    def __init__(self) -> None:
        """Initialize the edge compute filter."""
        pass

    def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
        """Compute edges for each incoming mesh.

        Parameters
        ----------
        items : Generator[Mesh]
            Stream of incoming meshes.

        Yields
        ------
        Mesh
            Mesh with ``global_data["edges"]`` populated.
        """
        from physicsnemo.mesh import Mesh
        from tensordict import TensorDict

        for mesh in items:
            if mesh.cells is None or mesh.n_cells == 0:
                logger.warning("EdgeComputeFilter: mesh has no cells, skipping edge computation")
                yield mesh
                continue

            # Compute edges using mesh's to_edge_graph method
            edge_mesh = mesh.to_edge_graph()
            edges = edge_mesh.cells  # (E, 2)

            if edges is None:
                logger.warning("EdgeComputeFilter: to_edge_graph returned no edges")
                yield mesh
                continue

            logger.debug(
                "EdgeComputeFilter: computed %d edges from %d cells",
                edges.shape[0],
                mesh.n_cells,
            )

            # Create or update global_data with edges
            if mesh.global_data is not None:
                new_global = TensorDict(
                    {str(k): mesh.global_data.get(k) for k in mesh.global_data.keys()},  # noqa: SIM118
                    batch_size=[],
                )
            else:
                new_global = TensorDict({}, batch_size=[])

            new_global["edges"] = edges

            # Yield new mesh with edges in global_data
            yield Mesh(
                points=mesh.points,
                cells=mesh.cells,
                point_data=mesh.point_data,
                cell_data=mesh.cell_data,
                global_data=new_global,
            )
