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

"""Point-data to cell-data conversion filter for mesh pipelines.

Wraps :meth:`physicsnemo.mesh.Mesh.point_data_to_cell_data` to move vertex
fields onto cells (averaging over each cell's vertices).  Useful for surface
boundary meshes that should carry cell-centered quantities.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from physicsnemo_curator.core.base import Filter, Param

if TYPE_CHECKING:
    from collections.abc import Generator

    from physicsnemo.mesh import Mesh

logger = logging.getLogger(__name__)


class PointDataToCellDataFilter(Filter["Mesh"]):
    """Convert ``point_data`` to ``cell_data`` by per-cell averaging.

    For each incoming mesh with cells and point data, the average of each
    point field over a cell's vertices is written to ``cell_data``.  By
    default the original ``point_data`` is then dropped so the mesh carries
    only cell-centered quantities (surface-boundary convention); set
    *drop_point_data* to ``False`` to keep both.

    For :class:`~physicsnemo.mesh.domain_mesh.DomainMesh` objects the
    conversion is applied to the interior and every boundary sub-mesh that
    has cells and point data.

    Parameters
    ----------
    overwrite_keys : bool
        If ``True``, overwrite existing ``cell_data`` keys; otherwise raise
        on conflict.  Default ``False``.
    drop_point_data : bool
        If ``True`` (default), remove ``point_data`` after conversion.

    Examples
    --------
    >>> filt = PointDataToCellDataFilter()  # doctest: +SKIP
    >>> pipeline = source.filter(filt).write(sink)  # doctest: +SKIP
    """

    name: ClassVar[str] = "Point Data to Cell Data"
    description: ClassVar[str] = "Move point_data onto cells by per-cell averaging"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the filter.

        Returns
        -------
        list[Param]
            The ``overwrite_keys`` and ``drop_point_data`` parameters.
        """
        return [
            Param(name="overwrite_keys", description="Overwrite existing cell_data keys", type=bool, default=False),
            Param(name="drop_point_data", description="Drop point_data after conversion", type=bool, default=True),
        ]

    def __init__(self, overwrite_keys: bool = False, drop_point_data: bool = True) -> None:
        self._overwrite_keys = overwrite_keys
        self._drop_point_data = drop_point_data

    def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
        """Convert point data to cell data for each mesh in the stream.

        Parameters
        ----------
        items : Generator[Mesh]
            Stream of incoming meshes (Mesh or DomainMesh).

        Yields
        ------
        Mesh
            Mesh with cell data populated from point data.
        """
        from physicsnemo.mesh.domain_mesh import DomainMesh as _DomainMesh

        for mesh in items:
            if isinstance(mesh, _DomainMesh):
                yield self._convert_domain_mesh(mesh)
            else:
                yield self._convert_mesh(mesh)

    def _convert_mesh(self, mesh: Mesh) -> Mesh:
        """Convert a single Mesh's point data to cell data.

        Parameters
        ----------
        mesh : Mesh
            The mesh to convert.

        Returns
        -------
        Mesh
            Converted mesh, or the original when it has no cells / point data.
        """
        if mesh.cells is None or mesh.n_cells == 0:
            return mesh
        if mesh.point_data is None or len(mesh.point_data.keys()) == 0:
            return mesh

        converted = mesh.point_data_to_cell_data(overwrite_keys=self._overwrite_keys)
        if self._drop_point_data and converted.point_data is not None:
            for key in list(converted.point_data.keys()):  # noqa: SIM118 - TensorDict needs .keys()
                del converted.point_data[key]
        return converted

    def _convert_domain_mesh(self, domain_mesh: object) -> object:
        """Convert all sub-meshes of a DomainMesh.

        Parameters
        ----------
        domain_mesh : DomainMesh
            The domain mesh to convert.

        Returns
        -------
        DomainMesh
            A new domain mesh with converted sub-meshes.
        """
        from physicsnemo.mesh.domain_mesh import DomainMesh as _DomainMesh

        interior = self._convert_mesh(domain_mesh.interior)  # ty: ignore[unresolved-attribute]
        boundaries = domain_mesh.boundaries  # ty: ignore[unresolved-attribute]
        new_boundaries = {}
        if boundaries is not None:
            for name in boundaries.keys():  # noqa: SIM118 - TensorDict needs .keys()
                new_boundaries[str(name)] = self._convert_mesh(boundaries[name])
        return _DomainMesh(
            interior=interior,
            boundaries=new_boundaries,
            global_data=domain_mesh.global_data,  # ty: ignore[unresolved-attribute]
        )
