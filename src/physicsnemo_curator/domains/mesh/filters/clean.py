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

"""Mesh cleaning filter for mesh pipelines.

Wraps :meth:`physicsnemo.mesh.Mesh.clean` to merge duplicate points, remove
duplicate cells, and drop unused points from meshes imported from external
sources (VTK, STL, CAD).  A **new** mesh is yielded.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from physicsnemo_curator.core.base import Filter, Param

if TYPE_CHECKING:
    from collections.abc import Generator

    from physicsnemo.mesh import Mesh

logger = logging.getLogger(__name__)


class CleanFilter(Filter["Mesh"]):
    """Clean and repair meshes via :meth:`Mesh.clean`.

    For each incoming mesh the filter optionally merges spatially-duplicate
    points, removes duplicate cells, and drops points not referenced by any
    cell.  Meshes without cells (point clouds) are passed through unchanged,
    since ``remove_unused_points`` would otherwise discard every point.

    For :class:`~physicsnemo.mesh.domain_mesh.DomainMesh` objects the interior
    and every boundary sub-mesh are cleaned independently.

    Parameters
    ----------
    tolerance : float
        Absolute L2 distance threshold for merging duplicate points.
        Default ``1e-12``.
    merge_points : bool
        Merge spatially-duplicate points (default ``True``).
    remove_duplicate_cells : bool
        Remove cells with identical vertex sets (default ``True``).
    remove_unused_points : bool
        Drop points not referenced by any cell (default ``True``).

    Examples
    --------
    >>> filt = CleanFilter()  # doctest: +SKIP
    >>> pipeline = source.filter(filt).write(sink)  # doctest: +SKIP
    """

    name: ClassVar[str] = "Mesh Clean"
    description: ClassVar[str] = "Merge duplicate points, remove duplicate cells, and drop unused points"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the clean filter.

        Returns
        -------
        list[Param]
            Cleaning option parameters.
        """
        return [
            Param(name="tolerance", description="L2 distance threshold for merging points", type=float, default=1e-12),
            Param(name="merge_points", description="Merge spatially-duplicate points", type=bool, default=True),
            Param(
                name="remove_duplicate_cells",
                description="Remove cells with identical vertex sets",
                type=bool,
                default=True,
            ),
            Param(
                name="remove_unused_points",
                description="Drop points not referenced by any cell",
                type=bool,
                default=True,
            ),
        ]

    def __init__(
        self,
        tolerance: float = 1e-12,
        merge_points: bool = True,
        remove_duplicate_cells: bool = True,
        remove_unused_points: bool = True,
    ) -> None:
        self._tolerance = tolerance
        self._merge_points = merge_points
        self._remove_duplicate_cells = remove_duplicate_cells
        self._remove_unused_points = remove_unused_points

    def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
        """Clean each mesh in the stream.

        Parameters
        ----------
        items : Generator[Mesh]
            Stream of incoming meshes (Mesh or DomainMesh).

        Yields
        ------
        Mesh
            Cleaned mesh (a new object when cleaning was applied).
        """
        from physicsnemo.mesh.domain_mesh import DomainMesh as _DomainMesh

        for mesh in items:
            if isinstance(mesh, _DomainMesh):
                yield self._clean_domain_mesh(mesh)
            else:
                yield self._clean_mesh(mesh)

    def _clean_mesh(self, mesh: Mesh) -> Mesh:
        """Clean a single Mesh, skipping point clouds.

        Parameters
        ----------
        mesh : Mesh
            The mesh to clean.

        Returns
        -------
        Mesh
            Cleaned mesh, or the original when it has no cells.
        """
        if mesh.cells is None or mesh.n_cells == 0:
            return mesh
        n_pts_before, n_cells_before = mesh.n_points, mesh.n_cells
        cleaned = mesh.clean(
            tolerance=self._tolerance,
            merge_points=self._merge_points,
            remove_duplicate_cells=self._remove_duplicate_cells,
            remove_unused_points=self._remove_unused_points,
        )
        logger.debug(
            "CleanFilter: points %d->%d, cells %d->%d",
            n_pts_before,
            cleaned.n_points,
            n_cells_before,
            cleaned.n_cells,
        )
        return cleaned

    def _clean_domain_mesh(self, domain_mesh: object) -> object:
        """Clean interior and boundary sub-meshes of a DomainMesh.

        Parameters
        ----------
        domain_mesh : DomainMesh
            The domain mesh to clean.

        Returns
        -------
        DomainMesh
            A new domain mesh with cleaned sub-meshes.
        """
        from physicsnemo.mesh.domain_mesh import DomainMesh as _DomainMesh

        interior = self._clean_mesh(domain_mesh.interior)  # ty: ignore[unresolved-attribute]
        boundaries = domain_mesh.boundaries  # ty: ignore[unresolved-attribute]
        new_boundaries = {}
        if boundaries is not None:
            for name in boundaries.keys():  # noqa: SIM118 - TensorDict needs .keys()
                new_boundaries[str(name)] = self._clean_mesh(boundaries[name])
        return _DomainMesh(
            interior=interior,
            boundaries=new_boundaries,
            global_data=domain_mesh.global_data,  # ty: ignore[unresolved-attribute]
        )
