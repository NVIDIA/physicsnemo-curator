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

"""Field selection filter for mesh pipelines.

Selectively includes or excludes fields from ``point_data`` and ``cell_data``
on Mesh and DomainMesh objects.  Useful for reducing output size when only a
subset of simulation fields is needed for training.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from physicsnemo_curator.core.base import Filter, Param

if TYPE_CHECKING:
    from collections.abc import Generator

    from physicsnemo.mesh import Mesh

logger = logging.getLogger(__name__)


def _filter_tensordict(td: object, keep_keys: set[str]) -> list[str]:
    """Remove keys from a TensorDict that are not in *keep_keys*.

    Parameters
    ----------
    td : object
        A TensorDict-like object (``point_data`` or ``cell_data``).
    keep_keys : set[str]
        Keys to retain.  All others are deleted.

    Returns
    -------
    list[str]
        Keys that were removed.
    """
    existing = set(td.keys())  # ty: ignore[unresolved-attribute]
    to_remove = existing - keep_keys
    for key in to_remove:
        del td[key]  # ty: ignore[not-subscriptable]
    return sorted(to_remove)


class FieldSelectFilter(Filter["Mesh"]):
    """Filter that selects which data fields to keep on each mesh.

    Provide *either* ``include`` (whitelist) or ``exclude`` (blacklist) to
    control which fields from ``point_data`` and ``cell_data`` are retained.
    Fields not matching are removed in-place before the mesh is yielded
    downstream.

    If both ``include`` and ``exclude`` are ``None``, meshes pass through
    unchanged.  Providing both simultaneously raises ``ValueError``.

    For :class:`~physicsnemo.mesh.domain_mesh.DomainMesh` objects the filter
    is applied to the interior mesh and every boundary sub-mesh.

    Parameters
    ----------
    include : list[str] or None
        If provided, only keep fields whose names appear in this list.
    exclude : list[str] or None
        If provided, remove fields whose names appear in this list.

    Examples
    --------
    Keep only a subset of volume fields:

    >>> filt = FieldSelectFilter(include=["CpMeanTrim", "nutMeanTrim", "pMeanTrim", "UMeanTrim"])  # doctest: +SKIP
    >>> pipeline = source.filter(filt).write(sink)  # doctest: +SKIP

    Remove turbulence fields you don't need:

    >>> filt = FieldSelectFilter(exclude=["turbulenceProperties:RMeanTrim", "UPrime2MeanTrim"])  # doctest: +SKIP
    """

    name: ClassVar[str] = "Field Select"
    description: ClassVar[str] = "Include or exclude data fields from meshes"

    @classmethod
    def params(cls) -> list[Param]:
        """Return configurable parameters for this filter.

        Returns
        -------
        list[Param]
            Parameter descriptors.
        """
        return [
            Param(
                name="include",
                description="Field names to keep (whitelist). Mutually exclusive with 'exclude'.",
                type=list,
                default=None,
            ),
            Param(
                name="exclude",
                description="Field names to remove (blacklist). Mutually exclusive with 'include'.",
                type=list,
                default=None,
            ),
        ]

    def __init__(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> None:
        """Initialize the field selection filter.

        Parameters
        ----------
        include : list[str] or None
            If provided, only these fields are kept.
        exclude : list[str] or None
            If provided, these fields are removed.

        Raises
        ------
        ValueError
            If both *include* and *exclude* are provided simultaneously.
        """
        if include is not None and exclude is not None:
            msg = "Cannot specify both 'include' and 'exclude'. Use one or the other."
            raise ValueError(msg)

        self._include: set[str] | None = set(include) if include is not None else None
        self._exclude: set[str] | None = set(exclude) if exclude is not None else None

    def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
        """Filter fields from each mesh in the stream.

        Parameters
        ----------
        items : Generator[Mesh]
            Stream of incoming meshes (Mesh or DomainMesh).

        Yields
        ------
        Mesh
            Meshes with fields filtered in-place.
        """
        from physicsnemo.mesh.domain_mesh import DomainMesh as _DomainMesh

        for mesh in items:
            if isinstance(mesh, _DomainMesh):
                self._filter_domain_mesh(mesh)
            else:
                self._filter_mesh(mesh)
            yield mesh

    def _keep_keys(self, existing: set[str]) -> set[str]:
        """Compute the set of keys to retain given existing field names.

        Parameters
        ----------
        existing : set[str]
            Currently available field names.

        Returns
        -------
        set[str]
            Field names to keep.
        """
        if self._include is not None:
            return existing & self._include
        if self._exclude is not None:
            return existing - self._exclude
        return existing

    def _filter_mesh(self, mesh: object) -> None:
        """Apply field selection to a single Mesh.

        Parameters
        ----------
        mesh : Mesh
            The mesh to filter (modified in-place).
        """
        removed: list[str] = []

        if mesh.point_data is not None and len(mesh.point_data.keys()) > 0:  # ty: ignore[unresolved-attribute]
            existing = set(mesh.point_data.keys())  # ty: ignore[unresolved-attribute]
            keep = self._keep_keys(existing)
            removed.extend(f"point_data/{k}" for k in _filter_tensordict(mesh.point_data, keep))  # ty: ignore[unresolved-attribute]

        if mesh.cell_data is not None and len(mesh.cell_data.keys()) > 0:  # ty: ignore[unresolved-attribute]
            existing = set(mesh.cell_data.keys())  # ty: ignore[unresolved-attribute]
            keep = self._keep_keys(existing)
            removed.extend(f"cell_data/{k}" for k in _filter_tensordict(mesh.cell_data, keep))  # ty: ignore[unresolved-attribute]

        if removed:
            logger.debug("FieldSelectFilter: removed %d field(s): %s", len(removed), removed)

    def _filter_domain_mesh(self, domain_mesh: object) -> None:
        """Apply field selection to all sub-meshes of a DomainMesh.

        Parameters
        ----------
        domain_mesh : DomainMesh
            The domain mesh to filter (modified in-place).
        """
        interior = domain_mesh.interior  # ty: ignore[unresolved-attribute]
        self._filter_mesh(interior)

        boundaries = domain_mesh.boundaries  # ty: ignore[unresolved-attribute]
        if boundaries is not None:
            for bnd_name in boundaries.keys():  # noqa: SIM118 - TensorDict needs .keys()
                boundary = boundaries[bnd_name]
                self._filter_mesh(boundary)
