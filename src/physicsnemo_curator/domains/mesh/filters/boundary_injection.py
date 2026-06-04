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

"""Boundary-condition injection filter for DomainMesh pipelines.

Applies a :class:`~physicsnemo_curator.domains.mesh.boundaries.BoundaryGenerator`
to each :class:`~physicsnemo.mesh.domain_mesh.DomainMesh` flowing through,
synthesizing the missing CFD-domain outer boundaries (inlet / outlet / walls /
symmetry) and merging them alongside the existing geometry boundary while
preserving ``interior`` and ``global_data``.

Plain :class:`~physicsnemo.mesh.Mesh` objects (no boundaries) pass through
unchanged.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from physicsnemo_curator.core.base import Filter, Param

if TYPE_CHECKING:
    from collections.abc import Generator

    from physicsnemo.mesh import Mesh

logger = logging.getLogger(__name__)


class BoundaryInjectionFilter(Filter["Mesh"]):
    """Inject synthesized outer boundaries into each DomainMesh.

    Parameters
    ----------
    generator : BoundaryGenerator
        Strategy that synthesizes ``{name: Mesh}`` outer boundaries from a
        domain (e.g.
        :class:`~physicsnemo_curator.domains.mesh.boundaries.BoxTunnelBoundaries`
        or
        :class:`~physicsnemo_curator.domains.mesh.boundaries.HemisphereBoundaries`).
    check_watertight : bool
        If ``True``, log :meth:`DomainMesh.is_boundary_watertight` after
        injection (informational; can be expensive).  Default ``False``.
    watertight_tolerance : float
        Tolerance forwarded to ``is_boundary_watertight``.  Default ``1e-3``.

    Examples
    --------
    >>> from physicsnemo_curator.domains.mesh.boundaries import HemisphereBoundaries
    >>> filt = BoundaryInjectionFilter(HemisphereBoundaries())  # doctest: +SKIP
    >>> pipeline = source.filter(filt).write(sink)  # doctest: +SKIP
    """

    name: ClassVar[str] = "Boundary Injection"
    description: ClassVar[str] = "Synthesize and inject CFD-domain outer boundaries into DomainMesh objects"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the filter.

        Returns
        -------
        list[Param]
            The ``generator``, ``check_watertight``, and
            ``watertight_tolerance`` parameters.
        """
        return [
            Param(name="generator", description="BoundaryGenerator strategy instance", type=object),
            Param(
                name="check_watertight",
                description="Log is_boundary_watertight after injection",
                type=bool,
                default=False,
            ),
            Param(
                name="watertight_tolerance",
                description="Tolerance for the watertight check",
                type=float,
                default=1e-3,
            ),
        ]

    def __init__(
        self,
        generator: Any,
        check_watertight: bool = False,
        watertight_tolerance: float = 1e-3,
    ) -> None:
        self._generator = generator
        self._check_watertight = check_watertight
        self._watertight_tolerance = watertight_tolerance

    def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
        """Inject boundaries into each DomainMesh in the stream.

        Parameters
        ----------
        items : Generator[Mesh]
            Stream of incoming meshes.  Non-DomainMesh items pass through.

        Yields
        ------
        Mesh
            DomainMesh with synthesized boundaries injected (or the original
            item when it is not a DomainMesh).
        """
        from physicsnemo.mesh.domain_mesh import DomainMesh as _DomainMesh

        from physicsnemo_curator.domains.mesh.boundaries import inject_boundaries

        for mesh in items:
            if not isinstance(mesh, _DomainMesh):
                logger.warning("BoundaryInjectionFilter received a non-DomainMesh item; passing through unchanged.")
                yield mesh
                continue

            injected = inject_boundaries(mesh, self._generator)

            if self._check_watertight:
                self._log_watertight(injected)

            yield injected

    def _log_watertight(self, domain: object) -> None:
        """Log the boundary watertightness, tolerant of physicsnemo API drift.

        Different physicsnemo versions expose either
        ``is_boundary_watertight(tolerance=...)`` or
        ``check_boundary_watertight()``.  The check is informational, so any
        failure is logged rather than raised.

        Parameters
        ----------
        domain : DomainMesh
            The injected domain mesh to check.
        """
        try:
            is_boundary_watertight = getattr(domain, "is_boundary_watertight", None)
            check_boundary_watertight = getattr(domain, "check_boundary_watertight", None)
            if is_boundary_watertight is not None:
                watertight = is_boundary_watertight(tolerance=self._watertight_tolerance)
            elif check_boundary_watertight is not None:
                watertight = check_boundary_watertight()
            else:
                logger.warning("BoundaryInjectionFilter: no watertight-check method on DomainMesh.")
                return
            logger.info("BoundaryInjectionFilter: boundary watertight=%s", watertight)
        except Exception as exc:  # noqa: BLE001 - informational check must not break the pipeline
            logger.warning("BoundaryInjectionFilter: watertight check failed: %s", exc)
