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

"""Boundary-condition injection into a DomainMesh."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from physicsnemo.mesh.domain_mesh import DomainMesh

    from physicsnemo_curator.domains.mesh.boundaries.generators import BoundaryGenerator

logger = logging.getLogger(__name__)


def inject_boundaries(domain: DomainMesh, generator: BoundaryGenerator) -> DomainMesh:
    """Return a new DomainMesh with synthesized boundaries injected.

    The existing ``interior`` and ``global_data`` are preserved, and the
    generator's synthesized boundaries are merged alongside the existing
    boundaries (e.g. the geometry ``vehicle`` surface).

    Parameters
    ----------
    domain : DomainMesh
        The input domain mesh.
    generator : BoundaryGenerator
        Strategy producing ``{name: Mesh}`` outer boundaries.

    Returns
    -------
    DomainMesh
        New domain mesh with original + synthesized boundaries.
    """
    from physicsnemo.mesh.domain_mesh import DomainMesh as _DomainMesh

    new_outer = generator.generate(domain)

    existing = domain.boundaries
    boundaries: dict[str, object] = {}
    if existing is not None:
        for name in existing.keys():  # noqa: SIM118 - TensorDict needs .keys()
            boundaries[str(name)] = existing[name]
    boundaries.update(new_outer)

    logger.info(
        "Injected %d boundary(ies) %s; total boundaries: %s",
        len(new_outer),
        sorted(new_outer.keys()),
        sorted(boundaries.keys()),
    )

    return _DomainMesh(
        interior=domain.interior,
        boundaries=boundaries,
        global_data=domain.global_data,
    )
