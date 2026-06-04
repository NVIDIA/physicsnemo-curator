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

"""Pluggable boundary-condition generators.

A :class:`BoundaryGenerator` synthesizes the missing CFD-domain outer
boundaries for a :class:`~physicsnemo.mesh.domain_mesh.DomainMesh` that ships
with only a geometry surface.  Datasets are specialized purely by choosing a
generator and its constants:

* :class:`BoxTunnelBoundaries` — rectangular wind tunnel (DrivAerML, ShiftSUV).
* :class:`HemisphereBoundaries` — hemispherical open-road domain with a
  symmetry plane (HighLiftAeroML).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from physicsnemo_curator.domains.mesh.boundaries import _geometry as geom

if TYPE_CHECKING:
    from physicsnemo.mesh import Mesh
    from physicsnemo.mesh.domain_mesh import DomainMesh

logger = logging.getLogger(__name__)


@runtime_checkable
class BoundaryGenerator(Protocol):
    """Protocol for boundary-condition generators.

    Implementations synthesize named outer-boundary meshes from a domain's
    known geometry (interior, the existing geometry surface, and
    ``global_data``).
    """

    def generate(self, domain: DomainMesh) -> dict[str, Mesh]:
        """Return a mapping of new boundary name -> synthesized Mesh.

        Parameters
        ----------
        domain : DomainMesh
            The input domain mesh (interior + existing boundaries + global
            data).

        Returns
        -------
        dict[str, Mesh]
            New outer boundaries to inject.
        """
        ...


class BoxTunnelBoundaries:
    """Generate rectangular wind-tunnel boundaries (inlet/outlet/slip/no_slip).

    The vertical floor height ``z_floor`` is, by default, inferred per sample
    from the minimum z of the geometry boundary (tire / contact patch); pass
    *z_floor* to override.

    Parameters
    ----------
    x_min, x_max : float
        Streamwise extents of the CFD domain.
    y_min, y_max : float
        Lateral extents.
    z_height : float
        Vertical extent above the floor.
    x_bl : float
        Streamwise coordinate of the slip-to-noslip floor transition.
    n_per_side : tuple[int, int]
        Per-face triangulation resolution.
    z_floor : float or None
        Fixed floor height; ``None`` infers from the geometry boundary.
    vehicle_key : str
        Boundary key of the geometry surface used for ``z_floor`` inference.
    """

    def __init__(
        self,
        *,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        z_height: float,
        x_bl: float,
        n_per_side: tuple[int, int] = (20, 20),
        z_floor: float | None = None,
        vehicle_key: str = "vehicle",
    ) -> None:
        self._x_min = x_min
        self._x_max = x_max
        self._y_min = y_min
        self._y_max = y_max
        self._z_height = z_height
        self._x_bl = x_bl
        self._n_per_side = n_per_side
        self._z_floor = z_floor
        self._vehicle_key = vehicle_key

    def generate(self, domain: DomainMesh) -> dict[str, Mesh]:
        """Synthesize inlet/outlet/slip/no_slip boundaries for *domain*."""
        if self._z_floor is not None:
            z_floor = self._z_floor
        else:
            vehicle = domain.boundaries[self._vehicle_key]
            z_floor = geom.z_floor_from_boundary(vehicle)

        bounds: geom.Bounds3D = (
            (self._x_min, self._x_max),
            (self._y_min, self._y_max),
            (z_floor, z_floor + self._z_height),
        )
        return geom.box_tunnel_boundaries(bounds=bounds, x_bl=self._x_bl, n_per_side=self._n_per_side)


class HemisphereBoundaries:
    """Generate hemispherical open-road boundaries (inlet/outlet/symmetry).

    The hemisphere radius is, by default, inferred per sample from the
    interior point-cloud bounding box; pass *radius* to override.  The
    inlet/outlet split is along the great half-circle perpendicular to the
    freestream direction read from ``global_data[freestream_key]``, so it is
    sample-dependent (encodes angle of attack).

    Parameters
    ----------
    radius : float or None
        Hemisphere radius; ``None`` infers from the interior bbox.
    n_theta, n_phi : int
        Hemisphere triangulation resolution.
    y_eps : float
        Tolerance for "vertex on the symmetry plane".
    freestream_key : str
        ``global_data`` key holding the freestream velocity vector.
    vehicle_key : str
        Boundary key of the geometry surface (provides the silhouette).
    """

    def __init__(
        self,
        *,
        radius: float | None = None,
        n_theta: int = 64,
        n_phi: int = 32,
        y_eps: float = 1e-3,
        freestream_key: str = "U_inf",
        vehicle_key: str = "vehicle",
    ) -> None:
        self._radius = radius
        self._n_theta = n_theta
        self._n_phi = n_phi
        self._y_eps = y_eps
        self._freestream_key = freestream_key
        self._vehicle_key = vehicle_key

    def generate(self, domain: DomainMesh) -> dict[str, Mesh]:
        """Synthesize inlet/outlet/symmetry boundaries for *domain*."""
        radius = self._radius if self._radius is not None else geom.radius_from_interior(domain.interior)

        hemisphere, equator_idx = geom.build_hemisphere(radius, n_theta=self._n_theta, n_phi=self._n_phi)

        u_inf = domain.global_data[self._freestream_key]
        inlet, outlet = geom.split_by_freestream(hemisphere, u_inf)

        vehicle = domain.boundaries[self._vehicle_key]
        loops = geom.silhouette_loops(vehicle, y_eps=self._y_eps)
        silhouette_xzs = [vehicle.points[loop][:, [0, 2]] for loop in loops]
        equator_xz = hemisphere.points[equator_idx][:, [0, 2]]
        symmetry = geom.constrained_delaunay_disk(equator_xz=equator_xz, silhouette_xzs=silhouette_xzs)

        return {"inlet": inlet, "outlet": outlet, "symmetry": symmetry}
