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

"""Random permutation filter for mesh pipelines.

Randomly shuffles the ordering of points and cells in each mesh to remove
any spatial or ordering bias.  Point coordinates, point data, cell data,
and cell connectivity are all permuted consistently so that the mesh
geometry is preserved.

The mesh is modified **in place** and then yielded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import torch
from tensordict import TensorDict, TensorDictBase

from physicsnemo_curator.core.base import Filter, Param

if TYPE_CHECKING:
    from collections.abc import Generator

    from physicsnemo.mesh import Mesh


def _permute_tensordict(
    td: TensorDictBase | None,
    perm: torch.Tensor,
    new_size: int,
) -> TensorDict | None:
    """Permute a TensorDict along its batch dimension.

    Parameters
    ----------
    td : TensorDictBase or None
        TensorDict to permute.
    perm : torch.Tensor
        Permutation index tensor of shape ``(N,)``.
    new_size : int
        Batch size after permutation (same as input size).

    Returns
    -------
    TensorDict or None
        Permuted TensorDict, or ``None`` if input is ``None``.
    """
    from tensordict import TensorDictBase

    if td is None:
        return None

    if not isinstance(td, TensorDictBase):
        return None

    permuted: dict[str, torch.Tensor | TensorDict] = {}
    for key in td.keys():  # noqa: SIM118
        child = td[key]
        if isinstance(child, TensorDictBase):
            permuted[key] = _permute_tensordict(child, perm, new_size)  # ty: ignore[invalid-assignment]
        else:
            permuted[key] = child[perm]  # ty: ignore[invalid-assignment]
    return TensorDict(permuted, batch_size=[new_size])  # ty: ignore[invalid-argument-type]


def _shuffle_mesh(mesh: object, rng: torch.Generator) -> None:
    """Shuffle point and cell ordering of a single mesh in place.

    Generates independent random permutations for points and cells.
    Point permutation reorders ``points``, ``point_data``, and remaps
    the point indices stored in ``cells``.  Cell permutation reorders
    ``cells`` (rows) and ``cell_data``.

    Parameters
    ----------
    mesh : Mesh
        The mesh to shuffle (modified in place).
    rng : torch.Generator
        Seeded random number generator.
    """
    n_points = mesh.points.shape[0] if mesh.points is not None else 0  # ty: ignore[unresolved-attribute]

    # --- Shuffle points ---
    if n_points > 1:
        perm_pts = torch.randperm(n_points, generator=rng)

        # Reorder point coordinates
        mesh.points = mesh.points[perm_pts]  # ty: ignore[unresolved-attribute]

        # Reorder point data
        if mesh.point_data is not None:  # ty: ignore[unresolved-attribute]
            mesh.point_data = _permute_tensordict(  # ty: ignore[unresolved-attribute]
                mesh.point_data,  # ty: ignore[unresolved-attribute]
                perm_pts,
                n_points,
            )

        # Remap cell connectivity: build inverse permutation so that
        # new_cells[i,j] = inv_perm[old_cells[i,j]] maps old point
        # indices to their new positions.
        if mesh.cells is not None:  # ty: ignore[unresolved-attribute]
            inv_perm = torch.empty_like(perm_pts)
            inv_perm[perm_pts] = torch.arange(n_points)
            mesh.cells = inv_perm[mesh.cells]  # ty: ignore[unresolved-attribute]

    # --- Shuffle cells ---
    n_cells = mesh.cells.shape[0] if mesh.cells is not None else 0  # ty: ignore[unresolved-attribute]

    if n_cells > 1:
        perm_cells = torch.randperm(n_cells, generator=rng)

        # Reorder cell connectivity rows
        mesh.cells = mesh.cells[perm_cells]  # ty: ignore[unresolved-attribute]

        # Reorder cell data
        if mesh.cell_data is not None:  # ty: ignore[unresolved-attribute]
            mesh.cell_data = _permute_tensordict(  # ty: ignore[unresolved-attribute]
                mesh.cell_data,  # ty: ignore[unresolved-attribute]
                perm_cells,
                n_cells,
            )


class RandomPermutationFilter(Filter["Mesh"]):
    """Randomly permute point and cell ordering in each mesh.

    For each incoming mesh the filter generates reproducible random
    permutations for the point and cell dimensions.  Point coordinates,
    point data, cell connectivity indices, and cell data are all
    reordered consistently so that the mesh geometry and topology are
    preserved — only the storage ordering changes.

    For :class:`~physicsnemo.mesh.domain_mesh.DomainMesh` objects each
    sub-mesh (interior and every boundary) is shuffled independently
    with its own derived seed.

    A per-mesh counter is combined with the base ``seed`` to produce a
    unique RNG state for every mesh processed, ensuring reproducibility
    regardless of the number of meshes or pipeline parallelism.

    Parameters
    ----------
    seed : int
        Base random seed.  The effective seed for the *k*-th mesh
        yielded is ``seed + k``.

    Examples
    --------
    Shuffle meshes with a fixed seed:

    >>> filt = RandomPermutationFilter(seed=42)  # doctest: +SKIP
    >>> pipeline = source.filter(filt).write(sink)  # doctest: +SKIP
    """

    name: ClassVar[str] = "Random Permutation"
    description: ClassVar[str] = "Randomly shuffle point and cell ordering in each mesh"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the random-permutation filter.

        Returns
        -------
        list[Param]
            The ``seed`` parameter.
        """
        return [
            Param(
                name="seed",
                description="Base random seed for reproducible permutations",
                type=int,
            ),
        ]

    def __init__(self, seed: int) -> None:
        """Initialise the random-permutation filter.

        Parameters
        ----------
        seed : int
            Base random seed.
        """
        self._seed = seed

    def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
        """Permute point and cell ordering for each mesh in the stream.

        Parameters
        ----------
        items : Generator[Mesh]
            Stream of incoming meshes.

        Yields
        ------
        Mesh
            The same mesh with points and cells shuffled in place.
        """
        from physicsnemo.mesh.domain_mesh import DomainMesh as _DomainMesh

        for counter, mesh in enumerate(items):
            effective_seed = self._seed + counter

            if isinstance(mesh, _DomainMesh):
                self._shuffle_domain_mesh(mesh, effective_seed)
            else:
                rng = torch.Generator()
                rng.manual_seed(effective_seed)
                _shuffle_mesh(mesh, rng)

            yield mesh

    # ------------------------------------------------------------------
    # DomainMesh helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _shuffle_domain_mesh(domain_mesh: object, effective_seed: int) -> None:
        """Shuffle all sub-meshes of a DomainMesh independently.

        Each sub-mesh receives a unique derived seed so that interior
        and boundary permutations are independent.

        Parameters
        ----------
        domain_mesh : DomainMesh
            The domain mesh to shuffle (modified in place).
        effective_seed : int
            Base seed for this mesh (further offset per sub-mesh).
        """
        sub_idx = 0

        # Interior
        interior = domain_mesh.interior  # ty: ignore[unresolved-attribute]
        rng = torch.Generator()
        rng.manual_seed(effective_seed + sub_idx)
        _shuffle_mesh(interior, rng)
        sub_idx += 1

        # Boundaries
        boundaries = domain_mesh.boundaries  # ty: ignore[unresolved-attribute]
        if boundaries is not None:
            for bnd_name in boundaries.keys():  # noqa: SIM118 - TensorDict needs .keys()
                boundary = boundaries[bnd_name]
                rng = torch.Generator()
                rng.manual_seed(effective_seed + sub_idx)
                _shuffle_mesh(boundary, rng)
                sub_idx += 1
