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

"""Global-data injection filter for mesh pipelines.

Injects constant ``global_data`` entries (e.g. freestream conditions such as
``U_inf``, ``rho_inf``, ``p_inf``, ``nu``, ``L_ref``) into each mesh flowing
through.  This is the generic, config-driven counterpart to the per-dataset
constants hardcoded in domain-specific sources.

The mesh is yielded with its ``global_data`` updated in place.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import torch

from physicsnemo_curator.core.base import Filter, Param

if TYPE_CHECKING:
    from collections.abc import Generator

    from physicsnemo.mesh import Mesh

logger = logging.getLogger(__name__)


def _to_tensor(value: object) -> torch.Tensor:
    """Coerce a scalar / sequence config value to a float32 tensor."""
    if isinstance(value, torch.Tensor):
        return value.to(torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


class GlobalDataFilter(Filter["Mesh"]):
    """Inject constant ``global_data`` values into each mesh.

    Each entry in *values* is converted to a float32 tensor and written to
    the mesh's ``global_data`` (scalars become 0-d tensors, sequences become
    1-d vectors).  For
    :class:`~physicsnemo.mesh.domain_mesh.DomainMesh` objects the values are
    written to the domain-level ``global_data``.

    Parameters
    ----------
    values : dict
        Mapping of ``global_data`` key to a scalar or sequence of floats
        (e.g. ``{"U_inf": [30.0, 0.0, 0.0], "rho_inf": 1.225}``).
    overwrite : bool
        If ``True`` (default), overwrite existing keys; if ``False``, skip
        keys already present.

    Examples
    --------
    >>> filt = GlobalDataFilter(values={"U_inf": [30.0, 0.0, 0.0], "rho_inf": 1.225})  # doctest: +SKIP
    >>> pipeline = source.filter(filt).write(sink)  # doctest: +SKIP
    """

    name: ClassVar[str] = "Global Data Injector"
    description: ClassVar[str] = "Inject constant global_data values (e.g. freestream conditions) into meshes"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the filter.

        Returns
        -------
        list[Param]
            The ``values`` mapping and ``overwrite`` flag.
        """
        return [
            Param(name="values", description="Mapping of global_data key to scalar or sequence", type=dict),
            Param(name="overwrite", description="Overwrite existing global_data keys", type=bool, default=True),
        ]

    def __init__(self, values: dict[str, object], overwrite: bool = True) -> None:
        if not values:
            msg = "GlobalDataFilter requires a non-empty 'values' mapping."
            raise ValueError(msg)
        self._values: dict[str, torch.Tensor] = {str(k): _to_tensor(v) for k, v in values.items()}
        self._overwrite = overwrite

    def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
        """Inject global data into each mesh and yield it.

        Parameters
        ----------
        items : Generator[Mesh]
            Stream of incoming meshes (Mesh or DomainMesh).

        Yields
        ------
        Mesh
            The same mesh with ``global_data`` updated.
        """
        for mesh in items:
            self._inject(mesh.global_data)
            yield mesh

    def _inject(self, global_data: object) -> None:
        """Write the configured values into a ``global_data`` TensorDict.

        Parameters
        ----------
        global_data : TensorDict
            The mesh's ``global_data`` (modified in place).
        """
        existing = set(global_data.keys())  # ty: ignore[unresolved-attribute]
        for key, tensor in self._values.items():
            if key in existing and not self._overwrite:
                continue
            global_data[key] = tensor  # ty: ignore[unresolved-attribute]
