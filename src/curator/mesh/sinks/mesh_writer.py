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

"""PhysicsNeMo Mesh writer sink.

Persists :class:`physicsnemo.mesh.Mesh` objects to disk using the native
tensordict memory-mapped format via :meth:`Mesh.save`.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, ClassVar

from curator.core.base import Param, Sink

if TYPE_CHECKING:
    from collections.abc import Generator

    from physicsnemo.mesh import Mesh


class MeshSink(Sink["Mesh"]):
    """Write :class:`~physicsnemo.mesh.Mesh` objects to disk.

    Each mesh is saved to a subdirectory of *output_dir* named
    ``mesh_{index:04d}_{seq}`` using the physicsnemo native format
    (:meth:`Mesh.save`).

    Parameters
    ----------
    output_dir : str
        Directory where mesh outputs will be written.

    Examples
    --------
    >>> sink = MeshSink(output_dir="./output/")
    >>> paths = sink(mesh_generator, index=0)
    >>> paths
    ['./output/mesh_0000_0']
    """

    name: ClassVar[str] = "PhysicsNeMo Mesh Writer"
    description: ClassVar[str] = "Save meshes in physicsnemo native format (tensordict memmap)"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the mesh sink.

        Returns
        -------
        list[Param]
            The ``output_dir`` parameter (required).
        """
        return [
            Param(name="output_dir", description="Output directory for mesh files", type=str),
        ]

    def __init__(self, output_dir: str) -> None:
        self._output_dir = pathlib.Path(output_dir)

    def __call__(self, items: Generator[Mesh], index: int) -> list[str]:
        """Consume meshes from the stream and save each to disk.

        Parameters
        ----------
        items : Generator[Mesh]
            Stream of meshes to persist.
        index : int
            Source index (used for naming output subdirectories).

        Returns
        -------
        list[str]
            Paths of the saved mesh directories.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []

        for seq, mesh in enumerate(items):
            subdir = self._output_dir / f"mesh_{index:04d}_{seq}"
            mesh.save(str(subdir))
            paths.append(str(subdir))

        return paths
