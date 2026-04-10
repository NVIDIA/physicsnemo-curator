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

import logging
import pathlib
from typing import TYPE_CHECKING, ClassVar

from physicsnemo_curator.core.base import Param, Sink

if TYPE_CHECKING:
    from collections.abc import Iterator

    from physicsnemo.mesh import Mesh

    from physicsnemo_curator.core.base import Source

logger = logging.getLogger(__name__)


class MeshSink(Sink["Mesh"]):
    """Write :class:`~physicsnemo.mesh.Mesh` objects to disk.

    Each mesh is saved to a subdirectory of *output_dir* using the physicsnemo
    native format (:meth:`Mesh.save`).  By default the subdirectory is named
    ``mesh_{index:04d}_{seq}``, but a custom *naming_template* can be provided
    to control output names — for example, to produce filenames that match the
    patterns expected by ``MeshReader`` in PhysicsNeMo.

    When the pipeline's source exposes a ``relative_path(index)`` method
    (e.g. :class:`~physicsnemo_curator.mesh.sources.vtk.VTKSource`), two
    additional template placeholders become available:

    * ``{relpath}`` — the parent directory of the source file relative to
      the source root (e.g. ``sim_a/subdir`` for a file at
      ``<root>/sim_a/subdir/mesh.vtu``).  Empty string when the file lives
      directly in the root.
    * ``{stem}`` — the filename stem of the source file (without extension,
      e.g. ``mesh`` for ``mesh.vtu``).

    These are wired automatically by the pipeline — no need to pass the
    source manually.  This enables **directory-mirroring** workflows where
    the output dataset reproduces the nested folder layout of the input.

    Parameters
    ----------
    output_dir : str
        Directory where mesh outputs will be written.
    naming_template : str or None
        Python format string used to generate subdirectory names.  The
        placeholders ``{index}`` (source index) and ``{seq}`` (sequence
        number within the source, starting at 0) are always available.
        When the source supports it, ``{relpath}`` and ``{stem}``
        are also available.  Standard format-spec syntax is supported
        (e.g. ``{index:04d}``).  When ``None`` (the default) the
        built-in pattern ``mesh_{index:04d}_{seq}`` is used.

    Examples
    --------
    Default naming:

    >>> sink = MeshSink(output_dir="./output/")  # doctest: +SKIP
    >>> paths = sink(mesh_generator, index=0)  # doctest: +SKIP
    >>> paths  # doctest: +SKIP
    ['./output/mesh_0000_0']

    Custom naming for compatibility with ``MeshReader``:

    >>> sink = MeshSink(  # doctest: +SKIP
    ...     output_dir="./output/",
    ...     naming_template="boundary_{index}.vtp.pmsh",
    ... )
    >>> paths = sink(mesh_generator, index=0)  # doctest: +SKIP
    >>> paths  # doctest: +SKIP
    ['./output/boundary_0.vtp.pmsh']

    Directory mirroring (preserves input folder structure, wired by pipeline):

    >>> pipeline = (  # doctest: +SKIP
    ...     VTKSource("./input/")
    ...     .write(MeshSink(output_dir="./output/", naming_template="{relpath}/{stem}"))
    ... )
    """

    name: ClassVar[str] = "PhysicsNeMo Mesh Writer"
    description: ClassVar[str] = "Save meshes in physicsnemo native format (tensordict memmap)"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the mesh sink.

        Returns
        -------
        list[Param]
            The ``output_dir`` and optional ``naming_template`` parameters.
        """
        return [
            Param(name="output_dir", description="Output directory for mesh files", type=str),
            Param(
                name="naming_template",
                description=(
                    "Format string for output names. "
                    "Placeholders: {index}, {seq}, {relpath}, {stem}. "
                    "Default: mesh_{index:04d}_{seq}"
                ),
                type=str,
                default=None,
            ),
        ]

    def __init__(
        self,
        output_dir: str,
        naming_template: str | None = None,
    ) -> None:
        """Initialise the mesh sink.

        Parameters
        ----------
        output_dir : str
            Directory where mesh outputs will be written.
        naming_template : str or None
            Python format string for subdirectory names.  See the class
            docstring for details.

        Raises
        ------
        ValueError
            If *naming_template* contains invalid placeholders.
        """
        self._output_dir = pathlib.Path(output_dir)
        self._naming_template = naming_template
        self._source: Source[Mesh] | None = None

        # Validate the template eagerly so users get a clear error at
        # construction time rather than deep inside a pipeline run.
        if naming_template is not None:
            try:
                naming_template.format(index=0, seq=0, relpath="test", stem="test")
            except (KeyError, IndexError, ValueError) as exc:
                msg = (
                    f"Invalid naming_template {naming_template!r}: {exc}. "
                    "Only {index}, {seq}, {relpath}, and {stem} placeholders are supported."
                )
                raise ValueError(msg) from exc

    def set_source(self, source: Source[Mesh]) -> None:
        """Inject the pipeline source for ``{relpath}``/``{stem}`` resolution.

        Called automatically by the :class:`~physicsnemo_curator.core.base.Pipeline`
        when the sink is attached via :meth:`Pipeline.write`.

        Parameters
        ----------
        source : Source[Mesh]
            The pipeline source.  If it exposes a ``relative_path(index)``
            method, the sink will use it to resolve naming placeholders.
        """
        self._source = source

    def __call__(self, items: Iterator[Mesh], index: int) -> list[str]:
        """Consume meshes from the stream and save each to disk.

        Parameters
        ----------
        items : Iterator[Mesh]
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

        # Resolve relpath / stem from source if available.
        relpath = ""
        stem = ""
        if self._source is not None and hasattr(self._source, "relative_path"):
            rel = self._source.relative_path(index)  # ty: ignore[call-non-callable]
            rel_path = pathlib.PurePosixPath(rel)
            stem = rel_path.stem
            relpath = str(rel_path.parent) if str(rel_path.parent) != "." else ""

        for seq, mesh in enumerate(items):
            if self._naming_template is not None:
                name = self._naming_template.format(
                    index=index,
                    seq=seq,
                    relpath=relpath,
                    stem=stem,
                )
            else:
                name = f"mesh_{index:04d}_{seq}"

            subdir = self._output_dir / name
            subdir.parent.mkdir(parents=True, exist_ok=True)
            mesh.save(str(subdir))  # ty: ignore[unresolved-attribute]  # @tensorclass adds .save() dynamically
            logger.debug("Saved mesh to %s", subdir)
            paths.append(str(subdir))

        return paths
