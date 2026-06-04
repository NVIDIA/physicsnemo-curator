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

"""Structured-grid sidecar sink.

Persists structured-grid :class:`tensordict.TensorDict` objects (produced by
:class:`~physicsnemo_curator.domains.mesh.sources.vti.VTISource`) to disk as
tensordict memmap directories.  Output is written as a **sidecar**: by
default the source's directory layout is mirrored under *output_dir* using
the ``{relpath}/{stem}`` naming, so each grid lands next to the mesh
(``.pmsh`` / ``.pdmsh``) outputs for the same sample.

Reload with :meth:`tensordict.TensorDict.load_memmap`.
"""

from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING, ClassVar

from physicsnemo_curator.core.base import Param, Sink

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tensordict import TensorDict

    from physicsnemo_curator.core.base import Source

logger = logging.getLogger(__name__)

#: Default output suffix for grid sidecar directories.
_GRID_SUFFIX = ".grid"


class GridSidecarSink(Sink["TensorDict"]):
    """Write structured-grid TensorDicts as tensordict memmap sidecars.

    Each grid is written to a ``<name>.grid`` directory under *output_dir*.
    When the pipeline source exposes ``relative_path(index)`` (e.g.
    :class:`~physicsnemo_curator.domains.mesh.sources.vti.VTISource`), the
    default ``{relpath}/{stem}`` template mirrors the input layout so the
    grid sits beside the corresponding mesh outputs.

    Parameters
    ----------
    output_dir : str
        Directory where grid sidecars will be written.
    naming_template : str or None
        Format string for output names.  Placeholders: ``{index}``,
        ``{seq}``, ``{relpath}``, ``{stem}``.  Default ``"{relpath}/{stem}"``
        when the source supports it, else ``"grid_{index:04d}_{seq}"``.
    group_readable : bool
        If ``True``, ``chmod g+r`` the written output.

    Examples
    --------
    >>> pipeline = VTISource("./grids/").write(GridSidecarSink(output_dir="./out/"))  # doctest: +SKIP
    """

    name: ClassVar[str] = "Grid Sidecar Writer"
    description: ClassVar[str] = "Write structured-grid TensorDicts as tensordict memmap sidecars"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the grid sidecar sink.

        Returns
        -------
        list[Param]
            The ``output_dir``, ``naming_template``, and ``group_readable``
            parameters.
        """
        return [
            Param(name="output_dir", description="Output directory for grid sidecars", type=str),
            Param(
                name="naming_template",
                description="Format string for output names ({index}, {seq}, {relpath}, {stem})",
                type=str,
                default=None,
            ),
            Param(
                name="group_readable",
                description="Make written outputs group-readable (chmod g+r)",
                type=bool,
                default=False,
            ),
        ]

    def __init__(
        self,
        output_dir: str,
        naming_template: str | None = None,
        group_readable: bool = False,
    ) -> None:
        from physicsnemo_curator.core.logging import get_logger

        self._output_dir = pathlib.Path(output_dir)
        self._naming_template = naming_template
        self._group_readable = group_readable
        self._source: Source[TensorDict] | None = None
        self._log = get_logger(self)

        if naming_template is not None:
            try:
                naming_template.format(index=0, seq=0, relpath="test", stem="test")
            except (KeyError, IndexError, ValueError) as exc:
                msg = (
                    f"Invalid naming_template {naming_template!r}: {exc}. "
                    "Only {index}, {seq}, {relpath}, and {stem} placeholders are supported."
                )
                raise ValueError(msg) from exc

    def set_source(self, source: Source[TensorDict]) -> None:
        """Inject the pipeline source for placeholder resolution.

        Parameters
        ----------
        source : Source[TensorDict]
            The pipeline source.  ``relative_path(index)`` is used when
            available to resolve ``{relpath}`` / ``{stem}``.
        """
        self._source = source

    def __call__(self, items: Iterator[TensorDict], index: int) -> list[str]:
        """Write structured-grid TensorDicts to memmap sidecars.

        Parameters
        ----------
        items : Iterator[TensorDict]
            Stream of grid TensorDicts to persist.
        index : int
            Source index (used for naming).

        Returns
        -------
        list[str]
            Paths of the written sidecar directories.
        """
        import shutil
        import tempfile
        import time

        self._output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []

        relpath = ""
        stem = ""
        has_relpath = self._source is not None and hasattr(self._source, "relative_path")
        if has_relpath and self._source is not None:
            rel = self._source.relative_path(index)  # ty: ignore[unresolved-attribute]
            rel_path = pathlib.PurePosixPath(rel)
            stem = rel_path.stem
            relpath = str(rel_path.parent) if str(rel_path.parent) != "." else ""

        for seq, grid in enumerate(items):
            if self._naming_template is not None:
                name = self._naming_template.format(index=index, seq=seq, relpath=relpath, stem=stem)
            elif has_relpath:
                name = f"{relpath}/{stem}" if relpath else stem
            else:
                name = f"grid_{index:04d}_{seq}"

            if not name.endswith(_GRID_SUFFIX):
                name = name + _GRID_SUFFIX

            subdir = self._output_dir / name
            subdir.parent.mkdir(parents=True, exist_ok=True)

            t0 = time.perf_counter()
            tmp_dir = tempfile.mkdtemp(prefix=".tmp_", dir=str(subdir.parent))
            try:
                grid.memmap(tmp_dir)
                if subdir.exists():
                    shutil.rmtree(subdir)
                pathlib.Path(tmp_dir).replace(subdir)
            except BaseException:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise

            if self._group_readable:
                from physicsnemo_curator.domains.mesh.sinks.mesh_writer import _fix_group_readable

                _fix_group_readable(subdir)

            self._log.debug("idx_%d: Saved grid to %s (%.2fs)", index, subdir, time.perf_counter() - t0)
            paths.append(str(subdir))

        return paths

    @property
    def output_dir(self) -> pathlib.Path:
        """Return the output directory path."""
        return self._output_dir
