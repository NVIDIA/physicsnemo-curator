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

"""Zarr writer sink emitting the physicsnemo mesh-zarr schema.

Writes :class:`physicsnemo.mesh.DomainMesh` (and plain
:class:`physicsnemo.mesh.Mesh`) objects as Zarr group trees readable by
``physicsnemo.datapipes.ZarrMeshReader`` -- the chunked, compressed,
random-access counterpart of the ``.pdmsh``/``.pmsh`` memmap output of
:class:`~physicsnemo_curator.domains.mesh.sinks.mesh_writer.MeshSink`.

DomainMesh schema (one Zarr group per case, mirroring the ``.pdmsh`` tree)::

    domain_run_1.zarr/            attrs: format="physicsnemo-domainmesh-zarr"
        global_data/<field>       # case-level metadata (single source of truth)
        interior/                 # mesh-schema subgroup
        boundaries/<name>/        # mesh-schema subgroups

Mesh schema (each mesh-schema group)::

    <group>/                      attrs: format="physicsnemo-mesh-zarr",
        points                    #       layout ("soup"|"indexed"), ...
        cells                     # absent for point clouds
        point_data/<field>
        cell_data/<field>
        global_data/<field>

Data fields mirror the ``TensorDict`` tree: a ``<field>`` is either an array
(a tensor leaf) or a subgroup (a nested ``TensorDict``), recursively. Field
names must be non-empty and must not contain ``"/"`` (the zarr path
separator), and non-tensor leaves (``NonTensorData``, e.g. strings) are not
representable in schema v1; the sink rejects both with clear errors rather
than corrupting the store. Groups are written as zarr format 3.

Chunks are sized along the cell/point axes (``chunk_cells`` /
``chunk_points``) so a training-time contiguous block subsample touches a
bounded number of chunks; align ``chunk_cells`` with the reader's
``subsample_n_cells``. With ``soup_boundaries=True`` boundary meshes are
denormalized to a per-cell vertex soup (~20% more raw bytes) so block reads
become fully contiguous -- recommended for cell-centric surface training on
meshes without point-index locality (e.g. DrivAerML). The ``layout`` attr is
verified from the written cells, never caller-asserted.
"""

from __future__ import annotations

import pathlib
import shutil
import tempfile
import time
from typing import TYPE_CHECKING, ClassVar

from physicsnemo_curator.core.base import Param, Sink
from physicsnemo_curator.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

    from physicsnemo.mesh import DomainMesh, Mesh

    from physicsnemo_curator.core.base import Source

MESH_FORMAT = "physicsnemo-mesh-zarr"
DOMAIN_MESH_FORMAT = "physicsnemo-domainmesh-zarr"
SCHEMA_VERSION = 1


def _to_cell_soup(mesh: Mesh) -> Mesh:
    """Denormalize a mesh to a per-cell vertex soup.

    Points are re-gathered so cell ``i`` references points
    ``[nodes_per_cell * i, nodes_per_cell * (i + 1))``, making every
    contiguous cell-block read a contiguous point-range read. Shared
    vertices (and their ``point_data``) are duplicated; connectivity
    between cells is lost.

    Parameters
    ----------
    mesh : Mesh
        Input mesh (returned unchanged if it has no cells).

    Returns
    -------
    Mesh
        Soup-layout mesh.
    """
    import torch
    from physicsnemo.mesh import Mesh as _Mesh

    if mesh.n_cells == 0:
        return mesh
    cells = mesh.cells
    n_cells, nodes_per_cell = cells.shape
    flat = cells.reshape(-1)
    points = mesh.points[flat].contiguous()
    # TensorDict batch-indexing gathers every leaf (nested included) at once.
    point_data = mesh.point_data[flat].contiguous()
    soup_cells = torch.arange(points.shape[0], dtype=torch.int64).reshape(n_cells, nodes_per_cell)
    return _Mesh(
        points=points,
        cells=soup_cells,
        point_data=point_data,
        cell_data=mesh.cell_data,
        global_data=mesh.global_data,
    )


def _is_soup(mesh: Mesh) -> bool:
    """Return True iff ``cells`` is exactly ``arange(n_points)`` reshaped.

    Parameters
    ----------
    mesh : Mesh
        Mesh to check.

    Returns
    -------
    bool
        Whether the mesh is in verified soup layout.
    """
    import torch

    if mesh.n_cells == 0:
        return False
    cells = mesh.cells
    if cells.numel() != mesh.n_points:
        return False
    expected = torch.arange(cells.numel(), dtype=cells.dtype, device=cells.device)
    return bool(torch.equal(cells.reshape(-1), expected))


class DomainMeshZarrSink(Sink["DomainMesh"]):
    """Write DomainMesh/Mesh objects as physicsnemo mesh-zarr schema groups.

    Each item is written to a separate ``.zarr`` directory: DomainMesh
    items as the DomainMesh schema (case-level ``global_data/`` +
    ``interior/`` + ``boundaries/<name>/``), plain Mesh items as a single
    mesh-schema group. Output is readable by
    ``physicsnemo.datapipes.ZarrMeshReader`` (use ``subpath`` and
    ``merge_global_data_from`` for DomainMesh groups).

    Parameters
    ----------
    output_dir : str
        Directory where Zarr stores will be written.
    compression_level : int
        Blosc/zstd compression level (0 disables compression). Default 3.
    chunk_cells : int
        Chunk length along the cell axis (``cells`` / ``cell_data``);
        align with the training reader's ``subsample_n_cells``.
        Default 200000.
    chunk_points : int
        Chunk length along the point axis (``points`` / ``point_data``).
        Default 600000.
    soup_boundaries : bool
        Denormalize boundary meshes to per-cell vertex soup before
        writing (contiguous block reads; recommended for cell-centric
        surface training). The interior is never souped. Default False.
    naming_template : str | None
        Format string for output names. Placeholders: ``{index}``,
        ``{seq}``, ``{run_id}`` (requires a source exposing
        ``run_id(index)``). Default: ``domain_{index:04d}_{seq}``.

    Examples
    --------
    >>> sink = DomainMeshZarrSink(output_dir="./out/", soup_boundaries=True)  # doctest: +SKIP
    >>> paths = sink(domain_mesh_generator, index=0)  # doctest: +SKIP
    """

    name: ClassVar[str] = "DomainMesh Zarr Writer"
    description: ClassVar[str] = "Write DomainMesh/Mesh objects as physicsnemo mesh-zarr schema Zarr groups"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the sink.

        Returns
        -------
        list[Param]
            Descriptors for configuration parameters.
        """
        return [
            Param(name="output_dir", description="Output directory for Zarr stores", type=str),
            Param(
                name="compression_level",
                description="Blosc/zstd compression level (0 disables compression)",
                type=int,
                default=3,
            ),
            Param(
                name="chunk_cells",
                description="Chunk length along the cell axis (align with reader subsample_n_cells)",
                type=int,
                default=200_000,
            ),
            Param(
                name="chunk_points",
                description="Chunk length along the point axis",
                type=int,
                default=600_000,
            ),
            Param(
                name="soup_boundaries",
                description="Denormalize boundary meshes to per-cell vertex soup for contiguous block reads",
                type=bool,
                default=False,
            ),
            Param(
                name="naming_template",
                description="Format string for output names ({index}, {seq}, {run_id})",
                type=str,
                default=None,
            ),
        ]

    def __init__(
        self,
        output_dir: str,
        compression_level: int = 3,
        chunk_cells: int = 200_000,
        chunk_points: int = 600_000,
        soup_boundaries: bool = False,
        naming_template: str | None = None,
    ) -> None:
        """Initialize the DomainMesh Zarr sink.

        Parameters
        ----------
        output_dir : str
            Directory where Zarr stores will be written.
        compression_level : int
            Blosc/zstd compression level (0 disables compression).
        chunk_cells : int
            Chunk length along the cell axis.
        chunk_points : int
            Chunk length along the point axis.
        soup_boundaries : bool
            Denormalize boundary meshes to per-cell vertex soup.
        naming_template : str | None
            Format string for output names.
        """
        self._log = get_logger(self)
        self._output_dir = pathlib.Path(output_dir)
        self._compression_level = compression_level
        self._chunk_cells = chunk_cells
        self._chunk_points = chunk_points
        self._soup_boundaries = soup_boundaries
        self._naming_template = naming_template
        self._source: Source[DomainMesh] | None = None

    def set_source(self, source: Source[DomainMesh]) -> None:
        """Set the source for resolving naming placeholders.

        Parameters
        ----------
        source : Source[DomainMesh]
            The pipeline source.
        """
        self._source = source

    def _compressors(self):
        """Return the configured Blosc compressor list (or None).

        Returns
        -------
        list | None
            Compressor list for ``zarr.Group.create_array``.
        """
        if self._compression_level <= 0:
            return None
        from zarr.codecs import BloscCodec

        return [BloscCodec(cname="zstd", clevel=self._compression_level, shuffle="bitshuffle")]

    def _write_array(self, target, name: str, tensor, chunk0: int) -> None:
        """Write one tensor as a zarr array with axis-0 chunking.

        Parameters
        ----------
        target : zarr.Group
            Group to create the array in.
        name : str
            Array name.
        tensor : torch.Tensor
            Tensor to write.
        chunk0 : int
            Chunk length along axis 0.
        """
        arr = tensor.detach().cpu().contiguous().numpy()
        # 0-d scalars and empty arrays fall back to zarr's auto chunking.
        chunks = (min(chunk0, arr.shape[0]), *arr.shape[1:]) if arr.ndim >= 1 and arr.shape[0] > 0 else "auto"
        z = target.create_array(
            name,
            shape=arr.shape,
            dtype=arr.dtype,
            chunks=chunks,
            compressors=self._compressors(),
        )
        if arr.size:
            z[...] = arr

    def _write_tensordict(self, group, td, chunk0: int) -> None:
        """Write a TensorDict as a zarr group tree, mirroring its nesting.

        Tensor leaves become arrays; nested TensorDicts become subgroups,
        recursively. Rejects field names containing ``"/"`` (zarr's path
        separator -- it would silently create hierarchy instead of a field)
        and non-tensor leaves (``NonTensorData`` is not representable in
        schema v1; an attrs-based encoding is reserved for v2).

        Parameters
        ----------
        group : zarr.Group
            Group to write into.
        td : tensordict.TensorDictBase
            TensorDict whose tree is mirrored into ``group``.
        chunk0 : int
            Chunk length along axis 0 for all leaf arrays.
        """
        import torch
        from tensordict import TensorDictBase

        # Not SIM118: iterating a TensorDict directly walks its batch
        # dimension, not its keys -- .keys() is semantically required.
        for key in td.keys():  # noqa: SIM118
            if not key or "/" in key:
                msg = (
                    f"field name {key!r} is not storable in the mesh-zarr schema: "
                    f"names must be non-empty and must not contain '/' (the zarr "
                    f"path separator). Rename the field before saving."
                )
                raise ValueError(msg)
            value = td.get(key)
            if isinstance(value, TensorDictBase):
                self._write_tensordict(group.create_group(key), value, chunk0)
            elif isinstance(value, torch.Tensor):
                self._write_array(group, key, value, chunk0)
            else:
                msg = (
                    f"field {key!r} has unsupported type {type(value).__name__}; "
                    f"mesh-zarr schema v1 stores tensors and nested TensorDicts "
                    f"only (NonTensorData / strings are reserved for v2)."
                )
                raise TypeError(msg)

    def _write_mesh_group(self, group, mesh: Mesh) -> None:
        """Write one mesh-schema group (arrays + verified attrs).

        Parameters
        ----------
        group : zarr.Group
            Target group.
        mesh : Mesh
            Mesh to serialize.
        """
        self._write_array(group, "points", mesh.points, self._chunk_points)
        if mesh.n_cells > 0:
            self._write_array(group, "cells", mesh.cells, self._chunk_cells)

        for sub_name, data, chunk0 in (
            ("point_data", mesh.point_data, self._chunk_points),
            ("cell_data", mesh.cell_data, self._chunk_cells),
            ("global_data", mesh.global_data, self._chunk_cells),
        ):
            if not len(data.keys()):
                continue
            self._write_tensordict(group.create_group(sub_name), data, chunk0)

        group.attrs["format"] = MESH_FORMAT
        group.attrs["schema_version"] = SCHEMA_VERSION
        group.attrs["n_points"] = int(mesh.n_points)
        group.attrs["n_cells"] = int(mesh.n_cells)
        if mesh.n_cells > 0:
            group.attrs["nodes_per_cell"] = int(mesh.cells.shape[1])
            # Verified, not caller-asserted: readers rely on it to skip
            # the cells read and synthesize indices.
            group.attrs["layout"] = "soup" if _is_soup(mesh) else "indexed"

    def _write_domain_mesh(self, root, domain: DomainMesh) -> None:
        """Write a DomainMesh-schema tree into an open root group.

        Parameters
        ----------
        root : zarr.Group
            Root group of the case.
        domain : DomainMesh
            Domain mesh to serialize.
        """
        root.attrs["format"] = DOMAIN_MESH_FORMAT
        root.attrs["schema_version"] = SCHEMA_VERSION
        root.attrs["boundary_names"] = list(domain.boundary_names)

        if len(domain.global_data.keys()):
            self._write_tensordict(root.create_group("global_data"), domain.global_data, self._chunk_cells)

        self._write_mesh_group(root.create_group("interior"), domain.interior)
        boundaries = root.create_group("boundaries")
        for bname in domain.boundary_names:
            mesh = domain.boundaries[bname]
            if self._soup_boundaries:
                mesh = _to_cell_soup(mesh)
            self._write_mesh_group(boundaries.create_group(bname), mesh)

    def __call__(self, items: Iterator[DomainMesh], index: int) -> list[str]:
        """Write meshes to Zarr stores.

        Parameters
        ----------
        items : Iterator[DomainMesh]
            Stream of DomainMesh (or Mesh) objects to write.
        index : int
            Pipeline source index.

        Returns
        -------
        list[str]
            Paths of the written Zarr stores.
        """
        import zarr
        from physicsnemo.mesh.domain_mesh import DomainMesh as _DomainMesh

        t0 = time.perf_counter()
        self._log.info("Writing index %d to mesh-zarr", index)

        run_id: int | str = ""
        has_run_id = self._source is not None and hasattr(self._source, "run_id")
        if has_run_id:
            run_id = self._source.run_id(index)  # ty: ignore[unresolved-attribute]
        if self._naming_template is not None and "{run_id" in self._naming_template and not has_run_id:
            msg = "Naming template uses {run_id} but the source does not expose a run_id(index) method."
            raise ValueError(msg)

        paths: list[str] = []
        for seq, mesh in enumerate(items):
            if self._naming_template is not None:
                name = self._naming_template.format(index=index, seq=seq, run_id=run_id)
            else:
                name = f"domain_{index:04d}_{seq}"
            if not name.endswith(".zarr"):
                name = f"{name}.zarr"

            output_path = self._output_dir / name
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to a temp directory in the same parent, then atomically
            # rename into place so an interrupted write never leaves a
            # partial/corrupt store behind.
            tmp_dir = tempfile.mkdtemp(prefix=".tmp_", dir=str(output_path.parent))
            try:
                # The schema requires zarr format 3 (zarr.json metadata);
                # pin it explicitly so a zarr.config default override
                # cannot silently produce a format-2 store.
                root = zarr.open_group(tmp_dir, mode="w", zarr_format=3)
                if isinstance(mesh, _DomainMesh):
                    self._write_domain_mesh(root, mesh)
                else:
                    self._write_mesh_group(root, mesh)
                if output_path.exists():
                    shutil.rmtree(output_path)
                pathlib.Path(tmp_dir).replace(output_path)
            except BaseException:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise

            paths.append(str(output_path))
            self._log.debug("Wrote %s to %s", type(mesh).__name__, output_path)

        self._log.info("Write complete: %d stores (%.2fs)", len(paths), time.perf_counter() - t0)
        return paths
