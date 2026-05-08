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

"""Zarr writer sink for PhysicsNeMo Mesh objects.

Writes :class:`physicsnemo.mesh.Mesh` objects to Zarr stores, producing
output compatible with deep learning workflows.  Each mesh is written
to a separate ``.zarr`` directory containing:

* ``mesh_pos`` — positions across all timesteps ``(T, N, 3)``
* ``edges`` — edge connectivity ``(E, 2)``
* ``thickness`` — per-node thickness ``(N,)``
* Additional point/cell data fields

The sink reconstructs ``mesh_pos`` from displacement fields
(``displacement_t{idx:03d}``) and the reference ``points`` tensor.
"""

from __future__ import annotations

import pathlib
import re
import time
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import zarr
from zarr.codecs import BloscCodec

from physicsnemo_curator.core.base import Param, Sink
from physicsnemo_curator.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

    from physicsnemo.mesh import Mesh

    from physicsnemo_curator.core.base import Source


def _compute_chunks(shape: tuple[int, ...], dtype: np.dtype, target_mb: float) -> tuple[int, ...]:
    """Compute chunk sizes targeting a given memory footprint.

    Parameters
    ----------
    shape : tuple[int, ...]
        Array shape.
    dtype : np.dtype
        Array data type.
    target_mb : float
        Target chunk size in megabytes.

    Returns
    -------
    tuple[int, ...]
        Chunk sizes per dimension.
    """
    item_size = dtype.itemsize
    target_bytes = int(target_mb * 1024 * 1024)
    ndim = len(shape)

    if ndim == 1:
        # 1D: chunk along the single dimension
        chunk_size = max(1, min(shape[0], target_bytes // item_size))
        return (chunk_size,)

    if ndim == 2:
        # 2D: keep full width, chunk rows
        row_bytes = shape[1] * item_size
        chunk_rows = max(1, min(shape[0], target_bytes // row_bytes))
        return (chunk_rows, shape[1])

    if ndim == 3:
        # 3D (T, N, D): preserve last dim, chunk T and N
        last_dim = shape[2]
        slice_bytes = shape[1] * last_dim * item_size

        if slice_bytes <= target_bytes:
            # Can fit full (N, D) slices - chunk only T
            chunk_t = max(1, min(shape[0], target_bytes // slice_bytes))
            return (chunk_t, shape[1], last_dim)
        else:
            # Need to chunk N as well
            row_bytes = last_dim * item_size
            chunk_n = max(1, target_bytes // row_bytes)
            return (1, min(shape[1], chunk_n), last_dim)

    # Fallback for 4D+: chunk first dimension adaptively
    rest_bytes = int(np.prod(shape[1:])) * item_size
    chunk_first = max(1, min(shape[0], target_bytes // rest_bytes))
    return (chunk_first,) + shape[1:]


class MeshZarrSink(Sink["Mesh"]):
    """Write :class:`~physicsnemo.mesh.Mesh` objects to Zarr stores.

    Each mesh is written to a separate ``.zarr`` directory.  The sink
    reconstructs position trajectories from displacement fields and
    stores them as ``mesh_pos``.  Edge connectivity is read from
    ``global_data["edges"]`` (computed by
    :class:`~physicsnemo_curator.domains.mesh.filters.edge_compute.EdgeComputeFilter`).

    Parameters
    ----------
    output_dir : str
        Directory where Zarr stores will be written.
    compression_level : int
        Blosc/zstd compression level (0-9).  Default is 3.
    chunk_size_mb : float
        Target chunk size in megabytes.  Default is 1.0.
    naming_template : str | None
        Format string for output names.  Placeholders: ``{index}``,
        ``{seq}``.  Default: ``mesh_{index:04d}``.

    Examples
    --------
    >>> sink = MeshZarrSink(output_dir="./output/")  # doctest: +SKIP
    >>> paths = sink(mesh_generator, index=0)  # doctest: +SKIP
    >>> paths  # doctest: +SKIP
    ['./output/mesh_0000.zarr']
    """

    name: ClassVar[str] = "Mesh Zarr Writer"
    description: ClassVar[str] = "Write Mesh objects to Zarr stores with configurable compression"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the Zarr sink.

        Returns
        -------
        list[Param]
            Descriptors for configuration parameters.
        """
        return [
            Param(name="output_dir", description="Output directory for Zarr stores", type=str),
            Param(
                name="compression_level",
                description="Blosc/zstd compression level (0-9)",
                type=int,
                default=3,
            ),
            Param(
                name="chunk_size_mb",
                description="Target chunk size in megabytes",
                type=float,
                default=1.0,
            ),
            Param(
                name="naming_template",
                description="Format string for output names ({index}, {seq})",
                type=str,
                default=None,
            ),
        ]

    def __init__(
        self,
        output_dir: str,
        compression_level: int = 3,
        chunk_size_mb: float = 1.0,
        naming_template: str | None = None,
    ) -> None:
        """Initialize the Zarr sink.

        Parameters
        ----------
        output_dir : str
            Directory where Zarr stores will be written.
        compression_level : int
            Blosc/zstd compression level (0-9).
        chunk_size_mb : float
            Target chunk size in megabytes.
        naming_template : str | None
            Format string for output names.
        """
        self._log = get_logger(self)
        self._output_dir = pathlib.Path(output_dir)
        self._compression_level = compression_level
        self._chunk_size_mb = chunk_size_mb
        self._naming_template = naming_template or "mesh_{index:04d}"
        self._source: Source[Mesh] | None = None

        if chunk_size_mb < 0.1:
            self._log.warning("chunk_size_mb=%.2f is very small, may cause performance issues", chunk_size_mb)
        if chunk_size_mb > 100.0:
            self._log.warning("chunk_size_mb=%.2f is very large, may reduce compression efficiency", chunk_size_mb)

    def set_source(self, source: Source[Mesh]) -> None:
        """Set the source for resolving naming placeholders.

        Parameters
        ----------
        source : Source[Mesh]
            The pipeline source.
        """
        self._source = source

    def __call__(self, items: Iterator[Mesh], index: int) -> list[str]:
        """Write meshes to Zarr stores.

        Parameters
        ----------
        items : Iterator[Mesh]
            Stream of Mesh objects to write.
        index : int
            Pipeline source index.

        Returns
        -------
        list[str]
            Paths of the written Zarr stores.
        """
        t0 = time.perf_counter()
        self._log.info("Writing index %d to Zarr", index)

        paths: list[str] = []

        for seq, mesh in enumerate(items):
            output_name = self._naming_template.format(index=index, seq=seq)
            if not output_name.endswith(".zarr"):
                output_name = f"{output_name}.zarr"

            output_path = self._output_dir / output_name
            self._write_mesh(mesh, output_path)
            paths.append(str(output_path))
            self._log.debug("Wrote mesh to %s", output_path)

        self._log.info("Write complete: %d files (%.2fs)", len(paths), time.perf_counter() - t0)
        return paths

    def _write_mesh(self, mesh: Mesh, output_path: pathlib.Path) -> None:
        """Write a single mesh to a Zarr store.

        Parameters
        ----------
        mesh : Mesh
            The mesh to write.
        output_path : pathlib.Path
            Path to the output Zarr store.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create compressor using zarr v3 codec
        compressors = [BloscCodec(cname="zstd", clevel=self._compression_level, shuffle="shuffle")]

        # Open/create Zarr store
        store = zarr.open_group(str(output_path), mode="w")

        # Extract displacement fields and reconstruct mesh_pos
        disp_keys = self._get_displacement_keys(mesh)
        num_timesteps = len(disp_keys)

        if num_timesteps > 0:
            mesh_pos = self._reconstruct_mesh_pos(mesh, disp_keys)
        else:
            # No displacement fields - just use points as single timestep
            mesh_pos = mesh.points.numpy()[np.newaxis, ...]  # (1, N, 3)
            num_timesteps = 1

        # Write mesh_pos
        mesh_pos = mesh_pos.astype(np.float32)
        chunks = _compute_chunks(mesh_pos.shape, mesh_pos.dtype, self._chunk_size_mb)
        store.create_array("mesh_pos", data=mesh_pos, chunks=chunks, compressors=compressors)

        # Write edges from global_data
        if mesh.global_data is not None and "edges" in mesh.global_data.keys():  # noqa: SIM118
            edges = mesh.global_data.get("edges").numpy().astype(np.int64)
            edge_chunks = _compute_chunks(edges.shape, edges.dtype, self._chunk_size_mb)
            store.create_array("edges", data=edges, chunks=edge_chunks, compressors=compressors)
            num_edges = edges.shape[0]
        else:
            num_edges = 0
            self._log.warning("No edges found in global_data - use EdgeComputeFilter before this sink")

        # Write thickness (zeros if not present)
        n_points = mesh.n_points
        if mesh.point_data is not None and "thickness" in mesh.point_data.keys():  # noqa: SIM118
            thickness = mesh.point_data.get("thickness").numpy().astype(np.float32)
        else:
            thickness = np.zeros(n_points, dtype=np.float32)

        thickness_chunks = _compute_chunks(thickness.shape, thickness.dtype, self._chunk_size_mb)
        store.create_array("thickness", data=thickness, chunks=thickness_chunks, compressors=compressors)

        # Write other point_data fields (excluding displacement_t* which are in mesh_pos)
        if mesh.point_data is not None:
            for key in mesh.point_data.keys():  # noqa: SIM118
                key_str = str(key)
                if key_str.startswith("displacement_t") or key_str == "thickness":
                    continue
                data = mesh.point_data.get(key).numpy().astype(np.float32)
                data_chunks = _compute_chunks(data.shape, data.dtype, self._chunk_size_mb)
                store.create_array(key_str, data=data, chunks=data_chunks, compressors=compressors)

        # Write cell_data fields
        if mesh.cell_data is not None:
            for key in mesh.cell_data.keys():  # noqa: SIM118
                key_str = str(key)
                data = mesh.cell_data.get(key).numpy().astype(np.float32)
                data_chunks = _compute_chunks(data.shape, data.dtype, self._chunk_size_mb)
                store.create_array(f"cell_{key_str}", data=data, chunks=data_chunks, compressors=compressors)

        # Write metadata attributes
        store.attrs["num_timesteps"] = num_timesteps
        store.attrs["num_nodes"] = n_points
        store.attrs["num_edges"] = num_edges
        store.attrs["thickness_min"] = float(np.min(thickness))
        store.attrs["thickness_max"] = float(np.max(thickness))
        store.attrs["thickness_mean"] = float(np.mean(thickness))

    def _get_displacement_keys(self, mesh: Mesh) -> list[str]:
        """Extract sorted displacement field keys from mesh point_data.

        Parameters
        ----------
        mesh : Mesh
            The mesh to inspect.

        Returns
        -------
        list[str]
            Sorted displacement keys (e.g., ["displacement_t000", ...]).
        """
        if mesh.point_data is None:
            return []

        pattern = re.compile(r"^displacement_t(\d+)$")
        keys = []
        for key in mesh.point_data.keys():  # noqa: SIM118
            key_str = str(key)
            if pattern.match(key_str):
                keys.append(key_str)

        return sorted(keys)

    def _reconstruct_mesh_pos(self, mesh: Mesh, disp_keys: list[str]) -> np.ndarray:
        """Reconstruct position trajectories from displacement fields.

        Parameters
        ----------
        mesh : Mesh
            The mesh with displacement fields.
        disp_keys : list[str]
            Sorted displacement field keys.

        Returns
        -------
        np.ndarray
            Position array of shape (T, N, 3).
        """
        points = mesh.points.numpy()  # (N, 3) reference positions
        n_timesteps = len(disp_keys)
        n_points = points.shape[0]

        mesh_pos = np.zeros((n_timesteps, n_points, 3), dtype=np.float32)

        for t, key in enumerate(disp_keys):
            disp = mesh.point_data[key].numpy()  # (N, 3)
            mesh_pos[t] = points + disp

        return mesh_pos

    @property
    def output_dir(self) -> pathlib.Path:
        """Return the output directory path."""
        return self._output_dir
