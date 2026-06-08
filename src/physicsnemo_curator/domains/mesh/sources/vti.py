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

"""Structured-grid (VTK ImageData / ``.vti``) source for mesh pipelines.

VTK ImageData describes a uniform rectilinear grid that is fully determined
by ``origin``, ``spacing``, and ``dimensions`` — there is no explicit point
list or connectivity.  Forcing such data through the unstructured
:class:`physicsnemo.mesh.Mesh` model would materialise redundant coordinates
and discard the regular lattice that grid models (CNN/U-Net/FNO) rely on.

Instead, this source reads each ``.vti`` file into a
:class:`tensordict.TensorDict` of **dense N-D field tensors**:

* ``point_data`` — a sub-TensorDict with ``batch_size = [Nz, Ny, Nx]``; each
  scalar field has shape ``(Nz, Ny, Nx)`` and each vector field
  ``(Nz, Ny, Nx, C)`` (VTK x-fastest ordering).
* ``cell_data`` — a sub-TensorDict with ``batch_size = [Cz, Cy, Cx]`` where
  ``Ci = max(dim_i - 1, 1)``.
* ``grid`` — a non-batched sub-TensorDict carrying ``origin`` (3,),
  ``spacing`` (3,), ``dimensions`` (3,, point counts), and ``direction``
  (3, 3).

The result is written as a tensordict memmap **sidecar** beside the mesh
outputs via
:class:`~physicsnemo_curator.domains.mesh.sinks.grid_sidecar.GridSidecarSink`.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from tensordict import TensorDict

from physicsnemo_curator.core.base import Param, Source

if TYPE_CHECKING:
    from collections.abc import Generator

#: File extension recognised as VTK ImageData.
_VTI_EXTENSIONS: frozenset[str] = frozenset({".vti"})


def _reshape_field(arr: np.ndarray, grid_shape: tuple[int, int, int]) -> torch.Tensor:
    """Reshape a flat VTK field array into a dense ``(Nz, Ny, Nx[, C])`` tensor.

    VTK stores ImageData with the x index varying fastest, then y, then z, so
    a C-order reshape into ``(Nz, Ny, Nx)`` recovers ``[z, y, x]`` indexing.

    Parameters
    ----------
    arr : numpy.ndarray
        Flat field array of shape ``(N,)`` (scalar) or ``(N, C)`` (vector).
    grid_shape : tuple[int, int, int]
        ``(Nz, Ny, Nx)`` target grid shape.

    Returns
    -------
    torch.Tensor
        Dense tensor of shape ``(Nz, Ny, Nx)`` or ``(Nz, Ny, Nx, C)``.
    """
    arr = np.ascontiguousarray(arr)
    nz, ny, nx = grid_shape
    reshaped = arr.reshape(nz, ny, nx) if arr.ndim == 1 else arr.reshape(nz, ny, nx, arr.shape[1])
    return torch.from_numpy(np.ascontiguousarray(reshaped))


def imagedata_to_griddict(image: Any, *, fp32: bool = False) -> TensorDict:
    """Convert a PyVista ImageData object to a structured-grid TensorDict.

    Parameters
    ----------
    image : pyvista.ImageData
        The loaded ImageData (uniform grid).
    fp32 : bool
        If ``True``, downcast float64 field/geometry arrays to float32.

    Returns
    -------
    TensorDict
        A scalar-batch TensorDict with ``grid`` metadata and (when present)
        ``point_data`` / ``cell_data`` sub-TensorDicts of dense field
        tensors.
    """
    nx, ny, nz = (int(d) for d in image.dimensions)
    point_shape = (nz, ny, nx)
    cell_shape = (max(nz - 1, 1), max(ny - 1, 1), max(nx - 1, 1))

    float_dtype = torch.float32 if fp32 else torch.float64

    def _maybe_cast(t: torch.Tensor) -> torch.Tensor:
        if fp32 and t.dtype == torch.float64:
            return t.float()
        return t

    point_fields: dict[str, torch.Tensor] = {}
    for name in image.point_data:
        point_fields[str(name)] = _maybe_cast(_reshape_field(np.asarray(image.point_data[name]), point_shape))

    cell_fields: dict[str, torch.Tensor] = {}
    for name in image.cell_data:
        cell_fields[str(name)] = _maybe_cast(_reshape_field(np.asarray(image.cell_data[name]), cell_shape))

    direction = getattr(image, "direction_matrix", None)
    if direction is None:
        direction_t = torch.eye(3, dtype=float_dtype)
    else:
        direction_t = torch.as_tensor(np.asarray(direction).reshape(3, 3), dtype=float_dtype)

    grid_meta = TensorDict(
        {
            "origin": torch.as_tensor(np.asarray(image.origin), dtype=float_dtype),
            "spacing": torch.as_tensor(np.asarray(image.spacing), dtype=float_dtype),
            "dimensions": torch.tensor([nx, ny, nz], dtype=torch.int64),
            "direction": direction_t,
        },
        batch_size=[],
    )

    contents: dict[str, TensorDict] = {"grid": grid_meta}
    if point_fields:
        contents["point_data"] = TensorDict(point_fields, batch_size=list(point_shape))
    if cell_fields:
        contents["cell_data"] = TensorDict(cell_fields, batch_size=list(cell_shape))

    return TensorDict(contents, batch_size=[])


class VTISource(Source[TensorDict]):
    """Read local VTK ImageData (``.vti``) files as structured-grid TensorDicts.

    Each discovered ``.vti`` file is converted (via
    :func:`imagedata_to_griddict`) into a :class:`tensordict.TensorDict` of
    dense N-D field tensors plus grid metadata, suitable for memmap storage
    as a sidecar alongside mesh outputs.

    Parameters
    ----------
    input_path : str
        Path to a local directory containing ``.vti`` files, or a single
        ``.vti`` file.
    file_pattern : str
        Glob pattern for filtering files inside a directory.  Defaults to
        ``"**/*"`` (recursive).
    fp32 : bool
        If ``True``, downcast float64 arrays to float32.

    Examples
    --------
    >>> source = VTISource("./grids/")  # doctest: +SKIP
    >>> grid = next(source[0])  # doctest: +SKIP
    >>> grid["point_data"].batch_size  # doctest: +SKIP
    torch.Size([64, 64, 64])
    """

    name: ClassVar[str] = "VTI Reader"
    description: ClassVar[str] = "Read VTK ImageData (.vti) as dense structured-grid TensorDicts"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the VTI source.

        Returns
        -------
        list[Param]
            The ``input_path``, ``file_pattern``, and ``fp32`` parameters.
        """
        return [
            Param(name="input_path", description="Path to .vti file or directory (local)", type=str),
            Param(name="file_pattern", description="Glob pattern for filtering files", type=str, default="**/*"),
            Param(name="fp32", description="Downcast float64 arrays to float32", type=bool, default=False),
        ]

    def __init__(self, input_path: str, file_pattern: str = "**/*", *, fp32: bool = False) -> None:
        self._fp32 = fp32

        root = pathlib.Path(input_path)
        if root.is_file():
            if root.suffix.lower() not in _VTI_EXTENSIONS:
                msg = f"File {root} does not have a recognised VTI extension {sorted(_VTI_EXTENSIONS)}."
                raise ValueError(msg)
            self._root = root.parent
            self._files: list[pathlib.Path] = [root.resolve()]
            return

        if not root.is_dir():
            msg = f"Path {root} is not a file or directory."
            raise FileNotFoundError(msg)

        self._root = root.resolve()
        discovered = sorted(
            p.resolve() for p in root.glob(file_pattern) if p.is_file() and p.suffix.lower() in _VTI_EXTENSIONS
        )
        if not discovered:
            msg = f"No .vti files found in {root} with pattern {file_pattern!r}."
            raise ValueError(msg)
        self._files = discovered

    def __len__(self) -> int:
        """Return the number of discovered ``.vti`` files."""
        return len(self._files)

    def __getitem__(self, index: int) -> Generator[TensorDict]:
        """Read the *index*-th ``.vti`` file and yield a structured-grid TensorDict.

        Parameters
        ----------
        index : int
            Zero-based file index.

        Yields
        ------
        TensorDict
            The converted structured-grid TensorDict.
        """
        import pyvista as pv

        image = pv.read(str(self._files[index]))
        yield imagedata_to_griddict(image, fp32=self._fp32)

    @property
    def root(self) -> pathlib.Path:
        """Return the root directory of this source."""
        return self._root

    def relative_path(self, index: int) -> str:
        """Return the *index*-th file path relative to the root (POSIX style)."""
        return self._files[index].relative_to(self._root).as_posix()
