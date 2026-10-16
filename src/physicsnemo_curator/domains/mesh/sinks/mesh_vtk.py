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

"""General VTK export with named modal vectors and nested field metadata."""

from __future__ import annotations

import json
import pathlib
import tempfile
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch

from physicsnemo_curator.core.base import Param
from physicsnemo_curator.domains.mesh.sinks.mesh_vtu import MeshVTUSink

if TYPE_CHECKING:
    from collections.abc import Iterator

    from physicsnemo.mesh import Mesh
    from tensordict import TensorDict


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    """Detach a tensor and retain supported numeric precision on the CPU."""
    if tensor.is_complex():
        raise TypeError("VTK export requires real-valued tensors")
    tensor = tensor.detach().cpu()
    if tensor.dtype in {torch.float16, torch.bfloat16}:
        tensor = tensor.float()
    data = tensor.numpy()
    return data.astype(np.uint8) if data.dtype == np.bool_ else data


def _add_arrays(
    fields: TensorDict,
    target: Any,
    schema: dict[str, Any],
    *,
    modes: list[int] | None = None,
    global_fields: bool = False,
) -> None:
    """Flatten tensor components, split modal vectors, and describe empty fields."""
    for key, tensor in fields.items(include_nested=True, leaves_only=True):
        parts = (key,) if isinstance(key, str) else key
        name = "/".join(part.replace("~", "~0").replace("/", "~1") for part in parts)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"VTK export requires tensor fields: {name}")
        data = _numpy(tensor)
        entry = {"path": list(parts), "shape": list(data.shape), "dtype": str(tensor.dtype), "arrays": []}
        schema[name] = entry
        if data.size == 0:
            # VTK cannot store empty arrays. Their shape/type remain explicit.
            continue
        if modes is not None and key in {"mode_shape", "mode_rotation"}:
            if data.ndim != 3 or data.shape[1:] != (len(modes), 3):
                raise ValueError(f"{name} must have shape (nodes, number of modes, 3)")
            arrays = {f"{name}_m{mode:04d}": data[:, i, :] for i, mode in enumerate(modes)}
        else:
            if global_fields:
                data = data.reshape(1) if data.ndim == 0 else data
            arrays = {name: data.reshape(data.shape[0], -1) if data.ndim > 2 else data}
        for array_name, array in arrays.items():
            if array_name in target:
                raise ValueError(f"VTK array name collision: {array_name}")
            target[array_name] = array
            entry["arrays"].append(array_name)


class MeshVTKSink(MeshVTUSink):
    """Export volume meshes as VTU/legacy VTK or their boundary surface as VTP.

    Numeric point/cell arrays retain integer IDs and floating precision.
    Modal fields become named three-component vectors such as
    ``mode_shape_m0001``. Nested global tensors (including constraint tables)
    become VTK field-data arrays. ``physicsnemo_schema`` records tensor paths,
    shapes, dtypes, and empty fields. VTP omits volume-interior points/cells;
    use VTU or VTK to retain the complete volume.

    Parameters
    ----------
    output_dir : str
        Destination directory.
    file_format : str
        ``"vtu"`` (default), ``"vtk"``, or ``"vtp"`` (boundary surface).
    naming_template : str | None
        Output stem template with index/seq and source-backed relpath/stem
        placeholders. Default ``"mesh_{index:04d}_{seq}"``.
    binary : bool
        Write binary files by default; False writes ASCII.
    """

    name: ClassVar[str] = "Mesh VTK Writer"
    description: ClassVar[str] = "Export VTU/VTK volumes or VTP surfaces with modes and metadata"
    _VTK_CELL_TYPES: ClassVar[dict[int, int]] = {**MeshVTUSink._VTK_CELL_TYPES, 1: 1, 2: 3}

    @classmethod
    def params(cls) -> list[Param]:
        """Declare format, naming, and binary-output parameters."""
        return [
            Param(name="output_dir", description="Output directory", type=str),
            Param(
                name="file_format",
                description="VTK format (vtp extracts surface)",
                type=str,
                default="vtu",
                choices=["vtu", "vtk", "vtp"],
            ),
            Param(
                name="naming_template",
                description="Name with {index}, {seq}, {relpath}, {stem}",
                type=str,
                default=None,
            ),
            Param(name="binary", description="Write binary rather than ASCII", type=bool, default=True),
        ]

    def __init__(
        self,
        output_dir: str,
        file_format: str = "vtu",
        naming_template: str | None = None,
        binary: bool = True,
    ) -> None:
        if file_format not in {"vtu", "vtk", "vtp"}:
            raise ValueError("file_format must be 'vtu', 'vtk', or 'vtp'")
        super().__init__(output_dir, naming_template or "mesh_{index:04d}_{seq}", flip_triangle_normals=False)
        self._file_format = file_format
        self._binary = binary

    def __call__(self, items: Iterator[Mesh], index: int) -> list[str]:
        """Write one atomic file per mesh and return all output paths."""
        paths = []
        for seq, mesh in enumerate(items):
            values: dict[str, Any] = {"index": index, "seq": seq}
            relative_path = getattr(self._source, "relative_path", None)
            if callable(relative_path):
                rel = pathlib.PurePosixPath(relative_path(index))
                values.update(relpath=str(rel.parent), stem=rel.stem)
            name = self._naming_template.format(**values)
            suffix = "." + self._file_format
            if not name.endswith(suffix):
                name += suffix
            path = self._output_dir / name
            if str(path) in paths:
                raise ValueError("naming_template produces duplicate output paths; include {seq}")
            self._write_mesh(mesh, path)
            paths.append(str(path))
        self._log.info("Wrote %d %s files for index %d", len(paths), self._file_format.upper(), index)
        return paths

    def _write_mesh(self, mesh: Mesh, output_path: pathlib.Path) -> None:
        """Convert geometry/fields and save with an atomic rename."""
        import pyvista as pv

        mesh = mesh.to("cpu")
        cells, types = self._build_cells(mesh)
        if (mesh.n_cells and len(types) == 0) or np.any(types == 0):
            raise ValueError("Unsupported cells for VTK export")
        grid = pv.UnstructuredGrid(cells, types, _numpy(mesh.points))
        mode_tensor = mesh.global_data.get("mode_numbers")
        modes = mode_tensor.detach().cpu().tolist() if mode_tensor is not None else None
        if modes is not None and len(modes) != len(set(modes)):
            raise ValueError("mode_numbers must be unique")
        schema: dict[str, Any] = {"version": 1, "surface_only": self._file_format == "vtp"}
        for association in ("point_data", "cell_data", "global_data"):
            schema[association] = {}
            _add_arrays(
                getattr(mesh, association),
                grid.field_data if association == "global_data" else getattr(grid, association),
                schema[association],
                modes=modes if association == "point_data" else None,
                global_fields=association == "global_data",
            )
        if "physicsnemo_schema" in grid.field_data:
            raise ValueError("physicsnemo_schema is reserved for VTK export metadata")
        if self._file_format == "vtp":
            from vtkmodules.vtkFiltersGeometry import vtkDataSetSurfaceFilter

            # Keep all original nodal/cell fields on their boundary subset.
            surface = vtkDataSetSurfaceFilter()
            surface.SetInputData(grid)
            surface.Update()
            grid = pv.wrap(surface.GetOutput())
            for association in ("point_data", "cell_data"):
                count = grid.n_points if association == "point_data" else grid.n_cells
                for entry in schema[association].values():
                    entry["shape"][0] = count
        grid.field_data["physicsnemo_schema"] = np.array([json.dumps(schema)])
        if modes and f"mode_shape_m{modes[0]:04d}" in grid.point_data:
            grid.set_active_vectors(f"mode_shape_m{modes[0]:04d}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=output_path.suffix, delete=False) as file:
            temp_path = pathlib.Path(file.name)
        try:
            grid.save(str(temp_path), binary=self._binary)
            if temp_path.stat().st_size == 0:
                raise OSError(f"VTK writer produced an empty file: {output_path}")
            temp_path.replace(output_path)
        finally:
            temp_path.unlink(missing_ok=True)
