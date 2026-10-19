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

"""VTK export with original Nastran cells and recoverable tensor metadata."""

from __future__ import annotations

import json
import pathlib
import tempfile
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from tensordict import TensorDict

from physicsnemo_curator.core.base import Param
from physicsnemo_curator.domains.mesh.sinks.mesh_vtu import MeshVTUSink

if TYPE_CHECKING:
    from collections.abc import Iterator

    from physicsnemo.mesh import Mesh


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    """Detach a tensor and retain supported numeric precision on the CPU."""
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
        if tensor.is_complex():
            flat = data.reshape(-1) if global_fields else data
            flat = flat.reshape(flat.shape[0], -1) if flat.ndim > 2 else flat
            arrays = {name + "__real": flat.real.copy(), name + "__imag": flat.imag.copy()}
            entry["encoding"] = "complex"
        elif modes is not None and key in {"mode_shape", "mode_rotation"}:
            if data.ndim != 3 or data.shape[1:] != (len(modes), 3):
                raise ValueError(f"{name} must have shape (nodes, number of modes, 3)")
            arrays = {f"{name}_m{mode:04d}": data[:, i, :] for i, mode in enumerate(modes)}
            entry["encoding"] = "modes"
        else:
            if global_fields:
                # Raw result tensors can have hundreds of thousands of
                # components per step. Store them as scalar field arrays:
                # VTK's vector norm cache can overflow on those layouts.
                data = data.reshape(-1) if data.ndim == 0 or data.ndim > 2 else data
            arrays = {name: data.reshape(data.shape[0], -1) if data.ndim > 2 else data}
        for array_name, array in arrays.items():
            if array_name in target:
                raise ValueError(f"VTK array name collision: {array_name}")
            target[array_name] = array
            entry["arrays"].append(array_name)


def _repair_norm_ranges(grid: Any) -> None:
    """Repair VTK norm-cache overflow without changing stored array values."""
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkCommonCore import vtkDataArray

    for fields in (grid.GetPointData(), grid.GetCellData(), grid.GetFieldData()):
        for index in range(fields.GetNumberOfArrays()):
            array = fields.GetArray(index)
            if array is None or array.GetNumberOfComponents() < 2:
                continue
            if np.isfinite(array.GetRange(-1)).all():
                continue
            # VTK can square large integer components in their original
            # dtype, producing NaN metadata that its XML reader rejects.
            norms = np.hypot.reduce(vtk_to_numpy(array).astype(np.float64), axis=1)
            finite = norms[np.isfinite(norms)]
            bounds = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 0.0)
            array.GetInformation().Set(vtkDataArray.L2_NORM_RANGE(), bounds, 2)


class MeshVTKSink(MeshVTUSink):
    """Export volume meshes as VTU/legacy VTK or their boundary surface as VTP.

    Numeric point/cell arrays retain integer IDs and floating precision.
    Modal fields become named three-component vectors such as
    ``mode_shape_m0001``. Nested global tensors (including constraint tables)
    become VTK field-data arrays. ``physicsnemo_schema`` records tensor paths,
    shapes, dtypes, and empty fields. Complex arrays use real/imaginary pairs.
    For Nastran sources, original higher-order cells and parent fields are
    exported by default. VTP carries a boundary subset of point/cell data;
    the complete global result/source archive remains in Field Data.

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
    use_original_geometry : bool
        Prefer retained Nastran cells (default True). False exports the
        native simplicial cells, including transformed cell fields.
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
            Param(
                name="use_original_geometry",
                description="Use original Nastran cells and parent fields",
                type=bool,
                default=True,
            ),
        ]

    def __init__(
        self,
        output_dir: str,
        file_format: str = "vtu",
        naming_template: str | None = None,
        binary: bool = True,
        use_original_geometry: bool = True,
    ) -> None:
        if file_format not in {"vtu", "vtk", "vtp"}:
            raise ValueError("file_format must be 'vtu', 'vtk', or 'vtp'")
        super().__init__(output_dir, naming_template or "mesh_{index:04d}_{seq}", flip_triangle_normals=False)
        self._file_format = file_format
        self._binary = binary
        self._use_original_geometry = use_original_geometry

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
        original_geometry = mesh.global_data.get(("nastran", "geometry")) if self._use_original_geometry else None
        cell_data = mesh.cell_data
        if original_geometry is not None:
            offsets = original_geometry["offsets"].tolist()
            node_index = {nid: i for i, nid in enumerate(mesh.point_data["node_id"].tolist())}
            original_ids = original_geometry["point_node_ids"][original_geometry["connectivity"]].tolist()
            if any(nid not in node_index for nid in original_ids):
                raise ValueError(
                    "Original Nastran nodes were removed; use_original_geometry=False exports the projection"
                )
            flat = [node_index[nid] for nid in original_ids]
            packed = []
            for start, end in zip(offsets[:-1], offsets[1:], strict=True):
                packed.extend([end - start, *flat[start:end]])
            cells = np.array(packed, dtype=np.int64)
            types = original_geometry["vtk_cell_type"].numpy()
            cell_data = mesh.global_data["nastran", "element_data"]
            parent_index = {eid: i for i, eid in enumerate(cell_data["element_id"].tolist())}
            if set(mesh.cell_data["element_id"].tolist()) != set(parent_index):
                raise ValueError("Projected elements changed; use_original_geometry=False to export the projection")
            if set(mesh.cell_data.keys()) != set(cell_data.keys()):
                raise ValueError("Projected cell fields changed; use_original_geometry=False to export those fields")
            rows = [parent_index[eid] for eid in mesh.cell_data["element_id"].tolist()]
            for key in cell_data.keys():  # noqa: SIM118 - TensorDict iterates batch rows, not keys.
                if not torch.allclose(mesh.cell_data[key], cell_data[key][rows], rtol=0, atol=0, equal_nan=True):
                    raise ValueError(
                        "Projected cell values changed; use_original_geometry=False to export those values"
                    )
        else:
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
                cell_data if association == "cell_data" else getattr(mesh, association),
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
            surface.SetNonlinearSubdivisionLevel(1)
            surface.Update()
            grid = pv.wrap(surface.GetOutput())
            for association in ("point_data", "cell_data"):
                count = grid.n_points if association == "point_data" else grid.n_cells
                for entry in schema[association].values():
                    entry["shape"][0] = count
        grid.field_data["physicsnemo_schema"] = np.array([json.dumps(schema)])
        if modes and f"mode_shape_m{modes[0]:04d}" in grid.point_data:
            grid.set_active_vectors(f"mode_shape_m{modes[0]:04d}")
        _repair_norm_ranges(grid)
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


def read_vtk_data(path: str) -> dict[str, TensorDict]:
    """Restore exported tensor fields, nested paths, complex values, and empty shapes.

    This returns point/cell/global TensorDicts rather than constructing a
    simplicial Mesh: Nastran VTU/VTK preserves original higher-order cells.
    """
    import pyvista as pv

    grid = pv.read(path)
    schema = json.loads(grid.field_data["physicsnemo_schema"][0])
    blocks = {}
    for association in ("point_data", "cell_data", "global_data"):
        arrays = grid.field_data if association == "global_data" else getattr(grid, association)
        count = grid.n_points if association == "point_data" else grid.n_cells
        batch_size = [] if association == "global_data" else [count]
        block = TensorDict({}, batch_size=batch_size)
        for entry in schema[association].values():
            dtype = getattr(torch, entry["dtype"].removeprefix("torch."))
            shape = entry["shape"]
            names = entry["arrays"]
            if not names:
                value = torch.empty(shape, dtype=dtype)
            else:
                parts = [np.asarray(arrays[name]) for name in names]
                if entry.get("encoding") == "complex":
                    data = parts[0] + 1j * parts[1]
                elif entry.get("encoding") == "modes":
                    data = np.stack(parts, axis=1)
                else:
                    data = parts[0]
                value = torch.from_numpy(np.array(data, copy=True).reshape(shape)).to(dtype)
            block[tuple(entry["path"])] = value
        blocks[association] = block
    return blocks
