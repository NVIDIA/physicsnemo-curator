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
import time
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
    tensor = tensor.detach().resolve_conj().resolve_neg().cpu()
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


def _canonical_projection(node_ids: torch.Tensor, element_ids: torch.Tensor) -> np.ndarray:
    """Sort simplex node IDs and rows while preserving duplicate multiplicities."""
    rows = np.column_stack((element_ids.numpy(), np.sort(node_ids.numpy(), axis=1)))
    return rows[np.lexsort(rows.T[::-1])]


def _original_cells(mesh: Mesh) -> tuple[np.ndarray, np.ndarray, TensorDict]:
    """Validate the native projection before recovering retained Nastran cells."""
    nastran = mesh.global_data["nastran"]
    guidance = "; use_original_geometry=False to export the projection"
    expected_nodes = nastran.get("projection_node_ids")
    expected_elements = nastran.get("projection_element_ids")
    if expected_nodes is None or expected_elements is None:
        raise ValueError("Original Nastran export requires its projection snapshot" + guidance)
    node_ids = mesh.point_data.get("node_id")
    element_ids = mesh.cell_data.get("element_id")
    if node_ids is None or element_ids is None or node_ids.ndim != 1 or element_ids.ndim != 1:
        raise ValueError("Original Nastran node/element IDs changed" + guidance)
    if node_ids.unique().numel() != mesh.n_points:
        raise ValueError("Original Nastran node IDs must be unique" + guidance)
    actual_nodes = node_ids[mesh.cells]
    if (
        actual_nodes.shape != expected_nodes.shape
        or element_ids.shape != expected_elements.shape
        or not np.array_equal(
            _canonical_projection(actual_nodes, element_ids),
            _canonical_projection(expected_nodes, expected_elements),
        )
    ):
        raise ValueError("Projected Nastran cells changed" + guidance)

    cell_data = nastran["element_data"]
    parent_index = {eid: i for i, eid in enumerate(cell_data["element_id"].tolist())}
    rows = [parent_index[eid] for eid in element_ids.tolist()]
    if set(mesh.cell_data.keys(True, True)) != set(cell_data.keys(True, True)):
        raise ValueError("Projected cell fields changed" + guidance)
    for key, value in cell_data.items(True, True):
        actual, expected = mesh.cell_data[key], value[rows]
        if (
            actual.dtype != expected.dtype
            or actual.shape != expected.shape
            or not torch.allclose(actual, expected, rtol=0, atol=0, equal_nan=True)
        ):
            raise ValueError("Projected cell values changed" + guidance)

    geometry = nastran["geometry"]
    node_index = {nid: i for i, nid in enumerate(node_ids.tolist())}
    original_ids = geometry["point_node_ids"][geometry["connectivity"]].tolist()
    if any(nid not in node_index for nid in original_ids):
        raise ValueError("Original Nastran nodes were removed" + guidance)
    flat = np.array([node_index[nid] for nid in original_ids], dtype=np.int64)
    offsets = geometry["offsets"].numpy()
    cells = np.empty(len(flat) + len(offsets) - 1, dtype=np.int64)
    sizes = np.diff(offsets)
    positions = offsets[:-1] + np.arange(len(sizes))
    cells[positions] = sizes
    mask = np.ones(len(cells), dtype=bool)
    mask[positions] = False
    cells[mask] = flat
    return cells, geometry["vtk_cell_type"].numpy(), cell_data


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
        Write binary files by default; False writes ASCII. Legacy VTK ASCII
        rejects NaN/infinity; VTU/VTP ASCII rejects negative infinity.
    use_original_geometry : bool
        Prefer retained Nastran cells (default True). False exports the
        native simplicial cells, including transformed cell fields. Original
        export rejects changed connectivity or cell fields.

    Examples
    --------
    >>> sink = MeshVTKSink(output_dir="./output", file_format="vtu")  # doctest: +SKIP
    >>> paths = sink(mesh_iterator, index=0)  # doctest: +SKIP
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
        """Initialize the VTK sink and validate output naming."""
        if file_format not in {"vtu", "vtk", "vtp"}:
            raise ValueError("file_format must be 'vtu', 'vtk', or 'vtp'")
        if naming_template is not None:
            try:
                naming_template.format(index=0, seq=0, relpath="test", stem="test")
            except (KeyError, IndexError, ValueError, AttributeError) as exc:
                raise ValueError(
                    f"Invalid naming_template {naming_template!r}: {exc}. "
                    "Only {index}, {seq}, {relpath}, and {stem} placeholders are supported."
                ) from exc
        super().__init__(output_dir, naming_template or "mesh_{index:04d}_{seq}", flip_triangle_normals=False)
        self._file_format = file_format
        self._binary = binary
        self._use_original_geometry = use_original_geometry

    def __call__(self, items: Iterator[Mesh], index: int) -> list[str]:
        """Consume meshes and write one atomic file per item.

        Parameters
        ----------
        items : Iterator[Mesh]
            Meshes to persist.
        index : int
            Source index used in output names.

        Returns
        -------
        list[str]
            Paths of the written files.
        """
        started = time.perf_counter()
        paths: list[str] = []
        seen: set[pathlib.Path] = set()
        values: dict[str, Any] = {"index": index, "relpath": "", "stem": ""}
        relative_path = getattr(self._source, "relative_path", None)
        if callable(relative_path):
            rel = pathlib.PurePosixPath(relative_path(index))
            values.update(relpath=str(rel.parent), stem=rel.stem)
        for seq, mesh in enumerate(items):
            if not callable(relative_path) and (
                "{relpath" in self._naming_template or "{stem" in self._naming_template
            ):
                raise ValueError("naming_template requires a source with relative_path(index)")
            values["seq"] = seq
            name = self._naming_template.format(**values)
            suffix = "." + self._file_format
            if not name.endswith(suffix):
                name += suffix
            path = self._output_dir / name
            resolved = path.resolve()
            if not resolved.is_relative_to(self._output_dir.resolve()):
                raise ValueError("naming_template must resolve within output_dir")
            if resolved in seen:
                raise ValueError("naming_template produces duplicate output paths; include {seq}")
            seen.add(resolved)
            self._write_mesh(mesh, path)
            paths.append(str(path))
        self._log.info(
            "Wrote %d %s files for index %d (%.2fs)",
            len(paths),
            self._file_format.upper(),
            index,
            time.perf_counter() - started,
        )
        return paths

    def _write_mesh(self, mesh: Mesh, output_path: pathlib.Path) -> None:
        """Convert geometry/fields and save with an atomic rename."""
        import pyvista as pv

        mesh = mesh.to("cpu")
        original_geometry = mesh.global_data.get(("nastran", "geometry")) if self._use_original_geometry else None
        cell_data = mesh.cell_data
        if original_geometry is not None:
            cells, types, cell_data = _original_cells(mesh)
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
        if not self._binary:
            arrays = (grid.points, *grid.point_data.values(), *grid.cell_data.values(), *grid.field_data.values())
            floating = [array for array in arrays if np.issubdtype(array.dtype, np.floating)]
            if self._file_format == "vtk":
                # Legacy readers differ across platforms; Linux rejects these
                # tokens and can stop before reading the remaining field data.
                if any(not np.isfinite(array).all() for array in floating):
                    raise ValueError("Legacy VTK ASCII cannot portably preserve NaN or infinity; use binary=True")
            elif any(np.isneginf(array).any() for array in floating):
                raise ValueError("VTK XML ASCII cannot preserve negative infinity; use binary=True")
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
            if self._file_format == "vtk" and not self._binary:
                from vtkmodules.vtkIOLegacy import vtkUnstructuredGridWriter

                writer = vtkUnstructuredGridWriter()
                writer.SetFileName(str(temp_path))
                writer.SetInputData(grid)
                writer.SetFileTypeToASCII()
                # VTK's default 11 digits truncate float64 source values.
                writer.SetPrecision(17)
                if not writer.Write():
                    raise OSError(f"VTK writer failed: {output_path}")
            else:
                grid.save(str(temp_path), binary=self._binary)
            if temp_path.stat().st_size == 0:
                raise OSError(f"VTK writer produced an empty file: {output_path}")
            temp_path.replace(output_path)
        finally:
            temp_path.unlink(missing_ok=True)

    @property
    def file_format(self) -> str:
        """Return the VTK output format."""
        return self._file_format

    @property
    def use_original_geometry(self) -> bool:
        """Return whether retained Nastran cells are preferred."""
        return self._use_original_geometry


def read_vtk_data(path: str) -> dict[str, TensorDict]:
    """Restore exported tensor fields, nested paths, complex values, and empty shapes.

    This returns point/cell/global TensorDicts rather than constructing a
    simplicial Mesh: Nastran VTU/VTK preserves original higher-order cells.

    Parameters
    ----------
    path : str
        File produced by :class:`MeshVTKSink`.

    Returns
    -------
    dict[str, TensorDict]
        Tensor fields keyed by ``point_data``, ``cell_data``, and ``global_data``.

    Raises
    ------
    ValueError
        If the file lacks a supported PhysicsNeMo tensor schema.
    """
    import pyvista as pv

    grid = pv.read(path)
    if "physicsnemo_schema" not in grid.field_data:
        raise ValueError("File has no physicsnemo_schema; read_vtk_data requires MeshVTKSink output")
    schema = json.loads(grid.field_data["physicsnemo_schema"][0])
    if schema.get("version") != 1:
        raise ValueError(f"Unsupported PhysicsNeMo VTK schema version: {schema.get('version')!r}")
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
                    # Arithmetic reconstruction mixes real/imaginary NaN/Inf.
                    data = np.empty(parts[0].shape, dtype=np.result_type(parts[0].dtype, np.complex64))
                    data.real = parts[0]
                    data.imag = parts[1]
                elif entry.get("encoding") == "modes":
                    data = np.stack(parts, axis=1)
                else:
                    data = parts[0]
                value = torch.from_numpy(np.array(data, copy=True).reshape(shape)).to(dtype)
            block[tuple(entry["path"])] = value
        blocks[association] = block
    return blocks
