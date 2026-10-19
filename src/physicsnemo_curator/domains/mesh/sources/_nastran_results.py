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

"""Discover OP2 tables and retain raw arrays before optional ID-based projections."""

from __future__ import annotations

import fnmatch
import hashlib
import json
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from physicsnemo_curator.domains.mesh.sources._nastran_metadata import _encode_json, _json_value


def _array_tensor(value: np.ndarray) -> torch.Tensor | None:
    """Preserve numeric array precision, using metadata for nonnumeric arrays."""
    if value.dtype.kind not in "biufc":
        return None
    return torch.from_numpy(np.array(value, dtype=value.dtype.newbyteorder("="), copy=True, order="C"))


def _archive_result(result: Any, name: str, key: Any) -> tuple[TensorDict, dict[str, Any]]:
    """Retain all numeric instance arrays and metadata for any backend result class."""
    arrays: dict[str, torch.Tensor] = {}
    sparse: dict[str, TensorDict] = {}
    metadata: dict[str, Any] = {
        "table": name,
        "key": _json_value(key),
        "class": type(result).__name__,
        "subcase_id": getattr(result, "isubcase", None),
        "attributes": {},
    }
    attributes = vars(result) if hasattr(result, "__dict__") else {"data": result}
    for attribute, value in attributes.items():
        if attribute in {"data_frame", "h5_file"}:
            continue
        if isinstance(value, np.ndarray):
            tensor = _array_tensor(value)
            if tensor is not None:
                arrays[attribute] = tensor
                continue
        elif hasattr(value, "tocoo"):
            # Monitor-point tables may store their values in sparse matrices.
            coo = value.tocoo()
            sparse[attribute] = TensorDict(
                {
                    "row": _array_tensor(coo.row),
                    "column": _array_tensor(coo.col),
                    "data": _array_tensor(coo.data),
                },
                batch_size=[],
            )
            metadata.setdefault("sparse_arrays", {})[attribute] = {
                "shape": list(coo.shape),
                "dtype": str(coo.dtype),
                "storage": "coo",
            }
            continue
        metadata["attributes"][attribute] = _json_value(value)
    if hasattr(result, "get_headers"):
        metadata["headers"] = list(result.get_headers())
    # Some result classes expose data as a property instead of an instance array.
    if "data" not in arrays and isinstance(getattr(result, "data", None), np.ndarray):
        tensor = _array_tensor(result.data)
        if tensor is not None:
            arrays["data"] = tensor
    metadata["arrays"] = {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()}
    archive = TensorDict(
        {"arrays": TensorDict(arrays, batch_size=[]), "metadata": _encode_json(metadata)}, batch_size=[]
    )
    if sparse:
        archive["sparse_arrays"] = TensorDict(sparse, batch_size=[])
    return archive, metadata


def _discover_results(op2: Any, patterns: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Inventory all nonempty backend tables and select families by glob, never by hardcoded names."""
    includes = [p.strip() for p in patterns.split(",") if p.strip()] or ["*"]
    records, catalog = [], []
    names = list(dict.fromkeys(op2.get_table_types()))
    seen = set(names)
    for name in names:
        table = op2.get_result(name)
        if table is None:
            continue
        # PSD collections expose a second level of table families. Other
        # singleton objects (GPDT, EQEXIN, optimization results) are results.
        if callable(getattr(table, "get_table_types", None)):
            children = table.get_table_types()
            for child in children:
                if child not in seen:
                    names.append(child)
                    seen.add(child)
            continue
        if name == "cstm" and table.data is None:
            continue  # pyNastran initializes an empty coordinate-table object.
        if isinstance(table, dict | list | tuple) and not table:
            continue
        if isinstance(table, np.ndarray) and not table.size:
            continue
        selected = any(fnmatch.fnmatchcase(name, pattern) for pattern in includes)
        entries = table.items() if isinstance(table, dict) else [(None, table)]
        for key, result in entries:
            if not selected:
                catalog.append({"table": name, "key": _json_value(key), "status": "excluded_by_user"})
                continue
            if not hasattr(result, "__dict__") and not isinstance(
                result, np.ndarray | np.generic | dict | list | tuple | str | bytes | int | float | complex | bool
            ):
                catalog.append(
                    {
                        "table": name,
                        "key": _json_value(key),
                        "status": "source_only",
                        "reason": "backend object has no instance attributes",
                    }
                )
                continue
            archive, metadata = _archive_result(result, name, key)
            identity = json.dumps([name, _json_value(key)], sort_keys=True).encode("utf-8")
            record_id = "table_" + hashlib.sha256(identity).hexdigest()[:16]
            record = {
                "id": record_id,
                "name": name,
                "key": key,
                "result": result,
                "archive": archive,
                "subcase_id": metadata["subcase_id"],
                "metadata": metadata,
            }
            records.append(record)
            catalog.append(
                {
                    "id": record_id,
                    "table": name,
                    "key": _json_value(key),
                    "subcase_id": metadata["subcase_id"],
                    "status": "raw_table_preserved",
                    "headers": metadata.get("headers", []),
                    "arrays": metadata["arrays"],
                }
            )
    # OP2 matrices live outside get_table_types(). Preserve sparse coordinates
    # directly instead of allocating a potentially enormous dense matrix.
    for name, matrix in getattr(op2, "matrices", {}).items():
        table_name = f"matrices.{name}"
        if not any(fnmatch.fnmatchcase(table_name, pattern) for pattern in includes):
            catalog.append({"table": table_name, "status": "excluded_by_user"})
            continue
        data = matrix.data
        metadata: dict[str, Any]
        if hasattr(data, "tocoo"):
            coo = data.tocoo()
            arrays = {
                "row": _array_tensor(coo.row),
                "column": _array_tensor(coo.col),
                "data": _array_tensor(coo.data),
            }
            metadata = {"table": f"matrices.{name}", "shape": list(data.shape), "storage": "coo", "subcase_id": None}
        else:
            arrays = {"data": _array_tensor(data)}
            metadata = {"table": f"matrices.{name}", "shape": list(data.shape), "storage": "dense", "subcase_id": None}
        metadata["attributes"] = {k: _json_value(v) for k, v in vars(matrix).items() if k != "data"}
        metadata["class"] = type(matrix).__name__
        record_id = "table_" + hashlib.sha256(table_name.encode("utf-8")).hexdigest()[:16]
        archive = TensorDict(
            {"arrays": TensorDict(arrays, batch_size=[]), "metadata": _encode_json(metadata)}, batch_size=[]
        )
        records.append(
            {
                "id": record_id,
                "name": f"matrices.{name}",
                "subcase_id": None,
                "result": matrix,
                "archive": archive,
                "metadata": metadata,
            }
        )
        catalog.append({"id": record_id, "table": f"matrices.{name}", "status": "raw_table_preserved"})
    return records, catalog


def _rotate_nodal(data: np.ndarray, model: Any, node_ids: list[int]) -> np.ndarray:
    """Rotate recognized displacement-like vectors, retaining raw table values separately."""
    data = data.copy()
    coordinate_rows: dict[int, list[int]] = defaultdict(list)
    for row, nid in enumerate(node_ids):
        coordinate_rows[model.nodes[nid].Cd()].append(row)
    for cid, rows in coordinate_rows.items():
        if cid == 0:
            continue
        coord = model.coords[cid]
        if coord.type not in {"CORD1R", "CORD2R"}:
            raise NotImplementedError(f"Output coordinate system {cid}: only rectangular CD projection is supported")
        for start in range(0, data.shape[-1], 3):
            data[rows, :, start : start + 3] = data[rows, :, start : start + 3] @ coord.beta()
    return data


def _project_result(
    record: dict[str, Any],
    model: Any,
    node_ids: list[int],
    element_ids: list[int],
) -> tuple[str | None, dict[str, torch.Tensor], str]:
    """Project only unambiguous GRID/centroid results, leaving layered/integration-point tables raw."""
    result = record["result"]
    values = getattr(result, "data", None)
    if not isinstance(values, np.ndarray) or values.ndim != 3:
        return None, {}, "raw table: no three-dimensional result array"
    # PyNastran normally canonicalizes SORT2, but never guess the layout.
    if getattr(result, "sort_method", 1) != 1:
        return None, {}, "raw table: SORT2 layout requires a dedicated adapter"
    if getattr(result, "superelement_adaptivity_index", None) not in (None, "", 0):
        return None, {}, "raw table: superelement/adaptive IDs require a dedicated association adapter"
    name = record["name"]
    field = record.get("field_name", name)
    node_gridtype = getattr(result, "node_gridtype", None)
    if isinstance(node_gridtype, np.ndarray) and node_gridtype.shape == (values.shape[1], 2):
        rows = [(i, int(nid)) for i, (nid, kind) in enumerate(node_gridtype) if kind == 1]
        ids = [nid for _, nid in rows]
        if len(ids) != len(set(ids)):
            return None, {}, "raw table: repeated GRID IDs"
        lookup = {nid: row for row, nid in rows}
        dtype = values.dtype.newbyteorder("=")
        missing = np.nan if dtype.kind in "fc" else 0
        aligned = np.full((len(node_ids), values.shape[0], values.shape[2]), missing, dtype=dtype)
        valid = np.zeros(len(node_ids), dtype=bool)
        for i, nid in enumerate(node_ids):
            if nid in lookup:
                aligned[i] = values[:, lookup[nid], :]
                valid[i] = True
        # No tensor-rank guess for arbitrary vector-like custom result tables.
        vector_families = {
            "eigenvectors",
            "displacements",
            "velocities",
            "accelerations",
            "load_vectors",
            "spc_forces",
            "mpc_forces",
            "force_vectors",
        }
        if name in vector_families and values.shape[2] in {3, 6}:
            if any(model.coords[model.nodes[nid].Cd()].type not in {"CORD1R", "CORD2R"} for nid in node_ids):
                return None, {}, "raw table: cylindrical/spherical output vectors require a dedicated adapter"
            aligned = _rotate_nodal(aligned, model, node_ids)
        return (
            "point",
            {field: torch.from_numpy(aligned), field + "__valid": torch.from_numpy(valid)},
            "GRID IDs; missing rows masked",
        )
    element = getattr(result, "element", None)
    if not isinstance(element, np.ndarray):
        return None, {}, "raw table: layered, nodal-on-element, or unrecognized association"
    if element.ndim == 1:
        if len(element) != values.shape[1]:
            return None, {}, "raw table: ambiguous element row count"
        element = np.broadcast_to(element, values.shape[:2])
    if element.shape != values.shape[:2]:
        return None, {}, "raw table: ambiguous element/time axes"
    # Multiple rows per element (layers, stations, integration points) cannot
    # be flattened to one cell value without losing physical meaning.
    for ids in element:
        positive = ids[ids > 0]
        if len(positive) != len(set(positive.tolist())):
            return None, {}, "raw table: repeated element IDs (layers/stations)"
    dtype = values.dtype.newbyteorder("=")
    missing = np.nan if dtype.kind in "fc" else 0
    aligned = np.full((len(element_ids), values.shape[0], values.shape[2]), missing, dtype=dtype)
    valid = np.zeros((len(element_ids), values.shape[0]), dtype=bool)
    lookup = {eid: i for i, eid in enumerate(element_ids)}
    for step in range(values.shape[0]):
        for row, eid in enumerate(element[step]):
            if int(eid) in lookup:
                aligned[lookup[int(eid)], step] = values[step, row]
                valid[lookup[int(eid)], step] = True
    return (
        "element",
        {field: torch.from_numpy(aligned), field + "__valid": torch.from_numpy(valid)},
        "parent element IDs; missing rows masked; no averaging",
    )
