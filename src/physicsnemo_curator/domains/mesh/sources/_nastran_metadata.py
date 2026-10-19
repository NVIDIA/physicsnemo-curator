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

"""Portable Nastran card metadata and byte-exact source preservation."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import pathlib
import tempfile
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from physicsnemo.mesh import Mesh


def _json_value(value: Any) -> Any:
    """Encode non-JSON scalars without pickle, executable representations, or NaN tokens."""
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite_float": str(value)}
    if isinstance(value, complex):
        return {"complex": [_json_value(value.real), _json_value(value.imag)]}
    if isinstance(value, bytes):
        return {"bytes_base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "values": _json_value(value.tolist())}
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_value(item) for item in value]
    if hasattr(value, "raw_fields"):
        return {"type": value.type, "fields": _json_value(value.raw_fields()), "comment": value.comment}
    # Non-array backend objects are described, never evaluated or unpickled.
    return {"python_type": type(value).__name__, "text": str(value)}


def _encode_json(value: Any) -> torch.Tensor:
    """Encode structured metadata as a UTF-8 byte tensor for all mesh writers."""
    data = json.dumps(_json_value(value), ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    return torch.frombuffer(bytearray(data), dtype=torch.uint8)


def decode_nastran_json(value: torch.Tensor) -> Any:
    """Decode a Nastran metadata byte tensor from a Mesh or VTK field-data archive."""
    return json.loads(value.detach().cpu().numpy().tobytes().decode("utf-8"))


def _deck_metadata(model: Any) -> dict[str, Any]:
    """Retain every parsed card's raw fields/comments and complete control text."""
    collections: dict[str, list[Any]] = {}
    seen: set[int] = set()

    def collect(value: Any, rows: list[Any], path: list[str]) -> None:
        if hasattr(value, "raw_fields"):
            if id(value) not in seen:
                seen.add(id(value))
                rows.append({"key": path, **_json_value(value)})
        elif isinstance(value, dict):
            for key, child in value.items():
                collect(child, rows, [*path, str(key)])
        elif isinstance(value, list | tuple):
            for index, child in enumerate(value):
                collect(child, rows, [*path, str(index)])

    for name, value in vars(model).items():
        if not name.startswith("_"):
            rows: list[Any] = []
            collect(value, rows, [])
            if rows:
                collections[name] = rows
    return {
        "solution": model.sol,
        "sol_method": model.sol_method,
        "system_control": model.system_command_lines,
        "executive_control": model.executive_control_lines,
        "case_control": str(model.case_control_deck),
        "card_count": dict(model.card_count),
        "rejected_lines": model.reject_lines,
        "rejected_cards": model.reject_cards,
        "cards": collections,
    }


def _source_files(model: Any, deck: pathlib.Path, op2: pathlib.Path | None) -> TensorDict:
    """Embed actual inputs and INCLUDE files with portable names and SHA-256 checksums."""
    paths = {deck.resolve(), *(pathlib.Path(p).resolve() for p in model.active_filenames)}
    if op2 is not None:
        paths.add(op2.resolve())
    base = pathlib.Path(os.path.commonpath([str(p.parent) for p in paths]))
    archive = TensorDict({}, batch_size=[])
    manifest = []
    for index, path in enumerate(sorted(paths)):
        content = path.read_bytes()
        key = f"file_{index:06d}"
        role = "deck" if path == deck.resolve() else "op2" if op2 and path == op2.resolve() else "include"
        manifest.append(
            {
                "key": key,
                "path": path.relative_to(base).as_posix(),
                "role": role,
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
        archive[key] = (
            torch.frombuffer(bytearray(content), dtype=torch.uint8) if content else torch.empty(0, dtype=torch.uint8)
        )
    archive["manifest"] = _encode_json(manifest)
    return archive


def restore_nastran_sources(mesh: Mesh | TensorDict, output_dir: str) -> list[str]:
    """Restore embedded deck/INCLUDE/OP2 inputs into a new directory after hash validation.

    ``mesh`` may also be the global-data TensorDict returned by
    :func:`physicsnemo_curator.domains.mesh.sinks.mesh_vtk.read_vtk_data`.
    Existing directories are never overwritten. Stored paths must remain
    inside the destination. No source text is executed.
    """
    global_data = mesh if isinstance(mesh, TensorDict) else mesh.global_data
    sources = global_data["nastran", "source_files"]
    manifest = decode_nastran_json(sources["manifest"])
    destination = pathlib.Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    names: set[str] = set()
    payloads = []
    for entry in manifest:
        relative = pathlib.PurePosixPath(entry["path"])
        if (
            relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
            or "\\" in entry["path"]
            or ":" in entry["path"]
            or relative.as_posix() in names
        ):
            raise ValueError(f"Unsafe or duplicate archived path: {entry['path']}")
        names.add(relative.as_posix())
        content = sources[entry["key"]].detach().cpu().numpy().tobytes()
        if len(content) != entry["size"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"Nastran source checksum mismatch: {relative}")
        payloads.append((relative, content))
    with tempfile.TemporaryDirectory(prefix=".nastran_restore_", dir=destination.parent) as temp:
        staged = pathlib.Path(temp) / "files"
        staged.mkdir()
        for relative, content in payloads:
            target = staged / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        staged.rename(destination)
    return [str(destination / relative) for relative, _ in payloads]
