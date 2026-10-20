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

"""Nastran geometry, discovered OP2 results, card metadata, and original inputs.

Raw result arrays retain their axes, IDs, precision, and component headers.
Unambiguous GRID/element tables also become mesh fields. Original cells
remain in global data alongside a native simplicial projection; source
bytes provide recovery of content not interpreted by the backend.
"""

from __future__ import annotations

from collections import Counter
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from tensordict import TensorDict

from physicsnemo_curator.core.base import Param, Source
from physicsnemo_curator.core.logging import get_logger
from physicsnemo_curator.domains.mesh.sources._nastran_geometry import _geometry_groups, _original_geometry
from physicsnemo_curator.domains.mesh.sources._nastran_metadata import (
    _deck_metadata,
    _encode_json,
    _source_files,
    decode_nastran_json,
    restore_nastran_sources,
)
from physicsnemo_curator.domains.mesh.sources._nastran_results import _discover_results, _project_result, _rotate_nodal

if TYPE_CHECKING:
    from collections.abc import Generator

    from physicsnemo.mesh import Mesh

_DECK_EXTENSIONS = {".bdf", ".dat", ".nas", ".deck", ".pch"}
__all__ = ["NastranSource", "decode_nastran_json", "restore_nastran_sources"]


def _read_deck(path: Path, encoding: str, punch: bool) -> Any:
    """Read a deck, including non-UTF-8 comments and relative INCLUDEs."""
    try:
        from pyNastran.bdf.bdf import BDF
    except ImportError as error:
        raise ImportError(
            "Nastran support requires the optional reader. Install it with: pip install 'pyNastran>=1.4.1,<1.5'."
        ) from error

    class _EncodedBDF(BDF):
        def _parse_primary_file_header(self, bdf_filename: str | StringIO) -> None:
            # pyNastran 1.4.1 opens this header with the system encoding even
            # when read_bdf(encoding=...) is supplied. Keep the actual file
            # reader/path intact so relative INCLUDE resolution still works.
            if isinstance(bdf_filename, StringIO):
                super()._parse_primary_file_header(bdf_filename)
            else:
                # Only leading comment lines can contain backend directives;
                # avoid reading the entire deck a second time for its header.
                header = []
                with Path(bdf_filename).open(encoding=encoding) as source:
                    for line in source:
                        if not line.startswith("$"):
                            break
                        header.append(line)
                super()._parse_primary_file_header(StringIO("".join(header)))

    model = _EncodedBDF(debug=None)
    model.read_bdf(str(path), encoding=encoding, punch=punch or path.suffix.lower() == ".pch")
    return model


def _constraints(model: Any) -> TensorDict:
    """Preserve constraint sets as numeric tables using original Nastran IDs."""
    spc_rows, spc_values = [], []
    for sid, cards in sorted(model.spcs.items()):
        for card in cards:
            if card.type == "SPC1":
                terms = [(nid, card.components, 0.0) for nid in card.node_ids]
            elif card.type == "SPC":
                terms = zip(card.node_ids, card.components, card.enforced, strict=True)
            else:
                continue  # Complete definitions remain in the generic card archive.
            for nid, components, value in terms:
                for component in str(components):
                    spc_rows.append([sid, nid, int(component)])
                    spc_values.append(value)
    mpc_sets, mpc_offsets, mpc_terms, mpc_coefficients = [], [0], [], []
    for sid, cards in sorted(model.mpcs.items()):
        for card in cards:
            if card.type != "MPC":
                continue
            mpc_sets.append(sid)
            for nid, component, coefficient in zip(card.node_ids, card.components, card.coefficients, strict=True):
                mpc_terms.append([nid, int(component)])
                mpc_coefficients.append(coefficient)
            mpc_offsets.append(len(mpc_terms))
    rbe2_rows, rbe2_alpha, rbe2_tref = [], [], []
    for eid, card in sorted(model.rigid_elements.items()):
        if card.type != "RBE2":
            continue
        for nid in card.Gmi:
            rbe2_rows.append([eid, card.gn, nid, int(card.cm)])
            rbe2_alpha.append(card.alpha)
            rbe2_tref.append(card.tref)
    data = TensorDict(
        {
            "spc": torch.tensor(spc_rows, dtype=torch.int64).reshape(-1, 3),
            "spc_value": torch.tensor(spc_values, dtype=torch.float64),
            "mpc_set_id": torch.tensor(mpc_sets, dtype=torch.int64),
            "mpc_offsets": torch.tensor(mpc_offsets, dtype=torch.int64),
            "mpc_terms": torch.tensor(mpc_terms, dtype=torch.int64).reshape(-1, 2),
            "mpc_coefficient": torch.tensor(mpc_coefficients, dtype=torch.float64),
            "rbe2": torch.tensor(rbe2_rows, dtype=torch.int64).reshape(-1, 4),
            "rbe2_alpha": torch.tensor(rbe2_alpha, dtype=torch.float64),
            "rbe2_tref": torch.tensor(rbe2_tref, dtype=torch.float64),
        },
        batch_size=[],
    )
    for name in ("spcadds", "mpcadds"):
        rows = [
            [sid, child] for sid, cards in sorted(getattr(model, name).items()) for card in cards for child in card.sets
        ]
        data[name] = torch.tensor(rows, dtype=torch.int64).reshape(-1, 2)
    return data


def _modal_data(result: Any, model: Any, node_ids: list[int]) -> tuple[torch.Tensor, TensorDict]:
    """Align a complete real eigenvector table by GRID ID and rotate to basic axes."""
    if getattr(result, "superelement_adaptivity_index", None) not in (None, "", 0):
        raise NotImplementedError("Superelement/adaptive output requires a dedicated ID association adapter")
    if np.iscomplexobj(result.data) or result.analysis_code != 2:
        raise NotImplementedError("Only real normal-mode eigenvectors (SOL 103) are supported")
    grid = result.node_gridtype[:, 1] == 1
    result_ids = result.node_gridtype[grid, 0].tolist()
    if len(result_ids) != len(set(result_ids)) or set(result_ids) != set(node_ids):
        raise ValueError("OP2 eigenvector GRID IDs must match deck GRID IDs exactly (no missing or duplicate nodes)")
    index = {nid: i for i, nid in enumerate(result_ids)}
    data = result.data[:, grid, :][:, [index[nid] for nid in node_ids], :].transpose(1, 0, 2)
    data = _rotate_nodal(data, model, node_ids)
    eigenvalues = np.asarray(result.eigns, dtype=np.float64)
    # Preserve solver-reported cycles, including near-zero rigid-body modes
    # whose eigenvalues can be slightly negative. Keep their sign in eigns.
    frequencies = np.asarray(result.mode_cycles, dtype=np.float64)
    global_data = TensorDict(
        {
            "subcase_id": torch.tensor(result.isubcase),
            "mode_numbers": torch.tensor(result.modes, dtype=torch.int64),
            "eigenvalues": torch.from_numpy(eigenvalues.copy()),
            "frequencies": torch.from_numpy(frequencies.copy()),
        },
        batch_size=[],
    )
    return torch.from_numpy(data), global_data


class NastranSource(Source["Mesh"]):
    """Read Nastran decks and optional OP2 files into meshes with a result archive.

    Parameters
    ----------
    input_path : str
        A deck file or a directory searched recursively for .bdf/.dat/.nas/.deck/.pch.
    file_pattern : str
        Deck discovery glob relative to a directory, default ``"**/*"``.
        Use a main-deck pattern when INCLUDE files also use deck extensions;
        discovery treats every matched file as an independent input.
    op2_pattern : str
        OP2 filename template relative to each deck's directory, with ``{stem}``.
        Empty selects a same-stem OP2, then a unique sibling OP2. Ambiguity is an error.
    read_results : bool
        Read all paired OP2 tables exposed by pyNastran (default True).
    result_patterns : str
        Comma-separated table-family globs, default ``"*"``. Exclusions
        remain explicit in the result catalog.
    preserve_source_files : bool
        Embed the original deck, INCLUDE files, and paired OP2 bytes
        (default True). Disable to reduce storage at the cost of exact
        recovery of uninterpreted content.
    encoding : str
        Deck encoding, default ``"latin1"`` (accepts arbitrary comment bytes).
    punch : bool
        True for bulk-data-only files without executive/case control. .pch
        files always use this mode. Default False.

    Notes
    -----
    Yields one mesh per subcase and topological dimension (lines, surfaces,
    volumes). Common rod/beam, triangle/quad, tetra/hex/wedge/pyramid cells
    are supported. Unsupported topology raises explicitly. Native cells
    are piecewise linear; original connectivity and higher-order types
    are retained separately for VTK export.

    Sparse floating results use NaN plus validity masks (integers use zero
    placeholders with masks). Layered/integration-point
    results stay in their original tables without averaging. Recognized
    nodal vectors are rotated from rectangular CD frames; other frames
    remain raw with a projection report. Units and solver normalization
    are unchanged. Complete real modes additionally expose modal aliases.
    Backend-unsupported content is recoverable from the embedded originals,
    but is not claimed as an interpreted mesh field.
    """

    name: ClassVar[str] = "Nastran"
    description: ClassVar[str] = (
        "Nastran geometry, discovered OP2 fields, complete card metadata, and source preservation"
    )

    @classmethod
    def params(cls) -> list[Param]:
        """Declare file discovery, pairing, and decoding parameters."""
        return [
            Param(name="input_path", description="Deck file or directory", type=str),
            Param(name="file_pattern", description="Recursive deck glob", type=str, default="**/*"),
            Param(
                name="op2_pattern", description="Relative OP2 template, e.g. Output_{stem}.op2", type=str, default=""
            ),
            Param(name="read_results", description="Read paired OP2 result tables", type=bool, default=True),
            Param(name="result_patterns", description="Comma-separated result family globs", type=str, default="*"),
            Param(
                name="preserve_source_files",
                description="Embed original deck/INCLUDE/OP2 bytes",
                type=bool,
                default=True,
            ),
            Param(name="encoding", description="Deck text encoding", type=str, default="latin1"),
            Param(name="punch", description="Bulk data only (no executive/case control)", type=bool, default=False),
        ]

    def __init__(
        self,
        input_path: str,
        file_pattern: str = "**/*",
        op2_pattern: str = "",
        read_results: bool = True,
        encoding: str = "latin1",
        punch: bool = False,
        result_patterns: str = "*",
        preserve_source_files: bool = True,
    ) -> None:
        """Discover input decks; defer geometry and result reads until indexing."""
        self._input_path = input_path
        self._file_pattern = file_pattern
        self._op2_pattern = op2_pattern
        self._read_results = read_results
        self._encoding = encoding
        self._punch = punch
        self._result_patterns = result_patterns
        self._preserve_source_files = preserve_source_files
        path = Path(input_path).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        self._root = path if path.is_dir() else path.parent
        candidates = path.glob(file_pattern) if path.is_dir() else [path]
        self._files = sorted(p for p in candidates if p.is_file() and p.suffix.lower() in _DECK_EXTENSIONS)
        if not self._files:
            raise ValueError(f"No Nastran decks found in {path}")
        self._log = get_logger(self)
        self._log.info("Discovered %d Nastran decks", len(self._files))

    @property
    def root(self) -> Path:
        """Return the directory used for relative output paths."""
        return self._root

    def relative_path(self, index: int) -> str:
        """Return the deck path relative to the source root for sink naming."""
        return self._files[index].relative_to(self._root).as_posix()

    def __len__(self) -> int:
        """Return the number of decks; one deck can yield multiple subcases."""
        return len(self._files)

    def _op2_path(self, deck: Path) -> Path:
        """Find the paired results without choosing an arbitrary sibling file."""
        if self._op2_pattern:
            path = deck.parent / self._op2_pattern.format(stem=deck.stem)
            if not path.is_file():
                raise FileNotFoundError(path)
            return path
        siblings = sorted(p for p in deck.parent.iterdir() if p.is_file() and p.suffix.lower() == ".op2")
        matching = [p for p in siblings if p.stem.lower() == deck.stem.lower()]
        if len(matching) == 1:
            return matching[0]
        if not matching and len(siblings) == 1:
            return siblings[0]
        if not siblings:
            raise FileNotFoundError(f"No OP2 paired with {deck}; use read_results=False for geometry only")
        raise ValueError(f"Ambiguous OP2 pairing for {deck}; set op2_pattern explicitly")

    def __getitem__(self, index: int) -> Generator[Mesh]:
        """Read one input deck and yield meshes with its result archive.

        Parameters
        ----------
        index : int
            Deck index. Negative indices count from the final discovered deck.

        Yields
        ------
        Mesh
            One mesh per result subcase and topological dimension.

        Raises
        ------
        IndexError
            If the index is outside the discovered decks.
        FileNotFoundError
            If results are requested and the paired OP2 does not exist.
        ValueError
            If pairing is ambiguous, no tables match, or geometry is invalid.
        NotImplementedError
            If the deck contains unsupported element topology.
        """
        from physicsnemo.mesh import Mesh

        deck = self._files[index]
        model = _read_deck(deck, self._encoding, self._punch)
        points, point_data, geometry, element_data = _original_geometry(model)
        groups = _geometry_groups(points, geometry, element_data)
        deck_metadata = _deck_metadata(model)
        base = TensorDict({"deck_metadata": _encode_json(deck_metadata)}, batch_size=[])
        records, catalog = [], []
        op2_path = self._op2_path(deck) if self._read_results else None
        if op2_path is not None:
            from pyNastran.op2.op2 import read_op2

            op2 = read_op2(str(op2_path), debug=None)
            records, catalog = _discover_results(op2, self._result_patterns)
            if not records:
                raise ValueError(f"No result tables matched {self._result_patterns!r} in {op2_path}")
            base["op2_metadata"] = _encode_json(
                {
                    "table_names": getattr(op2, "table_names", []),
                    "subcase_key": getattr(op2, "subcase_key", {}),
                    "version": getattr(op2, "nastran_format", None),
                }
            )
        if self._preserve_source_files:
            base["source_files"] = _source_files(model, deck, op2_path)
        constraints = _constraints(model)
        cases = sorted({r["subcase_id"] for r in records if r["subcase_id"] is not None}) or [0]
        node_ids = point_data["node_id"].tolist()
        for subcase_id in cases:
            selected_records = [r for r in records if r["subcase_id"] in {None, subcase_id}]
            family_counts = Counter(r["name"] for r in selected_records)
            reserved_fields = set(point_data.keys()) | set(element_data.keys()) | {"mode_shape", "mode_rotation"}
            for record in sorted(selected_records, key=lambda r: (r["name"], r["id"])):
                # Multiple result keys may share a family and subcase (e.g.
                # superelements). Never overwrite one with another.
                field = record["name"] + ("__" + record["id"] if family_counts[record["name"]] > 1 else "")
                if field in reserved_fields or field + "__valid" in reserved_fields:
                    field += "__" + record["id"]
                record["field_name"] = field
                reserved_fields.update((field, field + "__valid"))
            subcase = model.subcases.get(subcase_id, model.subcases[0])
            selections = {
                name.lower() + "_id": torch.tensor(subcase.params.get(name, [0])[0], dtype=torch.int64)
                for name in ("SPC", "MPC")
            }
            for group in groups:
                pd = point_data.clone()
                parent_data = group["element_data"].clone()
                report = []
                for record in selected_records:
                    location, fields, reason = _project_result(
                        record, model, node_ids, parent_data["element_id"].tolist()
                    )
                    if location == "point":
                        pd.update(fields)
                    elif location == "element":
                        parent_data.update(fields)
                    report.append(
                        {
                            "id": record["id"],
                            "table": record["name"],
                            "association": location,
                            "fields": list(fields),
                            "description": reason,
                        }
                    )
                # Retain the original public modal aliases for one real mode set.
                modal = [
                    r
                    for r in selected_records
                    if r["name"] == "eigenvectors"
                    and getattr(r["result"], "analysis_code", None) == 2
                    and not np.iscomplexobj(r["result"].data)
                ]
                gd = TensorDict(
                    {
                        "constraints": constraints.clone(),
                        "subcase_id": torch.tensor(subcase_id),
                        **{name: value.clone() for name, value in selections.items()},
                    },
                    batch_size=[],
                )
                if len(modal) == 1:
                    try:
                        modes, modal_global = _modal_data(modal[0]["result"], model, node_ids)
                    except (ValueError, NotImplementedError) as error:
                        report.append({"table": "eigenvectors", "modal_aliases": False, "reason": str(error)})
                    else:
                        pd["mode_shape"], pd["mode_rotation"] = modes[:, :, :3], modes[:, :, 3:]
                        gd.update(modal_global)
                nastran = base.clone(recurse=False)
                nastran["geometry"] = group["geometry"].clone()
                nastran["element_data"] = parent_data
                nastran["results"] = TensorDict({r["id"]: r["archive"] for r in selected_records}, batch_size=[])
                nastran["catalog"] = _encode_json(catalog)
                nastran["projection"] = _encode_json(
                    {
                        "dimension": group["dimension"],
                        "original_elements": len(parent_data),
                        "simplicial_cells": len(group["cells"]),
                        "results": report,
                        "cell_result_semantics": (
                            "Values refer to original parent elements; do not sum repeated children."
                        ),
                    }
                )
                nastran["parent_row"] = group["parent_row"].clone()
                nastran["projection_node_ids"] = point_data["node_id"][group["cells"]]
                nastran["projection_element_ids"] = parent_data["element_id"][group["parent_row"]]
                gd["nastran"] = nastran
                # Repeated values remain explicitly parent-associated; no
                # guessed integration, averaging, or energy redistribution.
                cell_data = parent_data[group["parent_row"]].clone()
                yield Mesh(
                    points=points.clone(),
                    cells=group["cells"].clone(),
                    point_data=pd,
                    cell_data=cell_data,
                    global_data=gd,
                )
