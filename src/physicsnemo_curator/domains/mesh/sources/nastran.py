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

"""Nastran bulk-data geometry and real modal OP2 results via pyNastran.

CTETRA4/10, PSOLID/MAT1, SPC/SPC1, MPC, and RBE2 are supported. Each
deck yields one mesh per real eigenvector result table (usually a subcase).
Quadratic tetrahedra are subdivided into eight linear tetrahedra, retaining
all nodes and repeating parent element data. This is a piecewise linear
approximation, not a quadratic finite-element representation.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from tensordict import TensorDict

from physicsnemo_curator.core.base import Param, Source
from physicsnemo_curator.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Generator

    from physicsnemo.mesh import Mesh

_DECK_EXTENSIONS = {".bdf", ".dat", ".nas", ".deck", ".pch"}
# Nastran CTETRA midside order: 12, 23, 31, 14, 24, 34.
_TET10_CELLS = np.array(
    [[0, 4, 6, 7], [4, 1, 5, 8], [6, 5, 2, 9], [7, 8, 9, 3], [4, 5, 6, 9], [4, 5, 8, 9], [4, 7, 8, 9], [4, 6, 7, 9]],
    dtype=np.int64,
)


def _read_deck(path: Path, encoding: str, punch: bool) -> Any:
    """Read a deck, including non-UTF-8 comments and relative INCLUDEs."""
    from pyNastran.bdf.bdf import BDF

    class _EncodedBDF(BDF):
        def _parse_primary_file_header(self, bdf_filename: str | StringIO) -> None:
            # pyNastran 1.4.1 opens this header with the system encoding even
            # when read_bdf(encoding=...) is supplied. Keep the actual file
            # reader/path intact so relative INCLUDE resolution still works.
            if isinstance(bdf_filename, StringIO):
                super()._parse_primary_file_header(bdf_filename)
            else:
                text = Path(bdf_filename).read_text(encoding=encoding)
                super()._parse_primary_file_header(StringIO(text))

    model = _EncodedBDF(debug=None)
    model.read_bdf(str(path), encoding=encoding, punch=punch or path.suffix.lower() == ".pch")
    return model


def _geometry(model: Any) -> tuple[torch.Tensor, torch.Tensor, TensorDict, TensorDict]:
    """Extract ID-aligned geometry and repeat parent data on subdivided cells."""
    node_ids = sorted(model.nodes)
    if not node_ids or not model.elements:
        raise ValueError("Nastran deck must contain GRID nodes and CTETRA elements")
    node_index = {nid: i for i, nid in enumerate(node_ids)}
    points = np.array([model.nodes[nid].get_position() for nid in node_ids], dtype=np.float64)
    bases = {}
    for cid in {model.nodes[nid].Cd() for nid in node_ids}:
        coord = model.coords[cid]
        if coord.type not in {"CORD1R", "CORD2R"}:
            raise NotImplementedError(f"Output coordinate system {cid}: only rectangular CD is supported")
        bases[cid] = coord.beta()
    cells, element_ids, property_ids, material_ids, material_values = [], [], [], [], []
    for eid, element in sorted(model.elements.items()):
        if element.type != "CTETRA" or any(nid is None for nid in element.node_ids):
            raise NotImplementedError(f"Element {eid}: only complete CTETRA4/10 elements are supported")
        prop = element.pid_ref
        if prop.type != "PSOLID":
            raise NotImplementedError(f"Element {eid}: only PSOLID with MAT1 is supported")
        material = prop.mid_ref
        if material.type != "MAT1":
            raise NotImplementedError(f"Element {eid}: only PSOLID with MAT1 is supported")
        ids = np.array([node_index[nid] for nid in element.node_ids], dtype=np.int64)
        if len(ids) == 4:
            child_cells = ids.reshape(1, 4)
        elif len(ids) == 10:
            child_cells = ids[_TET10_CELLS]
        else:
            raise NotImplementedError(f"Element {eid}: expected 4 or 10 nodes")
        cells.extend(child_cells)
        count = len(child_cells)
        element_ids.extend([eid] * count)
        property_ids.extend([prop.pid] * count)
        material_ids.extend([material.mid] * count)
        material_values.extend([[material.E(), material.Nu(), material.Rho()]] * count)
    connectivity = np.array(cells, dtype=np.int64)
    vertices = points[connectivity]
    signed_volume = np.linalg.det(vertices[:, 1:] - vertices[:, :1])
    negative = signed_volume < 0
    connectivity[negative, :2] = connectivity[negative, 1::-1]
    if np.any(signed_volume == 0):
        raise ValueError("Nastran mesh contains degenerate tetrahedra")
    permanent_spc = np.zeros((len(node_ids), 6), dtype=bool)
    for i, nid in enumerate(node_ids):
        for component in str(model.nodes[nid].ps):
            if component in "123456":
                permanent_spc[i, int(component) - 1] = True
    point_data = TensorDict(
        {
            "node_id": torch.tensor(node_ids),
            "input_coordinate_id": torch.tensor([model.nodes[nid].Cp() for nid in node_ids]),
            "output_coordinate_id": torch.tensor([model.nodes[nid].Cd() for nid in node_ids]),
            "output_coordinate_basis": torch.from_numpy(np.array([bases[model.nodes[nid].Cd()] for nid in node_ids])),
            "permanent_spc": torch.from_numpy(permanent_spc),
        },
        batch_size=[len(node_ids)],
    )
    values = torch.tensor(material_values, dtype=torch.float64)
    cell_data = TensorDict(
        {
            "element_id": torch.tensor(element_ids),
            "property_id": torch.tensor(property_ids),
            "material_id": torch.tensor(material_ids),
            "young_modulus": values[:, 0],
            "poisson_ratio": values[:, 1],
            "density": values[:, 2],
        },
        batch_size=[len(connectivity)],
    )
    return torch.from_numpy(points), torch.from_numpy(connectivity), point_data, cell_data


def _constraints(model: Any) -> TensorDict:
    """Preserve constraint sets as numeric tables using original Nastran IDs."""
    for name in ("suport", "suport1", "se_suport", "spcoffs"):
        if getattr(model, name, None):
            raise NotImplementedError(f"Unsupported constraints: {name}")
    spc_rows, spc_values = [], []
    for sid, cards in sorted(model.spcs.items()):
        for card in cards:
            if card.type == "SPC1":
                terms = [(nid, card.components, 0.0) for nid in card.node_ids]
            elif card.type == "SPC":
                terms = zip(card.node_ids, card.components, card.enforced, strict=True)
            else:
                raise NotImplementedError(f"Unsupported constraint: {card.type}")
            for nid, components, value in terms:
                for component in str(components):
                    spc_rows.append([sid, nid, int(component)])
                    spc_values.append(value)
    mpc_sets, mpc_offsets, mpc_terms, mpc_coefficients = [], [0], [], []
    for sid, cards in sorted(model.mpcs.items()):
        for card in cards:
            if card.type != "MPC":
                raise NotImplementedError(f"Unsupported constraint: {card.type}")
            mpc_sets.append(sid)
            for nid, component, coefficient in zip(card.node_ids, card.components, card.coefficients, strict=True):
                mpc_terms.append([nid, int(component)])
                mpc_coefficients.append(coefficient)
            mpc_offsets.append(len(mpc_terms))
    rbe2_rows, rbe2_alpha, rbe2_tref = [], [], []
    for eid, card in sorted(model.rigid_elements.items()):
        if card.type != "RBE2":
            raise NotImplementedError(f"Unsupported rigid element: {card.type}")
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
    if np.iscomplexobj(result.data) or result.analysis_code != 2:
        raise NotImplementedError("Only real normal-mode eigenvectors (SOL 103) are supported")
    grid = result.node_gridtype[:, 1] == 1
    result_ids = result.node_gridtype[grid, 0].tolist()
    if len(result_ids) != len(set(result_ids)) or set(result_ids) != set(node_ids):
        raise ValueError("OP2 eigenvector GRID IDs must match deck GRID IDs exactly (no missing or duplicate nodes)")
    index = {nid: i for i, nid in enumerate(result_ids)}
    data = result.data[:, grid, :][:, [index[nid] for nid in node_ids], :].transpose(1, 0, 2).copy()
    # GRID CD specifies the axes of both translations and rotations. Use the
    # aligned rows rather than assuming OP2 uses the deck's node ordering.
    for cid in {model.nodes[nid].Cd() for nid in node_ids} - {0}:
        coord = model.coords[cid]
        if coord.type not in {"CORD1R", "CORD2R"}:
            raise NotImplementedError(f"Output coordinate system {cid}: only rectangular CD is supported")
        indices = [i for i, nid in enumerate(node_ids) if model.nodes[nid].Cd() == cid]
        data[indices, :, :3] = data[indices, :, :3] @ coord.beta()
        data[indices, :, 3:] = data[indices, :, 3:] @ coord.beta()
    eigenvalues = np.asarray(result.eigns, dtype=np.float64)
    # Preserve solver-reported cycles, including near-zero rigid-body modes
    # whose eigenvalues can be slightly negative. Keep their sign in eigns.
    frequencies = np.asarray(result.mode_cycles, dtype=np.float64)
    subcase = model.subcases.get(result.isubcase, model.subcases[0])
    selections = {name.lower() + "_id": subcase.params.get(name, [0])[0] for name in ("SPC", "MPC")}
    global_data = TensorDict(
        {
            "subcase_id": torch.tensor(result.isubcase),
            "mode_numbers": torch.tensor(result.modes, dtype=torch.int64),
            "eigenvalues": torch.from_numpy(eigenvalues.copy()),
            "frequencies": torch.from_numpy(frequencies.copy()),
            **{name: torch.tensor(value, dtype=torch.int64) for name, value in selections.items()},
        },
        batch_size=[],
    )
    return torch.from_numpy(data), global_data


class NastranSource(Source["Mesh"]):
    """Read Nastran tetrahedral decks and optional paired modal OP2 files.

    Parameters
    ----------
    input_path : str
        A deck file or a directory searched recursively for .bdf/.dat/.nas/.deck/.pch.
    file_pattern : str
        Deck discovery glob relative to a directory, default ``"**/*"``.
    op2_pattern : str
        OP2 filename template relative to each deck's directory, with ``{stem}``.
        Empty selects a same-stem OP2, then a unique sibling OP2. Ambiguity is an error.
    read_results : bool
        Read real eigenmodes (default True); False yields geometry alone.
    encoding : str
        Deck encoding, default ``"latin1"`` (accepts arbitrary comment bytes).
    punch : bool
        True for bulk-data-only files without executive/case control. .pch
        files always use this mode. Default False.

    Notes
    -----
    Requires pyNastran. Units and modal normalization are retained from the
    solver. Frequencies are cycles per model time unit. Only complete GRID
    result tables and rectangular output coordinate systems are supported.
    Unsupported structural elements, properties, materials, or constraints
    raise an error rather than being silently discarded.
    """

    name: ClassVar[str] = "Nastran"
    description: ClassVar[str] = "Nastran CTETRA decks and real modal OP2 results with materials and constraints"

    @classmethod
    def params(cls) -> list[Param]:
        """Declare file discovery, pairing, and decoding parameters."""
        return [
            Param(name="input_path", description="Deck file or directory", type=str),
            Param(name="file_pattern", description="Recursive deck glob", type=str, default="**/*"),
            Param(
                name="op2_pattern", description="Relative OP2 template, e.g. Output_{stem}.op2", type=str, default=""
            ),
            Param(name="read_results", description="Read paired OP2 eigenmodes", type=bool, default=True),
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
    ) -> None:
        self._input_path = input_path
        self._file_pattern = file_pattern
        self._op2_pattern = op2_pattern
        self._read_results = read_results
        self._encoding = encoding
        self._punch = punch
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
        """Yield geometry alone or a mesh per real eigenvector table at this index."""
        from physicsnemo.mesh import Mesh

        deck = self._files[index]
        model = _read_deck(deck, self._encoding, self._punch)
        points, cells, point_data, cell_data = _geometry(model)
        global_data = TensorDict({"constraints": _constraints(model)}, batch_size=[])
        if not self._read_results:
            yield Mesh(points=points, cells=cells, point_data=point_data, cell_data=cell_data, global_data=global_data)
            return
        from pyNastran.op2.op2 import read_op2

        op2 = read_op2(str(self._op2_path(deck)), include_results=["eigenvectors"], debug=None)
        if not op2.eigenvectors:
            raise ValueError(f"No modal eigenvectors found in OP2 paired with {deck}")
        for result in op2.eigenvectors.values():
            modes, modal_global = _modal_data(result, model, point_data["node_id"].tolist())
            pd = point_data.clone()
            pd["mode_shape"] = modes[:, :, :3]
            pd["mode_rotation"] = modes[:, :, 3:]
            gd = global_data.clone()
            gd.update(modal_global)
            yield Mesh(
                points=points.clone(), cells=cells.clone(), point_data=pd, cell_data=cell_data.clone(), global_data=gd
            )
