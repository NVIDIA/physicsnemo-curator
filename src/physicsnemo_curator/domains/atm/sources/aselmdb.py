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

"""ASE LMDB data source for atomic/molecular pipelines.

Reads ``.aselmdb`` files and yields
:class:`~nvalchemi.data.AtomicData` objects for use in curator pipelines.

This source is designed for datasets stored in the ASE LMDB format, such
as the `Open Molecules 2025 (OMol25) <https://huggingface.co/facebook/OMol25>`_
dataset from Meta FAIR.  OMol25 contains over 100 million DFT calculations
at the ωB97M-V/def2-TZVPD level of theory, covering 83 elements and ~83M
unique molecular systems including small molecules, biomolecules, metal
complexes, and electrolytes (systems up to 350 atoms).

Each source index corresponds to one ``.aselmdb`` file.  The generator
returned by :meth:`__getitem__` iterates over every row in that database
file, converting each ASE :class:`~ase.Atoms` entry to an
:class:`~nvalchemi.data.AtomicData` instance.

Two read backends are supported:

- **python** (default): Uses a pure-Python reader (``lmdb`` + ``zlib`` +
  ``json``) to open the database, decompress entries, and convert
  ``__ndarray__`` markers to NumPy arrays.  Then constructs
  :class:`~nvalchemi.data.AtomicData` directly from the raw row dicts.
- **rust**: Uses a native Rust reader
  (:func:`physicsnemo_curator._lib.lmdb.read_lmdb`) for I/O, zlib
  decompression, and JSON parsing, then constructs
  :class:`~nvalchemi.data.AtomicData` directly from the raw row dicts.
  Avoids the :class:`ase.Atoms` intermediate and can be significantly
  faster for large datasets.  Falls back to **python** if the Rust
  extension is not available.

References
----------
- OMol25 dataset: https://huggingface.co/facebook/OMol25
- OMol25 paper: Levine et al., "The Open Molecules 2025 (OMol25) Dataset,
  Evaluations, and Models", arXiv:2505.08762 (2025).
  https://arxiv.org/abs/2505.08762
- OPoly26 extension: Levine et al., "The Open Polymers 2026 (OPoly26)
  Dataset and Evaluations", arXiv:2512.23117 (2025).
  https://arxiv.org/abs/2512.23117
- fairchem toolkit: https://github.com/facebookresearch/fairchem

"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np

from physicsnemo_curator.core.base import Param, Source
from physicsnemo_curator.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Generator

    import torch
    from nvalchemi.data import AtomicData

logger = get_logger("ASELMDBSource")

# Reserved LMDB keys that are not data rows (same set as the Rust reader).
_RESERVED_LMDB_KEYS = frozenset({"nextid", "deleted_ids", "metadata"})

# NumPy dtype lookup for __ndarray__ markers.
_NDARRAY_DTYPES: dict[str, np.dtype[np.generic]] = {
    "float64": np.dtype(np.float64),
    "float32": np.dtype(np.float32),
    "int64": np.dtype(np.int64),
    "int32": np.dtype(np.int32),
    "uint8": np.dtype(np.uint8),
    "bool": np.dtype(np.bool_),
}


def _decode_ndarray_markers(obj: object) -> object:
    """Recursively walk a JSON-decoded object and convert ``__ndarray__`` markers.

    The ASE LMDB format encodes NumPy arrays as
    ``{"__ndarray__": [shape, dtype_str, flat_data]}``.  This function
    converts those markers into real :class:`numpy.ndarray` objects
    in-place (for dicts and lists).

    Parameters
    ----------
    obj : object
        A JSON-decoded Python object (dict, list, scalar).

    Returns
    -------
    object
        The same structure with ``__ndarray__`` dicts replaced by
        ``numpy.ndarray`` instances.
    """
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            marker: object = obj["__ndarray__"]
            if not isinstance(marker, list) or len(marker) != 3:
                return obj  # pragma: no cover — defensive
            shape: list[int] = marker[0]  # ty: ignore[invalid-assignment]
            dtype_str: str = marker[1]  # ty: ignore[invalid-assignment]
            flat: list[object] = marker[2]  # ty: ignore[invalid-assignment]
            dt = _NDARRAY_DTYPES.get(dtype_str, np.dtype(dtype_str))
            arr = np.array(flat, dtype=dt)
            if len(shape) > 1:
                arr = arr.reshape(shape)
            return arr
        return {k: _decode_ndarray_markers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_ndarray_markers(v) for v in obj]
    return obj


def _count_aselmdb_rows(db_path: pathlib.Path) -> int:
    """Count data rows in a ``.aselmdb`` file without parsing values.

    Opens the LMDB environment, counts keys that are numeric row IDs
    (skipping reserved keys), and closes the environment.

    Parameters
    ----------
    db_path : pathlib.Path
        Path to the ``.aselmdb`` file.

    Returns
    -------
    int
        Number of data rows (structures) in the file.
    """
    import lmdb as lmdb_lib

    env = lmdb_lib.open(str(db_path), readonly=True, subdir=False, lock=False)
    try:
        count = 0
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            for key_bytes, _ in cursor:
                key_str = bytes(key_bytes).decode("utf-8")
                if key_str in _RESERVED_LMDB_KEYS:
                    continue
                try:
                    int(key_str)
                    count += 1
                except ValueError:
                    continue
        return count
    finally:
        env.close()


def _read_aselmdb_row_at(db_path: pathlib.Path, row_index: int) -> dict[str, object]:
    """Read a single row from a ``.aselmdb`` file by index.

    Opens the LMDB environment, iterates to the *row_index*-th data row
    (skipping reserved keys), and returns the parsed row dict.

    Parameters
    ----------
    db_path : pathlib.Path
        Path to the ``.aselmdb`` file.
    row_index : int
        Zero-based index of the row to read (among data rows only).

    Returns
    -------
    dict[str, object]
        Row dict with ``__ndarray__`` markers replaced by NumPy arrays
        and a synthetic ``"id"`` key added.

    Raises
    ------
    IndexError
        If *row_index* is out of range.
    """
    import json
    import zlib

    import lmdb as lmdb_lib

    env = lmdb_lib.open(str(db_path), readonly=True, subdir=False, lock=False)
    try:
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            # Collect valid (row_id, key_bytes, val_bytes) tuples, then sort
            valid_rows: list[tuple[int, bytes, bytes]] = []
            for key_bytes, val_bytes in cursor:
                key_str = bytes(key_bytes).decode("utf-8")
                if key_str in _RESERVED_LMDB_KEYS:
                    continue
                try:
                    row_id = int(key_str)
                except ValueError:
                    continue
                valid_rows.append((row_id, bytes(key_bytes), bytes(val_bytes)))

            # Sort by row_id for deterministic ordering
            valid_rows.sort(key=lambda x: x[0])

            if row_index < 0 or row_index >= len(valid_rows):
                msg = f"Row index {row_index} out of range for file with {len(valid_rows)} rows."
                raise IndexError(msg)

            row_id, _, val_bytes = valid_rows[row_index]
            json_bytes = zlib.decompress(val_bytes)
            raw: dict[str, object] = json.loads(json_bytes)
            decoded = _decode_ndarray_markers(raw)
            if not isinstance(decoded, dict):
                msg = f"Decoded row is not a dict: {type(decoded)}"
                raise TypeError(msg)
            row_dict: dict[str, object] = {str(k): v for k, v in decoded.items()}
            row_dict["id"] = row_id
            return row_dict
    finally:
        env.close()


def _read_aselmdb_rows(db_path: pathlib.Path) -> list[dict[str, object]]:
    """Read all data rows from a ``.aselmdb`` file using pure Python.

    Opens the LMDB environment, iterates over all entries, skips reserved
    keys (``nextid``, ``deleted_ids``, ``metadata``), decompresses each
    zlib-compressed JSON value, parses it, and converts ``__ndarray__``
    markers to NumPy arrays.

    Parameters
    ----------
    db_path : pathlib.Path
        Path to the ``.aselmdb`` file.

    Returns
    -------
    list[dict[str, object]]
        Row dicts sorted by ascending integer ID, with ``__ndarray__``
        markers replaced by actual NumPy arrays and a synthetic ``"id"``
        key added to each dict.
    """
    import json
    import zlib

    import lmdb as lmdb_lib

    env = lmdb_lib.open(str(db_path), readonly=True, subdir=False, lock=False)
    try:
        rows: list[tuple[int, dict[str, object]]] = []
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            for key_bytes, val_bytes in cursor:
                key_str = bytes(key_bytes).decode("utf-8")
                if key_str in _RESERVED_LMDB_KEYS:
                    continue
                try:
                    row_id = int(key_str)
                except ValueError:
                    continue
                json_bytes = zlib.decompress(bytes(val_bytes))
                raw: dict[str, object] = json.loads(json_bytes)
                decoded = _decode_ndarray_markers(raw)
                if not isinstance(decoded, dict):
                    continue  # pragma: no cover — defensive
                row_dict: dict[str, object] = {str(k): v for k, v in decoded.items()}
                row_dict["id"] = row_id
                rows.append((row_id, row_dict))

        rows.sort(key=lambda pair: pair[0])
        return [row for _, row in rows]
    finally:
        env.close()


def _encode_ndarray(arr: np.ndarray) -> dict[str, list[object]]:
    """Encode a NumPy array as an ASE ``__ndarray__`` marker dict.

    Parameters
    ----------
    arr : np.ndarray
        Array to encode.

    Returns
    -------
    dict
        ``{"__ndarray__": [shape, dtype_str, flat_data]}``
    """
    return {"__ndarray__": [list(arr.shape), str(arr.dtype), arr.ravel().tolist()]}


def _atoms_to_row_dict(
    atoms: object,
    row_id: int,
    *,
    key_value_pairs: dict[str, object] | None = None,
) -> dict[str, object]:
    """Convert an ASE :class:`~ase.Atoms` to an LMDB row dict.

    This encodes atomic fields (positions, numbers, cell, pbc, etc.)
    as ``__ndarray__`` markers compatible with the ASE LMDB format.

    Parameters
    ----------
    atoms : ase.Atoms
        Atoms object to encode.
    row_id : int
        Integer row ID (1-based).
    key_value_pairs : dict or None
        Optional key-value pairs to store alongside the row.

    Returns
    -------
    dict[str, object]
        JSON-serialisable row dict.
    """
    import uuid

    _numbers = np.asarray(atoms.numbers, dtype=np.int64)  # ty: ignore[unresolved-attribute]
    _positions = np.asarray(atoms.positions, dtype=np.float64)  # ty: ignore[unresolved-attribute]
    _cell = np.asarray(atoms.cell, dtype=np.float64).reshape(3, 3)  # ty: ignore[unresolved-attribute]
    _pbc = np.asarray(atoms.pbc, dtype=np.bool_)  # ty: ignore[unresolved-attribute]

    row: dict[str, object] = {
        "id": row_id,
        "unique_id": uuid.uuid4().hex,
        "ctime": 0.0,
        "mtime": 0.0,
        "user": "",
        "numbers": _encode_ndarray(_numbers),
        "positions": _encode_ndarray(_positions),
        "cell": _encode_ndarray(_cell),
        "pbc": _encode_ndarray(_pbc),
    }

    # Encode calculator results if present.
    calc = getattr(atoms, "calc", None)
    if calc is not None:
        results = getattr(calc, "results", {})
        if "energy" in results:
            row["energy"] = float(results["energy"])
        if "forces" in results:
            row["forces"] = _encode_ndarray(np.asarray(results["forces"], dtype=np.float64))
        if "stress" in results:
            row["stress"] = _encode_ndarray(np.asarray(results["stress"], dtype=np.float64))

    if key_value_pairs:
        row["key_value_pairs"] = key_value_pairs

    return row


def _write_aselmdb(
    db_path: pathlib.Path,
    rows: list[dict[str, object]],
) -> None:
    """Write rows to a ``.aselmdb`` file using pure Python.

    Creates a new LMDB environment and writes the given row dicts as
    zlib-compressed JSON, following the ASE LMDB format convention.

    Parameters
    ----------
    db_path : pathlib.Path
        Output file path.
    rows : list[dict[str, object]]
        Row dicts with ``__ndarray__`` markers (as produced by
        :func:`_atoms_to_row_dict` or similar).  Each must have an
        ``"id"`` key with an integer value.
    """
    import json
    import zlib

    import lmdb as lmdb_lib

    env = lmdb_lib.open(str(db_path), subdir=False, map_size=50 * 1024 * 1024)
    try:
        with env.begin(write=True) as txn:
            # Write reserved keys.
            next_id = max((r["id"] for r in rows), default=0) + 1  # type: ignore[type-var]
            txn.put(b"nextid", zlib.compress(json.dumps(next_id).encode()))
            txn.put(b"deleted_ids", zlib.compress(b"[]"))

            for row in rows:
                row_id = row["id"]
                # Remove "id" from the stored dict — ASE convention.
                row_copy = {k: v for k, v in row.items() if k != "id"}
                txn.put(
                    str(row_id).encode(),
                    zlib.compress(json.dumps(row_copy).encode()),
                )
    finally:
        env.close()


# Keys that the Rust reader returns but are not AtomicData fields.
_METADATA_KEYS = frozenset(
    {
        "id",
        "unique_id",
        "ctime",
        "mtime",
        "user",
        "calculator",
        "calculator_parameters",
        "key_value_pairs",
        "data",
    }
)

# Voigt notation (xx, yy, zz, yz, xz, xy) indices for 3×3 stress/virial.
_VOIGT_MAP: list[tuple[int, int]] = [
    (0, 0),
    (1, 1),
    (2, 2),
    (1, 2),
    (0, 2),
    (0, 1),
]


def _voigt_to_matrix(voigt: np.ndarray) -> np.ndarray:
    """Convert a 6-element Voigt-notation vector to a 3×3 symmetric matrix.

    Parameters
    ----------
    voigt : np.ndarray
        Flat array of length 6.

    Returns
    -------
    np.ndarray
        Symmetric 3×3 matrix.
    """
    mat = np.zeros((3, 3), dtype=voigt.dtype)
    for idx, (i, j) in enumerate(_VOIGT_MAP):
        mat[i, j] = voigt[idx]
        mat[j, i] = voigt[idx]
    return mat


def _to_f64(arr: np.ndarray) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Cast *arr* to float64, sharing memory when already that dtype.

    Parameters
    ----------
    arr : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Array with ``dtype=np.float64``.
    """
    return np.asarray(arr, dtype=np.float64)


def _to_i64(arr: np.ndarray) -> np.ndarray[tuple[int, ...], np.dtype[np.int64]]:
    """Cast *arr* to int64, sharing memory when already that dtype.

    Parameters
    ----------
    arr : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Array with ``dtype=np.int64``.
    """
    return np.asarray(arr, dtype=np.int64)


# Fields from the raw LMDB row that are consumed by _atomic_data_from_row
# and should NOT be forwarded to the ``info`` dict.
_CONSUMED_KEYS = _METADATA_KEYS | frozenset(
    {
        "numbers",
        "positions",
        "pbc",
        "cell",
        "energy",
        "forces",
        "stress",
        "virials",
        "dipole",
        "charges",
        "charge",
        "masses",
        "constraints",
        "initial_charges",
        "initial_magmoms",
        "momenta",
        "tags",
    }
)


def _atomic_data_from_row(
    row: dict[str, object],
    dtype: torch.dtype | None = None,
) -> AtomicData:
    """Build an :class:`~nvalchemi.data.AtomicData` from a raw Rust row dict.

    This mirrors the field extraction logic of
    :meth:`AtomicData.from_atoms` but works directly on the JSON-decoded
    dictionary returned by the Rust LMDB reader, skipping the
    :class:`ase.Atoms` intermediate entirely.

    Parameters
    ----------
    row : dict[str, object]
        Row dict as returned by :func:`physicsnemo_curator._lib.lmdb.read_lmdb`.
    dtype : torch.dtype or None, optional
        Torch dtype for floating-point tensors.  Defaults to
        ``torch.float32``.

    Returns
    -------
    AtomicData
        Constructed atomic data object.
    """
    import torch
    from nvalchemi.data import AtomicData

    if dtype is None:
        dtype = torch.float32

    # -- required fields ------------------------------------------------------
    numbers = row["numbers"]
    if not isinstance(numbers, np.ndarray):
        msg = f"Expected 'numbers' to be ndarray, got {type(numbers)}"
        raise TypeError(msg)
    atomic_numbers = torch.from_numpy(_to_i64(numbers))

    positions_raw = row["positions"]
    if not isinstance(positions_raw, np.ndarray):
        msg = f"Expected 'positions' to be ndarray, got {type(positions_raw)}"
        raise TypeError(msg)
    positions_t = torch.from_numpy(_to_f64(positions_raw)).to(dtype)

    # -- optional structure fields --------------------------------------------
    pbc_t: torch.Tensor | None = None
    cell_t: torch.Tensor | None = None
    energies_t: torch.Tensor | None = None
    forces_t: torch.Tensor | None = None
    stresses_t: torch.Tensor | None = None
    virials_t: torch.Tensor | None = None
    dipoles_t: torch.Tensor | None = None
    node_charges_t: torch.Tensor | None = None
    graph_charges_t: torch.Tensor | None = None
    atomic_masses_t: torch.Tensor | None = None

    pbc_arr = row.get("pbc")
    if isinstance(pbc_arr, np.ndarray) and bool(np.any(pbc_arr)):
        pbc_t = torch.from_numpy(pbc_arr).unsqueeze(0)

        cell_arr = row.get("cell")
        if isinstance(cell_arr, np.ndarray):
            cell_t = torch.from_numpy(_to_f64(cell_arr)).to(dtype).unsqueeze(0)

    # -- calculator / energy fields -------------------------------------------
    energy = row.get("energy")
    if energy is not None:
        if isinstance(energy, (int, float)):
            energies_t = torch.tensor([[energy]], dtype=dtype)
        elif isinstance(energy, np.ndarray):
            energies_t = torch.from_numpy(_to_f64(energy)).to(dtype).reshape(1, 1)

    forces_raw = row.get("forces")
    if isinstance(forces_raw, np.ndarray):
        forces_t = torch.from_numpy(_to_f64(forces_raw)).to(dtype)

    stress = row.get("stress")
    if isinstance(stress, np.ndarray):
        if stress.shape == (6,):
            stress = _voigt_to_matrix(stress)
        elif stress.shape == (9,):
            stress = np.asarray(stress).reshape(3, 3)
        stresses_t = torch.from_numpy(_to_f64(stress)).to(dtype).reshape(1, 3, 3)

    virials_raw = row.get("virials")
    if isinstance(virials_raw, np.ndarray):
        if virials_raw.shape == (6,):
            virials_raw = _voigt_to_matrix(virials_raw)
        elif virials_raw.shape == (9,):
            virials_raw = np.asarray(virials_raw).reshape(3, 3)
        virials_t = torch.from_numpy(_to_f64(virials_raw)).to(dtype).reshape(1, 3, 3)

    dipole = row.get("dipole")
    if isinstance(dipole, np.ndarray):
        dipoles_t = torch.from_numpy(_to_f64(dipole)).to(dtype).reshape(1, 3)

    charges = row.get("charges")
    if isinstance(charges, np.ndarray):
        node_charges_t = torch.from_numpy(_to_f64(charges)).to(dtype)
        if node_charges_t.ndim == 1:
            node_charges_t = node_charges_t.unsqueeze(-1)

    charge = row.get("charge")
    if charge is not None and isinstance(charge, (int, float)):
        graph_charges_t = torch.tensor([[float(charge)]], dtype=dtype)

    # -- masses (from row if available, else periodictable lookup) -------------
    masses = row.get("masses")
    if isinstance(masses, np.ndarray):
        atomic_masses_t = torch.from_numpy(_to_f64(masses)).to(dtype)

    # -- extra info fields (tensor-convertible values only) -------------------
    info_dict: dict[str, torch.Tensor] = {}
    for key, val in row.items():
        if key in _CONSUMED_KEYS:
            continue
        if isinstance(val, np.ndarray):
            info_dict[key] = torch.from_numpy(_to_f64(val)).to(dtype)
        elif isinstance(val, (int, float)):
            info_dict[key] = torch.tensor(val, dtype=dtype)

    return AtomicData(
        atomic_numbers=atomic_numbers,
        positions=positions_t,
        pbc=pbc_t,
        cell=cell_t,
        energies=energies_t,
        forces=forces_t,
        stresses=stresses_t,
        virials=virials_t,
        dipoles=dipoles_t,
        node_charges=node_charges_t,
        graph_charges=graph_charges_t,
        atomic_masses=atomic_masses_t,
        info=info_dict if info_dict else {},
    )


class ASELMDBSource(Source["AtomicData"]):
    """Read atomic data from ASE LMDB (``.aselmdb``) database files.

    The source discovers all ``.aselmdb`` files under *data_dir* matching
    *file_pattern*, sorted lexicographically.  Each file is treated as one
    source index containing many atomic structures.  Calling ``source[i]``
    returns a generator that opens the *i*-th database file and yields one
    :class:`~nvalchemi.data.AtomicData` per row.

    This source is compatible with any dataset stored in the ``.aselmdb``
    format, including local extracts
    of the `OMol25 <https://huggingface.co/facebook/OMol25>`_ and
    `OPoly26 <https://arxiv.org/abs/2512.23117>`_ datasets.

    An optional ``metadata.npz`` file (same directory or explicit path) is
    loaded eagerly if present.  It is not required for operation.

    Parameters
    ----------
    data_dir : str
        Directory containing ``.aselmdb`` files.
    file_pattern : str
        Glob pattern for file discovery.  Defaults to ``"**/*.aselmdb"``
        which recursively finds all ``.aselmdb`` files.  Use
        ``"*.aselmdb"`` to restrict to the top-level directory.
    metadata_path : str
        Optional path to a ``metadata.npz`` file.  Empty string (default)
        means auto-detect ``<data_dir>/metadata.npz``.
    backend : str
        Read backend: ``"python"`` (default) uses the ASE database API,
        ``"rust"`` uses the native Rust reader for faster I/O.  Falls
        back to ``"python"`` if the Rust extension is unavailable.

    Note
    ----
    - Dataset: `OMol25 <https://huggingface.co/facebook/OMol25>`_
    - License: `CC-BY-4.0 <https://creativecommons.org/licenses/by/4.0/>`_
      (dataset), `FAIR Chemistry License
      <https://huggingface.co/facebook/OMol25/blob/main/LICENSE>`_ (models)
    - Paper: `arXiv:2505.08762 <https://arxiv.org/abs/2505.08762>`_

    Examples
    --------
    >>> source = ASELMDBSource(data_dir="./val/")  # doctest: +SKIP
    >>> len(source)  # doctest: +SKIP
    80
    >>> atomic_data = next(source[0])  # doctest: +SKIP
    """

    name: ClassVar[str] = "ASE LMDB"
    description: ClassVar[str] = "Read atomic data from ASE LMDB (.aselmdb) files"

    @classmethod
    def params(cls) -> list[Param]:
        """Return configurable parameters for this source.

        Returns
        -------
        list[Param]
            Parameter list for CLI configuration.
        """
        return [
            Param(
                name="data_dir",
                description="Directory containing .aselmdb files",
                type=str,
            ),
            Param(
                name="file_pattern",
                description="Glob pattern for .aselmdb file discovery",
                type=str,
                default="**/*.aselmdb",
            ),
            Param(
                name="metadata_path",
                description="Path to metadata.npz (empty = auto-detect in data_dir)",
                type=str,
                default="",
            ),
            Param(
                name="backend",
                description="Read backend: 'python' (ASE) or 'rust' (native reader)",
                type=str,
                default="python",
                choices=["python", "rust"],
            ),
        ]

    def __init__(
        self,
        data_dir: str,
        file_pattern: str = "**/*.aselmdb",
        metadata_path: str = "",
        backend: Literal["python", "rust"] = "python",
    ) -> None:
        self._data_dir = pathlib.Path(data_dir)
        self._backend: Literal["python", "rust"] = backend

        # Validate backend choice and fall back gracefully.
        if backend == "rust":
            try:
                from physicsnemo_curator._lib.lmdb import read_lmdb as _  # noqa: F401
            except (ImportError, ModuleNotFoundError):
                logger.warning("Rust LMDB backend unavailable; falling back to 'python'.")
                self._backend = "python"

        root = pathlib.Path(data_dir)
        if not root.is_dir():
            msg = f"Path {root} is not a directory."
            raise FileNotFoundError(msg)

        self._root = root.resolve()

        # Discover .aselmdb files eagerly (sorted for deterministic ordering).
        self._db_files: list[pathlib.Path] = sorted(
            p.resolve() for p in root.glob(file_pattern) if p.is_file() and p.suffix == ".aselmdb"
        )
        if not self._db_files:
            msg = f"No .aselmdb files found in {self._data_dir} with pattern {file_pattern!r}"
            raise ValueError(msg)

        logger.info(
            "Discovered %d .aselmdb files in %s (backend=%s)",
            len(self._db_files),
            self._data_dir,
            self._backend,
        )

        # Build cumulative offset index for flat structure-level indexing.
        # _cumulative_counts[i] = total structures in files 0..i-1
        # So _cumulative_counts[0] = 0, _cumulative_counts[1] = count of file 0, etc.
        self._row_counts: list[int] = []
        self._cumulative_counts: list[int] = [0]
        logger.info("Counting structures in %d files...", len(self._db_files))
        for db_path in self._db_files:
            count = _count_aselmdb_rows(db_path)
            self._row_counts.append(count)
            self._cumulative_counts.append(self._cumulative_counts[-1] + count)
        self._total_structures = self._cumulative_counts[-1]
        logger.info(
            "Indexed %d total structures across %d files",
            self._total_structures,
            len(self._db_files),
        )

        # Load optional metadata.
        self._metadata: dict[str, np.ndarray] = {}
        meta_path = pathlib.Path(metadata_path) if metadata_path else self._data_dir / "metadata.npz"
        if meta_path.exists():
            with np.load(str(meta_path)) as npz:
                self._metadata = dict(npz)
            logger.info("Loaded metadata from %s (keys: %s)", meta_path, list(self._metadata.keys()))

    # -- Source interface -----------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of structures across all files."""
        return self._total_structures

    def __getitem__(self, index: int) -> Generator[AtomicData]:
        """Yield the *index*-th :class:`~nvalchemi.data.AtomicData` structure.

        Uses flat indexing across all files: index 0 is the first structure
        in the first file, and indices continue sequentially through all
        files.

        Parameters
        ----------
        index : int
            Zero-based structure index (supports negative indexing).

        Yields
        ------
        AtomicData
            A single atomic data object.

        Raises
        ------
        IndexError
            If *index* is out of range.
        """
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            msg = f"Index {index} out of range for source with {len(self)} structures."
            raise IndexError(msg)

        # Binary search to find which file contains this index
        file_index = self._find_file_for_index(index)
        row_index = index - self._cumulative_counts[file_index]

        db_path = self._db_files[file_index]
        logger.info(
            "Reading index %d from file %d (%s), row %d",
            index,
            file_index,
            db_path.name,
            row_index,
        )

        if self._backend == "rust":
            yield self._read_single_rust(db_path, row_index)
        else:
            yield self._read_single_python(db_path, row_index)

    def _find_file_for_index(self, index: int) -> int:
        """Find which file contains the given global structure index.

        Uses binary search on the cumulative counts.

        Parameters
        ----------
        index : int
            Global structure index.

        Returns
        -------
        int
            File index (0-based).
        """
        import bisect

        # bisect_right returns insertion point; subtract 1 to get file index
        return bisect.bisect_right(self._cumulative_counts, index) - 1

    def partition_indices(self, indices: list[int]) -> list[list[int]] | None:
        """Group indices by source ``.aselmdb`` file.

        Groups indices so that all structures from the same file are
        processed by the same worker.  This improves data locality
        (one env open per worker) and avoids redundant full-file scans
        when using the Python backend (which iterates all keys to find
        a row by index).

        Parameters
        ----------
        indices : list[int]
            The structure indices to partition.

        Returns
        -------
        list[list[int]] | None
            Groups of indices, one per file that is accessed.
            Returns ``None`` if all indices come from the same file
            (no partitioning needed).
        """
        from collections import defaultdict

        file_groups: dict[int, list[int]] = defaultdict(list)
        for idx in indices:
            file_idx = self._find_file_for_index(idx)
            file_groups[file_idx].append(idx)

        if len(file_groups) <= 1:
            return None

        # Return groups sorted by file order, indices sorted within.
        groups = [sorted(file_groups[k]) for k in sorted(file_groups)]
        return groups

    # -- Backend implementations ----------------------------------------------

    def _read_python(self, db_path: pathlib.Path) -> Generator[AtomicData]:
        """Read rows using a pure-Python LMDB reader.

        Opens the ``.aselmdb`` file directly with the ``lmdb`` package,
        decompresses each zlib-compressed JSON value, converts
        ``__ndarray__`` markers to NumPy arrays, and builds
        :class:`~nvalchemi.data.AtomicData` objects from the raw dicts.

        Parameters
        ----------
        db_path : pathlib.Path
            Path to the ``.aselmdb`` file.

        Yields
        ------
        AtomicData
            One atomic data object per database row.
        """
        for row in _read_aselmdb_rows(db_path):
            yield _atomic_data_from_row(row)

    def _read_rust(self, db_path: pathlib.Path) -> Generator[AtomicData]:
        """Read rows using the native Rust LMDB reader.

        Reads the entire database into memory using the Rust extension,
        then converts each raw row dict to an
        :class:`~nvalchemi.data.AtomicData` directly—bypassing the
        :class:`ase.Atoms` intermediate.

        Parameters
        ----------
        db_path : pathlib.Path
            Path to the ``.aselmdb`` file.

        Yields
        ------
        AtomicData
            One atomic data object per database row.
        """
        from physicsnemo_curator._lib.lmdb import read_lmdb

        rows = read_lmdb(str(db_path))
        for row in rows:
            yield _atomic_data_from_row(row)

    def _read_single_python(self, db_path: pathlib.Path, row_index: int) -> AtomicData:
        """Read a single row using a pure-Python LMDB reader.

        Parameters
        ----------
        db_path : pathlib.Path
            Path to the ``.aselmdb`` file.
        row_index : int
            Zero-based row index within this file.

        Returns
        -------
        AtomicData
            The atomic data object for the requested row.
        """
        row = _read_aselmdb_row_at(db_path, row_index)
        return _atomic_data_from_row(row)

    def _read_single_rust(self, db_path: pathlib.Path, row_index: int) -> AtomicData:
        """Read a single row using the native Rust LMDB reader.

        Currently reads all rows and indexes into them. A future optimization
        could add a Rust function to read a single row by index.

        Parameters
        ----------
        db_path : pathlib.Path
            Path to the ``.aselmdb`` file.
        row_index : int
            Zero-based row index within this file.

        Returns
        -------
        AtomicData
            The atomic data object for the requested row.
        """
        from physicsnemo_curator._lib.lmdb import read_lmdb

        rows = read_lmdb(str(db_path))
        if row_index < 0 or row_index >= len(rows):
            msg = f"Row index {row_index} out of range for file with {len(rows)} rows."
            raise IndexError(msg)
        return _atomic_data_from_row(rows[row_index])

    # -- Properties -----------------------------------------------------------

    @property
    def backend(self) -> str:
        """Return the active read backend name."""
        return self._backend

    @property
    def data_dir(self) -> pathlib.Path:
        """Return the data directory path."""
        return self._data_dir

    @property
    def db_files(self) -> list[pathlib.Path]:
        """Return the list of discovered database file paths."""
        return list(self._db_files)

    @property
    def metadata(self) -> dict[str, np.ndarray]:
        """Return loaded metadata arrays, if any."""
        return dict(self._metadata)

    @property
    def num_files(self) -> int:
        """Return the number of ``.aselmdb`` files discovered."""
        return len(self._db_files)

    @property
    def row_counts(self) -> list[int]:
        """Return the number of structures in each file.

        Returns
        -------
        list[int]
            List where ``row_counts[i]`` is the structure count in the *i*-th file.
        """
        return list(self._row_counts)

    @property
    def root(self) -> pathlib.Path:
        """Return the root directory of this source.

        Returns
        -------
        pathlib.Path
            The resolved root directory containing the discovered files.
        """
        return self._root

    def relative_path(self, index: int) -> str:
        """Return the relative path of the file containing structure *index*.

        This is used by sinks (e.g.
        :class:`~physicsnemo_curator.domains.atm.sinks.zarr_writer.AtomicDataZarrSink`)
        to resolve ``{relpath}`` and ``{stem}`` naming placeholders,
        enabling output directory layouts that mirror the input.

        Parameters
        ----------
        index : int
            Zero-based structure index (global across all files).

        Returns
        -------
        str
            POSIX-style relative path (e.g. ``"subdir/data.aselmdb"``).
        """
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            msg = f"Index {index} out of range for source with {len(self)} structures."
            raise IndexError(msg)
        file_index = self._find_file_for_index(index)
        return self._db_files[file_index].relative_to(self._root).as_posix()
