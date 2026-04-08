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

Reads ``.aselmdb`` files produced by `ASE database backends
<https://github.com/NVIDIA/ase-db-backends>`_ and yields
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

- **python** (default): Uses the ASE database API to read rows and
  :meth:`AtomicData.from_atoms` for conversion.  Guaranteed compatibility
  with all ASE row features (constraints, calculator results, etc.).
- **rust**: Uses a native Rust reader
  (:func:`physicsnemo.curator._lib.lmdb.read_lmdb`) for I/O, zlib
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
- ASE database backends: https://github.com/NVIDIA/ase-db-backends

Examples
--------
>>> source = ASELMDBSource(data_dir="./val/")  # doctest: +SKIP
>>> len(source)  # number of .aselmdb files  # doctest: +SKIP
80
>>> atomic_data = next(source[0])  # doctest: +SKIP

Use the Rust backend for faster reads:

>>> source = ASELMDBSource(data_dir="./val/", backend="rust")  # doctest: +SKIP
"""

from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np

from physicsnemo.curator.core.base import Param, Source

if TYPE_CHECKING:
    from collections.abc import Generator

    import torch
    from nvalchemi.data import AtomicData

logger = logging.getLogger(__name__)

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
        Row dict as returned by :func:`physicsnemo.curator._lib.lmdb.read_lmdb`.
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

    The source discovers all ``.aselmdb`` files under *data_dir*, sorted
    lexicographically.  Each file is treated as one source index containing
    many atomic structures.  Calling ``source[i]`` returns a generator that
    opens the *i*-th database file and yields one
    :class:`~nvalchemi.data.AtomicData` per row.

    This source is compatible with any dataset stored in the ``.aselmdb``
    format produced by `ase-db-backends
    <https://github.com/NVIDIA/ase-db-backends>`_, including local extracts
    of the `OMol25 <https://huggingface.co/facebook/OMol25>`_ and
    `OPoly26 <https://arxiv.org/abs/2512.23117>`_ datasets.

    An optional ``metadata.npz`` file (same directory or explicit path) is
    loaded eagerly if present.  It is not required for operation.

    Parameters
    ----------
    data_dir : str
        Directory containing ``.aselmdb`` files.
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
        metadata_path: str = "",
        backend: Literal["python", "rust"] = "python",
    ) -> None:
        self._data_dir = pathlib.Path(data_dir)
        self._backend: Literal["python", "rust"] = backend

        # Validate backend choice and fall back gracefully.
        if backend == "rust":
            try:
                from physicsnemo.curator._lib.lmdb import read_lmdb as _  # noqa: F401
            except (ImportError, ModuleNotFoundError):
                logger.warning("Rust LMDB backend unavailable; falling back to 'python'.")
                self._backend = "python"

        # Discover .aselmdb files eagerly (sorted for deterministic ordering).
        self._db_files: list[pathlib.Path] = sorted(self._data_dir.glob("*.aselmdb"))
        if not self._db_files:
            msg = f"No .aselmdb files found in {self._data_dir}"
            raise ValueError(msg)

        logger.info(
            "Discovered %d .aselmdb files in %s (backend=%s)",
            len(self._db_files),
            self._data_dir,
            self._backend,
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
        """Return the number of ``.aselmdb`` files discovered."""
        return len(self._db_files)

    def __getitem__(self, index: int) -> Generator[AtomicData]:
        """Yield :class:`~nvalchemi.data.AtomicData` for every row in the *index*-th database.

        Parameters
        ----------
        index : int
            Zero-based file index (supports negative indexing).

        Yields
        ------
        AtomicData
            One atomic data object per database row.

        Raises
        ------
        IndexError
            If *index* is out of range.
        """
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            msg = f"Index {index} out of range for source with {len(self)} items."
            raise IndexError(msg)

        db_path = self._db_files[index]
        logger.debug("Opening database: %s (backend=%s)", db_path, self._backend)

        if self._backend == "rust":
            yield from self._read_rust(db_path)
        else:
            yield from self._read_python(db_path)

    # -- Backend implementations ----------------------------------------------

    def _read_python(self, db_path: pathlib.Path) -> Generator[AtomicData]:
        """Read rows using the Python ASE database API.

        Parameters
        ----------
        db_path : pathlib.Path
            Path to the ``.aselmdb`` file.

        Yields
        ------
        AtomicData
            One atomic data object per database row.
        """
        import ase.db
        from nvalchemi.data import AtomicData

        db = ase.db.connect(str(db_path), type="aselmdb", readonly=True)
        try:
            for row in db.select():
                atoms = row.toatoms()
                yield AtomicData.from_atoms(atoms)
        finally:
            if hasattr(db, "close"):
                db.close()

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
        from physicsnemo.curator._lib.lmdb import read_lmdb

        rows = read_lmdb(str(db_path))
        for row in rows:
            yield _atomic_data_from_row(row)

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
