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
:class:`~nvalchemi.data.AtomicData` instance via
:meth:`AtomicData.from_atoms`.

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
"""

from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from physicsnemo_curator.core.base import Param, Source

if TYPE_CHECKING:
    from collections.abc import Generator

    from nvalchemi.data import AtomicData

logger = logging.getLogger(__name__)


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
        ]

    def __init__(
        self,
        data_dir: str,
        metadata_path: str = "",
    ) -> None:
        self._data_dir = pathlib.Path(data_dir)

        # Discover .aselmdb files eagerly (sorted for deterministic ordering).
        self._db_files: list[pathlib.Path] = sorted(self._data_dir.glob("*.aselmdb"))
        if not self._db_files:
            msg = f"No .aselmdb files found in {self._data_dir}"
            raise ValueError(msg)

        logger.info("Discovered %d .aselmdb files in %s", len(self._db_files), self._data_dir)

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

        import ase.db
        from nvalchemi.data import AtomicData

        db_path = self._db_files[index]
        logger.debug("Opening database: %s", db_path)

        db = ase.db.connect(str(db_path), type="aselmdb", readonly=True)
        try:
            for row in db.select():
                atoms = row.toatoms()
                yield AtomicData.from_atoms(atoms)
        finally:
            # Ensure the database connection is closed even if iteration
            # is interrupted.
            if hasattr(db, "close"):
                db.close()

    # -- Properties -----------------------------------------------------------

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
