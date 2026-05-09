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

"""Random atomic data source for testing and examples.

Generates synthetic :class:`~nvalchemi.data.AtomicData` objects with
random positions, forces, and energies.  Useful for unit tests, example
pipelines, and quick prototyping without needing real DFT data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

from physicsnemo_curator.core.base import Param, Source

if TYPE_CHECKING:
    from collections.abc import Generator

    from nvalchemi.data import AtomicData


class RandomAtomicSource(Source["AtomicData"]):
    """Generate random atomic structures with configurable properties.

    Each index yields a single :class:`~nvalchemi.data.AtomicData` with
    random positions, atomic numbers, forces, and energies.

    Parameters
    ----------
    n_samples : int
        Number of structures this source provides (i.e. ``len(source)``).
    n_atoms : int
        Number of atoms per structure.
    seed : int
        Base random seed.  Each index uses ``seed + index`` for
        reproducibility.

    Examples
    --------
    >>> from physicsnemo_curator.domains.atm.sources import RandomAtomicSource
    >>> source = RandomAtomicSource(n_samples=10, n_atoms=20)
    >>> len(source)
    10
    >>> atoms = next(source[0])
    >>> atoms.positions.shape
    torch.Size([20, 3])
    """

    name: ClassVar[str] = "Random Atomic"
    description: ClassVar[str] = (
        "Generate random atomic structures with positions, forces, and energies for testing and prototyping"
    )

    @classmethod
    def params(cls) -> list[Param]:
        """Declare configurable parameters."""
        return [
            Param(name="n_samples", description="Number of structures to generate", type=int, default=10),
            Param(name="n_atoms", description="Number of atoms per structure", type=int, default=20),
            Param(name="seed", description="Base random seed for reproducibility", type=int, default=42),
        ]

    def __init__(
        self,
        n_samples: int = 10,
        n_atoms: int = 20,
        seed: int = 42,
    ) -> None:
        """Initialize the random atomic source."""
        self._n_samples = n_samples
        self._n_atoms = n_atoms
        self._seed = seed

    def __len__(self) -> int:
        """Return the number of structures available."""
        return self._n_samples

    def __getitem__(self, index: int) -> Generator[AtomicData]:
        """Yield a random atomic structure for the given index.

        Parameters
        ----------
        index : int
            Zero-based index into the source.

        Yields
        ------
        AtomicData
            A randomly generated atomic structure.
        """
        from nvalchemi.data import AtomicData

        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self)})")

        rng = np.random.default_rng(self._seed + index)

        # Random atomic numbers from common elements (H, C, N, O, F, Si, S)
        elements = np.array([1, 6, 7, 8, 9, 14, 16], dtype=np.int64)
        atomic_numbers = torch.from_numpy(rng.choice(elements, size=self._n_atoms))

        # Random positions in a 10 Angstrom box
        positions = torch.from_numpy(rng.uniform(0.0, 10.0, size=(self._n_atoms, 3)).astype(np.float64))

        # Random unit cell (cubic, side length 10 A)
        cell = torch.eye(3, dtype=torch.float64) * 10.0

        # No periodic boundary conditions
        pbc = torch.zeros(3, dtype=torch.bool)

        # Random energy (eV, typical DFT range)
        energies = torch.tensor(rng.uniform(-500.0, -100.0), dtype=torch.float64)

        # Random forces (eV/A)
        forces = torch.from_numpy(rng.standard_normal((self._n_atoms, 3)).astype(np.float64) * 0.5)

        # Random stresses (6-component Voigt notation, GPa)
        stresses = torch.from_numpy(rng.standard_normal(6).astype(np.float64) * 0.01)

        yield AtomicData(
            atomic_numbers=atomic_numbers,
            positions=positions,
            cell=cell,
            pbc=pbc,
            energies=energies,
            forces=forces,
            stresses=stresses,
        )
