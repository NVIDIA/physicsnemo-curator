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

"""ASV benchmarks comparing Rust vs Python d3plot post-processing functions.

Measures three hot paths:
- K-file parsing (``_parse_k_file`` vs Rust ``parse_k_file``)
- Node thickness scatter-accumulate (``_compute_node_thickness`` vs Rust)
- Von Mises stress computation (``_von_mises_from_voigt`` vs Rust)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

# ── helpers ──────────────────────────────────────────────────────────────────


def _create_k_file(path: Path, n_parts: int) -> None:
    """Create a synthetic ``.k`` file with *n_parts* part/section pairs.

    Parameters
    ----------
    path : Path
        Output file path.
    n_parts : int
        Number of ``*PART`` / ``*SECTION_SHELL`` entries.
    """
    lines = ["$", "*KEYWORD"]
    for i in range(1, n_parts + 1):
        lines.append("*PART")
        lines.append(f"Part_{i}")
        lines.append(f"       {i}       {i}       {i}")
    lines.append("*SECTION_SHELL")
    for i in range(1, n_parts + 1):
        lines.append(f"       {i}")
        t = float(i) * 0.5
        lines.append(f"     {t:.1f}     {t:.1f}     {t:.1f}     {t:.1f}")
    lines.append("*END")
    path.write_text("\n".join(lines))


def _make_connectivity(n_elements: int, n_nodes: int, nodes_per_cell: int = 4) -> np.ndarray:
    """Create random element connectivity array.

    Parameters
    ----------
    n_elements : int
        Number of elements.
    n_nodes : int
        Number of nodes.
    nodes_per_cell : int
        Nodes per element.

    Returns
    -------
    np.ndarray
        Shape ``(n_elements, nodes_per_cell)``, dtype int64.
    """
    rng = np.random.default_rng(42)
    return rng.integers(0, n_nodes, size=(n_elements, nodes_per_cell), dtype=np.int64)


# ── K-file parsing benchmarks ───────────────────────────────────────────────


class TimeParseKFile:
    """Benchmark k-file parsing: Python vs Rust."""

    params = [10, 100, 500]
    param_names = ["n_parts"]

    def setup(self, n_parts: int) -> None:
        """Create a temporary k-file."""
        self._tmpdir = tempfile.TemporaryDirectory()
        self._k_file = Path(self._tmpdir.name) / "test.k"
        _create_k_file(self._k_file, n_parts)

    def teardown(self, n_parts: int) -> None:
        """Clean up temporary directory."""
        self._tmpdir.cleanup()

    def time_python(self, n_parts: int) -> None:
        """Parse k-file with Python backend."""
        from physicsnemo.curator.mesh.sources.d3plot import _parse_k_file

        _parse_k_file(self._k_file)

    def time_rust(self, n_parts: int) -> None:
        """Parse k-file with Rust backend."""
        from physicsnemo.curator.mesh.sources.d3plot import _parse_k_file_rust

        _parse_k_file_rust(self._k_file)


# ── Node thickness benchmarks ───────────────────────────────────────────────


class TimeComputeNodeThickness:
    """Benchmark node thickness scatter-accumulate: Python vs Rust."""

    params = [1_000, 10_000, 100_000]
    param_names = ["n_elements"]

    def setup(self, n_elements: int) -> None:
        """Create synthetic connectivity and part data."""
        n_nodes = n_elements * 2
        self._connectivity = _make_connectivity(n_elements, n_nodes)
        self._part_ids = np.random.default_rng(42).integers(1, 4, size=n_elements, dtype=np.int64)
        self._actual_part_ids = np.array([0, 10, 20, 30], dtype=np.int64)
        self._part_thickness = {10: 2.5, 20: 4.0, 30: 1.5}

    def time_python(self, n_elements: int) -> None:
        """Compute node thickness with Python backend."""
        from physicsnemo.curator.mesh.sources.d3plot import _compute_node_thickness

        _compute_node_thickness(self._connectivity, self._part_ids, self._part_thickness, self._actual_part_ids)

    def time_rust(self, n_elements: int) -> None:
        """Compute node thickness with Rust backend."""
        from physicsnemo.curator.mesh.sources.d3plot import _compute_node_thickness_rust

        _compute_node_thickness_rust(self._connectivity, self._part_ids, self._part_thickness, self._actual_part_ids)


# ── Von Mises stress benchmarks ─────────────────────────────────────────────


class TimeVonMises:
    """Benchmark von Mises stress computation: Python vs Rust."""

    params = [1_000, 10_000, 100_000]
    param_names = ["n_entries"]

    def setup(self, n_entries: int) -> None:
        """Create random Voigt stress tensor."""
        rng = np.random.default_rng(42)
        self._stress = rng.uniform(-200, 200, size=(n_entries, 6)).astype(np.float64)

    def time_python(self, n_entries: int) -> None:
        """Compute von Mises with Python (numpy vectorized)."""
        from physicsnemo.curator.mesh.sources.d3plot import _von_mises_from_voigt

        _von_mises_from_voigt(self._stress)

    def time_rust(self, n_entries: int) -> None:
        """Compute von Mises with Rust backend."""
        from physicsnemo.curator.mesh.sources.d3plot import _von_mises_from_voigt_rust

        _von_mises_from_voigt_rust(self._stress)


# ── Memory benchmarks ───────────────────────────────────────────────────────


class PeakmemNodeThickness:
    """Peak memory for node thickness computation."""

    params = [10_000, 100_000]
    param_names = ["n_elements"]

    def setup(self, n_elements: int) -> None:
        """Create synthetic data."""
        n_nodes = n_elements * 2
        self._connectivity = _make_connectivity(n_elements, n_nodes)
        self._part_ids = np.random.default_rng(42).integers(1, 4, size=n_elements, dtype=np.int64)
        self._actual_part_ids = np.array([0, 10, 20, 30], dtype=np.int64)
        self._part_thickness = {10: 2.5, 20: 4.0, 30: 1.5}

    def peakmem_python(self, n_elements: int) -> None:
        """Peak memory with Python backend."""
        from physicsnemo.curator.mesh.sources.d3plot import _compute_node_thickness

        _compute_node_thickness(self._connectivity, self._part_ids, self._part_thickness, self._actual_part_ids)

    def peakmem_rust(self, n_elements: int) -> None:
        """Peak memory with Rust backend."""
        from physicsnemo.curator.mesh.sources.d3plot import _compute_node_thickness_rust

        _compute_node_thickness_rust(self._connectivity, self._part_ids, self._part_thickness, self._actual_part_ids)
