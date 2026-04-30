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

"""Tests for D3PlotSource.

Unit tests use mock d3plot data created via lasso to avoid requiring
real LS-DYNA simulation files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

try:
    import lasso  # noqa: F401

    _has_lasso = True
except ImportError:
    _has_lasso = False

if TYPE_CHECKING:
    import pathlib


# ---------------------------------------------------------------------------
# Helpers — write mock d3plot directories
# ---------------------------------------------------------------------------


def _write_mock_d3plot_dir(
    root: pathlib.Path,
    n_runs: int = 2,
    n_points: int = 8,
    n_cells: int = 4,
    n_timesteps: int = 3,
    write_k_file: bool = True,
    write_stress: bool = True,
) -> None:
    """Create minimal mock run directories with a d3plot sentinel file.

    Since lasso.dyna.D3plot reads a proprietary binary format that we
    cannot easily create, tests mock the D3plot constructor.  However,
    the directory structure must exist for discovery to work.

    Parameters
    ----------
    root : pathlib.Path
        Base directory for all runs.
    n_runs : int
        Number of run subdirectories to create.
    n_points : int
        Number of mesh nodes.
    n_cells : int
        Number of shell elements.
    n_timesteps : int
        Number of timesteps.
    write_k_file : bool
        If True, create a dummy .k file in each run.
    write_stress : bool
        If True, the mock will include stress arrays.
    """
    for i in range(n_runs):
        run_dir = root / f"Run{i:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        # Create sentinel d3plot file (empty — we mock the reader).
        (run_dir / "d3plot").touch()

        if write_k_file:
            k_content = """$
*KEYWORD
*PART
Part_1
       1       1       1
*SECTION_SHELL
       1
     2.5     2.5     2.5     2.5
*END
"""
            (run_dir / f"run{i:03d}.k").write_text(k_content)


def _make_mock_d3plot(
    n_points: int = 8,
    n_cells: int = 4,
    n_timesteps: int = 3,
    include_stress: bool = True,
) -> MagicMock:
    """Create a mock ``lasso.dyna.D3plot`` instance.

    Parameters
    ----------
    n_points : int
        Number of nodes.
    n_cells : int
        Number of shell elements.
    n_timesteps : int
        Number of timesteps.
    include_stress : bool
        Include stress and strain arrays.

    Returns
    -------
    MagicMock
        Mock D3plot with realistic array shapes.
    """
    rng = np.random.default_rng(42)

    mock_dp = MagicMock()

    # Use string-based keys for ArrayType mock access.
    arrays = {}
    arrays["node_coordinates"] = rng.uniform(-10, 10, size=(n_points, 3)).astype(np.float64)
    arrays["node_displacement"] = rng.uniform(-5, 5, size=(n_timesteps, n_points, 3)).astype(np.float64)
    # Make first timestep zero displacement.
    arrays["node_displacement"][0] = 0.0

    # Shell connectivity: random indices.
    connectivity = np.zeros((n_cells, 4), dtype=np.int64)
    for c in range(n_cells):
        connectivity[c] = rng.choice(n_points, size=4, replace=False)
    arrays["element_shell_node_indexes"] = connectivity

    arrays["element_shell_part_indexes"] = np.ones(n_cells, dtype=np.int64)
    arrays["part_ids"] = np.array([0, 1], dtype=np.int64)  # index 0 is unused

    if include_stress:
        arrays["element_shell_stress"] = rng.uniform(0, 100, size=(n_timesteps, n_cells, 2, 6)).astype(np.float64)
        arrays["element_shell_effective_plastic_strain"] = rng.uniform(0, 1, size=(n_timesteps, n_cells, 2)).astype(
            np.float64
        )

    # Build a dict-like mock that also supports attribute access via ArrayType.
    class _ArrayDict(dict):
        """Dict that maps ArrayType enum values to arrays."""

        def __getitem__(self, key: object) -> np.ndarray:
            """Get array by ArrayType or string key."""
            name = key.name if hasattr(key, "name") else str(key)
            return super().__getitem__(name)

        def __contains__(self, key: object) -> bool:
            """Check if key exists."""
            name = key.name if hasattr(key, "name") else str(key)
            return super().__contains__(name)

        def get(self, key: object, default: object = None) -> object:
            """Get array with default."""
            name = key.name if hasattr(key, "name") else str(key)
            return super().get(name, default)

    mock_dp.arrays = _ArrayDict(arrays)
    return mock_dp


# ---------------------------------------------------------------------------
# Unit tests — parameter descriptors and metadata
# ---------------------------------------------------------------------------


@pytest.mark.requires("mesh")
class TestD3PlotSourceUnit:
    """Unit tests for D3PlotSource metadata and params."""

    def test_params_list(self) -> None:
        """params() should return a non-empty list of Param objects."""
        from physicsnemo_curator.domains.mesh.sources.d3plot import D3PlotSource

        params = D3PlotSource.params()
        assert len(params) > 0
        names = [p.name for p in params]
        assert "input_dir" in names
        assert "read_stress" in names
        assert "read_k_file" in names

    def test_name_and_description(self) -> None:
        """Class should have name and description ClassVars."""
        from physicsnemo_curator.domains.mesh.sources.d3plot import D3PlotSource

        assert D3PlotSource.name == "LS-DYNA D3Plot"
        assert len(D3PlotSource.description) > 0


# ---------------------------------------------------------------------------
# Unit tests — source with mock d3plot data
# ---------------------------------------------------------------------------


@pytest.mark.requires("mesh")
@pytest.mark.skipif(not _has_lasso, reason="lasso (lasso-python) not installed")
class TestD3PlotSourceLocal:
    """Unit tests using mocked lasso D3plot reader."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: pathlib.Path) -> None:
        """Write mock run directories."""
        self.mock_root = tmp_path / "crash_sims"
        _write_mock_d3plot_dir(self.mock_root, n_runs=3)
        self.tmp_path = tmp_path

    def test_nonexistent_dir_raises(self, tmp_path: pathlib.Path) -> None:
        """FileNotFoundError should be raised for non-existent input_dir."""
        from physicsnemo_curator.domains.mesh.sources.d3plot import D3PlotSource

        with pytest.raises(FileNotFoundError):
            D3PlotSource(input_dir=str(tmp_path / "nonexistent"))

    def test_len(self) -> None:
        """Length should equal the number of run directories."""
        from physicsnemo_curator.domains.mesh.sources.d3plot import D3PlotSource

        source = D3PlotSource(input_dir=str(self.mock_root))
        assert len(source) == 3

    @patch("lasso.dyna.D3plot")
    def test_getitem_returns_mesh(self, mock_d3plot_cls: MagicMock) -> None:
        """__getitem__ should yield a physicsnemo Mesh."""
        from physicsnemo.mesh import Mesh

        from physicsnemo_curator.domains.mesh.sources.d3plot import D3PlotSource

        mock_d3plot_cls.return_value = _make_mock_d3plot()
        source = D3PlotSource(input_dir=str(self.mock_root))
        mesh = next(source[0])
        assert isinstance(mesh, Mesh)

    @patch("lasso.dyna.D3plot")
    def test_mesh_geometry(self, mock_d3plot_cls: MagicMock) -> None:
        """Mesh should have correct point and cell counts."""
        from physicsnemo_curator.domains.mesh.sources.d3plot import D3PlotSource

        mock_d3plot_cls.return_value = _make_mock_d3plot(n_points=8, n_cells=4)
        source = D3PlotSource(input_dir=str(self.mock_root))
        mesh = next(source[0])
        assert mesh.n_points == 8
        assert mesh.n_cells == 4

    @patch("lasso.dyna.D3plot")
    def test_mesh_has_displacement_fields(self, mock_d3plot_cls: MagicMock) -> None:
        """Mesh should have displacement_t* fields in point_data."""
        from physicsnemo_curator.domains.mesh.sources.d3plot import D3PlotSource

        mock_d3plot_cls.return_value = _make_mock_d3plot(n_timesteps=3)
        source = D3PlotSource(input_dir=str(self.mock_root))
        mesh = next(source[0])
        keys = set(mesh.point_data.keys())
        assert "displacement_t000" in keys
        assert "displacement_t001" in keys
        assert "displacement_t002" in keys

    @patch("lasso.dyna.D3plot")
    def test_mesh_has_thickness(self, mock_d3plot_cls: MagicMock) -> None:
        """Mesh should have thickness in point_data."""
        from physicsnemo_curator.domains.mesh.sources.d3plot import D3PlotSource

        mock_d3plot_cls.return_value = _make_mock_d3plot()
        source = D3PlotSource(input_dir=str(self.mock_root))
        mesh = next(source[0])
        assert "thickness" in mesh.point_data

    @patch("lasso.dyna.D3plot")
    def test_mesh_has_global_data(self, mock_d3plot_cls: MagicMock) -> None:
        """Mesh should carry num_timesteps as global data."""
        from physicsnemo_curator.domains.mesh.sources.d3plot import D3PlotSource

        mock_d3plot_cls.return_value = _make_mock_d3plot(n_timesteps=5)
        source = D3PlotSource(input_dir=str(self.mock_root))
        mesh = next(source[0])
        assert "num_timesteps" in mesh.global_data
        assert mesh.global_data["num_timesteps"].item() == 5

    @patch("lasso.dyna.D3plot")
    def test_read_stress_produces_cell_data(self, mock_d3plot_cls: MagicMock) -> None:
        """With read_stress=True, mesh should have stress/strain cell data."""
        from physicsnemo_curator.domains.mesh.sources.d3plot import D3PlotSource

        mock_d3plot_cls.return_value = _make_mock_d3plot(n_timesteps=2, include_stress=True)
        source = D3PlotSource(input_dir=str(self.mock_root), read_stress=True)
        mesh = next(source[0])
        assert mesh.cell_data is not None
        keys = set(mesh.cell_data.keys())
        assert "stress_vm_t000" in keys
        assert "stress_vm_t001" in keys
        assert "effective_plastic_strain_t000" in keys

    @patch("lasso.dyna.D3plot")
    def test_no_stress_gives_no_cell_data(self, mock_d3plot_cls: MagicMock) -> None:
        """With read_stress=False, cell_data should have no fields."""
        from physicsnemo_curator.domains.mesh.sources.d3plot import D3PlotSource

        mock_d3plot_cls.return_value = _make_mock_d3plot()
        source = D3PlotSource(input_dir=str(self.mock_root), read_stress=False)
        mesh = next(source[0])
        # Mesh always creates a TensorDict for cell_data, but it should be empty.
        assert len(mesh.cell_data.keys()) == 0

    @patch("lasso.dyna.D3plot")
    def test_negative_index(self, mock_d3plot_cls: MagicMock) -> None:
        """Negative indexing should work."""
        from physicsnemo.mesh import Mesh

        from physicsnemo_curator.domains.mesh.sources.d3plot import D3PlotSource

        mock_d3plot_cls.return_value = _make_mock_d3plot()
        source = D3PlotSource(input_dir=str(self.mock_root))
        mesh = next(source[-1])
        assert isinstance(mesh, Mesh)

    def test_out_of_range_raises(self) -> None:
        """Out-of-range index should raise IndexError."""
        from physicsnemo_curator.domains.mesh.sources.d3plot import D3PlotSource

        source = D3PlotSource(input_dir=str(self.mock_root))
        with pytest.raises(IndexError):
            next(source[99])

    @patch("lasso.dyna.D3plot")
    def test_no_k_file_gives_zero_thickness(self, mock_d3plot_cls: MagicMock, tmp_path: pathlib.Path) -> None:
        """Without a .k file, thickness should be all zeros."""
        from physicsnemo_curator.domains.mesh.sources.d3plot import D3PlotSource

        # Create runs without .k files.
        no_k_root = tmp_path / "no_k"
        _write_mock_d3plot_dir(no_k_root, n_runs=1, write_k_file=False)

        mock_d3plot_cls.return_value = _make_mock_d3plot()
        source = D3PlotSource(input_dir=str(no_k_root))
        mesh = next(source[0])
        assert torch.all(mesh.point_data["thickness"] == 0.0)  # ty: ignore[no-matching-overload]


# ---------------------------------------------------------------------------
# Unit tests — helper functions
# ---------------------------------------------------------------------------


@pytest.mark.requires("mesh")
class TestD3PlotHelpers:
    """Tests for d3plot helper functions."""

    def test_parse_k_file(self, tmp_path: pathlib.Path) -> None:
        """_parse_k_file should extract part thickness mapping."""
        from physicsnemo_curator.domains.mesh.sources.d3plot import _parse_k_file

        k_content = """$
*KEYWORD
*PART
Part_1
       1       1       1
*PART
Part_2
       2       2       2
*SECTION_SHELL
       1
     2.0     2.0     2.0     2.0
       2
     3.0     3.0     3.0     3.0
*END
"""
        k_file = tmp_path / "test.k"
        k_file.write_text(k_content)
        result = _parse_k_file(k_file)
        assert result[1] == pytest.approx(2.0)
        assert result[2] == pytest.approx(3.0)

    def test_von_mises_from_voigt(self) -> None:
        """_von_mises_from_voigt should compute correct stress."""
        from physicsnemo_curator.domains.mesh.sources.d3plot import _von_mises_from_voigt

        # Uniaxial stress: sigma_x = 100, rest = 0.
        sig = np.array([[[100.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        vm = _von_mises_from_voigt(sig)
        # von Mises for uniaxial = sigma_x.
        assert vm[0, 0] == pytest.approx(100.0, rel=1e-5)

    def test_reduce_shell_layers_scalar(self) -> None:
        """_reduce_shell_layers_scalar should average two layers."""
        from physicsnemo_curator.domains.mesh.sources.d3plot import _reduce_shell_layers_scalar

        x = np.array([[[1.0, 3.0], [2.0, 4.0]]])  # (1, 2, 2)
        result = _reduce_shell_layers_scalar(x)
        assert result.shape == (1, 2)
        assert result[0, 0] == pytest.approx(2.0)
        assert result[0, 1] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Unit tests — registry integration
# ---------------------------------------------------------------------------


@pytest.mark.requires("mesh")
class TestD3PlotRegistry:
    """Verify the source is registered in the mesh registry."""

    def test_source_registered(self) -> None:
        """D3PlotSource should appear in the mesh registry."""
        import physicsnemo_curator.domains.mesh  # noqa: F401
        from physicsnemo_curator.core.registry import registry

        sources = registry.list_sources("mesh")
        source_names = {s.name for s in sources}
        assert "LS-DYNA D3Plot" in source_names


# ---------------------------------------------------------------------------
# Consistency tests — Rust vs Python backends
# ---------------------------------------------------------------------------


@pytest.mark.requires("mesh")
class TestD3PlotRustConsistency:
    """Verify Rust d3plot backend produces identical results to Python."""

    def test_parse_k_file_consistency(self, tmp_path: pathlib.Path) -> None:
        """Rust parse_k_file must match Python parse_k_file exactly."""
        from physicsnemo_curator.domains.mesh.sources.d3plot import (
            _parse_k_file,
            _parse_k_file_rust,
        )

        k_content = """$
*KEYWORD
*PART
Part_1
       1       1       1
*PART
Part_2
       2       2       2
*PART
Part_3
       3       1       3
*SECTION_SHELL
       1
     2.5     2.5     2.5     2.5
       2
     4.0     4.0     4.0     4.0
*END
"""
        k_file = tmp_path / "test.k"
        k_file.write_text(k_content)

        py_result = _parse_k_file(k_file)
        rust_result = _parse_k_file_rust(k_file)

        assert set(py_result.keys()) == set(rust_result.keys()), (
            f"Key mismatch: Python={sorted(py_result.keys())}, Rust={sorted(rust_result.keys())}"
        )
        for pid in py_result:
            assert py_result[pid] == pytest.approx(rust_result[pid], abs=1e-10), (
                f"Part {pid}: Python={py_result[pid]}, Rust={rust_result[pid]}"
            )

    def test_compute_node_thickness_consistency(self) -> None:
        """Rust compute_node_thickness must match Python exactly."""
        from physicsnemo_curator.domains.mesh.sources.d3plot import (
            _compute_node_thickness,
            _compute_node_thickness_rust,
        )

        rng = np.random.default_rng(42)
        n_elements = 500
        n_nodes = 200
        nodes_per_cell = 4

        connectivity = rng.integers(0, n_nodes, size=(n_elements, nodes_per_cell), dtype=np.int64)
        part_ids = rng.integers(1, 4, size=n_elements, dtype=np.int64)
        actual_part_ids = np.array([0, 10, 20, 30], dtype=np.int64)
        part_thickness_map = {10: 2.5, 20: 4.0, 30: 1.5}

        py_result = _compute_node_thickness(connectivity, part_ids, part_thickness_map, actual_part_ids)
        rust_result = _compute_node_thickness_rust(connectivity, part_ids, part_thickness_map, actual_part_ids)

        np.testing.assert_allclose(
            rust_result[: len(py_result)],
            py_result,
            atol=1e-10,
            err_msg="Rust and Python node thickness differ",
        )

    def test_compute_node_thickness_no_actual_ids(self) -> None:
        """Consistency when actual_part_ids is None."""
        from physicsnemo_curator.domains.mesh.sources.d3plot import (
            _compute_node_thickness,
            _compute_node_thickness_rust,
        )

        connectivity = np.array([[0, 1, 2, 3], [1, 2, 4, 5]], dtype=np.int64)
        part_ids = np.array([1, 2], dtype=np.int64)
        part_thickness_map = {10: 2.0, 20: 4.0}

        py_result = _compute_node_thickness(connectivity, part_ids, part_thickness_map, None)
        rust_result = _compute_node_thickness_rust(connectivity, part_ids, part_thickness_map, None)

        np.testing.assert_allclose(
            rust_result[: len(py_result)],
            py_result,
            atol=1e-10,
            err_msg="Rust and Python node thickness differ (no actual_part_ids)",
        )

    def test_von_mises_consistency_uniaxial(self) -> None:
        """Rust von Mises must match Python for uniaxial stress."""
        from physicsnemo_curator.domains.mesh.sources.d3plot import (
            _von_mises_from_voigt,
            _von_mises_from_voigt_rust,
        )

        sig = np.array([[[100.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        py_result = _von_mises_from_voigt(sig)
        rust_result = _von_mises_from_voigt_rust(sig)

        np.testing.assert_allclose(rust_result, py_result, atol=1e-10)

    def test_von_mises_consistency_random(self) -> None:
        """Rust von Mises must match Python for random stress tensors."""
        from physicsnemo_curator.domains.mesh.sources.d3plot import (
            _von_mises_from_voigt,
            _von_mises_from_voigt_rust,
        )

        rng = np.random.default_rng(123)
        # Shape (T=5, E=1000, 6) — realistic batch
        sig = rng.uniform(-200, 200, size=(5, 1000, 6))

        py_result = _von_mises_from_voigt(sig)
        rust_result = _von_mises_from_voigt_rust(sig)

        np.testing.assert_allclose(rust_result, py_result, rtol=1e-12, atol=1e-10)

    def test_von_mises_consistency_hydrostatic(self) -> None:
        """Hydrostatic stress should yield ~0 von Mises in both backends."""
        from physicsnemo_curator.domains.mesh.sources.d3plot import (
            _von_mises_from_voigt,
            _von_mises_from_voigt_rust,
        )

        p = 42.0
        sig = np.array([[[p, p, p, 0.0, 0.0, 0.0]]])
        py_result = _von_mises_from_voigt(sig)
        rust_result = _von_mises_from_voigt_rust(sig)

        np.testing.assert_allclose(rust_result, py_result, atol=1e-10)
        assert rust_result[0, 0] < 1e-10

    @patch("lasso.dyna.D3plot")
    @pytest.mark.skipif(not _has_lasso, reason="lasso (lasso-python) not installed")
    def test_source_backend_param(self, mock_d3plot_cls: MagicMock, tmp_path: pathlib.Path) -> None:
        """D3PlotSource should accept backend='rust' and backend='python'."""
        from physicsnemo_curator.domains.mesh.sources.d3plot import D3PlotSource

        mock_root = tmp_path / "sims"
        _write_mock_d3plot_dir(mock_root, n_runs=1)

        mock_d3plot_cls.return_value = _make_mock_d3plot()

        source_py = D3PlotSource(input_dir=str(mock_root), backend="python")
        mesh_py = next(source_py[0])

        mock_d3plot_cls.return_value = _make_mock_d3plot()

        source_rust = D3PlotSource(input_dir=str(mock_root), backend="rust")
        mesh_rust = next(source_rust[0])

        # Both should produce the same thickness values.
        np.testing.assert_allclose(  # ty: ignore[no-matching-overload]
            mesh_rust.point_data["thickness"].numpy(),
            mesh_py.point_data["thickness"].numpy(),
            atol=1e-10,
            err_msg="Rust and Python backends produce different thickness",
        )
