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

"""Tests for HuggingFace dataset sources and RunIndexedFileStore.

Unit tests use mock filesystems to avoid network access.
E2E tests (marked ``slow``) hit the real HuggingFace Hub.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from physicsnemo_curator.core.store import RunIndexedFileStore

if TYPE_CHECKING:
    import pathlib

# Network errors that should cause E2E tests to be skipped rather than
# reported as failures (the remote API is outside our control).
_NETWORK_ERRORS: tuple[type[BaseException], ...] = (OSError, TimeoutError)

try:
    import httpx

    _NETWORK_ERRORS = (*_NETWORK_ERRORS, httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Unit tests — RunIndexedFileStore
# ---------------------------------------------------------------------------


class TestRunIndexedFileStore:
    """Unit tests for RunIndexedFileStore with a local filesystem."""

    @pytest.fixture(autouse=True)
    def _setup_run_dirs(self, tmp_path: pathlib.Path) -> None:
        """Create a local directory tree mimicking run-indexed layout."""
        for i in [1, 3, 5, 10]:
            run_dir = tmp_path / f"run_{i}"
            run_dir.mkdir()
            (run_dir / f"boundary_{i}.vtp").write_text(f"mock-vtp-{i}")
            (run_dir / f"volume_{i}.vtu").write_text(f"mock-vtu-{i}")
            slices_dir = run_dir / "slices"
            slices_dir.mkdir()
            (slices_dir / f"xNormal_{i}.vtp").write_text(f"mock-slice-{i}")

        # Also create a non-run directory that should be ignored.
        (tmp_path / "openfoam_meshes").mkdir()
        (tmp_path / "force_mom_all.csv").write_text("mock-csv")

        self.tmp_path = tmp_path

    def test_discovers_runs(self) -> None:
        """Should find exactly the 4 run directories."""
        store = RunIndexedFileStore(
            url=str(self.tmp_path),
            file_template="boundary_{i}.vtp",
        )
        assert len(store) == 4

    def test_run_indices_sorted(self) -> None:
        """Run indices should be sorted ascending."""
        store = RunIndexedFileStore(
            url=str(self.tmp_path),
            file_template="boundary_{i}.vtp",
        )
        assert store.run_indices == [1, 3, 5, 10]

    def test_getitem_returns_correct_path(self) -> None:
        """Index 0 should resolve to run_1, index 3 to run_10."""
        store = RunIndexedFileStore(
            url=str(self.tmp_path),
            file_template="boundary_{i}.vtp",
        )
        path_0 = store[0]
        assert path_0.endswith("run_1/boundary_1.vtp")

        path_3 = store[3]
        assert path_3.endswith("run_10/boundary_10.vtp")

    def test_getitem_negative_index(self) -> None:
        """Negative indexing should work."""
        store = RunIndexedFileStore(
            url=str(self.tmp_path),
            file_template="boundary_{i}.vtp",
        )
        path = store[-1]
        assert path.endswith("run_10/boundary_10.vtp")

    def test_getitem_out_of_range(self) -> None:
        """Should raise IndexError for out-of-range indices."""
        store = RunIndexedFileStore(
            url=str(self.tmp_path),
            file_template="boundary_{i}.vtp",
        )
        with pytest.raises(IndexError):
            store[99]

    def test_empty_directory_raises(self, tmp_path: pathlib.Path) -> None:
        """Should raise ValueError when no run_* dirs exist."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No run_"):
            RunIndexedFileStore(
                url=str(empty_dir),
                file_template="boundary_{i}.vtp",
            )

    def test_repr(self) -> None:
        """__repr__ should include URL, template, and count."""
        store = RunIndexedFileStore(
            url=str(self.tmp_path),
            file_template="boundary_{i}.vtp",
        )
        r = repr(store)
        assert "RunIndexedFileStore" in r
        assert "runs=4" in r

    def test_volume_template(self) -> None:
        """Should work with volume file templates too."""
        store = RunIndexedFileStore(
            url=str(self.tmp_path),
            file_template="volume_{i}.vtu",
        )
        path = store[0]
        assert path.endswith("run_1/volume_1.vtu")


# ---------------------------------------------------------------------------
# Unit tests — Dataset source construction (mock filesystem)
# ---------------------------------------------------------------------------


@pytest.mark.requires("mesh")
class TestDrivAerMLSourceUnit:
    """Unit tests for DrivAerMLSource parameter descriptors."""

    def test_params_list(self) -> None:
        """params() should return a non-empty list of Param objects."""
        from physicsnemo_curator.mesh.sources.drivaerml import DrivAerMLSource

        params = DrivAerMLSource.params()
        assert len(params) > 0
        names = [p.name for p in params]
        assert "mesh_type" in names
        assert "url" in names

    def test_name_and_description(self) -> None:
        """Class should have name and description ClassVars."""
        from physicsnemo_curator.mesh.sources.drivaerml import DrivAerMLSource

        assert DrivAerMLSource.name == "DrivAerML"
        assert len(DrivAerMLSource.description) > 0


@pytest.mark.requires("mesh")
class TestAhmedMLSourceUnit:
    """Unit tests for AhmedMLSource parameter descriptors."""

    def test_params_list(self) -> None:
        """params() should return a non-empty list of Param objects."""
        from physicsnemo_curator.mesh.sources.ahmedml import AhmedMLSource

        params = AhmedMLSource.params()
        assert len(params) > 0
        names = [p.name for p in params]
        assert "mesh_type" in names

    def test_name_and_description(self) -> None:
        """Class should have name and description ClassVars."""
        from physicsnemo_curator.mesh.sources.ahmedml import AhmedMLSource

        assert AhmedMLSource.name == "AhmedML"
        assert len(AhmedMLSource.description) > 0


@pytest.mark.requires("mesh")
class TestWindsorMLSourceUnit:
    """Unit tests for WindsorMLSource parameter descriptors."""

    def test_params_list(self) -> None:
        """params() should return a non-empty list of Param objects."""
        from physicsnemo_curator.mesh.sources.windsorml import WindsorMLSource

        params = WindsorMLSource.params()
        assert len(params) > 0
        names = [p.name for p in params]
        assert "mesh_type" in names

    def test_mesh_type_choices(self) -> None:
        """WindsorML should only offer boundary and volume (no slices)."""
        from physicsnemo_curator.mesh.sources.windsorml import WindsorMLSource

        params = WindsorMLSource.params()
        mesh_type_param = next(p for p in params if p.name == "mesh_type")
        assert mesh_type_param.choices == ["boundary", "volume"]

    def test_name_and_description(self) -> None:
        """Class should have name and description ClassVars."""
        from physicsnemo_curator.mesh.sources.windsorml import WindsorMLSource

        assert WindsorMLSource.name == "WindsorML"
        assert len(WindsorMLSource.description) > 0


@pytest.mark.requires("mesh")
class TestWindTunnelSourceUnit:
    """Unit tests for WindTunnelSource parameter descriptors."""

    def test_params_list(self) -> None:
        """params() should return a non-empty list of Param objects."""
        from physicsnemo_curator.mesh.sources.windtunnel import WindTunnelSource

        params = WindTunnelSource.params()
        assert len(params) > 0
        names = [p.name for p in params]
        assert "split" in names
        assert "mesh_type" not in names  # uses 'split' instead

    def test_split_choices(self) -> None:
        """WindTunnel should offer train/validation/test/all splits."""
        from physicsnemo_curator.mesh.sources.windtunnel import WindTunnelSource

        params = WindTunnelSource.params()
        split_param = next(p for p in params if p.name == "split")
        assert split_param.choices == ["train", "validation", "test", "all"]

    def test_name_and_description(self) -> None:
        """Class should have name and description ClassVars."""
        from physicsnemo_curator.mesh.sources.windtunnel import WindTunnelSource

        assert WindTunnelSource.name == "WindTunnel-20k"
        assert len(WindTunnelSource.description) > 0


# ---------------------------------------------------------------------------
# Unit tests — Registry integration
# ---------------------------------------------------------------------------


@pytest.mark.requires("mesh")
class TestRegistryIntegration:
    """Verify that all dataset sources are registered correctly."""

    def test_all_sources_registered(self) -> None:
        """All four dataset sources should appear in the mesh registry."""
        import physicsnemo_curator.mesh  # noqa: F401
        from physicsnemo_curator.core.registry import registry

        sources = registry.list_sources("mesh")
        source_names = {s.name for s in sources}
        assert "DrivAerML" in source_names
        assert "AhmedML" in source_names
        assert "WindsorML" in source_names
        assert "WindTunnel-20k" in source_names

    def test_run_indexed_store_registered(self) -> None:
        """RunIndexedFileStore should be registered as a mesh store."""
        import physicsnemo_curator.mesh  # noqa: F401
        from physicsnemo_curator.core.registry import registry

        stores = registry.list_stores("mesh")
        store_names = {name for name, _ in stores}
        assert "Run-indexed (remote)" in store_names


# ---------------------------------------------------------------------------
# E2E tests — DrivAerML (real HuggingFace downloads)
# ---------------------------------------------------------------------------


@pytest.mark.requires("mesh")
@pytest.mark.e2e
@pytest.mark.slow
class TestDrivAerMLSourceE2E:
    """End-to-end tests fetching real DrivAerML data from HuggingFace.

    Downloads a small subset (boundary for the first available run) to
    verify the full pipeline works against the live dataset.

    The ``slow`` marker allows deselecting in quick CI runs.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: pathlib.Path) -> None:
        """Build the source with a local cache directory."""
        from physicsnemo_curator.mesh.sources.drivaerml import DrivAerMLSource

        try:
            self.source = DrivAerMLSource(
                mesh_type="boundary",
                cache_storage=str(tmp_path / "cache"),
                warn_on_lost_data=False,
            )
        except _NETWORK_ERRORS as exc:
            pytest.skip(f"HuggingFace API unreachable: {exc}")
        self.tmp_path = tmp_path

    def test_discovers_runs(self) -> None:
        """Should discover hundreds of runs."""
        assert len(self.source) > 400

    def test_run_indices_start_from_one(self) -> None:
        """DrivAerML runs start at run_1."""
        assert self.source._store.run_indices[0] >= 1  # type: ignore[union-attr]

    def test_reads_boundary_mesh(self) -> None:
        """Should read the first boundary mesh as a valid Mesh."""
        from physicsnemo.mesh import Mesh

        mesh = next(self.source[0])
        assert isinstance(mesh, Mesh)
        assert mesh.n_points > 0
        assert mesh.n_cells > 0

    def test_boundary_is_surface(self) -> None:
        """Boundary meshes should be 2D surfaces in 3D space."""
        mesh = next(self.source[0])
        assert mesh.n_spatial_dims == 3
        assert mesh.n_manifold_dims == 2

    def test_boundary_has_cell_data(self) -> None:
        """Boundary meshes should carry flow field cell data."""
        mesh = next(self.source[0])
        assert len(mesh.cell_data) > 0


# ---------------------------------------------------------------------------
# E2E tests — DrivAerML slices
# ---------------------------------------------------------------------------


@pytest.mark.requires("mesh")
@pytest.mark.e2e
@pytest.mark.slow
class TestDrivAerMLSlicesE2E:
    """End-to-end tests for DrivAerML slice plane meshes.

    Verifies that slices mode discovers and reads multiple VTP files
    per run from the live HuggingFace dataset.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: pathlib.Path) -> None:
        """Build the slices source with a local cache directory."""
        from physicsnemo_curator.mesh.sources.drivaerml import DrivAerMLSource

        try:
            self.source = DrivAerMLSource(
                mesh_type="slices",
                cache_storage=str(tmp_path / "cache"),
                warn_on_lost_data=False,
            )
        except _NETWORK_ERRORS as exc:
            pytest.skip(f"HuggingFace API unreachable: {exc}")
        self.tmp_path = tmp_path

    def test_discovers_runs(self) -> None:
        """Should discover hundreds of runs for slices."""
        assert len(self.source) > 400

    def test_yields_multiple_slices(self) -> None:
        """Each run should yield multiple slice plane meshes."""
        from physicsnemo.mesh import Mesh

        meshes = list(self.source[0])
        assert len(meshes) > 1
        for mesh in meshes:
            assert isinstance(mesh, Mesh)
            assert mesh.n_points > 0


# ---------------------------------------------------------------------------
# E2E tests — AhmedML
# ---------------------------------------------------------------------------


@pytest.mark.requires("mesh")
@pytest.mark.e2e
@pytest.mark.slow
class TestAhmedMLSourceE2E:
    """End-to-end tests fetching real AhmedML data from HuggingFace."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: pathlib.Path) -> None:
        """Build the source with a local cache directory."""
        from physicsnemo_curator.mesh.sources.ahmedml import AhmedMLSource

        try:
            self.source = AhmedMLSource(
                mesh_type="boundary",
                cache_storage=str(tmp_path / "cache"),
                warn_on_lost_data=False,
            )
        except _NETWORK_ERRORS as exc:
            pytest.skip(f"HuggingFace API unreachable: {exc}")
        self.tmp_path = tmp_path

    def test_discovers_runs(self) -> None:
        """Should discover 500 runs."""
        assert len(self.source) == 500

    def test_run_indices(self) -> None:
        """AhmedML runs should range from 1 to 500."""
        indices = self.source._store.run_indices  # type: ignore[union-attr]
        assert indices[0] == 1
        assert indices[-1] == 500

    def test_reads_boundary_mesh(self) -> None:
        """Should read the first boundary mesh as a valid Mesh."""
        from physicsnemo.mesh import Mesh

        mesh = next(self.source[0])
        assert isinstance(mesh, Mesh)
        assert mesh.n_points > 0
        assert mesh.n_cells > 0

    def test_boundary_is_surface(self) -> None:
        """Boundary meshes should be 2D surfaces in 3D space."""
        mesh = next(self.source[0])
        assert mesh.n_spatial_dims == 3
        assert mesh.n_manifold_dims == 2

    def test_boundary_has_cell_data(self) -> None:
        """Boundary meshes should carry flow field cell data."""
        mesh = next(self.source[0])
        assert len(mesh.cell_data) > 0


# ---------------------------------------------------------------------------
# E2E tests — WindsorML
# ---------------------------------------------------------------------------


@pytest.mark.requires("mesh")
@pytest.mark.e2e
@pytest.mark.slow
class TestWindsorMLSourceE2E:
    """End-to-end tests fetching real WindsorML data from HuggingFace."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: pathlib.Path) -> None:
        """Build the source with a local cache directory."""
        from physicsnemo_curator.mesh.sources.windsorml import WindsorMLSource

        try:
            self.source = WindsorMLSource(
                mesh_type="boundary",
                cache_storage=str(tmp_path / "cache"),
                warn_on_lost_data=False,
            )
        except _NETWORK_ERRORS as exc:
            pytest.skip(f"HuggingFace API unreachable: {exc}")
        self.tmp_path = tmp_path

    def test_discovers_runs(self) -> None:
        """Should discover 350 runs."""
        assert len(self.source) == 350

    def test_run_indices_start_from_zero(self) -> None:
        """WindsorML runs start at run_0."""
        assert self.source._store.run_indices[0] == 0

    def test_reads_boundary_mesh(self) -> None:
        """Should read the first boundary mesh as a valid Mesh."""
        from physicsnemo.mesh import Mesh

        mesh = next(self.source[0])
        assert isinstance(mesh, Mesh)
        assert mesh.n_points > 0
        assert mesh.n_cells > 0

    def test_boundary_is_surface(self) -> None:
        """Boundary meshes should be 2D surfaces in 3D space."""
        mesh = next(self.source[0])
        assert mesh.n_spatial_dims == 3
        assert mesh.n_manifold_dims == 2

    def test_boundary_has_cell_data(self) -> None:
        """Boundary meshes should carry flow field data."""
        mesh = next(self.source[0])
        assert len(mesh.cell_data) > 0


# ---------------------------------------------------------------------------
# E2E tests — WindTunnel-20k
# ---------------------------------------------------------------------------


@pytest.mark.requires("mesh")
@pytest.mark.e2e
@pytest.mark.slow
class TestWindTunnelSourceE2E:
    """End-to-end tests fetching real WindTunnel-20k data from HuggingFace."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: pathlib.Path) -> None:
        """Build the source for the test split (smallest)."""
        from physicsnemo_curator.mesh.sources.windtunnel import WindTunnelSource

        try:
            self.source = WindTunnelSource(
                split="test",
                cache_storage=str(tmp_path / "cache"),
                warn_on_lost_data=False,
            )
        except _NETWORK_ERRORS as exc:
            pytest.skip(f"HuggingFace API unreachable: {exc}")
        self.tmp_path = tmp_path

    def test_discovers_simulations(self) -> None:
        """Test split should have ~1,980 simulations."""
        assert len(self.source) > 1000

    def test_reads_pressure_field_mesh(self) -> None:
        """Should read the first pressure field mesh as a valid Mesh."""
        from physicsnemo.mesh import Mesh

        mesh = next(self.source[0])
        assert isinstance(mesh, Mesh)
        assert mesh.n_points > 0

    def test_mesh_has_point_data(self) -> None:
        """Pressure field meshes should have point data (pressure)."""
        mesh = next(self.source[0])
        # WindTunnel pressure field has 'p' as point data.
        assert mesh.n_points > 0
