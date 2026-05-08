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

"""Tests for MeshZarrSink."""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest
import torch

zarr = pytest.importorskip("zarr")

from tensordict import TensorDict  # noqa: E402

pytestmark = pytest.mark.requires("mesh")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_mesh():
    """Create a simple mesh with displacement fields."""
    from physicsnemo.mesh import Mesh

    n_points = 10
    n_cells = 4
    n_timesteps = 3

    rng = np.random.default_rng(42)

    points = torch.from_numpy(rng.uniform(-10, 10, size=(n_points, 3)).astype(np.float32))
    cells = torch.from_numpy(rng.integers(0, n_points, size=(n_cells, 3)).astype(np.int64))

    pd_dict: dict[str, torch.Tensor] = {}
    pd_dict["thickness"] = torch.from_numpy(rng.uniform(0.1, 1.0, size=(n_points,)).astype(np.float32))

    for t in range(n_timesteps):
        disp = rng.uniform(-1, 1, size=(n_points, 3)).astype(np.float32)
        pd_dict[f"displacement_t{t:03d}"] = torch.from_numpy(disp)

    point_data = TensorDict(pd_dict, batch_size=[n_points])  # ty: ignore[invalid-argument-type]

    # Cell data
    cd_dict: dict[str, torch.Tensor] = {}
    for t in range(n_timesteps):
        cd_dict[f"stress_vm_t{t:03d}"] = torch.from_numpy(rng.uniform(0, 100, size=(n_cells,)).astype(np.float32))
    cell_data = TensorDict(cd_dict, batch_size=[n_cells])  # ty: ignore[invalid-argument-type]

    # Global data with edges
    edges = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 3]], dtype=torch.int64)
    global_data = TensorDict(
        {
            "num_timesteps": torch.tensor([n_timesteps], dtype=torch.int64),
            "edges": edges,
        },
        batch_size=[],
    )

    return Mesh(
        points=points,
        cells=cells,
        point_data=point_data,
        cell_data=cell_data,
        global_data=global_data,
    )


@pytest.fixture
def mesh_without_edges():
    """Create a mesh without edges in global_data."""
    from physicsnemo.mesh import Mesh

    n_points = 5
    n_timesteps = 2

    points = torch.randn(n_points, 3, dtype=torch.float32)

    pd_dict: dict[str, torch.Tensor] = {}
    for t in range(n_timesteps):
        pd_dict[f"displacement_t{t:03d}"] = torch.randn(n_points, 3, dtype=torch.float32)
    point_data = TensorDict(pd_dict, batch_size=[n_points])  # ty: ignore[invalid-argument-type]

    global_data = TensorDict(
        {"num_timesteps": torch.tensor([n_timesteps], dtype=torch.int64)},
        batch_size=[],
    )

    return Mesh(
        points=points,
        cells=None,
        point_data=point_data,
        global_data=global_data,
    )


@pytest.fixture
def mesh_no_displacements():
    """Create a mesh without displacement fields."""
    from physicsnemo.mesh import Mesh

    n_points = 5

    points = torch.randn(n_points, 3, dtype=torch.float32)
    point_data = TensorDict(
        {"some_field": torch.randn(n_points, dtype=torch.float32)},
        batch_size=[n_points],
    )

    return Mesh(
        points=points,
        cells=None,
        point_data=point_data,
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestMeshZarrSinkUnit:
    """Unit tests for MeshZarrSink."""

    def test_params(self):
        """params() should return expected descriptors."""
        from physicsnemo_curator.domains.mesh.sinks.mesh_zarr import MeshZarrSink

        params = MeshZarrSink.params()
        names = [p.name for p in params]
        assert "output_dir" in names
        assert "compression_level" in names
        assert "chunk_size_mb" in names
        assert "naming_template" in names

    def test_init_default_naming(self):
        """MeshZarrSink should use default naming template."""
        from physicsnemo_curator.domains.mesh.sinks.mesh_zarr import MeshZarrSink

        with tempfile.TemporaryDirectory() as tmpdir:
            sink = MeshZarrSink(output_dir=tmpdir)
            assert sink._naming_template == "mesh_{index:04d}"

    def test_init_custom_naming(self):
        """MeshZarrSink should accept custom naming template."""
        from physicsnemo_curator.domains.mesh.sinks.mesh_zarr import MeshZarrSink

        with tempfile.TemporaryDirectory() as tmpdir:
            sink = MeshZarrSink(output_dir=tmpdir, naming_template="run_{index}")
            assert sink._naming_template == "run_{index}"

    def test_compute_chunks_1d(self):
        """_compute_chunks should handle 1D arrays."""
        from physicsnemo_curator.domains.mesh.sinks.mesh_zarr import _compute_chunks

        shape = (1000,)
        dtype = np.dtype(np.float32)
        chunks = _compute_chunks(shape, dtype, target_mb=0.001)
        assert len(chunks) == 1
        assert chunks[0] <= shape[0]

    def test_compute_chunks_2d(self):
        """_compute_chunks should handle 2D arrays."""
        from physicsnemo_curator.domains.mesh.sinks.mesh_zarr import _compute_chunks

        shape = (100, 3)
        dtype = np.dtype(np.float32)
        chunks = _compute_chunks(shape, dtype, target_mb=1.0)
        assert len(chunks) == 2
        assert chunks[1] == shape[1]  # Full width preserved

    def test_compute_chunks_3d(self):
        """_compute_chunks should handle 3D arrays."""
        from physicsnemo_curator.domains.mesh.sinks.mesh_zarr import _compute_chunks

        shape = (10, 1000, 3)
        dtype = np.dtype(np.float32)
        chunks = _compute_chunks(shape, dtype, target_mb=1.0)
        assert len(chunks) == 3
        assert chunks[2] == shape[2]  # Last dim preserved


class TestMeshZarrSinkIntegration:
    """Integration tests for MeshZarrSink."""

    def test_write_simple_mesh(self, simple_mesh):
        """MeshZarrSink should write a mesh to Zarr format."""
        from physicsnemo_curator.domains.mesh.sinks.mesh_zarr import MeshZarrSink

        with tempfile.TemporaryDirectory() as tmpdir:
            sink = MeshZarrSink(output_dir=tmpdir)

            def gen():
                yield simple_mesh

            paths = sink(gen(), index=0)

            assert len(paths) == 1
            zarr_path = pathlib.Path(paths[0])
            assert zarr_path.exists()
            assert zarr_path.suffix == ".zarr"

            # Load and verify
            store = zarr.open_group(str(zarr_path), mode="r")
            assert "mesh_pos" in store
            assert "edges" in store
            assert "thickness" in store

            mesh_pos = np.array(store["mesh_pos"])
            assert mesh_pos.shape == (3, 10, 3)  # (T, N, 3)
            assert mesh_pos.dtype == np.float32

    def test_write_mesh_without_edges_warns(self, mesh_without_edges, caplog):
        """MeshZarrSink should warn when edges are missing."""
        from physicsnemo_curator.domains.mesh.sinks.mesh_zarr import MeshZarrSink

        with tempfile.TemporaryDirectory() as tmpdir:
            sink = MeshZarrSink(output_dir=tmpdir)

            def gen():
                yield mesh_without_edges

            paths = sink(gen(), index=0)

            assert len(paths) == 1
            assert "No edges found" in caplog.text

    def test_write_mesh_no_displacements(self, mesh_no_displacements):
        """MeshZarrSink should handle meshes without displacement fields."""
        from physicsnemo_curator.domains.mesh.sinks.mesh_zarr import MeshZarrSink

        with tempfile.TemporaryDirectory() as tmpdir:
            sink = MeshZarrSink(output_dir=tmpdir)

            def gen():
                yield mesh_no_displacements

            paths = sink(gen(), index=0)

            assert len(paths) == 1

            # Load and verify - should have single timestep
            store = zarr.open_group(str(paths[0]), mode="r")
            mesh_pos = np.array(store["mesh_pos"])
            assert mesh_pos.shape[0] == 1  # Single timestep

    def test_mesh_pos_reconstruction(self, simple_mesh):
        """mesh_pos should be reconstructed from points + displacements."""
        from physicsnemo_curator.domains.mesh.sinks.mesh_zarr import MeshZarrSink

        with tempfile.TemporaryDirectory() as tmpdir:
            sink = MeshZarrSink(output_dir=tmpdir)

            def gen():
                yield simple_mesh

            paths = sink(gen(), index=0)

            store = zarr.open_group(str(paths[0]), mode="r")
            mesh_pos = np.array(store["mesh_pos"])

            # Verify t=0: mesh_pos[0] = points + disp_t0
            points = simple_mesh.points.numpy()
            disp_t0 = simple_mesh.point_data["displacement_t000"].numpy()
            expected_pos_t0 = points + disp_t0
            np.testing.assert_allclose(mesh_pos[0], expected_pos_t0, rtol=1e-5)

    def test_metadata_attributes(self, simple_mesh):
        """Zarr store should have metadata attributes."""
        from physicsnemo_curator.domains.mesh.sinks.mesh_zarr import MeshZarrSink

        with tempfile.TemporaryDirectory() as tmpdir:
            sink = MeshZarrSink(output_dir=tmpdir)

            def gen():
                yield simple_mesh

            paths = sink(gen(), index=0)

            store = zarr.open_group(str(paths[0]), mode="r")
            assert store.attrs["num_timesteps"] == 3
            assert store.attrs["num_nodes"] == 10
            assert store.attrs["num_edges"] == 4

    def test_cell_data_prefixed(self, simple_mesh):
        """Cell data arrays should be prefixed with 'cell_'."""
        from physicsnemo_curator.domains.mesh.sinks.mesh_zarr import MeshZarrSink

        with tempfile.TemporaryDirectory() as tmpdir:
            sink = MeshZarrSink(output_dir=tmpdir)

            def gen():
                yield simple_mesh

            paths = sink(gen(), index=0)

            store = zarr.open_group(str(paths[0]), mode="r")
            assert "cell_stress_vm_t000" in store

    def test_custom_compression_level(self, simple_mesh):
        """MeshZarrSink should use custom compression level."""
        from physicsnemo_curator.domains.mesh.sinks.mesh_zarr import MeshZarrSink

        with tempfile.TemporaryDirectory() as tmpdir:
            sink = MeshZarrSink(output_dir=tmpdir, compression_level=9)

            def gen():
                yield simple_mesh

            paths = sink(gen(), index=0)

            # Just verify it completes without error
            assert len(paths) == 1
