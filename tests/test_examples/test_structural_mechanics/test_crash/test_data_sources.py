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

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import pyvista as pv
import zarr
from data_sources import CrashD3PlotDataSource, CrashVTPDataSource, CrashZarrDataSource
from schemas import CrashExtractedDataInMemory, CrashMetadata


@pytest.fixture
def mock_crash_data():
    """Create mock crash simulation data for testing."""
    # Positions over time (timesteps, nodes, 3)
    pos_raw = np.array(
        [
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],  # t0
            [[0, 0, 0], [1.1, 0, 0], [0, 1, 0], [1.1, 1, 0]],  # t1
            [[0, 0, 0], [1.2, 0, 0], [0, 1, 0], [1.2, 1, 0]],  # t2
        ],
        dtype=np.float32,
    )

    # Mesh connectivity (one quad)
    mesh_connectivity = [[0, 1, 3, 2]]

    # Node thickness
    node_thickness = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    # Edges
    edges = {(0, 1), (1, 3), (2, 3), (0, 2)}

    return CrashExtractedDataInMemory(
        metadata=CrashMetadata(filename="test_run"),
        pos_raw=pos_raw,
        mesh_connectivity=mesh_connectivity,
        node_thickness=node_thickness,
        filtered_pos_raw=pos_raw,
        filtered_mesh_connectivity=mesh_connectivity,
        filtered_node_thickness=node_thickness,
        edges=edges,
    )


# Tests for CrashD3PlotDataSource


def test_d3plot_source_get_file_list():
    """Test that get_file_list finds all run folders with d3plot files."""
    from physicsnemo_curator.etl.processing_config import ProcessingConfig

    cfg = ProcessingConfig(num_processes=1)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create mock run folders with d3plot files
        (temp_path / "Run001").mkdir()
        (temp_path / "Run001" / "d3plot").touch()

        (temp_path / "Run002").mkdir()
        (temp_path / "Run002" / "d3plot").touch()

        # Create a folder without d3plot (should be ignored)
        (temp_path / "EmptyRun").mkdir()

        # Create a regular file (should be ignored)
        (temp_path / "some_file.txt").touch()

        source = CrashD3PlotDataSource(cfg, temp_dir)
        file_list = source.get_file_list()

        assert len(file_list) == 2, f"Expected 2 runs, got {len(file_list)}"
        assert "Run001" in file_list, "Run001 should be in file list"
        assert "Run002" in file_list, "Run002 should be in file list"
        assert "EmptyRun" not in file_list, "EmptyRun should not be in file list"


def test_d3plot_source_read_file():
    """Test reading a d3plot file."""
    from physicsnemo_curator.etl.processing_config import ProcessingConfig

    cfg = ProcessingConfig(num_processes=1)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create mock run folder
        run_dir = temp_path / "Run001"
        run_dir.mkdir()
        (run_dir / "d3plot").touch()

        # Create a mock .k file
        k_file = run_dir / "model.k"
        k_file.write_text("*PART\nPart 1\n1   1   1\n*SECTION_SHELL\n1\n2.5\n")

        # Mock the load_d3plot_data function
        mock_coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        mock_pos_raw = np.array(
            [
                [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                [[0, 0, 0], [1.1, 0, 0], [0, 1, 0]],
            ],
            dtype=np.float32,
        )
        mock_connectivity = [[0, 1, 2]]
        mock_part_ids = np.array([1])
        mock_actual_part_ids = np.array([0, 1])

        with patch("data_sources.load_d3plot_data") as mock_load:
            mock_load.return_value = (
                mock_coords,
                mock_pos_raw,
                mock_connectivity,
                mock_part_ids,
                mock_actual_part_ids,
            )

            source = CrashD3PlotDataSource(cfg, temp_dir)
            result = source.read_file("Run001")

            # Verify the result
            assert isinstance(result, CrashExtractedDataInMemory)
            assert result.metadata.filename == "Run001"
            assert result.pos_raw.shape == (2, 3, 3)
            assert len(result.mesh_connectivity) == 1
            assert len(result.node_thickness) == 3
            # Thickness should be 2.5 for all nodes (from .k file)
            np.testing.assert_array_almost_equal(
                result.node_thickness, np.array([2.5, 2.5, 2.5])
            )


def test_d3plot_source_read_file_no_k_file():
    """Test reading a d3plot file without a .k file."""
    from physicsnemo_curator.etl.processing_config import ProcessingConfig

    cfg = ProcessingConfig(num_processes=1)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create mock run folder without .k file
        run_dir = temp_path / "Run001"
        run_dir.mkdir()
        (run_dir / "d3plot").touch()

        # Mock the load_d3plot_data function
        mock_coords = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        mock_pos_raw = np.array(
            [[[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [1.1, 0, 0]]], dtype=np.float32
        )
        mock_connectivity = [[0, 1]]
        mock_part_ids = np.array([1])
        mock_actual_part_ids = None

        with patch("data_sources.load_d3plot_data") as mock_load:
            mock_load.return_value = (
                mock_coords,
                mock_pos_raw,
                mock_connectivity,
                mock_part_ids,
                mock_actual_part_ids,
            )

            source = CrashD3PlotDataSource(cfg, temp_dir)
            result = source.read_file("Run001")

            # Verify thickness is zero when no .k file
            assert len(result.node_thickness) == 2
            np.testing.assert_array_equal(result.node_thickness, np.array([0.0, 0.0]))


def test_d3plot_source_should_skip():
    """Test that CrashD3PlotDataSource never skips files."""
    from physicsnemo_curator.etl.processing_config import ProcessingConfig

    cfg = ProcessingConfig(num_processes=1)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        (temp_path / "Run001").mkdir()
        (temp_path / "Run001" / "d3plot").touch()

        source = CrashD3PlotDataSource(cfg, temp_dir)

        # Should never skip
        assert not source.should_skip("Run001")
        assert not source.should_skip("Run002")
        assert not source.should_skip("NonExistentRun")


def test_d3plot_source_write_not_implemented():
    """Test that write methods raise NotImplementedError."""
    from physicsnemo_curator.etl.processing_config import ProcessingConfig

    cfg = ProcessingConfig(num_processes=1)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        (temp_path / "Run001").mkdir()
        (temp_path / "Run001" / "d3plot").touch()

        source = CrashD3PlotDataSource(cfg, temp_dir)

        # Test that write raises NotImplementedError
        with pytest.raises(NotImplementedError, match="only supports reading"):
            source.write(None, "test")

        # Test that _get_output_path raises NotImplementedError
        with pytest.raises(NotImplementedError, match="only supports reading"):
            source._get_output_path("test")

        # Test that _write_impl_temp_file raises NotImplementedError
        with pytest.raises(NotImplementedError, match="only supports reading"):
            source._write_impl_temp_file(None, Path("test"))


def test_d3plot_source_nonexistent_directory():
    """Test that CrashD3PlotDataSource raises error for nonexistent directory."""
    from physicsnemo_curator.etl.processing_config import ProcessingConfig

    cfg = ProcessingConfig(num_processes=1)

    with pytest.raises(FileNotFoundError, match="Input directory does not exist"):
        CrashD3PlotDataSource(cfg, "/nonexistent/path")


# Tests for VTP and Zarr sinks


def test_vtp_sink_output_path():
    """Test VTP output path generation."""
    from physicsnemo_curator.etl.processing_config import ProcessingConfig

    cfg = ProcessingConfig(num_processes=1)

    with tempfile.TemporaryDirectory() as temp_dir:
        sink = CrashVTPDataSource(
            cfg, temp_dir, overwrite_existing=True, time_step=0.005
        )

        output_path = sink._get_output_path("Run100")
        assert output_path == Path(temp_dir) / "Run100.vtp"


def test_vtp_sink_write(mock_crash_data):
    """Test VTP file writing."""
    from physicsnemo_curator.etl.processing_config import ProcessingConfig

    cfg = ProcessingConfig(num_processes=1)

    with tempfile.TemporaryDirectory() as temp_dir:
        sink = CrashVTPDataSource(
            cfg, temp_dir, overwrite_existing=True, time_step=0.005
        )

        # Write the data
        sink.write(mock_crash_data, "test_run")

        # Check file was created
        output_file = Path(temp_dir) / "test_run.vtp"
        assert output_file.exists(), "VTP file was not created"

        # Load and verify the VTP file
        mesh = pv.read(str(output_file))

        # Check points (should be reference coordinates)
        assert mesh.n_points == 4, f"Expected 4 points, got {mesh.n_points}"

        # Check point data contains thickness
        assert "thickness" in mesh.point_data.keys(), "Thickness not found in VTP"
        np.testing.assert_array_almost_equal(
            mesh.point_data["thickness"], mock_crash_data.node_thickness
        )

        # Check displacement fields
        assert (
            "displacement_t0.000" in mesh.point_data.keys()
        ), "displacement_t0.000 not found"
        assert (
            "displacement_t0.005" in mesh.point_data.keys()
        ), "displacement_t0.005 not found"
        assert (
            "displacement_t0.010" in mesh.point_data.keys()
        ), "displacement_t0.010 not found"

        # First timestep should have zero displacement
        disp_t0 = mesh.point_data["displacement_t0.000"]
        np.testing.assert_array_almost_equal(disp_t0, np.zeros((4, 3)))


def test_vtp_sink_should_skip():
    """Test VTP sink skip logic."""
    from physicsnemo_curator.etl.processing_config import ProcessingConfig

    cfg = ProcessingConfig(num_processes=1)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a dummy file
        dummy_file = Path(temp_dir) / "existing_run.vtp"
        dummy_file.touch()

        # Test with overwrite_existing=True (should not skip)
        sink = CrashVTPDataSource(cfg, temp_dir, overwrite_existing=True)
        assert not sink.should_skip(
            "existing_run"
        ), "Should not skip when overwrite_existing=True"

        # Test with overwrite_existing=False (should skip)
        sink = CrashVTPDataSource(cfg, temp_dir, overwrite_existing=False)
        assert sink.should_skip(
            "existing_run"
        ), "Should skip when file exists and overwrite_existing=False"

        # Test with non-existent file
        assert not sink.should_skip(
            "non_existent_run"
        ), "Should not skip non-existent file"


def test_zarr_sink_output_path():
    """Test Zarr output path generation."""
    from physicsnemo_curator.etl.processing_config import ProcessingConfig

    cfg = ProcessingConfig(num_processes=1)

    with tempfile.TemporaryDirectory() as temp_dir:
        sink = CrashZarrDataSource(cfg, temp_dir, overwrite_existing=True)

        output_path = sink._get_output_path("Run100")
        assert output_path == Path(temp_dir) / "Run100.zarr"


def test_zarr_sink_write(mock_crash_data):
    """Test Zarr store writing."""
    from physicsnemo_curator.etl.processing_config import ProcessingConfig

    cfg = ProcessingConfig(num_processes=1)

    with tempfile.TemporaryDirectory() as temp_dir:
        sink = CrashZarrDataSource(
            cfg, temp_dir, overwrite_existing=True, compression_level=3
        )

        # Write the data
        sink.write(mock_crash_data, "test_run")

        # Check store was created
        output_store = Path(temp_dir) / "test_run.zarr"
        assert output_store.exists(), "Zarr store was not created"
        assert output_store.is_dir(), "Zarr store should be a directory"

        # Open and verify the Zarr store
        store = zarr.open(str(output_store), mode="r")

        # Check arrays exist
        assert "mesh_pos" in store, "mesh_pos not in Zarr store"
        assert "thickness" in store, "thickness not in Zarr store"
        assert "edges" in store, "edges not in Zarr store"

        # Check shapes
        assert store["mesh_pos"].shape == (
            3,
            4,
            3,
        ), f"Expected mesh_pos shape (3, 4, 3), got {store['mesh_pos'].shape}"
        assert store["thickness"].shape == (
            4,
        ), f"Expected thickness shape (4,), got {store['thickness'].shape}"

        # Check metadata
        assert "filename" in store.attrs, "filename not in metadata"
        assert store.attrs["filename"] == "test_run"
        assert store.attrs["num_timesteps"] == 3
        assert store.attrs["num_nodes"] == 4


def test_zarr_sink_should_skip():
    """Test Zarr sink skip logic."""
    from physicsnemo_curator.etl.processing_config import ProcessingConfig

    cfg = ProcessingConfig(num_processes=1)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a dummy zarr directory
        dummy_store = Path(temp_dir) / "existing_run.zarr"
        dummy_store.mkdir()

        # Test with overwrite_existing=True (should not skip)
        sink = CrashZarrDataSource(cfg, temp_dir, overwrite_existing=True)
        assert not sink.should_skip(
            "existing_run"
        ), "Should not skip when overwrite_existing=True"

        # Test with overwrite_existing=False (should skip)
        sink = CrashZarrDataSource(cfg, temp_dir, overwrite_existing=False)
        assert sink.should_skip(
            "existing_run"
        ), "Should skip when store exists and overwrite_existing=False"

        # Test with non-existent store
        assert not sink.should_skip(
            "non_existent_run"
        ), "Should not skip non-existent store"


def test_zarr_sink_metadata_statistics(mock_crash_data):
    """Test that Zarr store includes thickness statistics in metadata."""
    from physicsnemo_curator.etl.processing_config import ProcessingConfig

    cfg = ProcessingConfig(num_processes=1)

    with tempfile.TemporaryDirectory() as temp_dir:
        sink = CrashZarrDataSource(cfg, temp_dir, overwrite_existing=True)
        sink.write(mock_crash_data, "test_run")

        store = zarr.open(str(Path(temp_dir) / "test_run.zarr"), mode="r")

        # Check statistics
        assert "thickness_min" in store.attrs
        assert "thickness_max" in store.attrs
        assert "thickness_mean" in store.attrs

        assert store.attrs["thickness_min"] == 1.0
        assert store.attrs["thickness_max"] == 1.0
        assert store.attrs["thickness_mean"] == 1.0


def test_zarr_sink_chunk_size_parameter(mock_crash_data):
    """Test that chunk_size_mb parameter is respected."""
    from physicsnemo_curator.etl.processing_config import ProcessingConfig

    cfg = ProcessingConfig(num_processes=1)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with custom chunk size
        sink = CrashZarrDataSource(
            cfg, temp_dir, overwrite_existing=True, chunk_size_mb=2.0
        )

        assert sink.chunk_size_mb == 2.0, "chunk_size_mb should be set to 2.0"

        sink.write(mock_crash_data, "test_run")

        store = zarr.open(str(Path(temp_dir) / "test_run.zarr"), mode="r")

        # Check that chunk_size_mb is stored in metadata
        assert "chunk_size_mb" in store.attrs
        assert store.attrs["chunk_size_mb"] == 2.0


def test_zarr_sink_chunk_size_warnings():
    """Test that warnings are issued for problematic chunk sizes."""
    import warnings as warnings_module

    from physicsnemo_curator.etl.processing_config import ProcessingConfig

    cfg = ProcessingConfig(num_processes=1)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test warning for very small chunk size
        with pytest.warns(UserWarning, match="very small"):
            CrashZarrDataSource(
                cfg, temp_dir, overwrite_existing=True, chunk_size_mb=0.05
            )

        # Test warning for very large chunk size
        with pytest.warns(UserWarning, match="very large"):
            CrashZarrDataSource(
                cfg, temp_dir, overwrite_existing=True, chunk_size_mb=150.0
            )

        # Test no warning for reasonable chunk size
        with warnings_module.catch_warnings():
            warnings_module.simplefilter("error")
            # This should not raise any warnings
            CrashZarrDataSource(
                cfg, temp_dir, overwrite_existing=True, chunk_size_mb=1.0
            )


def test_zarr_sink_calculate_chunks():
    """Test the _calculate_chunks method for different array shapes."""
    from physicsnemo_curator.etl.processing_config import ProcessingConfig

    cfg = ProcessingConfig(num_processes=1)

    with tempfile.TemporaryDirectory() as temp_dir:
        sink = CrashZarrDataSource(
            cfg, temp_dir, overwrite_existing=True, chunk_size_mb=1.0
        )

        # Test 1D array (e.g., thickness)
        thickness_array = np.ones(1000, dtype=np.float32)
        chunks_1d = sink._calculate_chunks(thickness_array)
        assert len(chunks_1d) == 1, "1D chunks should have 1 dimension"
        assert chunks_1d[0] > 0, "Chunk size should be positive"
        assert chunks_1d[0] <= len(
            thickness_array
        ), "Chunk size should not exceed array size"

        # Test 2D array (e.g., edges)
        edges_array = np.ones((500, 2), dtype=np.int64)
        chunks_2d = sink._calculate_chunks(edges_array)
        assert len(chunks_2d) == 2, "2D chunks should have 2 dimensions"
        assert chunks_2d[0] > 0, "First chunk dimension should be positive"
        assert chunks_2d[1] == 2, "Second chunk dimension should match array shape"

        # Test 3D array (e.g., mesh_pos)
        mesh_pos_array = np.ones((10, 1000, 3), dtype=np.float32)
        chunks_3d = sink._calculate_chunks(mesh_pos_array)
        assert len(chunks_3d) == 3, "3D chunks should have 3 dimensions"
        assert chunks_3d[0] > 0, "First chunk dimension should be positive"
        assert chunks_3d[1] > 0, "Second chunk dimension should be positive"
        assert (
            chunks_3d[2] == 3
        ), "Third chunk dimension should match coordinate dimension"


def test_zarr_sink_chunks_stored_correctly(mock_crash_data):
    """Test that calculated chunks are actually used in Zarr store."""
    from physicsnemo_curator.etl.processing_config import ProcessingConfig

    cfg = ProcessingConfig(num_processes=1)

    with tempfile.TemporaryDirectory() as temp_dir:
        sink = CrashZarrDataSource(
            cfg, temp_dir, overwrite_existing=True, chunk_size_mb=0.5
        )
        sink.write(mock_crash_data, "test_run")

        store = zarr.open(str(Path(temp_dir) / "test_run.zarr"), mode="r")

        # Check that arrays have chunk information
        assert store["mesh_pos"].chunks is not None, "mesh_pos should have chunks"
        assert store["thickness"].chunks is not None, "thickness should have chunks"
        assert store["edges"].chunks is not None, "edges should have chunks"

        # Verify chunks have correct number of dimensions
        assert len(store["mesh_pos"].chunks) == 3, "mesh_pos chunks should be 3D"
        assert len(store["thickness"].chunks) == 1, "thickness chunks should be 1D"
        assert len(store["edges"].chunks) == 2, "edges chunks should be 2D"


def test_zarr_sink_chunks_match_calculated(mock_crash_data):
    """Test that stored chunks exactly match the calculated chunk sizes."""
    from physicsnemo_curator.etl.processing_config import ProcessingConfig

    cfg = ProcessingConfig(num_processes=1)

    with tempfile.TemporaryDirectory() as temp_dir:
        sink = CrashZarrDataSource(
            cfg, temp_dir, overwrite_existing=True, chunk_size_mb=1.0
        )

        # Calculate expected chunks BEFORE writing
        mesh_pos_data = mock_crash_data.filtered_pos_raw.astype(np.float32)
        thickness_data = mock_crash_data.filtered_node_thickness.astype(np.float32)
        edges_array = np.array(list(mock_crash_data.edges), dtype=np.int64)

        expected_mesh_pos_chunks = sink._calculate_chunks(mesh_pos_data)
        expected_thickness_chunks = sink._calculate_chunks(thickness_data)
        expected_edges_chunks = sink._calculate_chunks(edges_array)

        # Write data
        sink.write(mock_crash_data, "test_run")

        # Open store and verify chunks match exactly
        store = zarr.open(str(Path(temp_dir) / "test_run.zarr"), mode="r")

        assert store["mesh_pos"].chunks == expected_mesh_pos_chunks, (
            f"mesh_pos chunks mismatch: expected {expected_mesh_pos_chunks}, "
            f"got {store['mesh_pos'].chunks}"
        )

        assert store["thickness"].chunks == expected_thickness_chunks, (
            f"thickness chunks mismatch: expected {expected_thickness_chunks}, "
            f"got {store['thickness'].chunks}"
        )

        assert store["edges"].chunks == expected_edges_chunks, (
            f"edges chunks mismatch: expected {expected_edges_chunks}, "
            f"got {store['edges'].chunks}"
        )


def test_zarr_sink_chunks_physical_layout():
    """Test that different chunk sizes produce different physical layouts."""
    from physicsnemo_curator.etl.processing_config import ProcessingConfig

    cfg = ProcessingConfig(num_processes=1)

    # Create larger mock data to see chunk differences
    large_pos_raw = np.random.randn(20, 2000, 3).astype(np.float32)
    large_thickness = np.ones(2000, dtype=np.float32)
    large_edges = {(i, i + 1) for i in range(1999)}

    large_data = CrashExtractedDataInMemory(
        metadata=CrashMetadata(filename="large_test"),
        pos_raw=large_pos_raw,
        mesh_connectivity=[],
        node_thickness=large_thickness,
        filtered_pos_raw=large_pos_raw,
        filtered_mesh_connectivity=[],
        filtered_node_thickness=large_thickness,
        edges=large_edges,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        # Write with small chunks
        sink_small = CrashZarrDataSource(
            cfg, temp_dir, overwrite_existing=True, chunk_size_mb=0.1
        )
        sink_small.write(large_data, "small_chunks")

        # Write with large chunks
        sink_large = CrashZarrDataSource(
            cfg, temp_dir, overwrite_existing=True, chunk_size_mb=10.0
        )
        sink_large.write(large_data, "large_chunks")

        # Open both stores
        store_small = zarr.open(str(Path(temp_dir) / "small_chunks.zarr"), mode="r")
        store_large = zarr.open(str(Path(temp_dir) / "large_chunks.zarr"), mode="r")

        # Small chunks should have smaller chunk sizes than large chunks
        # (at least for the dimensions that get chunked)
        small_chunks = store_small["mesh_pos"].chunks
        large_chunks = store_large["mesh_pos"].chunks

        # The chunk sizes should be different
        assert (
            small_chunks != large_chunks
        ), "Different chunk_size_mb should produce different chunk layouts"

        # Small chunks should generally have smaller values
        # (though the last dimension is always 3 for coordinates)
        small_total = np.prod(small_chunks)
        large_total = np.prod(large_chunks)

        assert small_total < large_total, (
            f"Small chunks (product={small_total}) should have smaller total size "
            f"than large chunks (product={large_total})"
        )
