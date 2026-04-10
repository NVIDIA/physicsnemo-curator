# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""ASV benchmarks for the Mesh domain.

Covers end-to-end pipeline performance, VTK reader Rust-vs-Python comparisons,
d3plot component Rust-vs-Python comparisons, filter throughput, and sink write
time.
"""

from __future__ import annotations

from pathlib import Path

from ._helpers import (
    cleanup_temp_dir,
    create_temp_dir,
    make_connectivity,
    write_synthetic_k_file,
    write_synthetic_vtu,
)


# ---------------------------------------------------------------------------
# E2E: Full Mesh pipeline
# ---------------------------------------------------------------------------
class TimeMeshE2E:
    """End-to-end Mesh pipeline: VTKSource -> PrecisionFilter -> MeshInfoFilter -> MeshSink."""

    params = [10, 50]
    param_names = ["n_files"]

    def setup(self, n_files):
        """Generate synthetic VTU files and build the pipeline."""
        from physicsnemo_curator.core.base import Pipeline
        from physicsnemo_curator.core.store import LocalFileStore
        from physicsnemo_curator.mesh.filters.mesh_info import MeshInfoFilter
        from physicsnemo_curator.mesh.filters.precision import PrecisionFilter
        from physicsnemo_curator.mesh.sinks.mesh_writer import MeshSink
        from physicsnemo_curator.mesh.sources.vtk import VTKSource

        self._input_dir = create_temp_dir()
        self._output_dir = create_temp_dir()
        self._info_dir = create_temp_dir()
        n_points, n_cells = 1000, 500

        for i in range(n_files):
            write_synthetic_vtu(
                Path(self._input_dir) / f"mesh_{i:04d}.vtu",
                n_points,
                n_cells,
                seed=42 + i,
            )

        store = LocalFileStore(self._input_dir, extensions=frozenset({".vtu"}))
        source = VTKSource(store, backend="pyvista")
        precision = PrecisionFilter(target_dtype="float32")
        info = MeshInfoFilter(
            output=str(Path(self._info_dir) / "info.json"),
            log_level="warning",
        )
        sink = MeshSink(output_dir=self._output_dir)

        self.pipeline = Pipeline(
            source=source,
            filters=[precision, info],  # ty: ignore[invalid-argument-type]
            sink=sink,
            track_metrics=False,
            track_memory=False,
        )
        self.n_files = n_files

    def time_e2e(self, n_files):
        """Run the full pipeline for all indices."""
        for i in range(len(self.pipeline)):
            self.pipeline[i]

    def teardown(self, n_files):
        """Remove temporary directories."""
        cleanup_temp_dir(self._input_dir)
        cleanup_temp_dir(self._output_dir)
        cleanup_temp_dir(self._info_dir)


class MemMeshE2E:
    """Peak memory for full Mesh pipeline."""

    params = [10, 50]
    param_names = ["n_files"]

    def setup(self, n_files):
        """Generate synthetic VTU files and build the pipeline."""
        from physicsnemo_curator.core.base import Pipeline
        from physicsnemo_curator.core.store import LocalFileStore
        from physicsnemo_curator.mesh.filters.mesh_info import MeshInfoFilter
        from physicsnemo_curator.mesh.filters.precision import PrecisionFilter
        from physicsnemo_curator.mesh.sinks.mesh_writer import MeshSink
        from physicsnemo_curator.mesh.sources.vtk import VTKSource

        self._input_dir = create_temp_dir()
        self._output_dir = create_temp_dir()
        self._info_dir = create_temp_dir()
        n_points, n_cells = 1000, 500

        for i in range(n_files):
            write_synthetic_vtu(
                Path(self._input_dir) / f"mesh_{i:04d}.vtu",
                n_points,
                n_cells,
                seed=42 + i,
            )

        store = LocalFileStore(self._input_dir, extensions=frozenset({".vtu"}))
        source = VTKSource(store, backend="pyvista")
        precision = PrecisionFilter(target_dtype="float32")
        info = MeshInfoFilter(
            output=str(Path(self._info_dir) / "info.json"),
            log_level="warning",
        )
        sink = MeshSink(output_dir=self._output_dir)

        self.pipeline = Pipeline(
            source=source,
            filters=[precision, info],  # ty: ignore[invalid-argument-type]
            sink=sink,
            track_metrics=False,
            track_memory=False,
        )
        self.n_files = n_files

    def peakmem_e2e(self, n_files):
        """Run full pipeline, tracking peak RSS."""
        for i in range(len(self.pipeline)):
            self.pipeline[i]

    def teardown(self, n_files):
        """Remove temporary directories."""
        cleanup_temp_dir(self._input_dir)
        cleanup_temp_dir(self._output_dir)
        cleanup_temp_dir(self._info_dir)


# ---------------------------------------------------------------------------
# Component: VTKSource read
# ---------------------------------------------------------------------------
class TimeVTKSourceRead:
    """Time VTKSource.__getitem__ for different mesh sizes."""

    params = [100, 1000, 5000]
    param_names = ["n_points"]

    def setup(self, n_points):
        """Generate a single VTU file and create the source."""
        from physicsnemo_curator.core.store import LocalFileStore
        from physicsnemo_curator.mesh.sources.vtk import VTKSource

        self._tmpdir = create_temp_dir()
        n_cells = n_points // 2
        write_synthetic_vtu(Path(self._tmpdir) / "test.vtu", n_points, n_cells)
        store = LocalFileStore(self._tmpdir, extensions=frozenset({".vtu"}))
        self.source = VTKSource(store, backend="pyvista")

    def time_read(self, n_points):
        """Read a single mesh via VTKSource."""
        # Drain the generator
        for _ in self.source[0]:
            pass

    def teardown(self, n_points):
        """Remove temporary directory."""
        cleanup_temp_dir(self._tmpdir)


# ---------------------------------------------------------------------------
# Component: VTK reader Rust vs PyVista
# ---------------------------------------------------------------------------
class TimeVTKReaderComparison:
    """Compare Rust vtk.read_vtk vs PyVista pv.read on single files."""

    params = [100, 1000, 5000]
    param_names = ["n_points"]

    def setup(self, n_points):
        """Generate a single VTU file."""
        self._tmpdir = create_temp_dir()
        n_cells = n_points // 2
        self.test_file = str(Path(self._tmpdir) / "test.vtu")
        write_synthetic_vtu(Path(self.test_file), n_points, n_cells)

    def time_pyvista(self, n_points):
        """Read via PyVista."""
        import pyvista as pv

        pv.read(self.test_file)

    def time_rust(self, n_points):
        """Read via Rust VTK reader."""
        from physicsnemo_curator._lib import vtk

        vtk.read_vtk(self.test_file)

    def teardown(self, n_points):
        """Remove temporary directory."""
        cleanup_temp_dir(self._tmpdir)


class TimeVTKReaderParallel:
    """Compare Rust parallel VTK reader vs sequential reads."""

    params = [1, 4, 8]
    param_names = ["n_files"]

    def setup(self, n_files):
        """Generate multiple VTU files."""
        self._tmpdir = create_temp_dir()
        n_points, n_cells = 1000, 500
        self.files = []
        for i in range(n_files):
            p = str(Path(self._tmpdir) / f"test_{i:04d}.vtu")
            write_synthetic_vtu(Path(p), n_points, n_cells, seed=42 + i)
            self.files.append(p)

    def time_rust_parallel(self, n_files):
        """Read all files with Rust parallel reader."""
        from physicsnemo_curator._lib import vtk

        vtk.read_vtk_parallel(self.files)

    def time_rust_sequential(self, n_files):
        """Read all files with Rust sequential reader."""
        from physicsnemo_curator._lib import vtk

        for f in self.files:
            vtk.read_vtk(f)

    def time_pyvista_sequential(self, n_files):
        """Read all files with PyVista sequentially."""
        import pyvista as pv

        for f in self.files:
            pv.read(f)

    def teardown(self, n_files):
        """Remove temporary directory."""
        cleanup_temp_dir(self._tmpdir)


# ---------------------------------------------------------------------------
# PrecisionFilter benchmarks
# ---------------------------------------------------------------------------
class TimePrecisionFilter:
    """Time PrecisionFilter throughput for different mesh sizes."""

    params = [1000, 5000]
    param_names = ["n_points"]

    def setup(self, n_points):
        """Generate a VTU file and create source + filter."""
        from physicsnemo_curator.core.store import LocalFileStore
        from physicsnemo_curator.mesh.filters.precision import PrecisionFilter
        from physicsnemo_curator.mesh.sources.vtk import VTKSource

        self._tmpdir = create_temp_dir()
        n_cells = n_points // 2
        write_synthetic_vtu(Path(self._tmpdir) / "test.vtu", n_points, n_cells)
        store = LocalFileStore(self._tmpdir, extensions=frozenset({".vtu"}))
        self.source = VTKSource(store, backend="pyvista")
        self.filt = PrecisionFilter(target_dtype="float32")

    def time_filter(self, n_points):
        """Apply PrecisionFilter to a single mesh."""
        stream = self.source[0]
        for _ in self.filt(stream):
            pass

    def teardown(self, n_points):
        """Remove temporary directory."""
        cleanup_temp_dir(self._tmpdir)


# ---------------------------------------------------------------------------
# MeshInfoFilter benchmarks
# ---------------------------------------------------------------------------
class TimeMeshInfoFilter:
    """Time MeshInfoFilter throughput for different mesh sizes."""

    params = [1000, 5000]
    param_names = ["n_points"]

    def setup(self, n_points):
        """Generate a VTU file and create source + filter."""
        from physicsnemo_curator.core.store import LocalFileStore
        from physicsnemo_curator.mesh.filters.mesh_info import MeshInfoFilter
        from physicsnemo_curator.mesh.sources.vtk import VTKSource

        self._tmpdir = create_temp_dir()
        self._info_dir = create_temp_dir()
        n_cells = n_points // 2
        write_synthetic_vtu(Path(self._tmpdir) / "test.vtu", n_points, n_cells)
        store = LocalFileStore(self._tmpdir, extensions=frozenset({".vtu"}))
        self.source = VTKSource(store, backend="pyvista")
        self.filt = MeshInfoFilter(
            output=str(Path(self._info_dir) / "info.json"),
            log_level="warning",
        )

    def time_filter(self, n_points):
        """Apply MeshInfoFilter to a single mesh."""
        stream = self.source[0]
        for _ in self.filt(stream):
            pass

    def teardown(self, n_points):
        """Remove temporary directories."""
        cleanup_temp_dir(self._tmpdir)
        cleanup_temp_dir(self._info_dir)


# ---------------------------------------------------------------------------
# MeanFilter benchmarks
# ---------------------------------------------------------------------------
class TimeMeanFilter:
    """Time MeanFilter throughput for different mesh sizes."""

    params = [1000, 5000]
    param_names = ["n_points"]

    def setup(self, n_points):
        """Generate a VTU file and create source + filter."""
        from physicsnemo_curator.core.store import LocalFileStore
        from physicsnemo_curator.mesh.filters.mean import MeanFilter
        from physicsnemo_curator.mesh.sources.vtk import VTKSource

        self._tmpdir = create_temp_dir()
        self._output_dir = create_temp_dir()
        n_cells = n_points // 2
        write_synthetic_vtu(Path(self._tmpdir) / "test.vtu", n_points, n_cells)
        store = LocalFileStore(self._tmpdir, extensions=frozenset({".vtu"}))
        self.source = VTKSource(store, backend="pyvista")
        self.filt = MeanFilter(output=str(Path(self._output_dir) / "mean.parquet"))

    def time_filter(self, n_points):
        """Apply MeanFilter to a single mesh."""
        stream = self.source[0]
        for _ in self.filt(stream):
            pass

    def teardown(self, n_points):
        """Remove temporary directories."""
        cleanup_temp_dir(self._tmpdir)
        cleanup_temp_dir(self._output_dir)


# ---------------------------------------------------------------------------
# Component: D3Plot Rust vs Python
# ---------------------------------------------------------------------------
class TimeD3PlotComponents:
    """Compare Rust vs Python d3plot post-processing functions."""

    params = [1_000, 10_000, 100_000]
    param_names = ["n_elements"]

    def setup(self, n_elements):
        """Create synthetic connectivity, part data, and stress tensors."""
        import numpy as np

        self._tmpdir = create_temp_dir()
        n_parts = 50
        self.k_file = Path(self._tmpdir) / "test.k"
        write_synthetic_k_file(self.k_file, n_parts)

        n_nodes = n_elements * 2
        self.connectivity = make_connectivity(n_elements, n_nodes)

        rng = np.random.default_rng(42)
        self.part_ids = rng.integers(1, n_parts + 1, size=n_elements, dtype=np.int64)
        self.part_thickness = {i: float(i) * 0.5 for i in range(1, n_parts + 1)}
        self.stress = rng.random((n_elements, 6))

    def time_parse_k_file_python(self, n_elements):
        """Parse .k file with Python implementation."""
        from physicsnemo_curator.mesh.sources.d3plot import _parse_k_file

        _parse_k_file(self.k_file)

    def time_parse_k_file_rust(self, n_elements):
        """Parse .k file with Rust implementation."""
        from physicsnemo_curator.mesh.sources.d3plot import _parse_k_file_rust

        _parse_k_file_rust(self.k_file)

    def time_node_thickness_python(self, n_elements):
        """Compute node thickness with Python implementation."""
        from physicsnemo_curator.mesh.sources.d3plot import _compute_node_thickness

        _compute_node_thickness(self.connectivity, self.part_ids, self.part_thickness)

    def time_node_thickness_rust(self, n_elements):
        """Compute node thickness with Rust implementation."""
        from physicsnemo_curator.mesh.sources.d3plot import (
            _compute_node_thickness_rust,
        )

        _compute_node_thickness_rust(self.connectivity, self.part_ids, self.part_thickness)

    def time_von_mises_python(self, n_elements):
        """Compute von Mises stress with Python implementation."""
        from physicsnemo_curator.mesh.sources.d3plot import _von_mises_from_voigt

        _von_mises_from_voigt(self.stress)

    def time_von_mises_rust(self, n_elements):
        """Compute von Mises stress with Rust implementation."""
        from physicsnemo_curator.mesh.sources.d3plot import (
            _von_mises_from_voigt_rust,
        )

        _von_mises_from_voigt_rust(self.stress)

    def teardown(self, n_elements):
        """Remove temporary directory."""
        cleanup_temp_dir(self._tmpdir)


class PeakmemD3PlotComponents:
    """Peak memory for d3plot Rust vs Python post-processing."""

    params = [10_000, 100_000]
    param_names = ["n_elements"]

    def setup(self, n_elements):
        """Create synthetic connectivity and part data."""
        import numpy as np

        n_nodes = n_elements * 2
        n_parts = 50
        self.connectivity = make_connectivity(n_elements, n_nodes)
        rng = np.random.default_rng(42)
        self.part_ids = rng.integers(1, n_parts + 1, size=n_elements, dtype=np.int64)
        self.part_thickness = {i: float(i) * 0.5 for i in range(1, n_parts + 1)}

    def peakmem_node_thickness_python(self, n_elements):
        """Peak memory for Python node thickness."""
        from physicsnemo_curator.mesh.sources.d3plot import _compute_node_thickness

        _compute_node_thickness(self.connectivity, self.part_ids, self.part_thickness)

    def peakmem_node_thickness_rust(self, n_elements):
        """Peak memory for Rust node thickness."""
        from physicsnemo_curator.mesh.sources.d3plot import (
            _compute_node_thickness_rust,
        )

        _compute_node_thickness_rust(self.connectivity, self.part_ids, self.part_thickness)


# ---------------------------------------------------------------------------
# MeshSink benchmarks
# ---------------------------------------------------------------------------
class TimeMeshSink:
    """Time MeshSink write for different mesh sizes."""

    params = [1000, 5000]
    param_names = ["n_points"]

    def setup(self, n_points):
        """Generate a VTU file and create source + sink."""
        from physicsnemo_curator.core.store import LocalFileStore
        from physicsnemo_curator.mesh.sinks.mesh_writer import MeshSink
        from physicsnemo_curator.mesh.sources.vtk import VTKSource

        self._input_dir = create_temp_dir()
        self._output_dir = create_temp_dir()
        n_cells = n_points // 2
        write_synthetic_vtu(Path(self._input_dir) / "test.vtu", n_points, n_cells)
        store = LocalFileStore(self._input_dir, extensions=frozenset({".vtu"}))
        self.source = VTKSource(store, backend="pyvista")
        self.sink = MeshSink(output_dir=self._output_dir)

    def time_write(self, n_points):
        """Write a single mesh via MeshSink."""
        stream = self.source[0]
        self.sink(stream, 0)

    def teardown(self, n_points):
        """Remove temporary directories."""
        cleanup_temp_dir(self._input_dir)
        cleanup_temp_dir(self._output_dir)
