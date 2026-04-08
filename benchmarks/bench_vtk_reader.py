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

"""ASV benchmarks comparing Rust VTK reader vs PyVista.

These benchmarks measure:
- Single file read performance
- Parallel batch read performance
- Speedup factor of Rust vs PyVista
"""

from __future__ import annotations

import tempfile
from pathlib import Path

# ── benchmarks ───────────────────────────────────────────────────────────────


class TimeVTKReaderSingle:
    """Benchmark single-file VTK reading: Rust vs PyVista."""

    params = [100, 1000, 5000]
    param_names = ["n_points"]

    def setup(self, n_points: int) -> None:
        """Create a test VTK file for benchmarking."""
        self._tmpdir = tempfile.mkdtemp()
        n_cells = n_points // 2
        self.test_file = str(Path(self._tmpdir) / "test.vtu")
        self._write_test_vtu(Path(self.test_file), n_points=n_points, n_cells=n_cells)

    def teardown(self, n_points: int) -> None:
        """Clean up test files."""
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _write_test_vtu(self, path: Path, n_points: int, n_cells: int) -> None:
        """Write a test VTU file with specified size."""
        import numpy as np

        # Generate random points
        points = np.random.rand(n_points, 3).flatten()
        points_str = " ".join(f"{x:.6f}" for x in points)

        # Generate random triangles (3 points each)
        connectivity = np.random.randint(0, n_points, size=n_cells * 3)
        connectivity_str = " ".join(str(x) for x in connectivity)

        offsets = [3 * (i + 1) for i in range(n_cells)]
        offsets_str = " ".join(str(x) for x in offsets)

        types = [5] * n_cells  # VTK_TRIANGLE
        types_str = " ".join(str(x) for x in types)

        # Generate random point data
        temperature = np.random.rand(n_points)
        temp_str = " ".join(f"{x:.6f}" for x in temperature)

        xml = f"""<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1">
  <UnstructuredGrid>
    <Piece NumberOfPoints="{n_points}" NumberOfCells="{n_cells}">
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">
          {points_str}
        </DataArray>
      </Points>
      <Cells>
        <DataArray Name="connectivity" type="Int64" format="ascii">
          {connectivity_str}
        </DataArray>
        <DataArray Name="offsets" type="Int64" format="ascii">
          {offsets_str}
        </DataArray>
        <DataArray Name="types" type="UInt8" format="ascii">
          {types_str}
        </DataArray>
      </Cells>
      <PointData>
        <DataArray Name="Temperature" type="Float64" NumberOfComponents="1" format="ascii">
          {temp_str}
        </DataArray>
      </PointData>
    </Piece>
  </UnstructuredGrid>
</VTKFile>"""
        path.write_text(xml)

    def time_pyvista(self, n_points: int) -> None:
        """Benchmark PyVista reading a single file."""
        import pyvista as pv

        _ = pv.read(self.test_file)

    def time_rust(self, n_points: int) -> None:
        """Benchmark Rust reader reading a single file."""
        from physicsnemo.curator._lib import vtk

        _ = vtk.read_vtk(self.test_file)


class TimeVTKReaderParallel:
    """Benchmark parallel VTK reading: Rust parallel vs PyVista sequential."""

    params = [1, 4, 8, 16]
    param_names = ["n_files"]

    def setup(self, n_files: int) -> None:
        """Create test VTK files for benchmarking."""
        self._tmpdir = tempfile.mkdtemp()
        self.test_files: list[str] = []

        # Create n_files VTU files with moderate complexity
        for i in range(n_files):
            path = Path(self._tmpdir) / f"test_{i}.vtu"
            self._write_test_vtu(path, n_points=1000, n_cells=500)
            self.test_files.append(str(path))

        # Single file for single-file benchmarks
        self.single_file = self.test_files[0]

    def teardown(self, n_files: int) -> None:
        """Clean up test files."""
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _write_test_vtu(self, path: Path, n_points: int, n_cells: int) -> None:
        """Write a test VTU file with specified size."""
        import numpy as np

        # Generate random points
        points = np.random.rand(n_points, 3).flatten()
        points_str = " ".join(f"{x:.6f}" for x in points)

        # Generate random triangles (3 points each)
        connectivity = np.random.randint(0, n_points, size=n_cells * 3)
        connectivity_str = " ".join(str(x) for x in connectivity)

        offsets = [3 * (i + 1) for i in range(n_cells)]
        offsets_str = " ".join(str(x) for x in offsets)

        types = [5] * n_cells  # VTK_TRIANGLE
        types_str = " ".join(str(x) for x in types)

        # Generate random point data
        temperature = np.random.rand(n_points)
        temp_str = " ".join(f"{x:.6f}" for x in temperature)

        xml = f"""<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1">
  <UnstructuredGrid>
    <Piece NumberOfPoints="{n_points}" NumberOfCells="{n_cells}">
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">
          {points_str}
        </DataArray>
      </Points>
      <Cells>
        <DataArray Name="connectivity" type="Int64" format="ascii">
          {connectivity_str}
        </DataArray>
        <DataArray Name="offsets" type="Int64" format="ascii">
          {offsets_str}
        </DataArray>
        <DataArray Name="types" type="UInt8" format="ascii">
          {types_str}
        </DataArray>
      </Cells>
      <PointData>
        <DataArray Name="Temperature" type="Float64" NumberOfComponents="1" format="ascii">
          {temp_str}
        </DataArray>
      </PointData>
    </Piece>
  </UnstructuredGrid>
</VTKFile>"""
        path.write_text(xml)

    def time_pyvista_sequential(self, n_files: int) -> None:
        """Benchmark PyVista reading multiple files sequentially."""
        import pyvista as pv

        for f in self.test_files:
            _ = pv.read(f)

    def time_rust_parallel(self, n_files: int) -> None:
        """Benchmark Rust reader reading multiple files in parallel."""
        from physicsnemo.curator._lib import vtk

        _ = vtk.read_vtk_parallel(self.test_files)

    def time_rust_sequential(self, n_files: int) -> None:
        """Benchmark Rust reader reading multiple files sequentially."""
        from physicsnemo.curator._lib import vtk

        for f in self.test_files:
            _ = vtk.read_vtk(f)


class MemVTKReader:
    """Memory footprint of VTK readers."""

    params = [1000, 5000]
    param_names = ["n_points"]

    def setup(self, n_points: int) -> None:
        """Create a test VTK file for memory benchmarking."""
        self._tmpdir = tempfile.mkdtemp()
        n_cells = n_points // 2
        self.test_file = str(Path(self._tmpdir) / "test.vtu")
        self._write_test_vtu(Path(self.test_file), n_points=n_points, n_cells=n_cells)

    def teardown(self, n_points: int) -> None:
        """Clean up test files."""
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _write_test_vtu(self, path: Path, n_points: int, n_cells: int) -> None:
        """Write a test VTU file with specified size."""
        import numpy as np

        # Generate random points
        points = np.random.rand(n_points, 3).flatten()
        points_str = " ".join(f"{x:.6f}" for x in points)

        # Generate random triangles (3 points each)
        connectivity = np.random.randint(0, n_points, size=n_cells * 3)
        connectivity_str = " ".join(str(x) for x in connectivity)

        offsets = [3 * (i + 1) for i in range(n_cells)]
        offsets_str = " ".join(str(x) for x in offsets)

        types = [5] * n_cells  # VTK_TRIANGLE
        types_str = " ".join(str(x) for x in types)

        # Generate random point data
        temperature = np.random.rand(n_points)
        temp_str = " ".join(f"{x:.6f}" for x in temperature)

        xml = f"""<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1">
  <UnstructuredGrid>
    <Piece NumberOfPoints="{n_points}" NumberOfCells="{n_cells}">
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">
          {points_str}
        </DataArray>
      </Points>
      <Cells>
        <DataArray Name="connectivity" type="Int64" format="ascii">
          {connectivity_str}
        </DataArray>
        <DataArray Name="offsets" type="Int64" format="ascii">
          {offsets_str}
        </DataArray>
        <DataArray Name="types" type="UInt8" format="ascii">
          {types_str}
        </DataArray>
      </Cells>
      <PointData>
        <DataArray Name="Temperature" type="Float64" NumberOfComponents="1" format="ascii">
          {temp_str}
        </DataArray>
      </PointData>
    </Piece>
  </UnstructuredGrid>
</VTKFile>"""
        path.write_text(xml)

    def peakmem_rust(self, n_points: int):
        """Track peak RSS when reading with the Rust VTK reader."""
        from physicsnemo.curator._lib import vtk

        vtk.read_vtk(self.test_file)
