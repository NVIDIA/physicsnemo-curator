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

"""Benchmarks for the FileStore abstraction layer."""

import tempfile
from pathlib import Path

from physicsnemo_curator.core.store import LocalFileStore

# ── benchmarks ───────────────────────────────────────────────────────────────


class TimeLocalFileStoreCreation:
    """Benchmark LocalFileStore instantiation and file discovery."""

    params = [10, 100, 500]
    param_names = ["num_files"]

    def setup(self, num_files):
        """Create a temp directory with N dummy files."""
        self._tmpdir = tempfile.mkdtemp()
        for i in range(num_files):
            Path(self._tmpdir, f"file_{i:04d}.vtk").touch()

    def time_create_store(self, num_files):
        """Time constructing a LocalFileStore over a directory."""
        LocalFileStore(self._tmpdir, extensions=frozenset({".vtk"}))

    def teardown(self, num_files):
        """Remove temp directory."""
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)


class TimeLocalFileStoreIndexing:
    """Benchmark file path retrieval from a store."""

    params = [100, 1000]
    param_names = ["num_files"]

    def setup(self, num_files):
        """Create store with N files."""
        self._tmpdir = tempfile.mkdtemp()
        for i in range(num_files):
            Path(self._tmpdir, f"file_{i:04d}.vtk").touch()
        self.store = LocalFileStore(self._tmpdir, extensions=frozenset({".vtk"}))

    def time_index_all(self, num_files):
        """Time indexing every file path in the store."""
        for i in range(len(self.store)):
            self.store[i]

    def teardown(self, num_files):
        """Remove temp directory."""
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)


class MemLocalFileStore:
    """Memory footprint of a LocalFileStore."""

    params = [100, 1000]
    param_names = ["num_files"]

    def setup(self, num_files):
        """Create temp directory with N dummy files."""
        self._tmpdir = tempfile.mkdtemp()
        for i in range(num_files):
            Path(self._tmpdir, f"file_{i:04d}.vtk").touch()

    def mem_store(self, num_files):
        """Track memory of a LocalFileStore."""
        return LocalFileStore(self._tmpdir, extensions=frozenset({".vtk"}))

    def teardown(self, num_files):
        """Remove temp directory."""
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)
