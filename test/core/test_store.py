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

"""Unit tests for FileStore, LocalFileStore, and FsspecFileStore."""

from __future__ import annotations

import pathlib

import pytest

from physicsnemo.curator.core.store import FileStore, FsspecFileStore, LocalFileStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _populate_dir(tmp_path: pathlib.Path, names: list[str]) -> pathlib.Path:
    """Create empty files with the given names in *tmp_path*.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Target directory.
    names : list[str]
        File names to create.

    Returns
    -------
    pathlib.Path
        The *tmp_path* directory.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name in names:
        (tmp_path / name).write_bytes(b"")
    return tmp_path


# ---------------------------------------------------------------------------
# FileStore Protocol
# ---------------------------------------------------------------------------


class TestFileStoreProtocol:
    def test_local_file_store_is_file_store(self, tmp_path):
        _populate_dir(tmp_path, ["a.txt"])
        store = LocalFileStore(str(tmp_path))
        assert isinstance(store, FileStore)

    def test_fsspec_file_store_is_file_store(self, tmp_path):
        _populate_dir(tmp_path, ["a.txt"])
        store = FsspecFileStore(f"file://{tmp_path}")
        assert isinstance(store, FileStore)

    def test_custom_class_matches_protocol(self):
        class Custom:
            def __len__(self):
                return 0

            def __getitem__(self, index):
                raise IndexError

        assert isinstance(Custom(), FileStore)


# ---------------------------------------------------------------------------
# LocalFileStore
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestLocalFileStore:
    def test_discovers_all_files(self, tmp_path):
        _populate_dir(tmp_path, ["a.txt", "b.txt", "c.txt"])
        store = LocalFileStore(str(tmp_path))
        assert len(store) == 3

    def test_sorted_order(self, tmp_path):
        _populate_dir(tmp_path, ["c.dat", "a.dat", "b.dat"])
        store = LocalFileStore(str(tmp_path))
        names = [pathlib.Path(store[i]).name for i in range(len(store))]
        assert names == ["a.dat", "b.dat", "c.dat"]

    def test_extension_filter(self, tmp_path):
        _populate_dir(tmp_path, ["a.vtk", "b.csv", "c.vtk", "d.txt"])
        store = LocalFileStore(str(tmp_path), extensions=frozenset({".vtk"}))
        assert len(store) == 2
        assert all(pathlib.Path(store[i]).suffix == ".vtk" for i in range(len(store)))

    def test_glob_pattern(self, tmp_path):
        _populate_dir(tmp_path, ["sim_001.dat", "sim_002.dat", "other.dat"])
        store = LocalFileStore(str(tmp_path), pattern="sim_*")
        assert len(store) == 2

    def test_single_file(self, tmp_path):
        _populate_dir(tmp_path, ["single.vtk"])
        store = LocalFileStore(str(tmp_path / "single.vtk"))
        assert len(store) == 1
        assert pathlib.Path(store[0]).name == "single.vtk"

    def test_single_file_wrong_extension(self, tmp_path):
        _populate_dir(tmp_path, ["data.csv"])
        with pytest.raises(ValueError, match="does not match allowed extensions"):
            LocalFileStore(str(tmp_path / "data.csv"), extensions=frozenset({".vtk"}))

    def test_nonexistent_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            LocalFileStore(str(tmp_path / "nonexistent"))

    def test_empty_directory_raises(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        with pytest.raises(ValueError, match="No matching files"):
            LocalFileStore(str(tmp_path))

    def test_empty_after_filter_raises(self, tmp_path):
        _populate_dir(tmp_path, ["a.txt", "b.csv"])
        with pytest.raises(ValueError, match="No matching files"):
            LocalFileStore(str(tmp_path), extensions=frozenset({".vtk"}))

    def test_getitem_returns_str(self, tmp_path):
        _populate_dir(tmp_path, ["file.dat"])
        store = LocalFileStore(str(tmp_path))
        assert isinstance(store[0], str)

    def test_index_out_of_range(self, tmp_path):
        _populate_dir(tmp_path, ["only.dat"])
        store = LocalFileStore(str(tmp_path))
        with pytest.raises(IndexError):
            store[5]

    def test_negative_index(self, tmp_path):
        _populate_dir(tmp_path, ["a.dat", "b.dat", "c.dat"])
        store = LocalFileStore(str(tmp_path))
        assert pathlib.Path(store[-1]).name == "c.dat"

    def test_repr(self, tmp_path):
        _populate_dir(tmp_path, ["a.dat"])
        store = LocalFileStore(str(tmp_path))
        r = repr(store)
        assert "LocalFileStore" in r
        assert "files=1" in r


# ---------------------------------------------------------------------------
# FsspecFileStore — using memory:// and file:// (no network needed)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFsspecFileStore:
    def test_local_file_protocol(self, tmp_path):
        """FsspecFileStore with file:// should work like LocalFileStore."""
        _populate_dir(tmp_path, ["a.vtk", "b.vtk"])
        store = FsspecFileStore(f"file://{tmp_path}", extensions=frozenset({".vtk"}))
        assert len(store) == 2

    def test_local_protocol_returns_real_paths(self, tmp_path):
        """For file:// protocol, paths should be directly usable."""
        _populate_dir(tmp_path, ["data.txt"])
        store = FsspecFileStore(f"file://{tmp_path}")
        path = store[0]
        assert pathlib.Path(path).exists()

    def test_memory_filesystem(self):
        """FsspecFileStore should work with fsspec's in-memory filesystem."""
        import fsspec

        fs = fsspec.filesystem("memory")
        fs.mkdir("/testdir")
        fs.touch("/testdir/a.vtk")
        fs.touch("/testdir/b.vtk")
        fs.touch("/testdir/c.csv")

        # Write some content so they are actual files, not empty dirs.
        for name in ["a.vtk", "b.vtk"]:
            with fs.open(f"/testdir/{name}", "wb") as f:
                f.write(b"vtk content")
        with fs.open("/testdir/c.csv", "wb") as f:
            f.write(b"csv content")

        store = FsspecFileStore("memory:///testdir", extensions=frozenset({".vtk"}))
        assert len(store) == 2

    def test_memory_filesystem_caches_to_local(self):
        """Files from memory:// should be downloaded to cache_storage."""
        import fsspec

        fs = fsspec.filesystem("memory")
        fs.mkdir("/remote")
        with fs.open("/remote/mesh.vtk", "wb") as f:
            f.write(b"fake vtk data")

        import tempfile

        cache_dir = tempfile.mkdtemp(prefix="test_cache_")
        store = FsspecFileStore("memory:///remote", cache_storage=cache_dir)
        local_path = store[0]

        assert pathlib.Path(local_path).exists()
        assert pathlib.Path(local_path).read_bytes() == b"fake vtk data"

    def test_extension_filter(self):
        """Extension filtering should work for remote stores."""
        import fsspec

        fs = fsspec.filesystem("memory")
        fs.mkdir("/mixeddir")
        for name in ["a.vtk", "b.vtu", "c.csv", "d.vtk"]:
            with fs.open(f"/mixeddir/{name}", "wb") as f:
                f.write(b"data")

        store = FsspecFileStore("memory:///mixeddir", extensions=frozenset({".vtk", ".vtu"}))
        assert len(store) == 3

    def test_sorted_order(self):
        """Remote files should be sorted alphabetically."""
        import fsspec

        fs = fsspec.filesystem("memory")
        fs.mkdir("/sortdir")
        for name in ["c.dat", "a.dat", "b.dat"]:
            with fs.open(f"/sortdir/{name}", "wb") as f:
                f.write(b"data")

        store = FsspecFileStore("memory:///sortdir")
        names = [pathlib.PurePath(store[i]).name for i in range(len(store))]
        assert names == ["a.dat", "b.dat", "c.dat"]

    def test_no_matching_files_raises(self):
        """Empty results should raise ValueError."""
        import fsspec

        fs = fsspec.filesystem("memory")
        fs.mkdir("/emptydir")

        with pytest.raises(ValueError, match="No matching files"):
            FsspecFileStore("memory:///emptydir")

    def test_index_out_of_range(self):
        """Out-of-range index should raise IndexError."""
        import fsspec

        fs = fsspec.filesystem("memory")
        fs.mkdir("/smalldir")
        with fs.open("/smalldir/only.dat", "wb") as f:
            f.write(b"data")

        store = FsspecFileStore("memory:///smalldir")
        with pytest.raises(IndexError):
            store[5]

    def test_negative_index(self):
        """Negative indexing should work."""
        import fsspec

        fs = fsspec.filesystem("memory")
        fs.mkdir("/negdir")
        for name in ["a.dat", "b.dat"]:
            with fs.open(f"/negdir/{name}", "wb") as f:
                f.write(b"data")

        store = FsspecFileStore("memory:///negdir")
        # -1 should give the last file
        path = store[-1]
        assert pathlib.PurePath(path).name == "b.dat"

    def test_repr(self):
        """Repr should contain class name and file count."""
        import fsspec

        fs = fsspec.filesystem("memory")
        fs.mkdir("/reprdir")
        with fs.open("/reprdir/f.dat", "wb") as f:
            f.write(b"data")

        store = FsspecFileStore("memory:///reprdir")
        r = repr(store)
        assert "FsspecFileStore" in r
        assert "files=1" in r

    def test_storage_options_forwarded(self, tmp_path):
        """storage_options should be passed to the fsspec filesystem."""
        _populate_dir(tmp_path, ["a.dat"])
        # file:// doesn't use storage_options, but it shouldn't error.
        store = FsspecFileStore(f"file://{tmp_path}", storage_options={"auto_mkdir": True})
        assert len(store) == 1

    def test_cache_reuse(self):
        """Accessing the same file twice should serve from cache."""
        import fsspec

        fs = fsspec.filesystem("memory")
        fs.mkdir("/cachedir")
        with fs.open("/cachedir/data.vtk", "wb") as f:
            f.write(b"cached content")

        import tempfile

        cache_dir = tempfile.mkdtemp(prefix="test_reuse_")
        store = FsspecFileStore("memory:///cachedir", cache_storage=cache_dir)

        path1 = store[0]
        path2 = store[0]
        assert path1 == path2
        assert pathlib.Path(path1).exists()
