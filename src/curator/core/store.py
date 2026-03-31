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

"""File-discovery and access abstractions for pipeline sources.

This module provides the :class:`FileStore` protocol and two concrete
implementations:

* :class:`LocalFileStore` — discovers and serves files from a local
  directory.
* :class:`FsspecFileStore` — discovers and serves files from any
  ``fsspec``-compatible URL (S3, HuggingFace Hub, HTTPS, …),
  transparently caching them to a local directory.

Sources (e.g. :class:`~curator.mesh.sources.vtk.VTKSource`) accept a
:class:`FileStore` via dependency injection so that file-access logic is
decoupled from file-reading logic.
"""

from __future__ import annotations

import pathlib
import tempfile
from typing import Protocol, runtime_checkable


@runtime_checkable
class FileStore(Protocol):
    """Protocol for objects that map integer indices to local file paths.

    A :class:`FileStore` is a **sized**, **indexable** collection: it
    knows how many files it manages (``__len__``) and can return a local
    filesystem path for any valid index (``__getitem__``).

    Concrete implementations handle discovery, filtering, sorting, and —
    for remote stores — transparent caching / download.
    """

    def __len__(self) -> int:
        """Return the number of files in the store."""
        ...

    def __getitem__(self, index: int) -> str:
        """Return a local filesystem path for the file at *index*.

        Parameters
        ----------
        index : int
            Zero-based index into the sorted file list.

        Returns
        -------
        str
            Absolute or relative path on the local filesystem that can
            be opened directly (e.g. with :func:`pyvista.read`).

        Raises
        ------
        IndexError
            If *index* is out of range.
        """
        ...


# ---------------------------------------------------------------------------
# LocalFileStore
# ---------------------------------------------------------------------------


class LocalFileStore:
    """Discover and serve files from a local directory.

    Files are discovered once at construction time using
    :func:`pathlib.Path.glob` and optionally filtered by extension.

    Parameters
    ----------
    path : str
        Path to a directory or a single file.
    pattern : str
        Glob pattern applied when *path* is a directory.  Defaults to
        ``"*"`` (all files).
    extensions : frozenset[str] | None
        If given, only files whose lowercased suffix is in this set are
        kept.  Pass ``None`` to keep all files.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If no matching files are found.

    Examples
    --------
    >>> store = LocalFileStore("./data/", extensions=frozenset({".vtk", ".vtu"}))
    >>> len(store)
    42
    >>> store[0]
    '/absolute/path/to/data/mesh_0000.vtu'
    """

    def __init__(
        self,
        path: str,
        pattern: str = "*",
        extensions: frozenset[str] | None = None,
    ) -> None:
        self._root = pathlib.Path(path)
        self._pattern = pattern
        self._extensions = extensions
        self._files = self._discover()

    def __len__(self) -> int:
        """Return the number of discovered files."""
        return len(self._files)

    def __getitem__(self, index: int) -> str:
        """Return the local path for the file at *index*.

        Parameters
        ----------
        index : int
            Zero-based file index.

        Returns
        -------
        str
            Absolute path string.
        """
        return str(self._files[index])

    def __repr__(self) -> str:
        """Return a string representation of the store."""
        return f"LocalFileStore(root={self._root!r}, files={len(self._files)})"

    # -- internal ---------------------------------------------------------

    def _discover(self) -> list[pathlib.Path]:
        """Scan the root path and return sorted, filtered file paths.

        Returns
        -------
        list[pathlib.Path]
            Sorted list of matching file paths.

        Raises
        ------
        FileNotFoundError
            If *root* does not exist.
        ValueError
            If *root* is a file with a disallowed extension or no files
            match.
        """
        if self._root.is_file():
            if self._extensions is not None and self._root.suffix.lower() not in self._extensions:
                msg = f"File {self._root} does not match allowed extensions {self._extensions}."
                raise ValueError(msg)
            return [self._root]

        if not self._root.is_dir():
            msg = f"Path {self._root} does not exist."
            raise FileNotFoundError(msg)

        files = sorted(
            p
            for p in self._root.glob(self._pattern)
            if p.is_file() and (self._extensions is None or p.suffix.lower() in self._extensions)
        )

        if not files:
            msg = f"No matching files in {self._root} with pattern {self._pattern!r}."
            raise ValueError(msg)

        return files


# ---------------------------------------------------------------------------
# FsspecFileStore
# ---------------------------------------------------------------------------


class FsspecFileStore:
    """Discover and serve files from any ``fsspec``-compatible URL.

    Remote files are transparently cached to a local directory using
    ``fsspec``'s ``simplecache`` protocol so that each file is downloaded
    at most once.

    Parameters
    ----------
    url : str
        An ``fsspec``-compatible URL pointing to a directory.
        Examples: ``"s3://bucket/prefix/"``,
        ``"hf://datasets/org/repo/path/"``,
        ``"https://example.com/data/"``.
    pattern : str
        Glob pattern appended to *url* for file discovery.
        Defaults to ``"**"`` (all files, recursive).
    extensions : frozenset[str] | None
        If given, only files whose lowercased suffix is in this set are
        kept.
    storage_options : dict[str, object] | None
        Extra keyword arguments forwarded to the ``fsspec`` filesystem
        constructor (e.g. ``{"anon": True}`` for public S3 buckets, or
        ``{"token": "hf_..."}`` for HuggingFace).
    cache_storage : str | None
        Local directory for caching downloaded files.  If ``None``
        (default), a temporary directory is created automatically.

    Raises
    ------
    ValueError
        If no matching files are discovered at the URL.

    Examples
    --------
    >>> store = FsspecFileStore(
    ...     "hf://datasets/neashton/drivaerml/run_1/slices",
    ...     extensions=frozenset({".vtk", ".vtp", ".vtu"}),
    ... )
    >>> len(store)
    10
    >>> store[0]
    '/tmp/.../datasets/neashton/drivaerml/run_1/slices/slice_0.vtp'
    """

    def __init__(
        self,
        url: str,
        pattern: str = "**",
        extensions: frozenset[str] | None = None,
        storage_options: dict[str, object] | None = None,
        cache_storage: str | None = None,
    ) -> None:
        import fsspec

        self._url = url.rstrip("/")
        self._pattern = pattern
        self._extensions = extensions
        self._storage_options = storage_options or {}
        self._cache_storage = cache_storage or tempfile.mkdtemp(prefix="curator_cache_")

        # Resolve filesystem and root path from the URL.
        self._fs, self._root_path = fsspec.core.url_to_fs(self._url, **self._storage_options)
        self._protocol = self._fs.protocol if isinstance(self._fs.protocol, str) else self._fs.protocol[0]

        self._remote_files = self._discover()

    def __len__(self) -> int:
        """Return the number of discovered files."""
        return len(self._remote_files)

    def __getitem__(self, index: int) -> str:
        """Return a local cached path for the file at *index*.

        The file is downloaded on first access and served from cache on
        subsequent calls.

        Parameters
        ----------
        index : int
            Zero-based file index.

        Returns
        -------
        str
            Local filesystem path to the (cached) file.
        """
        if index < -len(self._remote_files) or index >= len(self._remote_files):
            msg = f"Index {index} out of range for store with {len(self._remote_files)} files."
            raise IndexError(msg)

        remote_path = self._remote_files[index]
        return self._ensure_local(remote_path)

    def __repr__(self) -> str:
        """Return a string representation of the store."""
        return f"FsspecFileStore(url={self._url!r}, files={len(self._remote_files)})"

    # -- internal ---------------------------------------------------------

    def _discover(self) -> list[str]:
        """List remote files matching the pattern and extensions.

        Returns
        -------
        list[str]
            Sorted list of remote file paths (relative to filesystem
            root, as returned by ``fs.glob``).

        Raises
        ------
        ValueError
            If no matching files are found.
        """
        glob_expr = f"{self._root_path}/{self._pattern}"
        all_files = self._fs.glob(glob_expr)

        # Filter by extension.
        if self._extensions is not None:
            all_files = [f for f in all_files if pathlib.PurePosixPath(f).suffix.lower() in self._extensions]

        # Remove directories (some fs.glob implementations include them).
        files = sorted(f for f in all_files if not self._fs.isdir(f))

        if not files:
            msg = f"No matching files at {self._url} with pattern {self._pattern!r}."
            raise ValueError(msg)

        return files

    def _ensure_local(self, remote_path: str) -> str:
        """Download *remote_path* to the cache (if not already present).

        Parameters
        ----------
        remote_path : str
            Path as returned by the fsspec filesystem (no protocol
            prefix).

        Returns
        -------
        str
            Local filesystem path.
        """
        # For local filesystems, just return the path directly.
        if self._protocol in ("file", ""):
            return remote_path

        # Build a deterministic local cache path.
        local_path = pathlib.Path(self._cache_storage) / remote_path.lstrip("/")
        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self._fs.get(remote_path, str(local_path))

        return str(local_path)
