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

"""VTK file source for mesh pipelines.

Reads VTK-format files (``.vtk``, ``.vtp``, ``.vtu``, ``.vts``, ``.vtm``)
via a :class:`~curator.core.store.FileStore` and converts each to a
:class:`physicsnemo.mesh.Mesh` using :func:`physicsnemo.mesh.io.from_pyvista`.

File discovery and access is delegated to the injected
:class:`~curator.core.store.FileStore`, allowing transparent support for
local directories, S3, HuggingFace Hub, HTTPS, and any other
``fsspec``-compatible backend.

The conversion supports multiple manifold dimensions (point clouds, lines,
surfaces, volumes) and two point-source modes (vertices or cell centroids).
See :func:`physicsnemo.mesh.io.from_pyvista` for full details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

import pyvista as pv
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.io import from_pyvista

from physicsnemo_curator.core.base import Param, Source
from physicsnemo_curator.core.store import FileStore, FsspecFileStore, LocalFileStore

if TYPE_CHECKING:
    from collections.abc import Generator

#: File extensions recognised as VTK formats.
_VTK_EXTENSIONS: frozenset[str] = frozenset({".vtk", ".vtp", ".vtu", ".vts", ".vtm"})

#: Valid backend options for VTK reading.
Backend = Literal["pyvista", "rust"]


class VTKSource(Source[Mesh]):
    """Read VTK files and yield :class:`~physicsnemo.mesh.Mesh` objects.

    File discovery and caching are handled by the injected
    :class:`~curator.core.store.FileStore`.  Use the convenience
    constructors :meth:`from_path` and :meth:`from_url` for the common
    cases of local directories and remote ``fsspec`` URLs.

    Parameters
    ----------
    store : FileStore
        A :class:`~curator.core.store.FileStore` that maps integer
        indices to local file paths.
    manifold_dim : int or {"auto"}
        Target manifold dimension passed to ``from_pyvista``:

        - ``"auto"`` (default): detect from cell types.
        - ``0``: point cloud (vertices only, no cells).
        - ``1``: line mesh (edge cells).
        - ``2``: surface mesh (triangulated).
        - ``3``: volume mesh (tetrahedralized).
    point_source : {"vertices", "cell_centroids"}
        Controls what becomes the Mesh points:

        - ``"vertices"`` (default): mesh vertices become points,
          ``point_data`` is preserved.
        - ``"cell_centroids"``: cell centroids become points,
          ``cell_data`` is mapped to ``point_data``.  Only
          ``manifold_dim`` 0 and 1 are valid in this mode.
    warn_on_lost_data : bool
        If *True* (default), emit a warning when the conversion discards
        non-empty data arrays (e.g. cell data lost during dimension
        reduction).
    backend : {"pyvista", "rust"}
        VTK reading backend:

        - ``"pyvista"`` (default): use PyVista for full-featured reading.
        - ``"rust"``: use the native Rust backend for faster reading.
          Note: The Rust backend only supports ASCII VTU/VTP files and
          does not support ``manifold_dim`` or ``point_source`` options.

    Examples
    --------
    Local directory:

    >>> source = VTKSource.from_path("./cfd_results/")
    >>> len(source)
    42
    >>> mesh = next(source[0])

    Remote HuggingFace dataset:

    >>> source = VTKSource.from_url(
    ...     "hf://datasets/neashton/drivaerml/run_1/slices"
    ... )

    Custom store (dependency injection):

    >>> store = MyCustomFileStore(...)
    >>> source = VTKSource(store=store, manifold_dim=2)

    Using the fast Rust backend:

    >>> source = VTKSource.from_path("./cfd_results/", backend="rust")

    Note
    ----
    - VTK format: `VTK File Formats <https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html>`_
    - PyVista: `PyVista documentation <https://docs.pyvista.org/>`_
    """

    name: ClassVar[str] = "VTK Reader"
    description: ClassVar[str] = "Read VTK files (.vtk, .vtp, .vtu, .vts, .vtm) and convert to physicsnemo Mesh"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the VTK source.

        Returns
        -------
        list[Param]
            Parameter list including file path, glob pattern, and
            ``from_pyvista`` conversion options.
        """
        return [
            Param(name="input_path", description="Path to VTK file or directory (local)", type=str),
            Param(name="url", description="fsspec URL (s3://, hf://, https://)", type=str, default=""),
            Param(name="file_pattern", description="Glob pattern for filtering files", type=str, default="*"),
            Param(
                name="manifold_dim",
                description="Target manifold dimension (auto, 0, 1, 2, 3)",
                type=str,
                default="auto",
                choices=["auto", "0", "1", "2", "3"],
            ),
            Param(
                name="point_source",
                description="Point source mode: vertices or cell_centroids",
                type=str,
                default="vertices",
                choices=["vertices", "cell_centroids"],
            ),
            Param(
                name="warn_on_lost_data",
                description="Warn when data arrays are discarded during conversion",
                type=bool,
                default=True,
            ),
            Param(
                name="backend",
                description="VTK reading backend: pyvista (default) or rust (faster)",
                type=str,
                default="pyvista",
                choices=["pyvista", "rust"],
            ),
        ]

    # -- Constructors --------------------------------------------------------

    def __init__(
        self,
        store: FileStore,
        manifold_dim: int | Literal["auto"] = "auto",
        point_source: Literal["vertices", "cell_centroids"] = "vertices",
        warn_on_lost_data: bool = True,
        backend: Backend = "pyvista",
    ) -> None:
        self._store = store
        self._manifold_dim = manifold_dim
        self._point_source = point_source
        self._warn_on_lost_data = warn_on_lost_data
        self._backend: Backend = backend

    @classmethod
    def from_path(
        cls,
        input_path: str,
        file_pattern: str = "*",
        *,
        manifold_dim: int | Literal["auto"] = "auto",
        point_source: Literal["vertices", "cell_centroids"] = "vertices",
        warn_on_lost_data: bool = True,
        backend: Backend = "pyvista",
    ) -> VTKSource:
        """Create a :class:`VTKSource` from a local directory or file.

        Parameters
        ----------
        input_path : str
            Path to a directory containing VTK files, or a single file.
        file_pattern : str
            Glob pattern for filtering files in a directory.
        manifold_dim : int or {"auto"}
            Target manifold dimension.
        point_source : {"vertices", "cell_centroids"}
            Point source mode.
        warn_on_lost_data : bool
            Warn when data arrays are discarded.
        backend : {"pyvista", "rust"}
            VTK reading backend.

        Returns
        -------
        VTKSource
            Configured source backed by a :class:`LocalFileStore`.

        Examples
        --------
        >>> source = VTKSource.from_path("./cfd_results/")
        >>> source = VTKSource.from_path("./data/", file_pattern="timestep_*")
        """
        store = LocalFileStore(input_path, pattern=file_pattern, extensions=_VTK_EXTENSIONS)
        return cls(
            store=store,
            manifold_dim=manifold_dim,
            point_source=point_source,
            warn_on_lost_data=warn_on_lost_data,
            backend=backend,
        )

    @classmethod
    def from_url(
        cls,
        url: str,
        file_pattern: str = "**",
        *,
        storage_options: dict[str, object] | None = None,
        cache_storage: str | None = None,
        manifold_dim: int | Literal["auto"] = "auto",
        point_source: Literal["vertices", "cell_centroids"] = "vertices",
        warn_on_lost_data: bool = True,
        backend: Backend = "pyvista",
    ) -> VTKSource:
        """Create a :class:`VTKSource` from an ``fsspec``-compatible URL.

        Remote files are transparently cached to a local directory.

        Parameters
        ----------
        url : str
            An ``fsspec`` URL such as ``"hf://datasets/org/repo/path"``,
            ``"s3://bucket/prefix/"``, or ``"https://example.com/data/"``.
        file_pattern : str
            Glob pattern for file discovery.  Defaults to ``"**"``
            (recursive).
        storage_options : dict[str, object] | None
            Extra keyword arguments for the ``fsspec`` filesystem
            (e.g. ``{"anon": True}``).
        cache_storage : str | None
            Local directory for cached downloads.  If ``None``, a
            temporary directory is used.
        manifold_dim : int or {"auto"}
            Target manifold dimension.
        point_source : {"vertices", "cell_centroids"}
            Point source mode.
        warn_on_lost_data : bool
            Warn when data arrays are discarded.
        backend : {"pyvista", "rust"}
            VTK reading backend.

        Returns
        -------
        VTKSource
            Configured source backed by a :class:`FsspecFileStore`.

        Examples
        --------
        >>> source = VTKSource.from_url(
        ...     "hf://datasets/neashton/drivaerml/run_1/slices"
        ... )
        >>> source = VTKSource.from_url(
        ...     "s3://my-bucket/data/", storage_options={"anon": True}
        ... )
        """
        store = FsspecFileStore(
            url=url,
            pattern=file_pattern,
            extensions=_VTK_EXTENSIONS,
            storage_options=storage_options,
            cache_storage=cache_storage,
        )
        return cls(
            store=store,
            manifold_dim=manifold_dim,
            point_source=point_source,
            warn_on_lost_data=warn_on_lost_data,
            backend=backend,
        )

    # -- Source interface -----------------------------------------------------

    def __len__(self) -> int:
        """Return the number of discovered VTK files."""
        return len(self._store)

    def __getitem__(self, index: int) -> Generator[Mesh]:
        """Read the *index*-th VTK file and yield a Mesh.

        The file is loaded with the configured backend and converted to
        a :class:`~physicsnemo.mesh.Mesh`. When using the PyVista backend,
        the mesh is converted via :func:`physicsnemo.mesh.io.from_pyvista`
        using the conversion parameters supplied at construction time.

        Parameters
        ----------
        index : int
            Zero-based file index.

        Yields
        ------
        Mesh
            The converted physicsnemo Mesh.
        """
        path = self._store[index]
        mesh = self._read_with_rust(path) if self._backend == "rust" else self._read_with_pyvista(path)
        yield mesh

    def _read_with_pyvista(self, path: str) -> Mesh:
        """Read VTK file using PyVista backend.

        Parameters
        ----------
        path : str
            Path to the VTK file.

        Returns
        -------
        Mesh
            Converted physicsnemo Mesh.
        """
        pv_mesh = pv.read(path)
        return from_pyvista(
            pv_mesh,
            manifold_dim=self._manifold_dim,
            point_source=self._point_source,
            warn_on_lost_data=self._warn_on_lost_data,
        )

    def _read_with_rust(self, path: str) -> Mesh:
        """Read VTK file using Rust backend.

        The Rust backend returns raw mesh data without the from_pyvista
        conversion. Currently only supports ASCII VTU/VTP files and does
        not support manifold_dim or point_source options.

        Parameters
        ----------
        path : str
            Path to the VTK file.

        Returns
        -------
        Mesh
            Raw mesh with points and point_data from the VTK file.
        """
        import torch
        from tensordict import TensorDict

        from physicsnemo_curator._lib import vtk

        rust_mesh = vtk.read_vtk(path)

        # Convert Rust mesh to physicsnemo Mesh
        points = torch.from_numpy(rust_mesh.points().reshape(-1, 3))

        # Build point_data TensorDict from Rust arrays
        point_data_dict = {}
        for name, (data, num_components) in rust_mesh.point_data().items():
            arr = torch.from_numpy(data)
            if num_components > 1:
                arr = arr.reshape(-1, num_components)
            point_data_dict[name] = arr

        point_data = TensorDict(point_data_dict, batch_size=[rust_mesh.n_points]) if point_data_dict else None

        return Mesh(
            points=points,
            point_data=point_data,
            # Note: cells/faces not converted - Rust backend is for raw data access
        )
