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

"""PLAID CFD dataset source for mesh pipelines.

Reads the native PLAID HuggingFace disk format produced by
``plaid.storage.download_from_hub`` and constructs one
:class:`~physicsnemo.mesh.domain_mesh.DomainMesh` per sample:

* ``interior`` — volume mesh (``volume_base``), coordinates + fields.
* ``boundaries["surface"]`` — surface mesh (``surface_base``), same
  convention.
* ``global_data`` — scalar domain-level quantities (e.g. Mach number,
  angle of attack).

Coordinates are read generically for 2-D or 3-D datasets (``CoordinateZ``
is included automatically when present). Either ``surface_base`` or
``volume_base`` may be ``None`` for datasets that only provide one of the
two (e.g. Rotor37 currently only ships a surface mesh); the missing side
is simply omitted (an empty point cloud stands in for ``interior`` when
only a surface is available, since :class:`DomainMesh` requires a non-null
interior). Per base, fields can be read from both ``VertexFields``
(-> ``point_data``) and ``CellCenterFields`` (-> ``cell_data``).

Default field names match the VKI-LS59 ``ProblemDefinition``; all parameters
are configurable for other PLAID-format datasets (e.g. Rotor37 — see the
``_ROTOR37_*`` constants below for a worked example).
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.domain_mesh import DomainMesh
from plaid.storage import download_from_hub, init_from_disk  # type: ignore[import-untyped]
from tensordict import TensorDict

from physicsnemo_curator.core.base import Param, Source
from physicsnemo_curator.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Generator

# ---------------------------------------------------------------------------
# LS59 feature-path defaults
# ---------------------------------------------------------------------------

#: Default surface base name in LS59 PLAID samples (1D topology, 2D physical).
_LS59_SURFACE_BASE = "Base_1_2"
#: Default volume base name in LS59 PLAID samples (2D topology, 2D physical).
_LS59_VOLUME_BASE = "Base_2_2"
#: Single zone name shared by both bases.
_LS59_ZONE = "Zone"
#: All scalar globals in the LS59 ProblemDefinition (inputs + outputs).
_LS59_GLOBALS = ["angle_in", "mach_out", "Pr", "Q", "Tr", "angle_out", "eth_is", "power"]
#: Surface vertex fields (inputs + outputs) for LS59.
_LS59_SURFACE_FIELDS = ["M_iso"]
#: Volume vertex fields (inputs + outputs) for LS59.
_LS59_VOLUME_FIELDS = ["sdf", "mach", "nut", "ro", "roe", "rou", "rov"]

# ---------------------------------------------------------------------------
# Rotor37 feature-path defaults (worked example — pass explicitly, these are
# not the active class defaults since LS59 remains the default dataset).
# ---------------------------------------------------------------------------

#: Rotor37 currently ships a single 3-D base (surface mesh of the blade).
_ROTOR37_SURFACE_BASE = "Base_2_3"
#: Rotor37 has no volume base yet.
_ROTOR37_VOLUME_BASE = None
#: Single zone name.
_ROTOR37_ZONE = "Zone"
#: Scalar globals in the Rotor37 ProblemDefinition.
_ROTOR37_GLOBALS = ["Compression_ratio", "Efficiency", "Massflow", "Omega", "P"]
#: Surface fields available both as VertexFields and CellCenterFields.
_ROTOR37_SURFACE_FIELDS = ["Density", "Pressure", "Temperature", "NormalsX", "NormalsY", "NormalsZ"]
#: Surface connectivity element type.
_ROTOR37_SURFACE_ELEMENT_TYPE = "QUAD_4"

# ---------------------------------------------------------------------------
# Non-simplicial element splits: local (sub-cell, node) index patterns used to
# decompose an element into simplicial cells (triangles / tetrahedra), since
# Mesh expects simplicial connectivity. Each row is one output sub-cell,
# indexing into the raw element's nodes (0-based, CGNS node order). The
# number of rows is also the CellCenterFields duplication factor: a value
# stored once per original (undivided) element must be repeated once per
# resulting sub-cell to stay aligned — see `_build_mesh`.
# ---------------------------------------------------------------------------
_CELL_SPLITS: dict[str, np.ndarray] = {
    # 4-node quad → 2 triangles, cut along the [0, 2] diagonal.
    "QUAD_4": np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
    # 8-node hexahedron → 6 tetrahedra sharing the (0, 6) space diagonal.
    # Requires standard CGNS/VTK hex node order (0-3 bottom face CCW, 4-7 top
    # face aligned above 0-3).
    "HEXA_8": np.array(
        [
            [0, 1, 2, 6],
            [0, 2, 3, 6],
            [0, 3, 7, 6],
            [0, 7, 4, 6],
            [0, 4, 5, 6],
            [0, 5, 1, 6],
        ],
        dtype=np.int64,
    ),
}


class PlaidSource(Source[DomainMesh]):
    """Read PLAID datasets from disk (native HuggingFace format) as DomainMesh objects.

    Reads the directory layout produced by ``plaid.storage.download_from_hub`` and
    constructs one :class:`~physicsnemo.mesh.domain_mesh.DomainMesh` per sample:

    * ``interior`` — volume mesh (``volume_base``), coordinates + fields.
    * ``boundaries["surface"]`` — surface mesh (``surface_base``), same convention.
    * ``global_data`` — scalar domain-level quantities (e.g. Mach, angle of attack).

    Coordinates are read generically: ``(N, 2)`` if only ``CoordinateX``/``CoordinateY``
    are present (e.g. VKI-LS59), ``(N, 3)`` if ``CoordinateZ`` is also present (e.g.
    Rotor37). Either ``surface_base`` or ``volume_base`` may be ``None`` for datasets
    that only provide one of the two — the missing side is omitted from the result
    (an empty point cloud with matching spatial dims stands in for ``interior`` when
    only a surface is available, since :class:`DomainMesh` requires a non-null
    interior). Each field name is looked up first in ``VertexFields`` (-> ``point_data``,
    validated against the point count) and separately in ``CellCenterFields`` (->
    ``cell_data``, validated against the cell count), so both associations can be read
    independently via ``*_fields`` (vertex) and ``*_cell_fields`` (cell-centered).

    Parameters
    ----------
    local_folder : str
        Path to the dataset root written by ``plaid.storage.download_from_hub``
        or ``plaid.storage.save_to_disk``.
    split : str
        Dataset split to load (e.g. ``"train"``, ``"test"``).  Must match a key
        returned by ``plaid.storage.init_from_disk``.
    surface_base : str | None
        CGNS base name for the surface mesh.  Defaults to ``"Base_1_2"``.  ``None``
        disables the surface mesh (``boundaries`` will be empty).
    volume_base : str | None
        CGNS base name for the volume mesh.  Defaults to ``"Base_2_2"``.  ``None``
        disables the volume mesh (``interior`` becomes an empty point cloud and the
        surface mesh, if present, is the only populated mesh — e.g. Rotor37, which
        currently ships a surface-only base; see ``_ROTOR37_*`` constants).
    zone : str
        Zone name shared by both bases.  Defaults to ``"Zone"``.
    surface_fields : list[str] | None
        Vertex field names to load from the surface zone's ``VertexFields``.
        ``None`` uses the LS59 defaults (``["M_iso"]``).
    volume_fields : list[str] | None
        Vertex field names to load from the volume zone's ``VertexFields``.
        ``None`` uses the LS59 defaults (``["sdf", "mach", "nut", "ro", "roe", "rou", "rov"]``).
    surface_cell_fields : list[str] | None
        Cell-centered field names to load from the surface zone's ``CellCenterFields``.
        ``None`` (default) loads none.
    volume_cell_fields : list[str] | None
        Cell-centered field names to load from the volume zone's ``CellCenterFields``.
        ``None`` (default) loads none.
    global_fields : list[str] | None
        Scalar global names to include in ``global_data``.  ``None`` uses the
        LS59 defaults.
    surface_element_type : str | None
        CGNS element-group name inside the surface zone, e.g. ``"BAR_2"`` for
        2-node line segments.  The connectivity is read from the path
        ``{surface_base}/{zone}/Elements_{surface_element_type}/ElementConnectivity``.
        ``None`` produces a point cloud (no connectivity).  Defaults to
        ``"BAR_2"``.
    volume_element_type : str | None
        CGNS element-group name inside the volume zone, e.g. ``"QUAD_4"`` for
        4-node quadrilaterals or ``"HEXA_8"`` for 8-node hexahedra.  Both are
        automatically split into simplicial cells (``QUAD_4`` -> 2 triangles,
        ``HEXA_8`` -> 6 tetrahedra; see ``_CELL_SPLITS``) since ``Mesh`` expects
        simplicial connectivity; any ``CellCenterFields`` read alongside are
        expanded to match. ``None`` produces a point cloud.  Defaults to
        ``"QUAD_4"``.

    Examples
    --------
    VKI-LS59 (defaults):

    >>> source = PlaidSource(local_folder="./downloaded_dataset")  # doctest: +SKIP
    >>> len(source)  # doctest: +SKIP
    800
    >>> dm = next(source[0])  # doctest: +SKIP
    >>> dm.interior.n_points  # volume nodes  # doctest: +SKIP
    36421
    >>> dm.boundaries["surface"].n_points  # surface nodes  # doctest: +SKIP
    244
    >>> list(dm.global_data.keys())  # doctest: +SKIP
    ['Pr', 'Q', 'Tr', 'angle_in', 'angle_out', 'eth_is', 'mach_out', 'power']
    >>> dm.boundaries["surface"].n_cells  # BAR_2 edges  # doctest: +SKIP
    243
    >>> dm.interior.n_cells  # triangles (QUAD_4 split 2-for-1)  # doctest: +SKIP
    72840

    Rotor37 (3-D, surface-only, cell-centered fields) — pass explicit overrides:

    >>> source = PlaidSource(  # doctest: +SKIP
    ...     local_folder="./downloaded_dataset",
    ...     surface_base=_ROTOR37_SURFACE_BASE,
    ...     volume_base=_ROTOR37_VOLUME_BASE,
    ...     surface_fields=_ROTOR37_SURFACE_FIELDS,
    ...     surface_cell_fields=_ROTOR37_SURFACE_FIELDS,
    ...     surface_element_type=_ROTOR37_SURFACE_ELEMENT_TYPE,
    ...     volume_element_type=None,
    ...     global_fields=_ROTOR37_GLOBALS,
    ... )
    >>> dm = next(source[0])  # doctest: +SKIP
    >>> dm.boundaries["surface"].points.shape  # (N, 3) — 3-D coords  # doctest: +SKIP
    torch.Size([29773, 3])
    >>> dm.interior.n_points  # no volume base -> empty point cloud  # doctest: +SKIP
    0
    """

    name: ClassVar[str] = "PLAID"
    description: ClassVar[str] = "PLAID CFD dataset — native HuggingFace format, surface+volume DomainMesh"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the PlaidSource.

        Returns
        -------
        list[Param]
            Parameter list for CLI configuration.
        """
        return [
            Param(
                name="local_folder",
                description="Path to the dataset root (output of download_from_hub)",
                type=str,
                default="./downloaded_dataset",
            ),
            Param(
                name="repo_id",
                description=(
                    "HuggingFace repo ID to download from if local_folder is absent "
                    "(e.g. 'PLAID-datasets/VKI-LS59'). None disables auto-download."
                ),
                type=str,
                default=None,
            ),
            Param(
                name="split",
                description="Dataset split to load (train / test)",
                type=str,
                default="train",
                choices=["train", "test"],
            ),
            Param(
                name="surface_base",
                description="CGNS base name for the surface mesh (None = no surface mesh)",
                type=str,
                default=_LS59_SURFACE_BASE,
            ),
            Param(
                name="volume_base",
                description="CGNS base name for the volume mesh (None = no volume mesh)",
                type=str,
                default=_LS59_VOLUME_BASE,
            ),
            Param(
                name="zone",
                description="Zone name shared by both bases",
                type=str,
                default=_LS59_ZONE,
            ),
            Param(
                name="surface_fields",
                description="Surface VertexFields names -> point_data (None = LS59 defaults)",
                type=list,
                default=None,
            ),
            Param(
                name="volume_fields",
                description="Volume VertexFields names -> point_data (None = LS59 defaults)",
                type=list,
                default=None,
            ),
            Param(
                name="surface_cell_fields",
                description="Surface CellCenterFields names -> cell_data (None = none loaded)",
                type=list,
                default=None,
            ),
            Param(
                name="volume_cell_fields",
                description="Volume CellCenterFields names -> cell_data (None = none loaded)",
                type=list,
                default=None,
            ),
            Param(
                name="global_fields",
                description="Scalar global field names (None = LS59 defaults)",
                type=list,
                default=None,
            ),
            Param(
                name="surface_element_type",
                description="CGNS element-group name for surface connectivity (None = point cloud)",
                type=str,
                default="BAR_2",
            ),
            Param(
                name="volume_element_type",
                description=(
                    "CGNS element-group name for volume connectivity; "
                    "QUAD_4/HEXA_8 are split into simplicial cells (None = point cloud)"
                ),
                type=str,
                default="QUAD_4",
            ),
        ]

    def __init__(
        self,
        local_folder: str = "./downloaded_dataset",
        split: str = "train",
        repo_id: str | None = None,
        surface_base: str | None = _LS59_SURFACE_BASE,
        volume_base: str | None = _LS59_VOLUME_BASE,
        zone: str = _LS59_ZONE,
        surface_fields: list[str] | None = None,
        volume_fields: list[str] | None = None,
        surface_cell_fields: list[str] | None = None,
        volume_cell_fields: list[str] | None = None,
        global_fields: list[str] | None = None,
        surface_element_type: str | None = "BAR_2",
        volume_element_type: str | None = "QUAD_4",
    ) -> None:
        """Initialise PlaidSource and load the requested split.

        If *local_folder* does not exist and *repo_id* is provided the dataset
        is downloaded automatically via ``plaid.storage.download_from_hub``
        before loading.

        Parameters
        ----------
        local_folder : str
            Path to the dataset root on disk.
        split : str
            Dataset split key (``"train"`` or ``"test"``).
        repo_id : str | None
            HuggingFace repository ID (e.g. ``"PLAID-datasets/VKI-LS59"``).
            When set and *local_folder* is absent the dataset is downloaded
            automatically.  ``None`` disables auto-download.
        surface_base : str | None
            CGNS base name for surface coordinates / fields.  ``None`` disables
            the surface mesh entirely.
        volume_base : str | None
            CGNS base name for volume coordinates / fields.  ``None`` disables
            the volume mesh entirely (``interior`` becomes an empty point cloud).
        zone : str
            Zone name within each base.
        surface_fields : list[str] | None
            ``VertexFields`` to load from the surface zone into ``point_data``.
        volume_fields : list[str] | None
            ``VertexFields`` to load from the volume zone into ``point_data``.
        surface_cell_fields : list[str] | None
            ``CellCenterFields`` to load from the surface zone into ``cell_data``.
        volume_cell_fields : list[str] | None
            ``CellCenterFields`` to load from the volume zone into ``cell_data``.
        global_fields : list[str] | None
            Global scalar names to include in ``global_data``.
        surface_element_type : str | None
            CGNS element-group name for the surface zone (e.g. ``"BAR_2"``).
            ``None`` leaves the surface as a point cloud.
        volume_element_type : str | None
            CGNS element-group name for the volume zone (e.g. ``"QUAD_4"``,
            ``"HEXA_8"``). Elements in ``_CELL_SPLITS`` are split into
            simplicial cells automatically. ``None`` leaves the volume as a
            point cloud.

        Raises
        ------
        ValueError
            If both *surface_base* and *volume_base* are ``None``.
        """
        self._log = get_logger(self)

        if surface_base is None and volume_base is None:
            msg = "At least one of `surface_base` or `volume_base` must be set."
            raise ValueError(msg)

        if not pathlib.Path(local_folder).exists():
            if repo_id is not None:
                self._log.info(
                    "Dataset not found at '%s' — downloading from HuggingFace '%s'",
                    local_folder,
                    repo_id,
                )
                download_from_hub(repo_id, local_folder)
            else:
                msg = (
                    f"Dataset folder '{local_folder}' does not exist. "
                    "Pass repo_id='<owner>/<dataset>' to download it automatically."
                )
                raise FileNotFoundError(msg)

        self._surface_base = surface_base
        self._volume_base = volume_base
        self._zone = zone
        self._surface_fields: list[str] = surface_fields if surface_fields is not None else list(_LS59_SURFACE_FIELDS)
        self._volume_fields: list[str] = volume_fields if volume_fields is not None else list(_LS59_VOLUME_FIELDS)
        self._surface_cell_fields: list[str] = list(surface_cell_fields or [])
        self._volume_cell_fields: list[str] = list(volume_cell_fields or [])
        self._global_fields: list[str] = global_fields if global_fields is not None else list(_LS59_GLOBALS)
        self._surface_element_type = surface_element_type
        self._volume_element_type = volume_element_type

        datasetdict, converterdict = init_from_disk(local_folder)
        if split not in datasetdict:
            available = list(datasetdict.keys())
            msg = f"Split '{split}' not found in {local_folder}. Available: {available}"
            raise KeyError(msg)

        self._dataset = datasetdict[split]
        self._converter = converterdict[split]
        self._log.info(
            "Loaded PlaidSource split='%s' with %d samples from %s",
            split,
            len(self._dataset),
            local_folder,
        )

    # -- Source interface -----------------------------------------------------

    def __len__(self) -> int:
        """Return the number of samples in the loaded split."""
        return len(self._dataset)

    def mesh_name(self, index: int, seq: int = 0) -> str:  # noqa: ARG002
        """Return a canonical name for sample *index*.

        Parameters
        ----------
        index : int
            Zero-based sample index.
        _seq : int
            Sequence number (always 0 — one DomainMesh per sample).

        Returns
        -------
        str
            Name like ``"sample_0"``.
        """
        return f"sample_{index}"

    def __getitem__(self, index: int) -> Generator[DomainMesh]:
        """Build a DomainMesh for sample *index*.

        Reads the PLAID sample at *index*, extracts surface and volume point
        coordinates together with their vertex fields and global scalars, and
        yields one :class:`~physicsnemo.mesh.domain_mesh.DomainMesh`.

        Parameters
        ----------
        index : int
            Zero-based index into the split dataset.  Negative indices are
            supported.

        Yields
        ------
        DomainMesh
            One DomainMesh per call, with ``interior`` (volume, or an empty
            point cloud if ``volume_base`` is ``None``), ``boundaries["surface"]``
            (surface, omitted if ``surface_base`` is ``None``), and ``global_data``.

        Raises
        ------
        IndexError
            If *index* is out of range.
        """
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            msg = f"Index {index} out of range for source with {len(self)} samples."
            raise IndexError(msg)

        plaid_sample = self._converter.to_plaid(self._dataset, index)
        time = plaid_sample.get_all_time_values()[0]
        raw_hf_row = self._dataset[index]

        surface_mesh = None
        if self._surface_base is not None:
            surface_mesh = self._build_mesh(
                plaid_sample,
                time,
                self._surface_base,
                self._surface_fields,
                self._surface_cell_fields,
                self._surface_element_type,
                raw_hf_row=raw_hf_row,
            )

        volume_mesh = None
        if self._volume_base is not None:
            volume_mesh = self._build_mesh(
                plaid_sample,
                time,
                self._volume_base,
                self._volume_fields,
                self._volume_cell_fields,
                self._volume_element_type,
                raw_hf_row=raw_hf_row,
            )
        if volume_mesh is None:
            # No volume base configured (e.g. Rotor37 currently ships a
            # surface-only base): DomainMesh requires a non-null interior, so
            # stand in with an empty point cloud matching the surface's dims.
            # __init__ guarantees surface_base is set whenever volume_base isn't.
            assert surface_mesh is not None  # noqa: S101
            n_spatial_dims = surface_mesh.points.shape[-1]
            volume_mesh = Mesh(points=torch.zeros((0, n_spatial_dims), dtype=torch.float32))

        global_data = self._build_global_data(plaid_sample, time)

        yield DomainMesh(
            interior=volume_mesh,
            boundaries={"surface": surface_mesh} if surface_mesh is not None else None,
            global_data=global_data,
        )

    # -- Internal helpers -----------------------------------------------------

    def _get_coords(self, plaid_sample: Any, time: float, base: str) -> torch.Tensor:
        """Extract point coordinates from a PLAID sample for *base*.

        Reads ``CoordinateX``/``CoordinateY`` (required) and ``CoordinateZ``
        (optional), so 2-D datasets (e.g. VKI-LS59) yield ``(N, 2)`` and 3-D
        datasets (e.g. Rotor37) yield ``(N, 3)``.

        Parameters
        ----------
        plaid_sample : Sample
            Loaded PLAID sample.
        time : float
            Time step to query.
        base : str
            CGNS base name (e.g. ``"Base_1_2"``).

        Returns
        -------
        torch.Tensor
            Float32 tensor of shape ``(N, 2)`` or ``(N, 3)``.
        """
        prefix = f"{base}/{self._zone}/GridCoordinates/Coordinate"
        x = plaid_sample.get_feature_by_path(f"{prefix}X", time)
        y = plaid_sample.get_feature_by_path(f"{prefix}Y", time)
        if x is None or y is None:
            return torch.zeros((0, 2), dtype=torch.float32)

        components = [np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)]
        z = plaid_sample.get_feature_by_path(f"{prefix}Z", time)
        if z is not None:
            components.append(np.asarray(z, dtype=np.float32))
        return torch.from_numpy(np.stack(components, axis=-1))

    def _get_cells(self, plaid_sample: Any, time: float, base: str, element_type: str) -> torch.Tensor | None:
        """Read element connectivity for *element_type* from a PLAID sample.

        Connectivity is stored flat in PLAID; this method reshapes it to
        ``(n_cells, nodes_per_cell)``.  Non-simplicial element types present in
        ``_CELL_SPLITS`` (``QUAD_4`` → 2 triangles, ``HEXA_8`` → 6 tetrahedra)
        are decomposed into simplicial cells, since :class:`Mesh` expects
        triangle/tetrahedra connectivity; each output sub-cell is grouped
        immediately after its siblings from the same original element (e.g.
        for ``QUAD_4``, cells ``2*i`` and ``2*i+1`` both come from quad *i*),
        matching the duplication order ``_build_mesh`` uses to expand
        ``CellCenterFields``.

        Parameters
        ----------
        plaid_sample : Sample
            Loaded PLAID sample.
        time : float
            Time step to query.
        base : str
            CGNS base name.
        element_type : str
            CGNS element-group name, e.g. ``"BAR_2"``, ``"QUAD_4"``, ``"HEXA_8"``.

        Returns
        -------
        torch.Tensor or None
            Long tensor of shape ``(n_cells, nodes_per_cell)``, or ``None`` if
            the connectivity path is absent in the sample.
        """
        path = f"{base}/{self._zone}/Elements_{element_type}/ElementConnectivity"
        raw = plaid_sample.get_feature_by_path(path, time)
        if raw is None:
            self._log.warning("Connectivity path '%s' not found — no cells added", path)
            return None

        conn = np.asarray(raw, dtype=np.int64).ravel()

        if element_type == "BAR_2":
            # 2-node line segments → (n_edges, 2)
            return torch.from_numpy(conn.reshape(-1, 2).copy())

        split_pattern = _CELL_SPLITS.get(element_type)
        if split_pattern is not None:
            raw_nodes_per_element = int(split_pattern.max()) + 1
            elements = conn.reshape(-1, raw_nodes_per_element)
            # Fancy indexing always allocates a fresh (writable) array, and
            # reshape keeps sub-cells from the same element consecutive.
            split_cells = elements[:, split_pattern].reshape(-1, split_pattern.shape[1])
            return torch.from_numpy(split_cells)

        # Generic fallback: try to infer nodes-per-cell from the element name suffix
        # (e.g. "TRI_3" → 3, "TETRA_4" → 4). Already-simplicial, no split needed.
        try:
            nodes_per_cell = int(element_type.split("_")[-1])
        except ValueError:
            self._log.warning(
                "Cannot infer nodes-per-cell from element type '%s' — no cells added",
                element_type,
            )
            return None
        return torch.from_numpy(conn.reshape(-1, nodes_per_cell).copy())

    def _read_field_group(
        self,
        plaid_sample: Any,
        time: float,
        base: str,
        group: str,
        fields: list[str],
        n_expected: int,
        raw_hf_row: dict | None,
        repeat: int = 1,
    ) -> TensorDict | None:
        """Read a group of scalar fields (``VertexFields`` or ``CellCenterFields``).

        Parameters
        ----------
        plaid_sample : Sample
            Loaded PLAID sample.
        time : float
            Time step to query.
        base : str
            CGNS base name.
        group : str
            CGNS field-group name, e.g. ``"VertexFields"`` or ``"CellCenterFields"``.
        fields : list[str]
            Field names to read from *group*.
        n_expected : int
            Expected array length as stored in PLAID (``n_points`` for
            ``VertexFields``; the original per-element count for
            ``CellCenterFields``, i.e. *before* any ``repeat`` expansion).
            Fields with a mismatched length are skipped.
        raw_hf_row : dict | None
            Raw HuggingFace dataset row used as a fallback when a field is
            not reachable via the CGNS tree (e.g. when the split's constants
            are missing ``_times`` metadata for that field).
        repeat : int
            Factor each value is duplicated by after reading (via
            ``repeat_interleave``). Used for ``CellCenterFields`` when the
            underlying elements were split into multiple cells (e.g. each
            ``QUAD_4`` -> 2 triangles), so a value stored once per original
            element is replicated to align with the split cells. ``1`` (the
            default) is a no-op, appropriate for ``VertexFields``.

        Returns
        -------
        TensorDict or None
            Batch-size-``[n_expected * repeat]`` TensorDict with one entry
            per found field, or ``None`` if no fields were found.
        """
        field_tensors: dict[str, torch.Tensor] = {}
        for fname in fields:
            path = f"{base}/{self._zone}/{group}/{fname}"
            val = plaid_sample.get_feature_by_path(path, time)
            if val is None and raw_hf_row is not None:
                raw = raw_hf_row.get(path)
                if raw is not None:
                    val = np.asarray(raw, dtype=np.float32)
            if val is not None:
                # np.array(..., copy=True) (not asarray) guards against handing
                # torch a read-only view into PLAID's memmapped storage.
                arr = np.array(val, dtype=np.float32, copy=True).ravel()
                if arr.shape[0] == n_expected:
                    tensor = torch.from_numpy(arr)
                    field_tensors[fname] = tensor.repeat_interleave(repeat) if repeat > 1 else tensor
                else:
                    self._log.warning(
                        "Field '%s' has %d values but expected %d — skipped",
                        fname,
                        arr.shape[0],
                        n_expected,
                    )

        if not field_tensors:
            return None
        return TensorDict(field_tensors, batch_size=[n_expected * repeat])

    def _build_mesh(
        self,
        plaid_sample: Any,
        time: float,
        base: str,
        vertex_fields: list[str],
        cell_fields: list[str],
        element_type: str | None = None,
        raw_hf_row: dict | None = None,
    ) -> Mesh:
        """Build a :class:`Mesh` for *base* from a PLAID sample.

        Parameters
        ----------
        plaid_sample : Sample
            Loaded PLAID sample.
        time : float
            Time step to query.
        base : str
            CGNS base name.
        vertex_fields : list[str]
            ``VertexFields`` names to include in ``point_data``.
        cell_fields : list[str]
            ``CellCenterFields`` names to include in ``cell_data``.
        element_type : str | None
            CGNS element-group name to read connectivity from (e.g.
            ``"BAR_2"``, ``"QUAD_4"``, ``"HEXA_8"``).  ``None`` produces a
            point cloud.
        raw_hf_row : dict | None
            Raw HuggingFace dataset row used as a fallback when a field is
            not reachable via the CGNS tree (e.g. when the split's constants
            are missing ``_times`` metadata for that field).

        Returns
        -------
        Mesh
            Mesh with coordinates, optional connectivity, optional
            ``point_data``, and optional ``cell_data``.
        """
        points = self._get_coords(plaid_sample, time, base)
        n_points = int(points.shape[0])

        cells: torch.Tensor | None = None
        cell_repeat = 1
        if element_type is not None:
            cells = self._get_cells(plaid_sample, time, base, element_type)
            if cells is not None and element_type in _CELL_SPLITS:
                # Each original element was split into `cell_repeat` simplicial
                # sub-cells (see _CELL_SPLITS / _get_cells), but CellCenterFields
                # stores one value per original element -> expand to align.
                cell_repeat = _CELL_SPLITS[element_type].shape[0]
        n_cells = 0 if cells is None else int(cells.shape[0])
        n_cell_records = n_cells // cell_repeat

        point_data = self._read_field_group(
            plaid_sample, time, base, "VertexFields", vertex_fields, n_points, raw_hf_row
        )
        cell_data = self._read_field_group(
            plaid_sample,
            time,
            base,
            "CellCenterFields",
            cell_fields,
            n_cell_records,
            raw_hf_row,
            repeat=cell_repeat,
        )
        return Mesh(points=points, cells=cells, point_data=point_data, cell_data=cell_data)

    def _build_global_data(self, plaid_sample: Any, time: float) -> TensorDict | None:
        """Build a scalar :class:`TensorDict` with global quantities.

        Parameters
        ----------
        plaid_sample : Sample
            Loaded PLAID sample.
        time : float
            Time step to query.

        Returns
        -------
        TensorDict or None
            Batch-size-``[]`` TensorDict with one entry per found global, or
            ``None`` if no globals were found.
        """
        scalars: dict[str, torch.Tensor] = {}
        for gname in self._global_fields:
            val = plaid_sample.get_feature_by_path(f"Global/{gname}", time)
            if val is not None:
                scalar = float(np.asarray(val).flat[0])
                scalars[gname] = torch.tensor(scalar, dtype=torch.float32)

        if not scalars:
            return None
        return TensorDict(scalars, batch_size=[])
