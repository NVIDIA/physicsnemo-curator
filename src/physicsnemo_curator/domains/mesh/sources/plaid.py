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

* ``interior`` — volume mesh (default base ``Base_2_2``), coordinates +
  vertex fields as ``point_data``, **no connectivity** (point cloud only
  until connectivity data is available in the dataset).
* ``boundaries["surface"]`` — surface mesh (default base ``Base_1_2``),
  same convention.
* ``global_data`` — scalar domain-level quantities (e.g. Mach number,
  angle of attack).

Default field names match the VKI-LS59 ``ProblemDefinition``; all parameters
are configurable for other PLAID-format datasets.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.domain_mesh import DomainMesh
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


class PlaidSource(Source[DomainMesh]):
    """Read PLAID datasets from disk (native HuggingFace format) as DomainMesh objects.

    Reads the directory layout produced by ``plaid.storage.download_from_hub`` and
    constructs one :class:`~physicsnemo.mesh.domain_mesh.DomainMesh` per sample:

    * ``interior`` — volume mesh (``volume_base``), coordinates + vertex fields as
      ``point_data``, **no connectivity** (point cloud only until connectivity data
      is available).
    * ``boundaries["surface"]`` — surface mesh (``surface_base``), same convention.
    * ``global_data`` — scalar domain-level quantities (e.g. Mach, angle of attack).

    Coordinates are always ``(N, 2)`` for 2-D datasets such as VKI-LS59.

    Parameters
    ----------
    local_folder : str
        Path to the dataset root written by ``plaid.storage.download_from_hub``
        or ``plaid.storage.save_to_disk``.
    split : str
        Dataset split to load (e.g. ``"train"``, ``"test"``).  Must match a key
        returned by ``plaid.storage.init_from_disk``.
    surface_base : str
        CGNS base name for the surface mesh.  Defaults to ``"Base_1_2"``.
    volume_base : str
        CGNS base name for the volume mesh.  Defaults to ``"Base_2_2"``.
    zone : str
        Zone name shared by both bases.  Defaults to ``"Zone"``.
    surface_fields : list[str] | None
        Vertex field names to load from the surface zone.  ``None`` uses the
        LS59 defaults (``["M_iso"]``).
    volume_fields : list[str] | None
        Vertex field names to load from the volume zone.  ``None`` uses the
        LS59 defaults (``["sdf", "mach", "nut", "ro", "roe", "rou", "rov"]``).
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
        4-node quadrilaterals.  ``QUAD_4`` elements are automatically split into
        two triangles each so the resulting ``Mesh`` uses simplicial cells.
        ``None`` produces a point cloud.  Defaults to ``"QUAD_4"``.

    Examples
    --------
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
    """

    name: ClassVar[str] = "PLAID"
    description: ClassVar[str] = (
        "PLAID CFD dataset — native HuggingFace format, surface+volume DomainMesh"
    )

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
                description="CGNS base name for the surface mesh",
                type=str,
                default=_LS59_SURFACE_BASE,
            ),
            Param(
                name="volume_base",
                description="CGNS base name for the volume mesh",
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
                description="Surface vertex field names (None = LS59 defaults)",
                type=list,
                default=None,
            ),
            Param(
                name="volume_fields",
                description="Volume vertex field names (None = LS59 defaults)",
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
                description="CGNS element-group name for volume connectivity; QUAD_4 is triangulated (None = point cloud)",
                type=str,
                default="QUAD_4",
            ),
        ]

    def __init__(
        self,
        local_folder: str = "./downloaded_dataset",
        split: str = "train",
        repo_id: str | None = None,
        surface_base: str = _LS59_SURFACE_BASE,
        volume_base: str = _LS59_VOLUME_BASE,
        zone: str = _LS59_ZONE,
        surface_fields: list[str] | None = None,
        volume_fields: list[str] | None = None,
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
        surface_base : str
            CGNS base name for surface coordinates / fields.
        volume_base : str
            CGNS base name for volume coordinates / fields.
        zone : str
            Zone name within each base.
        surface_fields : list[str] | None
            Vertex fields to load from the surface zone.
        volume_fields : list[str] | None
            Vertex fields to load from the volume zone.
        global_fields : list[str] | None
            Global scalar names to include in ``global_data``.
        surface_element_type : str | None
            CGNS element-group name for the surface zone (e.g. ``"BAR_2"``).
            ``None`` leaves the surface as a point cloud.
        volume_element_type : str | None
            CGNS element-group name for the volume zone (e.g. ``"QUAD_4"``).
            ``QUAD_4`` elements are split into triangles automatically.
            ``None`` leaves the volume as a point cloud.
        """
        from plaid.storage import init_from_disk  # type: ignore[import-untyped]

        self._log = get_logger(self)

        if not pathlib.Path(local_folder).exists():
            if repo_id is not None:
                from plaid.storage import download_from_hub  # type: ignore[import-untyped]

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
        self._surface_fields: list[str] = (
            surface_fields if surface_fields is not None else list(_LS59_SURFACE_FIELDS)
        )
        self._volume_fields: list[str] = (
            volume_fields if volume_fields is not None else list(_LS59_VOLUME_FIELDS)
        )
        self._global_fields: list[str] = (
            global_fields if global_fields is not None else list(_LS59_GLOBALS)
        )
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
            One DomainMesh per call, with ``interior`` (volume),
            ``boundaries["surface"]`` (surface), and ``global_data``.

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

        surface_mesh = self._build_mesh(
            plaid_sample, time, self._surface_base, self._surface_fields, self._surface_element_type
        )
        volume_mesh = self._build_mesh(
            plaid_sample, time, self._volume_base, self._volume_fields, self._volume_element_type
        )
        global_data = self._build_global_data(plaid_sample, time)

        yield DomainMesh(
            interior=volume_mesh,
            boundaries={"surface": surface_mesh},
            global_data=global_data,
        )

    # -- Internal helpers -----------------------------------------------------

    def _get_coords(self, plaid_sample: Any, time: float, base: str) -> torch.Tensor:
        """Extract 2-D point coordinates from a PLAID sample for *base*.

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
            Float32 tensor of shape ``(N, 2)``.
        """
        x = plaid_sample.get_feature_by_path(
            f"{base}/{self._zone}/GridCoordinates/CoordinateX", time
        )
        y = plaid_sample.get_feature_by_path(
            f"{base}/{self._zone}/GridCoordinates/CoordinateY", time
        )
        if x is None or y is None:
            return torch.zeros((0, 2), dtype=torch.float32)
        return torch.from_numpy(
            np.stack(
                [np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)],
                axis=-1,
            )
        )

    def _get_cells(
        self, plaid_sample: Any, time: float, base: str, element_type: str
    ) -> torch.Tensor | None:
        """Read element connectivity for *element_type* from a PLAID sample.

        Connectivity is stored flat in PLAID; this method reshapes it to
        ``(n_cells, nodes_per_cell)``.  ``QUAD_4`` elements (4-node
        quadrilaterals) are split into two triangles each by cutting along the
        diagonal ``[0, 2]``, producing a ``(2*n_quads, 3)`` tensor.

        Parameters
        ----------
        plaid_sample : Sample
            Loaded PLAID sample.
        time : float
            Time step to query.
        base : str
            CGNS base name.
        element_type : str
            CGNS element-group name, e.g. ``"BAR_2"`` or ``"QUAD_4"``.

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

        if element_type == "QUAD_4":
            # 4-node quads → split each [A,B,C,D] into [A,B,C] and [A,C,D]
            quads = conn.reshape(-1, 4)
            tris = np.empty((len(quads) * 2, 3), dtype=np.int64)
            tris[0::2] = quads[:, [0, 1, 2]]
            tris[1::2] = quads[:, [0, 2, 3]]
            return torch.from_numpy(tris)  # tris is already a fresh writable array

        # Generic fallback: try to infer nodes-per-cell from the element name suffix
        # (e.g. "TRI_3" → 3, "TETRA_4" → 4).
        try:
            nodes_per_cell = int(element_type.split("_")[-1])
        except ValueError:
            self._log.warning(
                "Cannot infer nodes-per-cell from element type '%s' — no cells added",
                element_type,
            )
            return None
        return torch.from_numpy(conn.reshape(-1, nodes_per_cell).copy())

    def _build_mesh(
        self,
        plaid_sample: Any,
        time: float,
        base: str,
        fields: list[str],
        element_type: str | None = None,
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
        fields : list[str]
            Vertex field names to include in ``point_data``.
        element_type : str | None
            CGNS element-group name to read connectivity from (e.g.
            ``"BAR_2"``, ``"QUAD_4"``).  ``None`` produces a point cloud.

        Returns
        -------
        Mesh
            Mesh with coordinates, optional connectivity, and optional
            ``point_data``.
        """
        points = self._get_coords(plaid_sample, time, base)
        n_points = int(points.shape[0])

        cells: torch.Tensor | None = None
        if element_type is not None:
            cells = self._get_cells(plaid_sample, time, base, element_type)

        field_tensors: dict[str, torch.Tensor] = {}
        for fname in fields:
            path = f"{base}/{self._zone}/VertexFields/{fname}"
            val = plaid_sample.get_feature_by_path(path, time)
            if val is not None:
                arr = np.asarray(val, dtype=np.float32).ravel()
                if arr.shape[0] == n_points:
                    field_tensors[fname] = torch.from_numpy(arr)
                else:
                    self._log.warning(
                        "Field '%s' has %d values but mesh has %d points — skipped",
                        fname,
                        arr.shape[0],
                        n_points,
                    )

        point_data = (
            TensorDict(field_tensors, batch_size=[n_points]) if field_tensors else None
        )
        return Mesh(points=points, cells=cells, point_data=point_data)

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
