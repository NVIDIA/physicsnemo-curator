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

"""Tests for PlaidSource (PLAID HuggingFace disk-format reader)."""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.requires("mesh")

# ---------------------------------------------------------------------------
# Constants that mirror VKI-LS59 defaults
# ---------------------------------------------------------------------------

_N_SURFACE = 8  # boundary nodes
_N_VOLUME = 20  # interior nodes
_N_EDGES = 7  # BAR_2 edges
_N_QUADS = 10  # QUAD_4 quads  →  20 triangles after splitting


# ---------------------------------------------------------------------------
# Helpers: build a fake plaid Sample
# ---------------------------------------------------------------------------


def _make_plaid_sample(
    *,
    time: float = 0.0,
    surface_base: str = "Base_1_2",
    volume_base: str = "Base_2_2",
    zone: str = "Zone",
    add_surface_conn: bool = True,
    add_volume_conn: bool = True,
    add_globals: bool = True,
    add_surface_z: bool = False,
    add_surface_cell_fields: bool = False,
    add_volume_cell_fields: bool = False,
) -> MagicMock:
    """Return a mock plaid.Sample that mimics the VKI-LS59 feature layout."""
    rng = np.random.default_rng(42)

    # Flat connectivity arrays (1-based per CGNS convention; PlaidSource uses
    # them 0-based internally, but the source itself does not offset — the raw
    # values are used directly as array indices, so we keep them 0-based here
    # to avoid confusion in the tests).
    bar2_conn = rng.integers(0, _N_SURFACE, size=_N_EDGES * 2, dtype=np.int64)
    quad4_conn = rng.integers(0, _N_VOLUME, size=_N_QUADS * 4, dtype=np.int64)

    def _get_feature(path: str, t: float | None = None):  # noqa: ARG001
        mapping: dict[str, np.ndarray | None] = {
            # Surface coords
            f"{surface_base}/{zone}/GridCoordinates/CoordinateX": rng.random(_N_SURFACE).astype(np.float32),
            f"{surface_base}/{zone}/GridCoordinates/CoordinateY": rng.random(_N_SURFACE).astype(np.float32),
            # Volume coords
            f"{volume_base}/{zone}/GridCoordinates/CoordinateX": rng.random(_N_VOLUME).astype(np.float32),
            f"{volume_base}/{zone}/GridCoordinates/CoordinateY": rng.random(_N_VOLUME).astype(np.float32),
            # Surface connectivity
            f"{surface_base}/{zone}/Elements_BAR_2/ElementConnectivity": bar2_conn if add_surface_conn else None,
            # Volume connectivity
            f"{volume_base}/{zone}/Elements_QUAD_4/ElementConnectivity": quad4_conn if add_volume_conn else None,
            # Surface fields
            f"{surface_base}/{zone}/VertexFields/M_iso": rng.random(_N_SURFACE).astype(np.float32),
            # Volume fields
            f"{volume_base}/{zone}/VertexFields/sdf": rng.random(_N_VOLUME).astype(np.float32),
            f"{volume_base}/{zone}/VertexFields/mach": rng.random(_N_VOLUME).astype(np.float32),
            f"{volume_base}/{zone}/VertexFields/nut": rng.random(_N_VOLUME).astype(np.float32),
            f"{volume_base}/{zone}/VertexFields/ro": rng.random(_N_VOLUME).astype(np.float32),
            f"{volume_base}/{zone}/VertexFields/roe": rng.random(_N_VOLUME).astype(np.float32),
            f"{volume_base}/{zone}/VertexFields/rou": rng.random(_N_VOLUME).astype(np.float32),
            f"{volume_base}/{zone}/VertexFields/rov": rng.random(_N_VOLUME).astype(np.float32),
            # Globals
            "Global/angle_in": np.array(0.1, dtype=np.float32) if add_globals else None,
            "Global/mach_out": np.array(0.7, dtype=np.float32) if add_globals else None,
            "Global/Pr": np.array(0.71, dtype=np.float32) if add_globals else None,
            "Global/Q": np.array(1.0, dtype=np.float32) if add_globals else None,
            "Global/Tr": np.array(300.0, dtype=np.float32) if add_globals else None,
            "Global/angle_out": np.array(0.05, dtype=np.float32) if add_globals else None,
            "Global/eth_is": np.array(0.9, dtype=np.float32) if add_globals else None,
            "Global/power": np.array(5e6, dtype=np.float32) if add_globals else None,
        }
        if add_surface_z:
            mapping[f"{surface_base}/{zone}/GridCoordinates/CoordinateZ"] = rng.random(_N_SURFACE).astype(np.float32)
        if add_surface_cell_fields:
            # Rotor37-like: same field also exposed as a CellCenterFields value,
            # sized to the surface's BAR_2 edge count (_N_EDGES).
            mapping[f"{surface_base}/{zone}/CellCenterFields/Pressure"] = rng.random(_N_EDGES).astype(np.float32)
        if add_volume_cell_fields:
            # Rotor37-like: one value per original QUAD_4 element (_N_QUADS),
            # *not* per post-triangulation cell (_N_QUADS * 2).
            mapping[f"{volume_base}/{zone}/CellCenterFields/Density"] = rng.random(_N_QUADS).astype(np.float32)
        return mapping.get(path)

    sample = MagicMock()
    sample.get_all_time_values.return_value = [time]
    sample.get_feature_by_path.side_effect = _get_feature
    return sample


def _make_storage_mocks(n_samples: int = 3, **sample_kwargs):
    """Return (dataset_mock, converter_mock) with *n_samples* fake entries."""
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=n_samples)

    converter = MagicMock()
    converter.to_plaid.side_effect = lambda ds, idx: _make_plaid_sample(**sample_kwargs)  # noqa: ARG005

    return dataset, converter


def _patch_init_from_disk(tmp_path: pathlib.Path, n_samples: int = 3, **sample_kwargs):
    """Context-manager patch that fakes plaid.storage.init_from_disk."""
    dataset, converter = _make_storage_mocks(n_samples, **sample_kwargs)

    def _fake_init(local_folder):  # noqa: ARG001
        return {"train": dataset, "test": dataset}, {"train": converter, "test": converter}

    tmp_path.mkdir(parents=True, exist_ok=True)
    return patch(
        "physicsnemo_curator.domains.mesh.sources.plaid.init_from_disk",
        side_effect=_fake_init,
    )


# ---------------------------------------------------------------------------
# Unit tests — no filesystem access
# ---------------------------------------------------------------------------


class TestPlaidSourceUnit:
    """Metadata / parameter introspection tests (no I/O)."""

    def test_params_list(self) -> None:
        """params() returns descriptors covering the expected parameter names."""
        from physicsnemo_curator.domains.mesh.sources.plaid import PlaidSource

        params = PlaidSource.params()
        assert len(params) > 0
        names = [p.name for p in params]
        assert "local_folder" in names
        assert "split" in names
        assert "repo_id" in names
        assert "surface_base" in names
        assert "volume_base" in names

    def test_name_and_description(self) -> None:
        """ClassVar name and description are non-empty strings."""
        from physicsnemo_curator.domains.mesh.sources.plaid import PlaidSource

        assert isinstance(PlaidSource.name, str) and PlaidSource.name
        assert isinstance(PlaidSource.description, str) and PlaidSource.description


# ---------------------------------------------------------------------------
# Integration tests — mocked PLAID storage backend
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPlaidSourceLocal:
    """Integration tests against a mocked PLAID storage backend."""

    # -- construction & length ------------------------------------------------

    def test_len(self, tmp_path: pathlib.Path) -> None:
        """Source length equals the number of samples in the split."""
        with _patch_init_from_disk(tmp_path, n_samples=5):
            from physicsnemo_curator.domains.mesh.sources.plaid import PlaidSource

            src = PlaidSource(local_folder=str(tmp_path))
            assert len(src) == 5

    def test_invalid_split_raises(self, tmp_path: pathlib.Path) -> None:
        """KeyError is raised when the requested split is not in the dataset."""
        with _patch_init_from_disk(tmp_path):
            from physicsnemo_curator.domains.mesh.sources.plaid import PlaidSource

            with pytest.raises(KeyError, match="nosuchsplit"):
                PlaidSource(local_folder=str(tmp_path), split="nosuchsplit")

    def test_missing_folder_no_repo_id_raises(self, tmp_path: pathlib.Path) -> None:
        """FileNotFoundError when folder is absent and repo_id is None."""
        from physicsnemo_curator.domains.mesh.sources.plaid import PlaidSource

        missing = str(tmp_path / "does_not_exist")
        with pytest.raises(FileNotFoundError, match="repo_id"):
            PlaidSource(local_folder=missing)

    def test_auto_download_called_when_missing(self, tmp_path: pathlib.Path) -> None:
        """download_from_hub is invoked when folder is absent and repo_id set."""
        missing = tmp_path / "new_dataset"

        def _fake_download(repo_id, local_dir, **kwargs):  # noqa: ARG001
            pathlib.Path(local_dir).mkdir(parents=True, exist_ok=True)

        dataset, converter = _make_storage_mocks()

        def _fake_init(local_folder):  # noqa: ARG001
            return {"train": dataset}, {"train": converter}

        with (
            patch(
                "physicsnemo_curator.domains.mesh.sources.plaid.download_from_hub",
                side_effect=_fake_download,
            ) as mock_dl,
            patch(
                "physicsnemo_curator.domains.mesh.sources.plaid.init_from_disk",
                side_effect=_fake_init,
            ),
        ):
            from physicsnemo_curator.domains.mesh.sources.plaid import PlaidSource

            PlaidSource(local_folder=str(missing), repo_id="PLAID-datasets/VKI-LS59")
            mock_dl.assert_called_once_with("PLAID-datasets/VKI-LS59", str(missing))

    # -- DomainMesh structure -------------------------------------------------

    def test_getitem_yields_domain_mesh(self, tmp_path: pathlib.Path) -> None:
        """__getitem__ yields one DomainMesh per sample."""
        from physicsnemo.mesh.domain_mesh import DomainMesh

        with _patch_init_from_disk(tmp_path):
            from physicsnemo_curator.domains.mesh.sources.plaid import PlaidSource

            src = PlaidSource(local_folder=str(tmp_path))
            items = list(src[0])
            assert len(items) == 1
            assert isinstance(items[0], DomainMesh)

    def test_interior_shape(self, tmp_path: pathlib.Path) -> None:
        """Interior mesh has the expected number of points."""
        with _patch_init_from_disk(tmp_path):
            from physicsnemo_curator.domains.mesh.sources.plaid import PlaidSource

            src = PlaidSource(local_folder=str(tmp_path))
            dm = next(src[0])
            assert dm.interior.n_points == _N_VOLUME

    def test_surface_shape(self, tmp_path: pathlib.Path) -> None:
        """Surface boundary mesh has the expected number of points."""
        with _patch_init_from_disk(tmp_path):
            from physicsnemo_curator.domains.mesh.sources.plaid import PlaidSource

            src = PlaidSource(local_folder=str(tmp_path))
            dm = next(src[0])
            assert "surface" in dm.boundaries
            assert dm.boundaries["surface"].n_points == _N_SURFACE

    def test_coordinates_are_2d(self, tmp_path: pathlib.Path) -> None:
        """Coordinate tensors have shape (N, 2) — 2-D dataset."""
        with _patch_init_from_disk(tmp_path):
            from physicsnemo_curator.domains.mesh.sources.plaid import PlaidSource

            src = PlaidSource(local_folder=str(tmp_path))
            dm = next(src[0])
            assert dm.interior.points.shape == (_N_VOLUME, 2), "Volume coords must be (N, 2)"
            assert dm.boundaries["surface"].points.shape == (_N_SURFACE, 2), "Surface coords must be (N, 2)"

    # -- Connectivity ----------------------------------------------------------

    def test_surface_connectivity_bar2(self, tmp_path: pathlib.Path) -> None:
        """Surface mesh has BAR_2 edge cells with shape (n_edges, 2)."""
        with _patch_init_from_disk(tmp_path):
            from physicsnemo_curator.domains.mesh.sources.plaid import PlaidSource

            src = PlaidSource(local_folder=str(tmp_path))
            dm = next(src[0])
            surf = dm.boundaries["surface"]
            assert surf.cells is not None
            assert surf.cells.shape == (_N_EDGES, 2)
            assert surf.cells.dtype == torch.int64

    def test_volume_connectivity_quad4_triangulated(self, tmp_path: pathlib.Path) -> None:
        """Volume mesh splits QUAD_4 elements into 2× triangles."""
        with _patch_init_from_disk(tmp_path):
            from physicsnemo_curator.domains.mesh.sources.plaid import PlaidSource

            src = PlaidSource(local_folder=str(tmp_path))
            dm = next(src[0])
            assert dm.interior.cells is not None
            # Each QUAD_4 → 2 triangles
            assert dm.interior.cells.shape == (_N_QUADS * 2, 3)
            assert dm.interior.cells.dtype == torch.int64

    def test_point_cloud_when_element_type_none(self, tmp_path: pathlib.Path) -> None:
        """Passing element_type=None produces a point cloud (no cells)."""
        with _patch_init_from_disk(tmp_path):
            from physicsnemo_curator.domains.mesh.sources.plaid import PlaidSource

            src = PlaidSource(
                local_folder=str(tmp_path),
                surface_element_type=None,
                volume_element_type=None,
            )
            dm = next(src[0])
            assert dm.interior.cells is None
            assert dm.boundaries["surface"].cells is None

    # -- Fields ---------------------------------------------------------------

    def test_volume_point_data_keys(self, tmp_path: pathlib.Path) -> None:
        """Volume mesh point_data contains all default LS59 fields."""
        from physicsnemo_curator.domains.mesh.sources.plaid import _LS59_VOLUME_FIELDS

        with _patch_init_from_disk(tmp_path):
            from physicsnemo_curator.domains.mesh.sources.plaid import PlaidSource

            src = PlaidSource(local_folder=str(tmp_path))
            dm = next(src[0])
            pd = dm.interior.point_data
            assert pd is not None
            for fname in _LS59_VOLUME_FIELDS:
                assert fname in pd, f"Volume field '{fname}' missing"

    def test_surface_point_data_keys(self, tmp_path: pathlib.Path) -> None:
        """Surface mesh point_data contains the default LS59 field M_iso."""
        with _patch_init_from_disk(tmp_path):
            from physicsnemo_curator.domains.mesh.sources.plaid import PlaidSource

            src = PlaidSource(local_folder=str(tmp_path))
            dm = next(src[0])
            pd = dm.boundaries["surface"].point_data
            assert pd is not None
            assert "M_iso" in pd

    def test_global_data_keys(self, tmp_path: pathlib.Path) -> None:
        """global_data TensorDict contains all LS59 scalar globals."""
        from physicsnemo_curator.domains.mesh.sources.plaid import _LS59_GLOBALS

        with _patch_init_from_disk(tmp_path):
            from physicsnemo_curator.domains.mesh.sources.plaid import PlaidSource

            src = PlaidSource(local_folder=str(tmp_path))
            dm = next(src[0])
            assert dm.global_data is not None
            for gname in _LS59_GLOBALS:
                assert gname in dm.global_data, f"Global '{gname}' missing"

    def test_global_data_none_when_absent(self, tmp_path: pathlib.Path) -> None:
        """global_data is None when no scalar globals are found."""
        with _patch_init_from_disk(tmp_path, add_globals=False):
            from physicsnemo_curator.domains.mesh.sources.plaid import PlaidSource

            src = PlaidSource(local_folder=str(tmp_path))
            dm = next(src[0])
            assert dm.global_data is None

    # -- Index handling -------------------------------------------------------

    def test_mesh_name(self, tmp_path: pathlib.Path) -> None:
        """mesh_name returns 'sample_{index}'."""
        with _patch_init_from_disk(tmp_path):
            from physicsnemo_curator.domains.mesh.sources.plaid import PlaidSource

            src = PlaidSource(local_folder=str(tmp_path))
            assert src.mesh_name(0) == "sample_0"
            assert src.mesh_name(2) == "sample_2"

    def test_negative_index(self, tmp_path: pathlib.Path) -> None:
        """Negative index wraps to the correct sample."""
        with _patch_init_from_disk(tmp_path, n_samples=4):
            from physicsnemo_curator.domains.mesh.sources.plaid import PlaidSource

            src = PlaidSource(local_folder=str(tmp_path))
            dm_neg = next(src[-1])
            dm_pos = next(src[3])
            # Both items come from the same mock so they are equal
            assert dm_neg.interior.n_points == dm_pos.interior.n_points

    def test_index_out_of_bounds(self, tmp_path: pathlib.Path) -> None:
        """IndexError is raised for out-of-range index."""
        with _patch_init_from_disk(tmp_path, n_samples=3):
            from physicsnemo_curator.domains.mesh.sources.plaid import PlaidSource

            src = PlaidSource(local_folder=str(tmp_path))
            with pytest.raises(IndexError):
                next(src[3])

    def test_negative_index_out_of_bounds(self, tmp_path: pathlib.Path) -> None:
        """IndexError is raised for a negative index past the start."""
        with _patch_init_from_disk(tmp_path, n_samples=3):
            from physicsnemo_curator.domains.mesh.sources.plaid import PlaidSource

            src = PlaidSource(local_folder=str(tmp_path))
            with pytest.raises(IndexError):
                next(src[-4])


# ---------------------------------------------------------------------------
# Generic (Rotor37-like) dataset support: 3-D coords, cell_data, optional bases
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPlaidSourceGenericDataset:
    """Tests for dataset layouts that differ from VKI-LS59 (e.g. Rotor37)."""

    def test_3d_coordinates_when_z_present(self, tmp_path: pathlib.Path) -> None:
        """CoordinateZ is included automatically when present in the sample.

        Only the surface base is 3-D in this mock (Rotor37-like); volume_base
        is disabled so DomainMesh doesn't reject mismatched spatial dims
        between a 3-D surface and the mock's 2-D volume base.
        """
        with _patch_init_from_disk(tmp_path, add_surface_z=True):
            from physicsnemo_curator.domains.mesh.sources.plaid import PlaidSource

            src = PlaidSource(local_folder=str(tmp_path), volume_base=None)
            dm = next(src[0])
            assert dm.boundaries["surface"].points.shape == (_N_SURFACE, 3)
            # No volume_base -> empty point cloud matching the surface's 3 dims.
            assert dm.interior.points.shape == (0, 3)

    def test_surface_cell_data(self, tmp_path: pathlib.Path) -> None:
        """CellCenterFields are read into cell_data, validated against n_cells."""
        with _patch_init_from_disk(tmp_path, add_surface_cell_fields=True):
            from physicsnemo_curator.domains.mesh.sources.plaid import PlaidSource

            src = PlaidSource(
                local_folder=str(tmp_path),
                surface_cell_fields=["Pressure"],
            )
            dm = next(src[0])
            surf = dm.boundaries["surface"]
            assert surf.cell_data is not None
            assert "Pressure" in surf.cell_data
            assert surf.cell_data["Pressure"].shape == (_N_EDGES,)

    def test_cell_field_wrong_length_skipped(self, tmp_path: pathlib.Path) -> None:
        """A CellCenterFields entry with a mismatched length is skipped, not raised."""
        with _patch_init_from_disk(tmp_path, add_surface_cell_fields=True):
            from physicsnemo_curator.domains.mesh.sources.plaid import PlaidSource

            # Surface has no cell connectivity here -> n_cells == 0, so the
            # mocked Pressure array (length _N_EDGES) mismatches and is dropped.
            src = PlaidSource(
                local_folder=str(tmp_path),
                surface_cell_fields=["Pressure"],
                surface_element_type=None,
            )
            dm = next(src[0])
            assert dm.boundaries["surface"].cell_data is None or ("Pressure" not in dm.boundaries["surface"].cell_data)

    def test_quad4_cell_data_expanded_to_split_triangles(self, tmp_path: pathlib.Path) -> None:
        """QUAD_4 CellCenterFields (one value/quad) are duplicated 2x to match triangulated cells.

        Regression test for Rotor37: CellCenterFields values are stored once
        per original QUAD_4 element, but volume_element_type="QUAD_4" splits
        each quad into 2 triangles, so cell_data must be expanded to align.
        """
        with _patch_init_from_disk(tmp_path, add_volume_cell_fields=True):
            from physicsnemo_curator.domains.mesh.sources.plaid import PlaidSource

            src = PlaidSource(local_folder=str(tmp_path), volume_cell_fields=["Density"])
            dm = next(src[0])
            assert dm.interior.cells.shape == (_N_QUADS * 2, 3)
            assert dm.interior.cell_data is not None
            assert "Density" in dm.interior.cell_data
            assert dm.interior.cell_data["Density"].shape == (_N_QUADS * 2,)
            # Each split-triangle pair shares its parent quad's value.
            density = dm.interior.cell_data["Density"]
            assert torch.equal(density[0::2], density[1::2])

    def test_hexa8_split_into_6_tetrahedra(self) -> None:
        """HEXA_8 elements are split into 6 simplicial tetrahedra each.

        Exercises `_get_cells` directly (no dataset I/O needed): a single
        unit-cube hexahedron, standard CGNS/VTK node order, should decompose
        into 6 tetrahedra whose volumes sum back to exactly 1.0.
        """
        from unittest.mock import MagicMock

        from physicsnemo_curator.domains.mesh.sources.plaid import PlaidSource

        src = PlaidSource.__new__(PlaidSource)
        src._zone = "Zone"
        src._log = MagicMock()

        hexa_conn = np.arange(8, dtype=np.int64)
        sample = MagicMock()
        sample.get_feature_by_path.side_effect = lambda path, t=None: hexa_conn if "Elements_HEXA_8" in path else None

        cells = src._get_cells(sample, 0.0, "Base", "HEXA_8")
        assert cells.shape == (6, 4)
        assert cells.dtype == torch.int64

        # Sub-cells from the same hex are consecutive (grouping required for
        # `_build_mesh`'s repeat_interleave-based cell_data expansion).
        assert torch.equal(cells.unique(), torch.arange(8, dtype=torch.int64))

        points = torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=torch.float64,
        )
        total_volume = 0.0
        for tet in cells:
            p0, p1, p2, p3 = points[tet]
            total_volume += abs(torch.dot(p1 - p0, torch.linalg.cross(p2 - p0, p3 - p0)).item()) / 6
        assert total_volume == pytest.approx(1.0)

    def test_volume_base_none_yields_empty_interior(self, tmp_path: pathlib.Path) -> None:
        """volume_base=None -> interior is an empty point cloud matching surface dims."""
        with _patch_init_from_disk(tmp_path, add_surface_z=True):
            from physicsnemo_curator.domains.mesh.sources.plaid import PlaidSource

            src = PlaidSource(local_folder=str(tmp_path), volume_base=None)
            dm = next(src[0])
            assert dm.interior.n_points == 0
            assert dm.interior.points.shape[-1] == 3  # matches 3-D surface
            assert dm.boundaries["surface"].n_points == _N_SURFACE

    def test_surface_base_none_omits_boundaries(self, tmp_path: pathlib.Path) -> None:
        """surface_base=None -> boundaries is empty, interior is still the volume mesh."""
        with _patch_init_from_disk(tmp_path):
            from physicsnemo_curator.domains.mesh.sources.plaid import PlaidSource

            src = PlaidSource(local_folder=str(tmp_path), surface_base=None)
            dm = next(src[0])
            assert dm.n_boundaries == 0
            assert dm.interior.n_points == _N_VOLUME

    def test_both_bases_none_raises(self, tmp_path: pathlib.Path) -> None:
        """Both surface_base and volume_base None is a configuration error."""
        with _patch_init_from_disk(tmp_path):
            from physicsnemo_curator.domains.mesh.sources.plaid import PlaidSource

            with pytest.raises(ValueError, match="surface_base"):
                PlaidSource(local_folder=str(tmp_path), surface_base=None, volume_base=None)
