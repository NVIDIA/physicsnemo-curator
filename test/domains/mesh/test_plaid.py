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
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

if TYPE_CHECKING:
    pass

pytestmark = pytest.mark.requires("mesh")

# ---------------------------------------------------------------------------
# Constants that mirror VKI-LS59 defaults
# ---------------------------------------------------------------------------

_N_SURFACE = 8   # boundary nodes
_N_VOLUME = 20   # interior nodes
_N_EDGES = 7     # BAR_2 edges
_N_QUADS = 10    # QUAD_4 quads  →  20 triangles after splitting


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
            assert dm.interior.points.shape == (
                _N_VOLUME, 2
            ), "Volume coords must be (N, 2)"
            assert dm.boundaries["surface"].points.shape == (
                _N_SURFACE, 2
            ), "Surface coords must be (N, 2)"

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
