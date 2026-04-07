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

"""Shared pytest configuration, fixtures, and custom markers.

Markers
-------
- ``@pytest.mark.unit`` — fast, isolated unit tests (no I/O, no GPU).
- ``@pytest.mark.integration`` — tests that touch the filesystem, network, or
  multiple components working together.
- ``@pytest.mark.e2e`` — end-to-end tests that exercise a full pipeline from
  source through sink.
- ``@pytest.mark.device`` — parametrised across compute devices. Tests decorated
  with this marker are automatically collected once per available device.
- ``@pytest.mark.requires("group")`` — declares the dependency group needed
  to run a test file.  Tests are **skipped at collection time** when the
  required packages are not importable.  Use as a module-level
  ``pytestmark``::

      pytestmark = pytest.mark.requires("mesh")

  CI can select tests for a specific environment with::

      pytest -m 'requires("mesh")'    # only mesh tests
      pytest -m 'not requires'         # core-only (no optional deps)

Device fixture
--------------
The ``device`` fixture yields ``"cpu"`` and, when CUDA is available,
``"cuda"``.  Use it together with ``@pytest.mark.device``::

    @pytest.mark.device
    def test_something(device: str):
        tensor = torch.zeros(3, device=device)
        ...
"""

from __future__ import annotations

import importlib

import pytest

# ---------------------------------------------------------------------------
# Dependency-group → sentinel imports
# ---------------------------------------------------------------------------
# Maps each dependency-group name to a list of top-level packages that MUST
# be importable for tests in that group to run.  When any import fails the
# whole file is skipped with a clear message.
_GROUP_SENTINELS: dict[str, list[str]] = {
    "mesh": ["pyvista", "torch", "pyarrow", "physicsnemo.mesh"],
    "da": ["xarray", "earth2studio"],
    "mdt": ["torch"],
    "alch": ["nvalchemi.data", "ase", "torch"],
    "cli": ["click", "questionary"],
    "loky": ["joblib"],
    "dask": ["dask"],
    "prefect": ["prefect"],
}


def _group_available(group: str) -> tuple[bool, str]:
    """Check whether all sentinel packages for *group* can be imported.

    Parameters
    ----------
    group : str
        Dependency group name (must be a key in ``_GROUP_SENTINELS``).

    Returns
    -------
    tuple[bool, str]
        ``(True, "")`` when available, or ``(False, reason)`` when not.
    """
    sentinels = _GROUP_SENTINELS.get(group)
    if sentinels is None:
        return True, ""
    for mod in sentinels:
        try:
            importlib.import_module(mod)
        except ImportError:
            return False, f"dependency group {group!r} requires {mod!r}"
    return True, ""


# ---------------------------------------------------------------------------
# Collection hooks
# ---------------------------------------------------------------------------
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Post-collection hook: auto-tag unit tests and skip unavailable groups.

    1. Tests decorated with ``@pytest.mark.requires("group")`` are skipped
       when the sentinel imports for that group fail.  This happens at
       **collection time** so the test module's optional imports never
       execute in the wrong environment.
    2. Tests without an explicit category marker (``unit``, ``integration``,
       ``e2e``, ``benchmark``) are tagged ``unit`` by default.
    """
    category_markers = {"unit", "integration", "e2e", "benchmark"}

    # Cache availability checks per group so we import at most once.
    _cache: dict[str, tuple[bool, str]] = {}

    for item in items:
        item_markers = {m.name for m in item.iter_markers()}

        # --- requires: add group name as marker + skip when deps missing ---
        for marker in item.iter_markers(name="requires"):
            group: str = marker.args[0]
            # Add the group name itself as a marker so that
            # ``pytest -m mesh`` selects exactly mesh tests.
            item.add_marker(getattr(pytest.mark, group))

            if group not in _cache:
                _cache[group] = _group_available(group)
            ok, reason = _cache[group]
            if not ok:
                item.add_marker(pytest.mark.skip(reason=reason))

        # --- auto-tag unit ---
        if not item_markers & category_markers:
            item.add_marker(pytest.mark.unit)


# ---------------------------------------------------------------------------
# Device fixture & helpers
# ---------------------------------------------------------------------------
def _available_devices() -> list[str]:
    """Return the list of devices to test on."""
    devices = ["cpu"]
    try:
        import torch

        if torch.cuda.is_available():
            devices.append("cuda")
    except ImportError:  # pragma: no cover
        pass
    return devices


@pytest.fixture(params=_available_devices(), scope="session")
def device(request: pytest.FixtureRequest) -> str:
    """Yield each available compute device (``"cpu"``, ``"cuda"``, ...).

    Parameters
    ----------
    request : pytest.FixtureRequest
        Injected by pytest.

    Returns
    -------
    str
        Device string usable by ``torch.device()``.
    """
    return request.param
