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

"""Unit tests for :mod:`physicsnemo_curator.run` — base classes and configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from physicsnemo_curator.run import RunConfig, list_backends, register_backend
from physicsnemo_curator.run.base import RunBackend

if TYPE_CHECKING:
    from physicsnemo_curator.core.base import Pipeline

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# RunConfig tests
# ---------------------------------------------------------------------------


class TestRunConfig:
    """Tests for RunConfig dataclass."""

    def test_positive_passthrough(self):
        """Positive n_jobs should pass through unchanged."""
        config = RunConfig(n_jobs=4)
        assert config.resolved_n_jobs == 4

    def test_one_is_one(self):
        """n_jobs=1 should resolve to 1."""
        config = RunConfig(n_jobs=1)
        assert config.resolved_n_jobs == 1

    def test_negative_one_uses_all_cpus(self):
        """n_jobs=-1 should use all available CPUs."""
        import os

        expected = os.cpu_count() or 1
        config = RunConfig(n_jobs=-1)
        assert config.resolved_n_jobs == expected

    def test_negative_two_is_cpus_minus_one(self):
        """n_jobs=-2 should use cpu_count - 1."""
        import os

        cpu = os.cpu_count() or 1
        config = RunConfig(n_jobs=-2)
        assert config.resolved_n_jobs == max(1, cpu - 1)

    def test_very_negative_floors_at_one(self):
        """Very negative n_jobs should floor at 1."""
        config = RunConfig(n_jobs=-999)
        assert config.resolved_n_jobs >= 1

    def test_default_values(self):
        """Default values should be sensible."""
        config = RunConfig()
        assert config.n_jobs == 1
        assert config.progress is True
        assert config.indices is None
        assert config.backend_options == {}

    def test_backend_options(self):
        """Backend options should be stored correctly."""
        config = RunConfig(backend_options={"retries": 3, "timeout": 60})
        assert config.backend_options["retries"] == 3
        assert config.backend_options["timeout"] == 60


# ---------------------------------------------------------------------------
# Backend registry tests
# ---------------------------------------------------------------------------


class TestBackendRegistry:
    """Tests for backend registration and discovery."""

    def test_list_backends_contains_builtins(self):
        """Built-in backends should be registered."""
        backends = list_backends()
        assert "sequential" in backends
        assert "thread_pool" in backends
        assert "process_pool" in backends
        assert "loky" in backends
        assert "dask" in backends
        assert "prefect" in backends

    def test_backend_info_structure(self):
        """Backend info should contain expected fields."""
        backends = list_backends()
        for name, info in backends.items():
            assert "description" in info, f"Backend {name} missing description"
            assert "available" in info, f"Backend {name} missing available"
            assert "requires" in info, f"Backend {name} missing requires"

    def test_builtin_backends_always_available(self):
        """Sequential, thread_pool, process_pool should always be available."""
        backends = list_backends()
        assert backends["sequential"]["available"] is True
        assert backends["thread_pool"]["available"] is True
        assert backends["process_pool"]["available"] is True

    def test_register_custom_backend(self):
        """Custom backends can be registered."""

        class CustomTestBackend(RunBackend):
            name: ClassVar[str] = "custom_test_backend"
            description: ClassVar[str] = "Test backend for unit tests"
            requires: ClassVar[tuple[str, ...]] = ()

            def run(self, pipeline: Pipeline[Any], config: RunConfig) -> list[list[str]]:
                return []

        # Register the backend
        register_backend(CustomTestBackend)

        # Verify it appears in list
        backends = list_backends()
        assert "custom_test_backend" in backends
        assert backends["custom_test_backend"]["available"] is True

    def test_register_duplicate_raises(self):
        """Registering a backend with duplicate name should raise."""

        class DuplicateBackend(RunBackend):
            name: ClassVar[str] = "sequential"  # Already registered
            description: ClassVar[str] = "Duplicate"

            def run(self, pipeline: Pipeline[Any], config: RunConfig) -> list[list[str]]:
                return []

        with pytest.raises(ValueError, match="already registered"):
            register_backend(DuplicateBackend)


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestImports:
    """Tests for module import paths."""

    def test_import_from_core(self):
        """run_pipeline should be importable from core."""
        from physicsnemo_curator.core import run_pipeline as rp

        assert callable(rp)

    def test_import_from_top_level(self):
        """run_pipeline should be importable from top-level."""
        from physicsnemo_curator import run_pipeline as rp

        assert callable(rp)

    def test_import_from_run_module(self):
        """run_pipeline should be importable from run module."""
        from physicsnemo_curator.run import run_pipeline as rp

        assert callable(rp)

    def test_import_backend_classes(self):
        """Backend classes should be importable."""
        from physicsnemo_curator.run import (
            DaskBackend,
            LokyBackend,
            PrefectBackend,
            ProcessPoolBackend,
            SequentialBackend,
            ThreadPoolBackend,
        )

        assert SequentialBackend.name == "sequential"
        assert ThreadPoolBackend.name == "thread_pool"
        assert ProcessPoolBackend.name == "process_pool"
        assert LokyBackend.name == "loky"
        assert DaskBackend.name == "dask"
        assert PrefectBackend.name == "prefect"
