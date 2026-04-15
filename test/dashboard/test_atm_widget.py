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

"""Tests for AtomicStatsScatterWidget."""

from __future__ import annotations

import pytest

pytest.importorskip("panel")
pytest.importorskip("holoviews")


class TestAtomicStatsScatterWidget:
    """Tests for AtomicStatsScatterWidget."""

    def test_instantiation(self) -> None:
        """Widget can be instantiated."""
        from physicsnemo_curator.dashboard.widgets.atm import AtomicStatsScatterWidget

        widget = AtomicStatsScatterWidget()
        assert widget.name == "Atomic Statistics Scatter"
        assert widget.filter_name == "AtomicStatsFilter"
