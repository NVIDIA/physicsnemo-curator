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

"""AtomicStatsFilter artifact visualization widget."""

from __future__ import annotations

import panel as pn


class AtomicStatsScatterWidget:
    """Interactive scatter plot widget for AtomicStatsFilter parquet output.

    Displays a scatter plot where users can select X/Y axes from available
    statistics columns, color points by categorical dimensions, and filter
    by level (node/edge/system/extra).
    """

    name: str = "Atomic Statistics Scatter"
    filter_name: str = "AtomicStatsFilter"

    def panel(
        self,
        artifact_paths: list[str],
        selected_index: int | None = None,
    ) -> pn.viewable.Viewable:
        """Return a Panel component visualizing AtomicStatsFilter artifacts.

        Parameters
        ----------
        artifact_paths : list[str]
            Paths to Parquet files produced by AtomicStatsFilter.
        selected_index : int or None
            Currently selected pipeline index, if any.

        Returns
        -------
        pn.viewable.Viewable
            A Panel Row containing sidebar controls and scatter plot.
        """
        if not artifact_paths:
            return pn.pane.Markdown("*No AtomicStatsFilter artifacts found.*")

        return pn.pane.Markdown("*Widget not yet implemented.*")
