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

"""Widget provider protocol for filter-specific artifact visualizations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import panel as pn


@runtime_checkable
class WidgetProvider(Protocol):
    """Protocol for filter-specific artifact visualization widgets.

    Implementations must declare *name* and *filter_name* class attributes
    and implement :meth:`panel` and :meth:`layout_hints` methods.
    """

    name: str
    """Human-readable name for the widget."""

    filter_name: str
    """The filter class name this widget handles (e.g. ``'MeanFilter'``)."""

    def panel(
        self,
        artifact_paths: list[str],
        selected_index: int | None = None,
    ) -> pn.viewable.Viewable:
        """Return a Panel component visualizing the artifacts.

        Parameters
        ----------
        artifact_paths : list[str]
            Paths to artifact files produced by the filter.
        selected_index : int or None
            Currently selected pipeline index, if any.

        Returns
        -------
        pn.viewable.Viewable
            A Panel component (pane, widget, row, column, etc.).
        """
        ...

    def layout_hints(self) -> dict[str, int]:
        """Declare grid space preferences for GridStack placement.

        Returns
        -------
        dict[str, int]
            ``cols``: number of GridStack columns to span (1-12).
            ``rows``: number of GridStack rows to span (1+).
        """
        ...
