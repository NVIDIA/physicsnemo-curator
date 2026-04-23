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

"""Widget registry mapping filter names to filter classes with dashboard widgets."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class WidgetRegistry:
    """Registry mapping filter names to filter classes for dashboard widgets.

    Filters that override :meth:`~physicsnemo_curator.core.base.Filter.dashboard_panel`
    are auto-discovered on construction.  Additional filter classes can be
    registered at runtime via :meth:`register`.
    """

    def __init__(self) -> None:
        """Initialize the registry and discover built-in filter widgets."""
        self._filter_classes: dict[str, type] = {}
        self._auto_discover()

    def register(self, filter_cls: type) -> None:
        """Register a filter class that provides a dashboard widget.

        Parameters
        ----------
        filter_cls : type
            A :class:`~physicsnemo_curator.core.base.Filter` subclass
            whose ``dashboard_panel`` classmethod returns non-None.
        """
        self._filter_classes[filter_cls.name] = filter_cls

    def get_panel(
        self,
        filter_name: str,
        artifact_paths: list[str],
        selected_index: int | None = None,
    ) -> Any:
        """Look up a filter class by name and render its dashboard widget.

        Parameters
        ----------
        filter_name : str
            The filter's ``name`` class variable (as stored in the DB).
        artifact_paths : list[str]
            Paths to artifact files produced by the filter.
        selected_index : int or None
            Currently selected pipeline index, if any.

        Returns
        -------
        pn.viewable.Viewable or None
            The rendered widget, or ``None`` if no widget is registered.
        """
        cls = self._filter_classes.get(filter_name)
        if cls is None:
            return None
        return cls.dashboard_panel(artifact_paths, selected_index)

    def get_layout_hints(self, filter_name: str) -> dict[str, int]:
        """Look up layout hints for a filter name.

        Parameters
        ----------
        filter_name : str
            The filter's ``name`` class variable.

        Returns
        -------
        dict[str, int]
            Grid column and row span preferences.
        """
        cls = self._filter_classes.get(filter_name)
        if cls is None:
            return {"cols": 6, "rows": 2}
        return cls.dashboard_layout_hints()

    def list_providers(self) -> dict[str, str]:
        """Return a mapping of filter name to filter name for all registered widgets.

        Returns
        -------
        dict[str, str]
            ``{filter_name: filter_name}`` for all registered filters.
        """
        return {name: name for name in self._filter_classes}

    def _auto_discover(self) -> None:
        """Import known filter classes with dashboard widgets.

        Imports are deferred so the registry works even when optional
        domain dependencies are not installed.
        """
        try:
            from physicsnemo_curator.domains.atm.filters.stats import AtomicStatsFilter

            self.register(AtomicStatsFilter)
        except Exception:  # noqa: BLE001
            logger.debug("AtomicStatsFilter not available", exc_info=True)

        try:
            from physicsnemo_curator.domains.mesh.filters.mean import MeanFilter

            self.register(MeanFilter)
        except Exception:  # noqa: BLE001
            logger.debug("MeanFilter not available", exc_info=True)
