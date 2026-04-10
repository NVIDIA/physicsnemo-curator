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

"""Widget registry for filter-specific artifact visualizations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from physicsnemo_curator.dashboard.widgets.base import WidgetProvider

logger = logging.getLogger(__name__)


class WidgetRegistry:
    """Registry mapping filter names to their visualization widgets.

    Built-in widgets are auto-discovered on construction.  Additional
    widgets can be registered at runtime via :meth:`register`.
    """

    def __init__(self) -> None:
        """Initialize the registry and discover built-in widgets."""
        self._providers: dict[str, WidgetProvider] = {}
        self._auto_discover()

    def register(self, provider: WidgetProvider) -> None:
        """Register a widget provider for a filter name.

        Parameters
        ----------
        provider : WidgetProvider
            Widget provider instance to register.
        """
        self._providers[provider.filter_name] = provider

    def get(self, filter_name: str) -> WidgetProvider | None:
        """Look up a widget provider by filter name.

        Parameters
        ----------
        filter_name : str
            The filter class name (e.g. ``'MeanFilter'``).

        Returns
        -------
        WidgetProvider or None
            The registered provider, or ``None`` if not found.
        """
        return self._providers.get(filter_name)

    def list_providers(self) -> dict[str, str]:
        """Return a mapping of filter name to widget display name.

        Returns
        -------
        dict[str, str]
            ``{filter_name: widget.name}`` for all registered widgets.
        """
        return {k: v.name for k, v in self._providers.items()}

    def _auto_discover(self) -> None:
        """Register built-in widgets.

        Imports are deferred so the registry works even when optional
        domain dependencies are not installed.
        """
        try:
            from physicsnemo_curator.dashboard.widgets.mesh import MeanFilterWidget

            self.register(MeanFilterWidget())
        except Exception:  # noqa: BLE001
            logger.debug("MeanFilterWidget not available", exc_info=True)
