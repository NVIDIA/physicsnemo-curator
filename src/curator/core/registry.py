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

"""Component registry for pipeline discovery and CLI auto-generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SubmoduleEntry:
    """Registry entry for a single submodule (e.g. ``mesh``, ``xr``, ``mdt``).

    Parameters
    ----------
    name : str
        Submodule identifier (e.g. ``"mesh"``).
    description : str
        Human-readable description.
    import_check : str
        Dotted module path to probe availability (e.g.
        ``"physicsnemo.mesh"``).
    sources : dict[str, type]
        Registered source classes keyed by their :attr:`~Source.name`.
    filters : dict[str, type]
        Registered filter classes keyed by their :attr:`~Filter.name`.
    sinks : dict[str, type]
        Registered sink classes keyed by their :attr:`~Sink.name`.
    """

    name: str
    description: str
    import_check: str
    sources: dict[str, type] = field(default_factory=dict)
    filters: dict[str, type] = field(default_factory=dict)
    sinks: dict[str, type] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        """Return ``True`` if the required dependency is importable."""
        import importlib

        try:
            importlib.import_module(self.import_check)
        except ImportError:
            return False
        return True


class Registry:
    """Global registry of submodules and their pipeline components.

    Components register themselves at import time so the interactive CLI
    can dynamically discover available sources, filters, and sinks.

    Examples
    --------
    >>> from curator.core.registry import registry
    >>> registry.register_submodule("mesh", "Mesh processing", "physicsnemo.mesh")
    >>> registry.register_source("mesh", VTKSource)
    """

    def __init__(self) -> None:
        self._submodules: dict[str, SubmoduleEntry] = {}

    # -- Registration --------------------------------------------------------

    def register_submodule(self, name: str, description: str, import_check: str) -> None:
        """Register a new submodule.

        Parameters
        ----------
        name : str
            Submodule identifier.
        description : str
            Human-readable description.
        import_check : str
            Dotted module path used to check if the dependency is installed.
        """
        if name not in self._submodules:
            self._submodules[name] = SubmoduleEntry(
                name=name,
                description=description,
                import_check=import_check,
            )

    def register_source(self, submodule: str, cls: type) -> None:
        """Register a source class under *submodule*.

        Parameters
        ----------
        submodule : str
            Target submodule name (must already be registered).
        cls : type
            Source class with a ``name`` class attribute.
        """
        self._ensure_submodule(submodule)
        self._submodules[submodule].sources[cls.name] = cls

    def register_filter(self, submodule: str, cls: type) -> None:
        """Register a filter class under *submodule*.

        Parameters
        ----------
        submodule : str
            Target submodule name (must already be registered).
        cls : type
            Filter class with a ``name`` class attribute.
        """
        self._ensure_submodule(submodule)
        self._submodules[submodule].filters[cls.name] = cls

    def register_sink(self, submodule: str, cls: type) -> None:
        """Register a sink class under *submodule*.

        Parameters
        ----------
        submodule : str
            Target submodule name (must already be registered).
        cls : type
            Sink class with a ``name`` class attribute.
        """
        self._ensure_submodule(submodule)
        self._submodules[submodule].sinks[cls.name] = cls

    # -- Querying ------------------------------------------------------------

    def submodules(self) -> dict[str, SubmoduleEntry]:
        """Return all registered submodules.

        Returns
        -------
        dict[str, SubmoduleEntry]
            Mapping of submodule name to entry.
        """
        return dict(self._submodules)

    def sources(self, submodule: str) -> dict[str, type]:
        """Return registered sources for *submodule*.

        Parameters
        ----------
        submodule : str
            Submodule name.

        Returns
        -------
        dict[str, type]
            Mapping of source name to class.
        """
        return dict(self._submodules[submodule].sources)

    def filters(self, submodule: str) -> dict[str, type]:
        """Return registered filters for *submodule*.

        Parameters
        ----------
        submodule : str
            Submodule name.

        Returns
        -------
        dict[str, type]
            Mapping of filter name to class.
        """
        return dict(self._submodules[submodule].filters)

    def sinks(self, submodule: str) -> dict[str, type]:
        """Return registered sinks for *submodule*.

        Parameters
        ----------
        submodule : str
            Submodule name.

        Returns
        -------
        dict[str, type]
            Mapping of sink name to class.
        """
        return dict(self._submodules[submodule].sinks)

    # -- Internal ------------------------------------------------------------

    def _ensure_submodule(self, name: str) -> None:
        """Raise ``KeyError`` if *name* is not a registered submodule.

        Parameters
        ----------
        name : str
            Submodule name to check.
        """
        if name not in self._submodules:
            msg = f"Submodule {name!r} is not registered. Call registry.register_submodule() first."
            raise KeyError(msg)

    def __repr__(self) -> str:
        """Return a string representation of the registry."""
        parts: list[str] = []
        for name, entry in self._submodules.items():
            status = "available" if entry.available else "not installed"
            parts.append(
                f"  {name} ({status}): "
                f"{len(entry.sources)} sources, "
                f"{len(entry.filters)} filters, "
                f"{len(entry.sinks)} sinks"
            )
        body = "\n".join(parts) if parts else "  (empty)"
        return f"Registry(\n{body}\n)"


#: Global singleton registry used by all submodules.
registry: Any = Registry()
