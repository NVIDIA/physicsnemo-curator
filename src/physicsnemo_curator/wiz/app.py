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

"""Textual application shell for the interactive pipeline wizard.

Houses :class:`CuratorApp`, the top-level ``App``, and :class:`WizardState`,
the mutable dataclass that carries pipeline configuration between screens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header

if TYPE_CHECKING:
    from physicsnemo_curator.core.base import Filter, Pipeline, Sink, Source


@dataclass
class WizardState:
    """Mutable state shared between wizard screens."""

    mode: str | None = None  #: ``"build"`` or ``"load"``.
    submodule: str | None = None  #: Selected submodule name.
    source_cls: type | None = None  #: Selected source class.
    source_kwargs: dict[str, Any] = field(default_factory=dict)  #: User-supplied source parameters.
    source_instance: Source | None = None  #: Constructed source instance.
    filter_classes: list[type] = field(default_factory=list)  #: Selected filter classes (ordered).
    filter_kwargs: list[dict[str, Any]] = field(default_factory=list)  #: User-supplied filter parameters.
    filter_instances: list[Filter] = field(default_factory=list)  #: Constructed filter instances.
    sink_cls: type | None = None  #: Selected sink class.
    sink_kwargs: dict[str, Any] = field(default_factory=dict)  #: User-supplied sink parameters.
    sink_instance: Sink | None = None  #: Constructed sink instance.
    pipeline: Pipeline | None = None  #: The fully assembled pipeline.

    def reset(self) -> None:
        """Reset all fields to their defaults for a new wizard run."""
        self.mode = None
        self.submodule = None
        self.source_cls = None
        self.source_kwargs = {}
        self.source_instance = None
        self.filter_classes = []
        self.filter_kwargs = []
        self.filter_instances = []
        self.sink_cls = None
        self.sink_kwargs = {}
        self.sink_instance = None
        self.pipeline = None


class CuratorApp(App[None]):
    """Top-level Textual application for the PhysicsNeMo Curator wizard.

    Manages the screen stack and shared :class:`WizardState`.
    """

    TITLE = "PhysicsNeMo Curator"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("down", "focus_next", "Next"),
        ("up", "focus_previous", "Previous"),
    ]

    def __init__(self) -> None:
        """Initialise the curator app and create an empty wizard state."""
        super().__init__()
        self.state = WizardState()

    def compose(self) -> ComposeResult:
        """Yield the global header and footer."""
        yield Header()
        yield Footer()

    def on_mount(self) -> None:
        """Push the welcome screen on startup."""
        from physicsnemo_curator.wiz.screens.welcome import WelcomeScreen

        self.push_screen(WelcomeScreen())
