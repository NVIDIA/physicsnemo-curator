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

"""Interactive pipeline builder using questionary prompts.

Walks the user through selecting a submodule, source, filters, and sink,
then constructs and executes a :class:`~curator.core.base.Pipeline`.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

import click
import questionary

from curator.core.base import Filter, Param, Pipeline, Sink, Source
from curator.core.registry import registry

# Map submodule names to their Python module paths so we can import them
# on demand (triggering component registration).
_SUBMODULE_IMPORTS: dict[str, str] = {
    "mesh": "curator.mesh",
    "xr": "curator.xr",
    "mdt": "curator.mdt",
}


def _ensure_submodules_registered() -> None:
    """Import submodule packages so they register with the global registry.

    Only submodules whose dependencies are installed will succeed; others
    are silently skipped (they will appear as "not installed" in the prompt).
    """
    for name, module_path in _SUBMODULE_IMPORTS.items():
        try:
            importlib.import_module(module_path)
        except ImportError:
            # Register a placeholder so the CLI can show it as unavailable.
            dep_map = {"mesh": "physicsnemo.mesh", "xr": "xarray", "mdt": "torch"}
            desc_map = {
                "mesh": "Mesh processing (physicsnemo.mesh.Mesh)",
                "xr": "xarray dataset processing",
                "mdt": "Molecular dynamics tensor tuples",
            }
            registry.register_submodule(
                name,
                desc_map.get(name, name),
                dep_map.get(name, name),
            )


def _prompt_params(component_cls: type) -> dict[str, Any]:
    """Prompt the user for each parameter declared by *component_cls*.

    Parameters
    ----------
    component_cls : type
        A Source, Filter, or Sink subclass with a ``params()`` classmethod.

    Returns
    -------
    dict[str, Any]
        Mapping of parameter name to user-supplied value.
    """
    params: list[Param] = component_cls.params()
    values: dict[str, Any] = {}

    for param in params:
        default_text = "" if param.required else f" [{param.default}]"
        prompt_msg = f"  {param.description}{default_text}"

        if param.choices:
            answer = questionary.select(prompt_msg, choices=param.choices).ask()
        else:
            answer = questionary.text(
                prompt_msg,
                default="" if param.required else str(param.default),
            ).ask()

        if answer is None:
            # User pressed Ctrl-C.
            click.echo("Aborted.")
            sys.exit(1)

        # Use the default if the user left the input empty.
        if answer == "" and not param.required:
            values[param.name] = param.default
        else:
            values[param.name] = answer

    return values


def run_interactive() -> None:
    """Run the full interactive pipeline builder flow."""
    click.echo()
    click.echo("Welcome to PhysicsNeMo Curator!")
    click.echo()

    # ------------------------------------------------------------------
    # 1. Discover and register available submodules
    # ------------------------------------------------------------------
    _ensure_submodules_registered()
    submodules = registry.submodules()

    if not submodules:
        click.echo("No submodules registered. Install an extra (e.g. curator[mesh]).")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Select submodule
    # ------------------------------------------------------------------
    choices = []
    available_names: list[str] = []
    for name, entry in submodules.items():
        if entry.available:
            choices.append(questionary.Choice(title=f"{name} — {entry.description}", value=name))
            available_names.append(name)
        else:
            choices.append(
                questionary.Choice(
                    title=f"{name} — {entry.description} (not installed)",
                    value=name,
                    disabled=f"install curator[{name}]",
                )
            )

    if not available_names:
        click.echo("No submodules with installed dependencies found.")
        click.echo("Install an extra, e.g.: pip install 'curator[mesh]'")
        sys.exit(1)

    submodule_name: str | None = questionary.select("Select a submodule:", choices=choices).ask()
    if submodule_name is None:
        click.echo("Aborted.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Select source
    # ------------------------------------------------------------------
    sources = registry.sources(submodule_name)
    if not sources:
        click.echo(f"No sources registered for {submodule_name!r}.")
        sys.exit(1)

    source_choices = [
        questionary.Choice(title=f"{cls.name} — {cls.description}", value=cls_name) for cls_name, cls in sources.items()
    ]
    source_name: str | None = questionary.select("Select a source/reader:", choices=source_choices).ask()
    if source_name is None:
        click.echo("Aborted.")
        sys.exit(1)

    source_cls = sources[source_name]
    click.echo(f"\nConfigure {source_cls.name}:")
    source_kwargs = _prompt_params(source_cls)
    source_instance: Source[Any] = source_cls(**source_kwargs)

    click.echo(f"  Found {len(source_instance)} item(s) in source.")
    click.echo()

    # ------------------------------------------------------------------
    # 4. Select filters (multi-select)
    # ------------------------------------------------------------------
    filters = registry.filters(submodule_name)
    filter_instances: list[Filter[Any]] = []

    if filters:
        filter_choices = [
            questionary.Choice(title=f"{cls.name} — {cls.description}", value=cls_name, checked=False)
            for cls_name, cls in filters.items()
        ]
        selected_filter_names: list[str] | None = questionary.checkbox(
            "Select filters (space to toggle, enter to confirm):", choices=filter_choices
        ).ask()

        if selected_filter_names is None:
            click.echo("Aborted.")
            sys.exit(1)

        for fname in selected_filter_names:
            filter_cls = filters[fname]
            click.echo(f"\nConfigure {filter_cls.name}:")
            fkwargs = _prompt_params(filter_cls)
            filter_instances.append(filter_cls(**fkwargs))
    else:
        click.echo("No filters available for this submodule.")

    click.echo()

    # ------------------------------------------------------------------
    # 5. Select sink
    # ------------------------------------------------------------------
    sinks = registry.sinks(submodule_name)
    if not sinks:
        click.echo(f"No sinks registered for {submodule_name!r}.")
        sys.exit(1)

    sink_choices = [
        questionary.Choice(title=f"{cls.name} — {cls.description}", value=cls_name) for cls_name, cls in sinks.items()
    ]
    sink_name: str | None = questionary.select("Select a sink/writer:", choices=sink_choices).ask()
    if sink_name is None:
        click.echo("Aborted.")
        sys.exit(1)

    sink_cls = sinks[sink_name]
    click.echo(f"\nConfigure {sink_cls.name}:")
    sink_kwargs = _prompt_params(sink_cls)
    sink_instance: Sink[Any] = sink_cls(**sink_kwargs)

    # ------------------------------------------------------------------
    # 6. Build pipeline
    # ------------------------------------------------------------------
    click.echo()
    chain_parts = [source_cls.name]
    chain_parts.extend(f.name for f in filter_instances)
    chain_parts.append(sink_cls.name)
    click.echo(f"Pipeline: {' → '.join(chain_parts)}")
    click.echo()

    pipeline: Pipeline[Any] = Pipeline(source=source_instance, filters=filter_instances, sink=sink_instance)

    # ------------------------------------------------------------------
    # 7. Execute pipeline
    # ------------------------------------------------------------------
    n = len(pipeline)
    click.echo(f"Processing {n} item(s)...")

    all_paths: list[list[str]] = []
    for i in range(n):
        paths = pipeline[i]
        all_paths.append(paths)
        click.echo(f"  [{i + 1}/{n}] → {', '.join(paths)}")

    click.echo()

    # ------------------------------------------------------------------
    # 8. Flush any stateful filters (e.g. MeanFilter)
    # ------------------------------------------------------------------
    for f in filter_instances:
        if hasattr(f, "flush"):
            result = f.flush()
            if result:
                click.echo(f"Statistics saved to {result}")

    total_outputs = sum(len(p) for p in all_paths)
    click.echo(f"\nDone! {n} source item(s) processed, {total_outputs} output(s) written.")
