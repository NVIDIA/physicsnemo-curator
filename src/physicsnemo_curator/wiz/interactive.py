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

"""Interactive pipeline wizard using questionary prompts.

Walks the user through selecting a submodule, source, filters, and
sink, then constructs and executes a :class:`~curator.core.base.Pipeline`.
Optionally loads a saved pipeline or saves the built pipeline to YAML.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from physicsnemo_curator.core.base import Filter, Param, Pipeline, Sink, Source
from physicsnemo_curator.core.registry import registry
from physicsnemo_curator.core.serialization import load_pipeline, save_pipeline

# Custom theme with NVIDIA green accent
CURATOR_THEME = Theme(
    {
        "info": "cyan",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "header": "bold magenta",
        "highlight": "bold cyan",
        "dim": "dim white",
        "step": "bold blue",
        "nvidia": "bold #76B900",  # NVIDIA green
    }
)

# Shared console for colored output
console = Console(theme=CURATOR_THEME)

# Map submodule names to their Python module paths so we can import them
# on demand (triggering component registration).
_SUBMODULE_IMPORTS: dict[str, str] = {
    "mesh": "physicsnemo_curator.domains.mesh",
    "da": "physicsnemo_curator.domains.da",
    "atm": "physicsnemo_curator.domains.atm",
}


def _print_banner() -> None:
    """Print the welcome banner with styling."""
    banner_text = Text()
    banner_text.append("PhysicsNeMo ", style="nvidia")
    banner_text.append("Curator", style="bold white")

    subtitle = Text("Interactive ETL Pipeline Wizard", style="dim")

    panel = Panel(
        Text.assemble(banner_text, "\n", subtitle),
        border_style="nvidia",
        padding=(1, 2),
    )
    console.print(panel)


def _print_step(step_num: int, total: int, description: str) -> None:
    """Print a step header."""
    console.print()
    console.print(f"[step]Step {step_num}/{total}:[/step] [highlight]{description}[/highlight]")


def _print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"  [success]\u2713[/success] {message}")


def _print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"  [error]\u2717[/error] {message}")


def _print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"  [info]\u2022[/info] {message}")


def _print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"  [warning]\u26a0[/warning] {message}")


def _ensure_submodules_registered() -> None:
    """Import submodule packages so they register with the global registry.

    Only submodules whose dependencies are installed will succeed; others
    are silently skipped (they will appear as "not installed" in the prompt).
    """
    for name, module_path in _SUBMODULE_IMPORTS.items():
        try:
            importlib.import_module(module_path)
        except ImportError:
            # Register a placeholder so the wizard can show it as unavailable.
            dep_map = {"mesh": "physicsnemo.mesh", "da": "xarray", "atm": "nvalchemi.data"}
            desc_map = {
                "mesh": "Mesh data curation (physicsnemo.mesh.Mesh)",
                "da": "DataArray data curation (xarray.DataArray)",
                "atm": "Atomic data curation (nvalchemi.data.AtomicData)",
            }
            registry.register_submodule(
                name,
                desc_map.get(name, name),
                dep_map.get(name, name),
            )


def _prompt_params(component_cls: Any) -> dict[str, Any]:
    """Prompt the user for each parameter declared by *component_cls*.

    Parameters
    ----------
    component_cls : Any
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
            _print_error("Aborted.")
            sys.exit(1)

        # Use the default if the user left the input empty.
        if answer == "" and not param.required:
            values[param.name] = param.default
        else:
            values[param.name] = answer

    return values


def _build_source(source_cls: type, kwargs: dict[str, Any]) -> Source[Any]:
    """Construct a source instance from user-provided parameters.

    Parameters
    ----------
    source_cls : type
        The Source subclass.
    kwargs : dict[str, Any]
        Parameter values collected from the user.

    Returns
    -------
    Source[Any]
        Constructed source instance.
    """
    return source_cls(**kwargs)


def _print_pipeline_summary(source_name: str, filter_names: list[str], sink_name: str) -> None:
    """Print a nicely formatted pipeline summary."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="dim")
    table.add_column(style="bold")

    # Build pipeline chain
    chain_parts = [f"[cyan]{source_name}[/cyan]"]
    for fname in filter_names:
        chain_parts.append(f"[yellow]{fname}[/yellow]")
    chain_parts.append(f"[green]{sink_name}[/green]")

    pipeline_str = " [dim]\u2192[/dim] ".join(chain_parts)

    console.print()
    console.print(Panel(pipeline_str, title="[bold]Pipeline[/bold]", border_style="blue"))


def _build_pipeline_interactive() -> Pipeline[Any]:  # noqa: C901, PLR0912, PLR0915
    """Walk through the interactive build steps and return a Pipeline.

    Steps: submodule selection, source selection, filter
    selection, and sink selection.

    Returns
    -------
    Pipeline[Any]
        The fully configured pipeline ready for execution or saving.
    """
    total_steps = 4

    # ------------------------------------------------------------------
    # 1. Discover and register available submodules
    # ------------------------------------------------------------------
    _print_step(1, total_steps, "Select Submodule")
    _ensure_submodules_registered()
    submodules = registry.submodules()

    if not submodules:
        _print_error("No submodules registered. Install an extra (e.g. curator[mesh]).")
        sys.exit(1)

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
        _print_error("No submodules with installed dependencies found.")
        _print_info("Install an extra, e.g.: pip install 'curator[mesh]'")
        sys.exit(1)

    submodule_name: str | None = questionary.select("Select a submodule:", choices=choices).ask()
    if submodule_name is None:
        _print_error("Aborted.")
        sys.exit(1)

    _print_success(f"Selected submodule: [highlight]{submodule_name}[/highlight]")

    # ------------------------------------------------------------------
    # 2. Select source
    # ------------------------------------------------------------------
    _print_step(2, total_steps, "Select Source/Reader")
    sources = registry.sources(submodule_name)
    if not sources:
        _print_error(f"No sources registered for {submodule_name!r}.")
        sys.exit(1)

    source_choices = [
        questionary.Choice(title=f"{cls.name} — {cls.description}", value=cls_name) for cls_name, cls in sources.items()
    ]
    source_name: str | None = questionary.select("Select a source/reader:", choices=source_choices).ask()
    if source_name is None:
        _print_error("Aborted.")
        sys.exit(1)

    source_cls = sources[source_name]
    console.print(f"\n  [header]Configure {source_cls.name}:[/header]")
    source_kwargs = _prompt_params(source_cls)

    source_instance: Source[Any] = _build_source(source_cls, source_kwargs)

    _print_success(f"Found [highlight]{len(source_instance)}[/highlight] item(s) in source")

    # ------------------------------------------------------------------
    # 3. Select filters (multi-select)
    # ------------------------------------------------------------------
    _print_step(3, total_steps, "Select Filters")
    filters = registry.filters(submodule_name)
    filter_instances: list[Filter[Any]] = []
    filter_names: list[str] = []

    if filters:
        filter_choices = [
            questionary.Choice(title=f"{cls.name} — {cls.description}", value=cls_name, checked=False)
            for cls_name, cls in filters.items()
        ]
        selected_filter_names: list[str] | None = questionary.checkbox(
            "Select filters (space to toggle, enter to confirm):", choices=filter_choices
        ).ask()

        if selected_filter_names is None:
            _print_error("Aborted.")
            sys.exit(1)

        for fname in selected_filter_names:
            filter_cls = filters[fname]
            console.print(f"\n  [header]Configure {filter_cls.name}:[/header]")
            fkwargs = _prompt_params(filter_cls)
            filter_instances.append(filter_cls(**fkwargs))
            filter_names.append(filter_cls.name)

        if selected_filter_names:
            _print_success(f"Selected [highlight]{len(selected_filter_names)}[/highlight] filter(s)")
        else:
            _print_info("No filters selected")
    else:
        _print_warning("No filters available for this submodule")

    # ------------------------------------------------------------------
    # 4. Select sink
    # ------------------------------------------------------------------
    _print_step(4, total_steps, "Select Sink/Writer")
    sinks = registry.sinks(submodule_name)
    if not sinks:
        _print_error(f"No sinks registered for {submodule_name!r}.")
        sys.exit(1)

    sink_choices = [
        questionary.Choice(title=f"{cls.name} — {cls.description}", value=cls_name) for cls_name, cls in sinks.items()
    ]
    sink_name: str | None = questionary.select("Select a sink/writer:", choices=sink_choices).ask()
    if sink_name is None:
        _print_error("Aborted.")
        sys.exit(1)

    sink_cls = sinks[sink_name]
    console.print(f"\n  [header]Configure {sink_cls.name}:[/header]")
    sink_kwargs = _prompt_params(sink_cls)
    sink_instance: Sink[Any] = sink_cls(**sink_kwargs)

    _print_success(f"Configured sink: [highlight]{sink_cls.name}[/highlight]")

    pipeline: Pipeline[Any] = Pipeline(source=source_instance, filters=filter_instances, sink=sink_instance)

    _print_pipeline_summary(source_cls.name, filter_names, sink_cls.name)

    return pipeline


def run_interactive() -> None:  # noqa: C901, PLR0912, PLR0915
    """Run the full interactive pipeline wizard flow."""
    _print_banner()

    # ------------------------------------------------------------------
    # 0. Load or build?
    # ------------------------------------------------------------------
    mode: str | None = questionary.select(
        "How would you like to start?",
        choices=[
            questionary.Choice(title="Build a new pipeline", value="build"),
            questionary.Choice(title="Load a saved pipeline (YAML/JSON)", value="load"),
        ],
    ).ask()
    if mode is None:
        _print_error("Aborted.")
        sys.exit(1)

    pipeline: Pipeline[Any]

    if mode == "load":
        path: str | None = questionary.text("  Path to pipeline file:").ask()
        if path is None:
            _print_error("Aborted.")
            sys.exit(1)

        try:
            pipeline = load_pipeline(path)
        except Exception as exc:  # noqa: BLE001
            _print_error(f"Failed to load pipeline: {exc}")
            sys.exit(1)

        _print_success(f"Loaded pipeline from [highlight]{path}[/highlight]")
        _print_info(f"Source items: [highlight]{len(pipeline)}[/highlight]")
    else:
        pipeline = _build_pipeline_interactive()

    # ------------------------------------------------------------------
    # Save pipeline?
    # ------------------------------------------------------------------
    save_answer: str | None = questionary.select(
        "Save pipeline to file before executing?",
        choices=[
            questionary.Choice(title="No, just execute", value="no"),
            questionary.Choice(title="Yes, save to YAML", value="yaml"),
            questionary.Choice(title="Yes, save to JSON", value="json"),
        ],
    ).ask()
    if save_answer is None:
        _print_error("Aborted.")
        sys.exit(1)

    if save_answer in ("yaml", "json"):
        default_name = f"pipeline.{save_answer}"
        save_path: str | None = questionary.text(
            f"  Save path [{default_name}]:",
            default=default_name,
        ).ask()
        if save_path is None:
            _print_error("Aborted.")
            sys.exit(1)

        try:
            save_pipeline(pipeline, save_path)
            _print_success(f"Pipeline saved to [highlight]{save_path}[/highlight]")
        except Exception as exc:  # noqa: BLE001
            _print_warning(f"Could not save pipeline: {exc}")

    # ------------------------------------------------------------------
    # Execute pipeline with progress bar
    # ------------------------------------------------------------------
    n = len(pipeline)
    console.print()

    all_paths: list[list[str]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TextColumn("[dim]{task.fields[current_file]}[/dim]"),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(
            "[cyan]Processing...[/cyan]",
            total=n,
            current_file="",
        )

        for i in range(n):
            paths = pipeline[i]
            all_paths.append(paths)
            progress.update(
                task,
                advance=1,
                current_file=paths[0] if paths else "",
            )

    # ------------------------------------------------------------------
    # Flush any stateful filters (e.g. MeanFilter)
    # ------------------------------------------------------------------
    for f in pipeline.filters:
        if hasattr(f, "flush"):
            flush = getattr(f, "flush")  # noqa: B009
            result = flush()
            if result:
                _print_info(f"Statistics saved to {result}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    console.print()
    total_outputs = sum(len(p) for p in all_paths)

    summary_table = Table(show_header=False, box=None, padding=(0, 2))
    summary_table.add_column(style="dim")
    summary_table.add_column(style="bold")

    summary_table.add_row("Source items processed:", f"[cyan]{n}[/cyan]")
    summary_table.add_row("Outputs written:", f"[green]{total_outputs}[/green]")
    if pipeline.track_metrics and pipeline.db_path is not None:
        summary_table.add_row("Pipeline DB:", f"[dim]{pipeline.db_path}[/dim]")

    console.print(
        Panel(
            summary_table,
            title="[bold green]\u2713 Complete[/bold green]",
            border_style="green",
        )
    )
