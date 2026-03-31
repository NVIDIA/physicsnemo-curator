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

Walks the user through selecting a submodule, store, source, filters, and
sink, then constructs and executes a :class:`~curator.core.base.Pipeline`.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

import click
import questionary

from curator.core.base import Filter, Param, Pipeline, Sink, Source
from curator.core.registry import registry
from curator.core.store import FileStore, FsspecFileStore, LocalFileStore

# Map submodule names to their Python module paths so we can import them
# on demand (triggering component registration).
_SUBMODULE_IMPORTS: dict[str, str] = {
    "mesh": "curator.mesh",
    "da": "curator.da",
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
            dep_map = {"mesh": "physicsnemo.mesh", "da": "xarray", "mdt": "torch"}
            desc_map = {
                "mesh": "Mesh processing (physicsnemo.mesh.Mesh)",
                "da": "DataArray processing (xarray.DataArray)",
                "mdt": "Molecular dynamics tensor tuples",
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
            click.echo("Aborted.")
            sys.exit(1)

        # Use the default if the user left the input empty.
        if answer == "" and not param.required:
            values[param.name] = param.default
        else:
            values[param.name] = answer

    return values


def _build_source(source_cls: type, store: FileStore, kwargs: dict[str, Any]) -> Source[Any]:
    """Construct a source with an already-built :class:`FileStore`.

    If *source_cls* accepts a ``store`` init parameter (the recommended
    pattern), the store is injected directly.  Otherwise the kwargs are
    passed through to ``__init__``.

    Parameters
    ----------
    source_cls : type
        The Source subclass.
    store : FileStore
        Pre-built file store to inject.
    kwargs : dict[str, Any]
        Additional parameter values collected from the user (e.g.
        conversion options such as ``manifold_dim``).

    Returns
    -------
    Source[Any]
        Constructed source instance.
    """
    return source_cls(store=store, **kwargs)


def _build_store(store_cls: type) -> FileStore:
    """Prompt for store-specific inputs and construct the store.

    For :class:`LocalFileStore` the user provides a local path and optional
    glob pattern.  For :class:`FsspecFileStore` the user provides a URL and
    optional pattern / cache directory.

    Parameters
    ----------
    store_cls : type
        The FileStore class to build.

    Returns
    -------
    FileStore
        Constructed file store instance.
    """
    if store_cls is LocalFileStore:
        path = questionary.text("  Path to file or directory:").ask()
        if path is None:
            click.echo("Aborted.")
            sys.exit(1)
        pattern = questionary.text("  Glob pattern [*]:", default="*").ask()
        if pattern is None:
            click.echo("Aborted.")
            sys.exit(1)
        return LocalFileStore(path=path, pattern=pattern or "*")

    if store_cls is FsspecFileStore:
        url = questionary.text("  Remote URL (s3://, hf://, https://):").ask()
        if url is None:
            click.echo("Aborted.")
            sys.exit(1)
        pattern = questionary.text("  Glob pattern [**]:", default="**").ask()
        if pattern is None:
            click.echo("Aborted.")
            sys.exit(1)
        cache = questionary.text("  Local cache directory (leave empty for temp):", default="").ask()
        return FsspecFileStore(
            url=url,
            pattern=pattern or "**",
            cache_storage=cache or None,
        )

    # Generic fallback for user-registered stores.
    click.echo("  (custom store — no interactive prompts, constructing with defaults)")
    return store_cls()


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
    # 3. Select and build store (data location)
    # ------------------------------------------------------------------
    stores = registry.stores(submodule_name)
    store_instance: FileStore | None = None

    if stores:
        store_choices = [questionary.Choice(title=name, value=name) for name in stores]
        store_name: str | None = questionary.select("Select a data store:", choices=store_choices).ask()
        if store_name is None:
            click.echo("Aborted.")
            sys.exit(1)

        store_cls = stores[store_name]
        click.echo(f"\nConfigure {store_name}:")
        store_instance = _build_store(store_cls)
        click.echo(f"  Found {len(store_instance)} file(s) in store.")
        click.echo()

    # ------------------------------------------------------------------
    # 4. Select source
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

    # Remove store-related params that are now handled by the store step.
    source_kwargs.pop("input_path", None)
    source_kwargs.pop("url", None)
    source_kwargs.pop("file_pattern", None)

    if store_instance is not None:
        source_instance: Source[Any] = _build_source(source_cls, store_instance, source_kwargs)
    else:
        # No stores registered — fall back to passing kwargs directly.
        source_instance = source_cls(**source_kwargs)

    click.echo(f"  Found {len(source_instance)} item(s) in source.")
    click.echo()

    # ------------------------------------------------------------------
    # 5. Select filters (multi-select)
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
    # 6. Select sink
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
    # 7. Build pipeline
    # ------------------------------------------------------------------
    click.echo()
    chain_parts = [source_cls.name]
    chain_parts.extend(f.name for f in filter_instances)
    chain_parts.append(sink_cls.name)
    click.echo(f"Pipeline: {' → '.join(chain_parts)}")
    click.echo()

    pipeline: Pipeline[Any] = Pipeline(source=source_instance, filters=filter_instances, sink=sink_instance)

    # ------------------------------------------------------------------
    # 8. Execute pipeline
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
    # 9. Flush any stateful filters (e.g. MeanFilter)
    # ------------------------------------------------------------------
    for f in filter_instances:
        if hasattr(f, "flush"):
            flush = getattr(f, "flush")  # noqa: B009
            result = flush()
            if result:
                click.echo(f"Statistics saved to {result}")

    total_outputs = sum(len(p) for p in all_paths)
    click.echo(f"\nDone! {n} source item(s) processed, {total_outputs} output(s) written.")
