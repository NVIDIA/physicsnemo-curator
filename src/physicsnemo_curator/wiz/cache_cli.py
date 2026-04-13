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

"""CLI subcommands for managing the pipeline database cache.

Provides ``psnc cache list``, ``psnc cache info``, ``psnc cache rm``,
and ``psnc cache path`` commands.
"""

from __future__ import annotations

import re

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def _human_size(n_bytes: int) -> str:
    """Format bytes as a human-readable string."""
    size: float = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size) < 1024:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _parse_duration(text: str) -> float:
    """Parse a human duration like '7d', '2h', '30m' into seconds.

    Parameters
    ----------
    text : str
        Duration string (e.g. ``7d``, ``24h``, ``30m``).

    Returns
    -------
    float
        Duration in seconds.

    Raises
    ------
    click.BadParameter
        If the format is not recognized.
    """
    match = re.fullmatch(r"(\d+)\s*([dhm])", text.strip().lower())
    if not match:
        msg = f"Invalid duration format: '{text}'. Use e.g. 7d, 24h, 30m."
        raise click.BadParameter(msg)

    value = int(match.group(1))
    unit = match.group(2)
    multipliers = {"d": 86400, "h": 3600, "m": 60}
    return value * multipliers[unit]


@click.group("cache")
def cache_group() -> None:
    """Manage cached pipeline databases."""


@cache_group.command("path")
def cache_path_cmd() -> None:
    """Print the resolved cache directory path."""
    from physicsnemo_curator.core.cache import default_cache_dir

    console.print(str(default_cache_dir()))


@cache_group.command("list")
def cache_list_cmd() -> None:
    """List all cached pipeline databases with metadata."""
    from physicsnemo_curator.core.cache import cache_size, default_cache_dir, list_databases

    cache_dir = default_cache_dir()
    databases = list_databases(cache_dir)

    if not databases:
        console.print(f"[dim]No cached databases in {cache_dir}[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("HASH", style="cyan", no_wrap=True)
    table.add_column("CREATED", style="dim")
    table.add_column("SIZE", justify="right")
    table.add_column("DONE", justify="right")
    table.add_column("SOURCE", style="green")
    table.add_column("SINK", style="yellow")

    for db in databases:
        table.add_row(
            db.hash_prefix[:8],
            db.created.strftime("%Y-%m-%d %H:%M"),
            _human_size(db.size_bytes),
            f"{db.completed}/{db.total}",
            db.source_name,
            db.sink_name,
        )

    total_size = cache_size(cache_dir=cache_dir)
    panel = Panel(
        table,
        title=f"[bold]Pipeline Cache ({cache_dir})[/bold]",
        subtitle=f"{len(databases)} database{'s' if len(databases) != 1 else ''}, {_human_size(total_size)} total",
        border_style="dim",
    )
    console.print(panel)


@cache_group.command("info")
@click.argument("hash_prefix")
def cache_info_cmd(hash_prefix: str) -> None:
    """Show detailed info for a cached pipeline database.

    HASH_PREFIX is the beginning of the database hash (at least 4 chars).
    """
    from physicsnemo_curator.core.cache import default_cache_dir, list_databases

    databases = list_databases(default_cache_dir())
    matches = [db for db in databases if db.hash_prefix.startswith(hash_prefix)]

    if not matches:
        console.print(f"[red]No database matching '{hash_prefix}'[/red]")
        raise SystemExit(1)
    if len(matches) > 1:
        console.print(f"[red]Ambiguous prefix '{hash_prefix}', matches:[/red]")
        for m in matches:
            console.print(f"  {m.hash_prefix[:8]}  {m.source_name}")
        raise SystemExit(1)

    db = matches[0]
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="bold")

    table.add_row("Hash:", db.hash_prefix)
    table.add_row("Path:", str(db.path))
    table.add_row("Size:", _human_size(db.size_bytes))
    table.add_row("Created:", db.created.strftime("%Y-%m-%d %H:%M:%S"))
    table.add_row("Source:", db.source_name)
    table.add_row("Sink:", db.sink_name)
    table.add_row("Filters:", ", ".join(db.filter_names) if db.filter_names else "none")
    table.add_row("Completed:", f"{db.completed}/{db.total}")
    table.add_row("Failed:", str(db.failed))

    console.print(Panel(table, title="[bold]Database Info[/bold]", border_style="cyan"))


@cache_group.command("rm")
@click.argument("hash_prefixes", nargs=-1)
@click.option("--older-than", "older_than", default=None, help="Remove databases older than duration (e.g. 7d, 24h).")
@click.option("--all", "remove_all", is_flag=True, help="Remove all cached databases.")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
def cache_rm_cmd(
    hash_prefixes: tuple[str, ...],
    older_than: str | None,
    remove_all: bool,
    yes: bool,
) -> None:
    """Remove cached pipeline databases.

    Provide HASH_PREFIXES to remove specific databases, or use --older-than
    or --all for bulk removal.
    """
    from datetime import timedelta

    from physicsnemo_curator.core.cache import clear_cache, remove_databases, remove_older_than

    if not hash_prefixes and older_than is None and not remove_all:
        console.print("[red]Provide hash prefixes, --older-than, or --all[/red]")
        raise SystemExit(1)

    if remove_all:
        if not yes:
            click.confirm("Remove all cached pipeline databases?", abort=True)
        removed = clear_cache()
        console.print(f"Removed {removed} database{'s' if removed != 1 else ''}")
        return

    if older_than is not None:
        seconds = _parse_duration(older_than)
        if not yes:
            click.confirm(f"Remove databases older than {older_than}?", abort=True)
        removed = remove_older_than(timedelta(seconds=seconds))
        console.print(f"Removed {removed} database{'s' if removed != 1 else ''}")
        return

    try:
        removed = remove_databases(list(hash_prefixes))
        console.print(f"Removed {removed} database{'s' if removed != 1 else ''}")
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from None
