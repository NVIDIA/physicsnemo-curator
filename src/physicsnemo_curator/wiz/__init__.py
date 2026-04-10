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

"""CLI entry point for PhysicsNeMo Curator.

Run ``curator`` to launch the interactive pipeline wizard (default).
Run ``curator dashboard <db_path>`` to open the metrics dashboard.

Requires the ``curator[wiz]`` extra (click, questionary, rich).
"""

from __future__ import annotations

import click
from rich.console import Console

from physicsnemo_curator.wiz.interactive import run_interactive

# Shared console for colored output
console = Console()


@click.group(invoke_without_command=True)
@click.version_option(package_name="physicsnemo-curator")
@click.pass_context
def main(ctx: click.Context) -> None:
    """PhysicsNeMo Curator — interactive ETL pipeline toolkit."""
    if ctx.invoked_subcommand is None:
        run_interactive()


@main.command("wizard")
def wizard_cmd() -> None:
    """Launch the interactive pipeline wizard."""
    run_interactive()


# Lazily register the dashboard subcommand so that importing
# physicsnemo_curator.dashboard (which requires panel) is deferred
# until the user actually invokes the command.
try:
    from physicsnemo_curator.dashboard._cli import dashboard_cmd

    main.add_command(dashboard_cmd)
except ImportError:
    pass
