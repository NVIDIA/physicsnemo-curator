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

"""Entry point for the PhysicsNeMo Curator CLI.

Run ``psnc`` to launch the full-screen Textual TUI wizard.
Run ``psnc dashboard <db_path>`` to launch the metrics dashboard.
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    """Execute the ``psnc`` CLI."""
    parser = argparse.ArgumentParser(
        prog="psnc",
        description="PhysicsNeMo Curator CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Wizard subcommand (default when no command given)
    subparsers.add_parser("wizard", help="Launch the interactive pipeline wizard (default)")

    # Dashboard subcommand
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Launch the metrics dashboard for a pipeline run",
    )
    dashboard_parser.add_argument(
        "db_path",
        help="Path to .db file, pipeline config file, or hash prefix",
    )
    dashboard_parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=5006,
        help="Server port (default: 5006)",
    )
    dashboard_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open a browser window automatically",
    )

    args = parser.parse_args()

    if args.command == "dashboard":
        _run_dashboard(args.db_path, port=args.port, open_browser=not args.no_browser)
    else:
        # Default to wizard (covers both `psnc` and `psnc wizard`)
        _run_wizard()


def _run_wizard() -> None:
    """Launch the Curator Textual wizard application."""
    from physicsnemo_curator.wiz.app import CuratorApp

    CuratorApp().run()


def _run_dashboard(db_path: str, *, port: int, open_browser: bool) -> None:
    """Launch the metrics dashboard.

    Parameters
    ----------
    db_path : str
        Path to .db file, pipeline config file, or hash prefix.
    port : int
        Server port.
    open_browser : bool
        Whether to open a browser window.
    """
    try:
        from physicsnemo_curator.dashboard._cli import launch_dashboard
    except ImportError as exc:
        print(  # noqa: T201
            "Dashboard dependencies not installed. Install with: uv sync --group dashboard",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    try:
        launch_dashboard(db_path, port=port, open_browser=open_browser)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)  # noqa: T201
        raise SystemExit(1) from exc
