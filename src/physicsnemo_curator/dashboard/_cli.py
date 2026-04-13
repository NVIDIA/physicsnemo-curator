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

"""CLI subcommand for launching the dashboard."""

from __future__ import annotations

import pathlib

import click


def _resolve_db_path(db_path: str) -> str:
    """Resolve *db_path* as a file path or cache hash prefix.

    If *db_path* is an existing file, return it as-is.  Otherwise treat
    it as a hash prefix and look up a matching ``.db`` in the default
    cache directory.

    Parameters
    ----------
    db_path : str
        File path or hash prefix.

    Returns
    -------
    str
        Resolved absolute path to the ``.db`` file.

    Raises
    ------
    click.BadParameter
        If no matching database is found or the prefix is ambiguous.
    """
    p = pathlib.Path(db_path)
    if p.exists():
        return str(p)

    # Try resolving as hash prefix in cache dir
    from physicsnemo_curator.core.cache import default_cache_dir

    cache_dir = default_cache_dir()
    if not cache_dir.is_dir():
        msg = f"'{db_path}' is not an existing file and cache dir {cache_dir} does not exist"
        raise click.BadParameter(msg)

    matches = list(cache_dir.glob(f"{db_path}*.db"))
    if len(matches) == 0:
        msg = f"No database matching '{db_path}' in {cache_dir}"
        raise click.BadParameter(msg)
    if len(matches) > 1:
        stems = ", ".join(m.stem[:8] for m in matches)
        msg = f"Ambiguous prefix '{db_path}', matches: {stems}"
        raise click.BadParameter(msg)

    return str(matches[0])


@click.command("dashboard")
@click.argument("db_path")
@click.option("--port", default=5006, type=int, help="Server port.")
@click.option("--no-browser", is_flag=True, help="Don't open a browser window.")
def dashboard_cmd(db_path: str, port: int, no_browser: bool) -> None:
    """Launch the pipeline metrics dashboard.

    DB_PATH is a path to a PipelineStore SQLite database file, or a hash
    prefix that will be looked up in the default cache directory.
    """
    resolved = _resolve_db_path(db_path)
    from physicsnemo_curator.dashboard import launch

    launch(resolved, port=port, open_browser=not no_browser)
