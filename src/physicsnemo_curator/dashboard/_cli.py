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

import json
import pathlib

import click

_PIPELINE_SUFFIXES = {".yaml", ".yml", ".json"}


def _resolve_db_path(db_path: str) -> str:
    """Resolve *db_path* to a database file path.

    Resolution order:

    1. **Existing ``.db`` file** — returned as-is.
    2. **Pipeline file** (``.yaml`` / ``.yml`` / ``.json``) — the config
       hash is computed from the serialized pipeline and the matching
       database is looked up in the cache directory.
    3. **Hash prefix** — glob-matched against ``*.db`` in the cache
       directory.

    Parameters
    ----------
    db_path : str
        File path, pipeline file path, or hash prefix.

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

    # 1. Existing .db file
    if p.exists() and p.suffix not in _PIPELINE_SUFFIXES:
        return str(p)

    # 2. Pipeline file → compute config hash → look up .db
    if p.exists() and p.suffix.lower() in _PIPELINE_SUFFIXES:
        return _resolve_from_pipeline_file(p)

    # 3. Hash prefix lookup in cache dir
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


def _resolve_from_pipeline_file(path: pathlib.Path) -> str:
    """Compute the config hash from a serialized pipeline file and resolve the DB.

    Parameters
    ----------
    path : pathlib.Path
        Path to a ``.yaml``, ``.yml``, or ``.json`` pipeline file.

    Returns
    -------
    str
        Resolved absolute path to the ``.db`` file.

    Raises
    ------
    click.BadParameter
        If the file cannot be parsed or no matching DB exists.
    """
    from physicsnemo_curator.core.cache import default_cache_dir
    from physicsnemo_curator.core.pipeline_store import _config_hash

    suffix = path.suffix.lower()
    try:
        text = path.read_text()
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                msg = "PyYAML is required to read pipeline YAML files. Install with: pip install pyyaml"
                raise click.BadParameter(msg) from exc
            config = yaml.safe_load(text)
        else:
            config = json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Cannot read pipeline file '{path}': {exc}"
        raise click.BadParameter(msg) from exc

    if not isinstance(config, dict):
        msg = f"Pipeline file '{path}' does not contain a valid config dict"
        raise click.BadParameter(msg)

    # Strip serialization-only keys before hashing (matches _pipeline_config output)
    config.pop("version", None)

    hash_ = _config_hash(config)
    db_name = f"{hash_[:16]}.db"

    cache_dir = default_cache_dir()
    db_path = cache_dir / db_name
    if not db_path.exists():
        msg = f"No database for pipeline '{path.name}' (hash {hash_[:16]}) in {cache_dir}"
        raise click.BadParameter(msg)

    return str(db_path)


@click.command("dashboard")
@click.argument("db_path")
@click.option("--port", default=5006, type=int, help="Server port.")
@click.option("--no-browser", is_flag=True, help="Don't open a browser window.")
def dashboard_cmd(db_path: str, port: int, no_browser: bool) -> None:
    r"""Launch the pipeline metrics dashboard.

    DB_PATH can be:

    \b
    - A path to a PipelineStore .db file
    - A path to a serialized pipeline (.yaml / .json)
    - A hash prefix to look up in the cache directory
    """
    resolved = _resolve_db_path(db_path)
    from physicsnemo_curator.dashboard import launch

    launch(resolved, port=port, open_browser=not no_browser)
