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

import click


@click.command("dashboard")
@click.argument("db_path", type=click.Path(exists=True))
@click.option("--port", default=5006, type=int, help="Server port.")
@click.option("--no-browser", is_flag=True, help="Don't open a browser window.")
def dashboard_cmd(db_path: str, port: int, no_browser: bool) -> None:
    """Launch the pipeline metrics dashboard.

    DB_PATH is the path to a PipelineStore SQLite database file
    produced by a pipeline run.
    """
    from physicsnemo_curator.dashboard import launch

    launch(db_path, port=port, open_browser=not no_browser)
