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

"""Plot per-variable statistics from a GFS stats Zarr store.

Supports two layouts:

1. Legacy per-variable groups:
   ``<stats_path>/<variable>/<stat_name>``
2. Consolidated arrays:
   ``<stats_path>/<stat_name>`` with leading ``variable`` dimension.

Writes one image per statistic (mean, variance, skewness, min, max),
with all variables shown in a subplot grid.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr
from zarr.storage import LocalStore

DEFAULT_STATS = ("mean", "variance", "skewness", "min", "max")


def _read_layout(stats_path: Path) -> tuple[str, list[str]]:
    """Return (layout, variables) for the stats store.

    Layout is either:
    - ``legacy`` for ``<stats>/<variable>/<stat>`` groups
    - ``consolidated`` for root arrays with leading ``variable`` dimension
    """
    root = zarr.open_group(LocalStore(str(stats_path)), mode="r")
    groups = sorted(root.group_keys())
    if groups:
        return "legacy", groups

    if "variables" in root.attrs:
        vars_attr = root.attrs["variables"]
        variables = [str(v) for v in list(vars_attr)]
        return "consolidated", variables

    msg = (
        f"Could not determine stats store layout at {stats_path}. Expected variable groups or root attrs['variables']."
    )
    raise ValueError(msg)


def _load_stat_array(
    stats_path: Path,
    layout: str,
    variables: list[str],
    variable: str,
    stat_name: str,
) -> np.ndarray | None:
    """Load one statistic array for a variable from either layout."""
    if layout == "legacy":
        group = zarr.open_group(LocalStore(str(stats_path / variable)), mode="r")
        if stat_name not in group:
            return None
        return np.asarray(group[stat_name][:])

    root = zarr.open_group(LocalStore(str(stats_path)), mode="r")
    if stat_name not in root:
        return None

    arr = root[stat_name]
    try:
        var_idx = variables.index(variable)
    except ValueError:
        return None

    return np.asarray(arr[var_idx])


def _prepare_plot_data(data: np.ndarray) -> np.ndarray:
    """Reduce data to 0D/1D/2D for plotting.

    For 3D+ arrays (e.g., lead_time, lat, lon), repeatedly select the
    first index on leading dimensions until only 2 dimensions remain.
    """
    out = data
    while out.ndim > 2:
        out = out[0]
    return out


def plot_stat_grids(
    stats_path: Path,
    output_dir: Path,
    max_cols: int = 9,
    dpi: int = 150,
) -> list[Path]:
    """Create one grid plot image per statistic across all variables."""
    layout, variables = _read_layout(stats_path)
    if not variables:
        msg = f"No variables found in stats store: {stats_path}"
        raise ValueError(msg)

    output_dir.mkdir(parents=True, exist_ok=True)

    n_vars = len(variables)
    n_cols = min(max_cols, n_vars)
    n_rows = math.ceil(n_vars / n_cols)

    written: list[Path] = []

    for stat_name in DEFAULT_STATS:
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * 3.0, n_rows * 2.6),
            squeeze=False,
        )
        fig.suptitle(f"GFS {stat_name} Across Variables", fontsize=14, fontweight="bold")

        for i, var_name in enumerate(variables):
            ax = axes[i // n_cols, i % n_cols]
            data = _load_stat_array(stats_path, layout, variables, var_name, stat_name)

            if data is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8)
                ax.set_title(var_name, fontsize=8)
                ax.axis("off")
                continue

            data = _prepare_plot_data(data)
            if data.ndim == 2:
                im = ax.imshow(data, cmap="viridis", aspect="auto")
                fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02).ax.tick_params(labelsize=6)
            elif data.ndim == 1:
                ax.plot(data, linewidth=1.0)
            else:
                ax.text(0.5, 0.5, f"{data.ndim}D", ha="center", va="center", fontsize=8)

            ax.set_title(var_name, fontsize=8)
            ax.tick_params(labelsize=6)

        # Hide any unused axes in the grid.
        for i in range(n_vars, n_rows * n_cols):
            axes[i // n_cols, i % n_cols].axis("off")

        plt.tight_layout()
        out_path = output_dir / f"{stat_name}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)

    return written


def main() -> None:
    """Parse arguments and generate stat grid plots."""
    parser = argparse.ArgumentParser(description="Plot GFS stats.zarr into one image per statistic.")
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=Path("outputs/stats_consolidated.zarr"),
        help="Path to stats Zarr directory (default: outputs/stats_consolidated.zarr)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stats_plots"),
        help="Directory for output PNGs (default: outputs/stats_plots)",
    )
    parser.add_argument(
        "--max-cols",
        type=int,
        default=9,
        help="Maximum columns per grid (default: 9)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI (default: 150)",
    )
    args = parser.parse_args()

    if not args.stats_path.exists():
        raise FileNotFoundError(args.stats_path)

    written = plot_stat_grids(
        stats_path=args.stats_path,
        output_dir=args.output_dir,
        max_cols=args.max_cols,
        dpi=args.dpi,
    )

    print(f"Wrote {len(written)} stat images to {args.output_dir}")  # noqa: T201
    for path in written:
        print(f"  - {path}")  # noqa: T201


if __name__ == "__main__":
    main()
