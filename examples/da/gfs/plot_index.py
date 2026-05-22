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

"""Plot all variables from a GFS Zarr store for a given time index.

See README.md for usage instructions.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import zarr
from dotenv import load_dotenv
from zarr.storage import FsspecStore, LocalStore

# Load .env from this directory
load_dotenv(Path(__file__).parent / ".env")


def _build_s3_storage_options() -> dict[str, object]:
    """Build S3 storage options from ZARR_S3_* environment variables."""
    options: dict[str, object] = {}

    if key := os.environ.get("ZARR_S3_ACCESS_KEY_ID"):
        options["key"] = key
    if secret := os.environ.get("ZARR_S3_SECRET_ACCESS_KEY"):
        options["secret"] = secret
    if endpoint := os.environ.get("ZARR_S3_ENDPOINT_URL"):
        options["endpoint_url"] = endpoint
    if region := os.environ.get("ZARR_S3_REGION"):
        options["client_kwargs"] = {"region_name": region}

    return options


def _open_store(zarr_path: str, storage_options: dict[str, object] | None = None) -> zarr.Group:
    """Open the Zarr store root group."""
    if zarr_path.startswith(("s3://", "gs://", "az://")):
        store = FsspecStore.from_url(zarr_path, storage_options=storage_options or {})
    else:
        store = LocalStore(zarr_path)
    return zarr.open_group(store, mode="r")


def plot_variable_grid(
    zarr_path: str,
    index: int,
    max_cols: int = 9,
    storage_options: dict[str, object] | None = None,
    output: str | None = None,
) -> None:
    """Plot all variables from the Zarr store at a given time index.

    Parameters
    ----------
    zarr_path : str
        Path to Zarr store (local or s3://).
    index : int
        Time index to plot.
    max_cols : int
        Maximum columns in the grid.
    storage_options : dict or None
        Storage options for remote stores.
    output : str or None
        Output filename. Defaults to sample{index}.png.
    """
    root = _open_store(zarr_path, storage_options)
    variables = sorted(root.array_keys())
    n_vars = len(variables)

    if n_vars == 0:
        print("No variables found in the Zarr store.")  # noqa: T201
        return

    print(f"Found {n_vars} variables")  # noqa: T201

    # Grid layout
    n_cols = min(max_cols, n_vars)
    n_rows = math.ceil(n_vars / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.5), squeeze=False)
    fig.suptitle(f"GFS Variables at Index {index}", fontsize=14, fontweight="bold")

    # Plot each variable
    for i, var_name in enumerate(variables):
        ax = axes[i // n_cols, i % n_cols]
        try:
            data = root[var_name][index, :, :]
            im = ax.imshow(data, aspect="auto", cmap="viridis")
            fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02).ax.tick_params(labelsize=6)
            ax.set_title(var_name, fontsize=8)
            ax.tick_params(labelsize=6)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{e!s}", ha="center", va="center", fontsize=8)
            ax.set_title(var_name, fontsize=8)

    # Hide unused subplots
    for i in range(n_vars, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].axis("off")

    plt.tight_layout()
    output_file = output or f"sample{index}.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_file}")  # noqa: T201


def main() -> None:
    """Parse arguments and generate the plot."""
    parser = argparse.ArgumentParser(description="Plot GFS variables from a Zarr store.")
    parser.add_argument("--index", "-i", type=int, default=0, help="Time index to plot (default: 0)")
    parser.add_argument("--zarr-path", "-z", type=str, default="s3://gfs/data.zarr", help="Path to Zarr store")
    parser.add_argument("--max-cols", type=int, default=9, help="Maximum columns in grid (default: 9)")
    parser.add_argument("--output", "-o", type=str, help="Output filename (default: sample{index}.png)")
    args = parser.parse_args()

    # Build storage options for S3
    storage_options = _build_s3_storage_options() if args.zarr_path.startswith("s3://") else None

    plot_variable_grid(
        zarr_path=args.zarr_path,
        index=args.index,
        max_cols=args.max_cols,
        storage_options=storage_options,
        output=args.output,
    )


if __name__ == "__main__":
    main()
