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

Usage::

    python plot_index.py --index 0 --zarr-path outputs/gfs/data.zarr
    python plot_index.py --index 10 --zarr-path s3://bucket/gfs/data.zarr
    python plot_index.py --index 5 --output plot.png

S3 credentials can be configured via a ``.env`` file in this directory::

    ZARR_S3_ACCESS_KEY_ID=your-access-key
    ZARR_S3_SECRET_ACCESS_KEY=your-secret-key
    ZARR_S3_REGION=us-east-1
    ZARR_S3_ENDPOINT_URL=https://s3.amazonaws.com

"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr
from dotenv import load_dotenv
from zarr.storage import FsspecStore, LocalStore

# Load .env from this directory (does not overwrite existing env vars)
load_dotenv(Path(__file__).parent / ".env")


def _build_s3_storage_options() -> dict[str, object]:
    """Build S3 storage options from environment variables."""
    options: dict[str, object] = {}

    access_key = os.environ.get("ZARR_S3_ACCESS_KEY_ID")
    secret_key = os.environ.get("ZARR_S3_SECRET_ACCESS_KEY")
    region = os.environ.get("ZARR_S3_REGION")
    endpoint_url = os.environ.get("ZARR_S3_ENDPOINT_URL")

    if access_key:
        options["key"] = access_key
    if secret_key:
        options["secret"] = secret_key
    if endpoint_url:
        options["endpoint_url"] = endpoint_url
    if region:
        options["client_kwargs"] = {"region_name": region}

    return options


def _open_store(zarr_path: str, storage_options: dict[str, object] | None = None) -> zarr.Group:
    """Open the Zarr store root group."""
    if zarr_path.startswith(("s3://", "gs://", "az://")):
        store = FsspecStore.from_url(zarr_path, storage_options=storage_options or {})
    else:
        store = LocalStore(zarr_path)

    return zarr.open_group(store, mode="r+")


def _list_variable_arrays(
    root: zarr.Group | None, zarr_path: str, storage_options: dict[str, object] | None = None
) -> list[str]:
    """List variable arrays in the Zarr store."""
    # Try using zarr array_keys first (new format: arrays directly under root)
    if root is not None:
        try:
            variables = list(root.array_keys())
            if variables:
                return sorted(variables)
        except Exception:
            pass

        # Fallback: try group_keys for legacy format (groups with nested "data" array)
        try:
            variables = list(root.group_keys())
            if variables:
                return sorted(variables)
        except Exception:
            pass

    # Final fallback: list directories via fsspec
    import fsspec

    if zarr_path.startswith("s3://"):
        fs = fsspec.filesystem("s3", **(storage_options or {}))
        path = zarr_path[5:]
    elif zarr_path.startswith("gs://"):
        fs = fsspec.filesystem("gcs", **(storage_options or {}))
        path = zarr_path[5:]
    elif zarr_path.startswith("az://"):
        fs = fsspec.filesystem("az", **(storage_options or {}))
        path = zarr_path[5:]
    else:
        fs = fsspec.filesystem("file")
        path = zarr_path

    try:
        entries = fs.ls(path, detail=False)
    except FileNotFoundError:
        return []

    variables = []
    for entry in entries:
        name = entry.rstrip("/").split("/")[-1]
        if not name.startswith(".") and not name.startswith("zarr") and fs.isdir(entry):
            variables.append(name)

    return sorted(variables)


def _open_variable_array(
    root: zarr.Group | None,
    zarr_path: str,
    var_name: str,
    storage_options: dict[str, object] | None = None,
) -> zarr.Array | zarr.Group:
    """Open a single variable's Zarr array (or group for legacy format)."""
    # Try using root group first
    if root is not None:
        try:
            item = root[var_name]
            # Could be an array (new format) or group (legacy format)
            return item
        except (KeyError, Exception):
            pass

    # Fallback: open variable path directly
    var_path = f"{zarr_path}/{var_name}"
    if var_path.startswith(("s3://", "gs://", "az://")):
        store = FsspecStore.from_url(var_path, storage_options=storage_options or {})
    else:
        store = LocalStore(var_path)

    # Try opening as array first (new format), then as group (legacy)
    try:
        return zarr.open_array(store, mode="r")
    except Exception:
        return zarr.open_group(store, mode="r")


def plot_variable_grid(
    zarr_path: str,
    index: int,
    max_cols: int = 9,
    storage_options: dict[str, object] | None = None,
) -> None:
    """Plot all variables from the Zarr store at a given time index."""
    # Open the root store
    try:
        root = _open_store(zarr_path, storage_options)
    except Exception as e:
        print(f"Warning: Could not open root group: {e}")  # noqa: T201
        root = None

    # List variable arrays
    variables = _list_variable_arrays(root, zarr_path, storage_options)
    n_vars = len(variables)

    if n_vars == 0:
        print("No variables found in the Zarr store.")  # noqa: T201
        return

    print(f"Found {n_vars} variables: {variables[:5]}{'...' if n_vars > 5 else ''}")  # noqa: T201

    # Calculate grid dimensions
    n_cols = min(max_cols, n_vars)
    n_rows = math.ceil(n_vars / n_cols)

    # Create figure
    fig_width = n_cols * 3
    fig_height = n_rows * 2.5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)

    # Try to get time coordinate for title
    time_str = f"Index {index}"
    try:
        first_var = _open_variable_array(root, zarr_path, variables[0], storage_options)
        # For new format (array), there's no time coordinate stored
        # For legacy format (group with "data" array), check for time array
        if isinstance(first_var, zarr.Group) and "time" in first_var.array_keys():
            time_arr = first_var["time"][:]  # ty: ignore[not-subscriptable]
            if index < len(time_arr):  # ty: ignore[invalid-argument-type]
                time_val = time_arr[index]  # ty: ignore[not-subscriptable]
                time_str = str(np.datetime_as_string(time_val, unit="h"))  # ty: ignore[no-matching-overload]
    except Exception:
        pass

    fig.suptitle(f"GFS Variables at {time_str}", fontsize=14, fontweight="bold")

    # Plot each variable
    for i, var_name in enumerate(variables):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        try:
            var_item = _open_variable_array(root, zarr_path, var_name, storage_options)

            # Handle both new format (array directly) and legacy format (group with "data")
            data_arr = var_item if isinstance(var_item, zarr.Array) else var_item["data"]

            data = data_arr[index, :, :]

            # For legacy format, try to get lat/lon coordinates
            lat: np.ndarray | None = None
            lon: np.ndarray | None = None
            if isinstance(var_item, zarr.Group):
                if "lat" in var_item.array_keys():
                    lat = np.asarray(var_item["lat"][:])
                if "lon" in var_item.array_keys():
                    lon = np.asarray(var_item["lon"][:])

            if lat is not None and lon is not None:
                im = ax.imshow(
                    data,
                    origin="upper" if lat[0] > lat[-1] else "lower",
                    extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                    aspect="auto",
                    cmap="viridis",
                )
            else:
                im = ax.imshow(data, aspect="auto", cmap="viridis")

            cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cbar.ax.tick_params(labelsize=6)
            ax.set_title(var_name, fontsize=8)
            ax.tick_params(labelsize=6)

        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{e!s}", ha="center", va="center", fontsize=8)
            ax.set_title(var_name, fontsize=8)

    # Hide unused subplots
    for i in range(n_vars, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis("off")

    plt.tight_layout()
    output_file = f"sample{index}.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_file}")  # noqa: T201


def main() -> None:
    """Parse arguments and generate the plot."""
    parser = argparse.ArgumentParser(
        description="Plot all GFS variables from a Zarr store at a given time index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--index", "-i", type=int, default=0, help="Time index to plot (default: 0)")
    parser.add_argument("--zarr-path", "-z", type=str, default="s3://gfs/data.zarr", help="Path to Zarr store")
    parser.add_argument("--max-cols", type=int, default=9, help="Maximum columns in grid (default: 9)")

    args = parser.parse_args()

    # Build storage options for remote paths
    if args.zarr_path.startswith(("s3://", "gs://", "az://")):
        zarr_path = args.zarr_path
        storage_options = _build_s3_storage_options() if args.zarr_path.startswith("s3://") else None
    else:
        zarr_path = str(Path(__file__).parent / args.zarr_path)
        storage_options = None

    plot_variable_grid(
        zarr_path=zarr_path,
        index=args.index,
        max_cols=args.max_cols,
        storage_options=storage_options,
    )


if __name__ == "__main__":
    main()
