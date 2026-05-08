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

"""Plot ERA5 surface wind fields (u10m, v10m) at 12-hour intervals."""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr


def plot_era5(output_dir: Path, step_hours: int = 12, out_file: str = "era5.jpg") -> None:
    """Plot u10m and v10m on lat-lon grids at regular time intervals.

    Parameters
    ----------
    output_dir : Path
        Directory containing the ERA5 dataset.zarr store.
    step_hours : int
        Hours between plotted frames.
    out_file : str
        Output JPEG filename.
    """
    zarr_path = output_dir / "dataset.zarr"
    if not zarr_path.exists():
        raise FileNotFoundError(f"No dataset.zarr found in {output_dir}")

    # Load coordinate arrays from one variable group
    lat = np.asarray(zarr.open_array(str(zarr_path / "u10m" / "lat"), mode="r"))
    lon = np.asarray(zarr.open_array(str(zarr_path / "u10m" / "lon"), mode="r"))

    # Load time coordinate and decode to datetime
    time_arr = np.asarray(zarr.open_array(str(zarr_path / "u10m" / "time"), mode="r"))
    time_meta = zarr.open_array(str(zarr_path / "u10m" / "time"), mode="r")
    units = str(time_meta.metadata.attributes.get("units", "hours since 1970-01-01"))
    # Parse "days since YYYY-MM-DD HH:MM:SS" format
    parts = units.split("since")
    epoch = datetime.fromisoformat(parts[1].strip()) if len(parts) == 2 else datetime(1970, 1, 1)
    scale = "days" if "days" in parts[0] else "hours"

    # Load data arrays (time, lat, lon)
    u10m = zarr.open_array(str(zarr_path / "u10m" / "data"), mode="r")
    v10m = zarr.open_array(str(zarr_path / "v10m" / "data"), mode="r")
    n_times = u10m.shape[0]

    # Select timesteps at the requested interval (1 index = 1 hour)
    indices = list(range(0, n_times, step_hours))
    if not indices:
        indices = [0]
    n_cols = len(indices)

    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 6), squeeze=False, constrained_layout=True)

    for col, t in enumerate(indices):
        # Decode timestamp to ISO format
        if t < len(time_arr):
            offset = float(time_arr[t])
            dt = epoch + timedelta(days=offset) if scale == "days" else epoch + timedelta(hours=offset)
            title = dt.isoformat(timespec="hours")
        else:
            title = f"t+{t}h"

        # u10m row
        ax_u = axes[0, col]
        data_u = np.asarray(u10m[t])
        im_u = ax_u.pcolormesh(lon, lat, data_u, cmap="PiYG", shading="auto", vmin=-15, vmax=15)
        ax_u.set_xticks([])
        ax_u.set_yticks([])
        ax_u.set_title(title, fontsize=8)
        if col == 0:
            ax_u.set_ylabel("u10m", fontsize=10)

        # v10m row
        ax_v = axes[1, col]
        data_v = np.asarray(v10m[t])
        im_v = ax_v.pcolormesh(lon, lat, data_v, cmap="PuOr", shading="auto", vmin=-15, vmax=15)
        ax_v.set_xticks([])
        ax_v.set_yticks([])
        if col == 0:
            ax_v.set_ylabel("v10m", fontsize=10)

    fig.colorbar(im_u, ax=axes[0, :].tolist(), shrink=0.8, label="m/s")
    fig.colorbar(im_v, ax=axes[1, :].tolist(), shrink=0.8, label="m/s")
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ERA5 u10m/v10m wind fields")
    parser.add_argument(
        "--output", type=Path, default=Path("output/era5_surface"), help="ERA5 output directory with dataset.zarr"
    )
    parser.add_argument("--step", type=int, default=12, help="Hours between plotted frames (default: 12)")
    parser.add_argument("--out", type=str, default="sample.jpg", help="Output JPEG filename")
    args = parser.parse_args()

    plot_era5(args.output, step_hours=args.step, out_file=args.out)
