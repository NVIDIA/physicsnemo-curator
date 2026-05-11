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

"""Plot HRRR analysis fields (t2m, q2m, tcwv) at 12-hour intervals."""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr


def plot_hrrr(output_dir: Path, step_hours: int = 12, out_file: str = "hrrr.jpg") -> None:
    """Plot t2m, q2m, and tcwv on HRRR grid at regular time intervals.

    Parameters
    ----------
    output_dir : Path
        Directory containing the HRRR dataset.zarr store.
    step_hours : int
        Hours between plotted frames.
    out_file : str
        Output JPEG filename.
    """
    zarr_path = output_dir / "dataset.zarr"
    if not zarr_path.exists():
        raise FileNotFoundError(f"No dataset.zarr found in {output_dir}")

    # Load time coordinate and decode to datetime
    time_arr = np.asarray(zarr.open_array(str(zarr_path / "t2m" / "time"), mode="r"))
    time_meta = zarr.open_array(str(zarr_path / "t2m" / "time"), mode="r")
    units = str(time_meta.metadata.attributes.get("units", "hours since 1970-01-01"))
    # Parse "days since YYYY-MM-DD HH:MM:SS" format
    parts = units.split("since")
    epoch = datetime.fromisoformat(parts[1].strip()) if len(parts) == 2 else datetime(1970, 1, 1)
    scale = "days" if "days" in parts[0] else "hours"

    # Load data arrays (time, hrrr_y, hrrr_x)
    t2m = zarr.open_array(str(zarr_path / "t2m" / "data"), mode="r")
    q2m = zarr.open_array(str(zarr_path / "q2m" / "data"), mode="r")
    tcwv = zarr.open_array(str(zarr_path / "tcwv" / "data"), mode="r")
    n_times = t2m.shape[0]

    # Select timesteps at the requested interval (1 index = 1 hour)
    indices = list(range(0, n_times, step_hours))
    if not indices:
        indices = [0]
    n_cols = len(indices)

    fig, axes = plt.subplots(3, n_cols, figsize=(3.5 * n_cols, 9), squeeze=False, constrained_layout=True)

    for col, t in enumerate(indices):
        # Decode timestamp to ISO format
        if t < len(time_arr):
            offset = float(time_arr[t])
            dt = epoch + timedelta(days=offset) if scale == "days" else epoch + timedelta(hours=offset)
            title = dt.isoformat(timespec="hours")
        else:
            title = f"t+{t}h"

        # t2m row (2-metre temperature in K)
        # Data shape after indexing: (hrrr_y, hrrr_x) which maps to (rows, cols).
        # origin="lower" puts hrrr_y index 0 at the bottom of the plot.
        ax_t = axes[0, col]
        data_t = np.asarray(t2m[t])
        im_t = ax_t.imshow(data_t, cmap="RdYlBu_r", origin="lower", aspect="auto")
        ax_t.set_xticks([])
        ax_t.set_yticks([])
        ax_t.set_title(title, fontsize=8)
        if col == 0:
            ax_t.set_ylabel("t2m", fontsize=10)

        # q2m row (2-metre specific humidity in kg/kg)
        ax_q = axes[1, col]
        data_q = np.asarray(q2m[t])
        im_q = ax_q.imshow(data_q, cmap="YlGnBu", origin="lower", aspect="auto")
        ax_q.set_xticks([])
        ax_q.set_yticks([])
        if col == 0:
            ax_q.set_ylabel("q2m", fontsize=10)

        # tcwv row (total column water vapour in kg/m^2)
        ax_w = axes[2, col]
        data_w = np.asarray(tcwv[t])
        im_w = ax_w.imshow(data_w, cmap="Blues", origin="lower", aspect="auto")
        ax_w.set_xticks([])
        ax_w.set_yticks([])
        if col == 0:
            ax_w.set_ylabel("tcwv", fontsize=10)

    fig.colorbar(im_t, ax=axes[0, :].tolist(), shrink=0.8, label="K")
    fig.colorbar(im_q, ax=axes[1, :].tolist(), shrink=0.8, label="kg/kg")
    fig.colorbar(im_w, ax=axes[2, :].tolist(), shrink=0.8, label="kg/m²")
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot HRRR t2m/q2m/tcwv analysis fields")
    parser.add_argument(
        "--output", type=Path, default=Path("output/hrrr_analysis"), help="HRRR output directory with dataset.zarr"
    )
    parser.add_argument("--step", type=int, default=12, help="Hours between plotted frames (default: 12)")
    parser.add_argument("--out", type=str, default="sample.jpg", help="Output JPEG filename")
    args = parser.parse_args()

    plot_hrrr(args.output, step_hours=args.step, out_file=args.out)
