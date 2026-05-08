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

"""Plot drop test meshes at several timesteps showing deformation over time."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv


def plot_meshes(output_dir: Path, n_timesteps: int = 5, field: str = "Von_Mises", out_file: str = "drop.jpg") -> None:
    """Render mesh point clouds at evenly spaced timesteps.

    Parameters
    ----------
    output_dir : Path
        Directory containing .vtu files from the drop test ETL pipeline.
    n_timesteps : int
        Number of timestep columns to display.
    field : str
        Field prefix to color by (e.g. 'Von_Mises', 'stress_voigt').
        Uses the matching timestep field for each column.
    out_file : str
        Output JPEG filename.
    """
    vtu_files = sorted(output_dir.glob("*.vtu"))
    if not vtu_files:
        raise FileNotFoundError(f"No .vtu files found in {output_dir}")

    n_runs = len(vtu_files)
    fig, axes = plt.subplots(n_runs, n_timesteps, figsize=(3 * n_timesteps, 3 * n_runs), squeeze=False)

    for row, vtu_path in enumerate(vtu_files):
        mesh = pv.read(str(vtu_path))
        base_pts = np.array(mesh.points)

        # Find displacement fields
        disp_keys = sorted(k for k in mesh.point_data if k.startswith("displacement_t"))
        # Find coloring field keys
        field_keys = sorted(k for k in mesh.point_data if k.startswith(f"{field}_t"))
        n_frames = len(disp_keys)

        if n_frames == 0:
            continue

        # Pick evenly spaced timestep indices
        indices = np.linspace(0, n_frames - 1, n_timesteps, dtype=int)

        for col, t in enumerate(indices):
            ax = axes[row, col]
            disp = np.array(mesh.point_data[disp_keys[t]])
            pts = base_pts + disp

            # Get field values for coloring
            if t < len(field_keys):
                color_vals = np.array(mesh.point_data[field_keys[t]])
                if color_vals.ndim == 2:
                    color_vals = np.linalg.norm(color_vals, axis=1)
            else:
                color_vals = pts[:, 2]

            # Subsample for speed
            step = max(1, pts.shape[0] // 5000)
            vmin, vmax = np.percentile(color_vals[::step], [2, 98])
            ax.scatter(
                pts[::step, 0], pts[::step, 1], s=0.1, c=color_vals[::step], cmap="coolwarm", vmin=vmin, vmax=vmax
            )
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"t={t}", fontsize=9)
            if col == 0:
                ax.set_ylabel(vtu_path.stem, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot drop test meshes over time")
    parser.add_argument("--output", type=Path, default=Path("output/drop"), help="VTU output directory")
    parser.add_argument("--timesteps", type=int, default=5, help="Number of timestep columns")
    parser.add_argument("--field", type=str, default="Von_Mises", help="Field prefix to color by (default: Von_Mises)")
    parser.add_argument("--out", type=str, default="sample.jpg", help="Output JPEG filename")
    args = parser.parse_args()

    plot_meshes(args.output, n_timesteps=args.timesteps, field=args.field, out_file=args.out)
