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

"""Plot crash simulation meshes at several timesteps."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr


def plot_meshes(output_dir: Path, n_timesteps: int = 5, out_file: str = "crashsim.jpg") -> None:
    """Render mesh point clouds at evenly spaced timesteps.

    Parameters
    ----------
    output_dir : Path
        Directory containing .zarr mesh stores from the ETL pipeline.
    n_timesteps : int
        Number of timestep columns to display.
    out_file : str
        Output JPEG filename.
    """
    zarr_dirs = sorted(output_dir.glob("*.zarr"))
    if not zarr_dirs:
        raise FileNotFoundError(f"No .zarr stores found in {output_dir}")

    n_runs = len(zarr_dirs)
    fig, axes = plt.subplots(n_runs, n_timesteps, figsize=(3 * n_timesteps, 3 * n_runs), squeeze=False)

    for row, zdir in enumerate(zarr_dirs):
        store = zarr.open_group(str(zdir), mode="r")
        mesh_pos = np.asarray(store["mesh_pos"])  # (T, N, 3)
        n_frames = mesh_pos.shape[0]

        # Pick evenly spaced timestep indices
        indices = np.linspace(0, n_frames - 1, n_timesteps, dtype=int)

        for col, t in enumerate(indices):
            ax = axes[row, col]
            pts = mesh_pos[t]
            # Subsample for speed if large
            step = max(1, pts.shape[0] // 5000)
            ax.scatter(pts[::step, 0], pts[::step, 1], s=0.1, c=pts[::step, 2], cmap="magma")
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"t={t}", fontsize=9)
            if col == 0:
                ax.set_ylabel(zdir.stem, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot crash simulation meshes")
    parser.add_argument("--output", type=Path, default=Path("output/crashsim"), help="Zarr output directory")
    parser.add_argument("--timesteps", type=int, default=5, help="Number of timestep columns")
    parser.add_argument("--out", type=str, default="sample.jpg", help="Output JPEG filename")
    args = parser.parse_args()

    plot_meshes(args.output, n_timesteps=args.timesteps, out_file=args.out)
