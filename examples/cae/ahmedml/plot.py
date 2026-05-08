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

"""Plot AhmedML interior slices via direct memmap reads (fast, no DomainMesh load)."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _interior_path(pdmsh: Path) -> Path:
    """Return the interior tensordict directory inside a .pdmsh."""
    return pdmsh / "_tensordict" / "interior" / "_tensordict"


def _load_points(interior: Path, n_points: int) -> np.ndarray:
    """Memory-map the points array (N, 3) float32."""
    return np.memmap(interior / "points.memmap", dtype=np.float32, mode="r", shape=(n_points, 3))


def _load_field(interior: Path, name: str, shape: tuple[int, ...]) -> np.ndarray:
    """Memory-map a point_data field."""
    path = interior / "point_data" / f"{name}.memmap"
    return np.memmap(path, dtype=np.float32, mode="r", shape=shape)


def _read_meta(interior: Path) -> tuple[int, dict[str, tuple[int, ...]]]:
    """Parse meta.json to get n_points and field shapes.

    Returns
    -------
    tuple[int, dict[str, tuple]]
        (n_points, {field_name: shape_tuple})
    """
    meta = json.loads((interior / "meta.json").read_text())
    n_points = meta["points"]["shape"][0]

    pd_meta = json.loads((interior / "point_data" / "meta.json").read_text())
    fields: dict[str, tuple[int, ...]] = {}
    for key, val in pd_meta.items():
        if isinstance(val, dict) and "shape" in val:
            fields[key] = tuple(val["shape"])
    return n_points, fields


def plot_meshes(
    output_dir: Path,
    n_runs: int = 2,
    fields: list[str] | None = None,
    slice_axis: str = "y",
    slice_tol: float = 0.05,
    max_pts: int = 50_000,
    out_file: str = "ahmedml.jpg",
) -> None:
    """Plot thin slices of interior point clouds colored by different fields.

    Parameters
    ----------
    output_dir : Path
        Output directory containing run_<id>/ subdirectories with .pdmsh files.
    n_runs : int
        Number of runs to display (rows).
    fields : list[str] or None
        Field names to plot (columns). Auto-detected if None.
    slice_axis : str
        Axis normal to the slice plane: 'x', 'y', or 'z'.
    slice_tol : float
        Half-thickness of the slice around the midplane.
    max_pts : int
        Maximum points to scatter per panel.
    out_file : str
        Output image filename.
    """
    pdmsh_dirs = sorted(output_dir.rglob("*.pdmsh"))[:n_runs]
    if not pdmsh_dirs:
        raise FileNotFoundError(f"No .pdmsh files found in {output_dir}")

    # Read metadata from first run to discover fields
    interior0 = _interior_path(pdmsh_dirs[0])
    n_points0, available_fields = _read_meta(interior0)

    if fields is None:
        # Pick scalar fields (1D) first, then magnitude of vectors
        fields = [k for k, s in available_fields.items() if len(s) == 1][:4]
        if not fields:
            fields = list(available_fields.keys())[:4]

    if not fields:
        raise ValueError("No fields found in interior point_data")

    axis_map = {"x": 0, "y": 1, "z": 2}
    ax_idx = axis_map[slice_axis.lower()]
    plot_axes = [i for i in range(3) if i != ax_idx]
    axis_labels = ["X", "Y", "Z"]

    n_fields = len(fields)
    actual_runs = len(pdmsh_dirs)

    fig, axes = plt.subplots(actual_runs, n_fields, figsize=(4 * n_fields, 2.5 * actual_runs), squeeze=False)

    for row, pdmsh in enumerate(pdmsh_dirs):
        interior = _interior_path(pdmsh)
        n_points, field_shapes = _read_meta(interior)
        pts = _load_points(interior, n_points)

        # Subsample first to avoid scanning all 22M+ points for the slice
        rng = np.random.default_rng(42)
        sample_size = min(n_points, max_pts * 10)  # oversample then slice
        sample_idx = rng.choice(n_points, sample_size, replace=False)
        sample_idx.sort()  # sequential access for memmap performance
        pts_sample = pts[sample_idx]

        # Slice the subsample around the midplane
        col_vals = pts_sample[:, ax_idx]
        mid = (col_vals.min() + col_vals.max()) / 2.0
        slice_mask = np.abs(col_vals - mid) < slice_tol
        pts_slice = pts_sample[slice_mask]
        slice_idx = sample_idx[slice_mask]

        # Further subsample if still too many
        if pts_slice.shape[0] > max_pts:
            keep = rng.choice(pts_slice.shape[0], max_pts, replace=False)
            pts_slice = pts_slice[keep]
            slice_idx = slice_idx[keep]

        run_label = pdmsh.parent.name

        for col, field in enumerate(fields):
            ax = axes[row, col]

            if field not in field_shapes:
                ax.text(0.5, 0.5, f"{field}\nN/A", transform=ax.transAxes, ha="center")
                ax.set_aspect("equal")
                continue

            shape = field_shapes[field]
            arr = _load_field(interior, field, shape)
            arr_plot = arr[slice_idx]

            # Compute magnitude for vector fields
            if len(shape) == 2:
                arr_plot = np.linalg.norm(arr_plot, axis=1)

            vmin, vmax = np.percentile(arr_plot, [2, 98])
            sc = ax.scatter(
                pts_slice[:, plot_axes[0]],
                pts_slice[:, plot_axes[1]],
                c=arr_plot,
                s=0.2,
                cmap="coolwarm",
                vmin=vmin,
                vmax=vmax,
                rasterized=True,
            )
            ax.set_aspect("equal")
            ax.set_xlim(-2, 2)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)

            if row == 0:
                ax.set_title(field, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{run_label}\n{axis_labels[plot_axes[1]]}", fontsize=9)
            else:
                ax.set_ylabel("")
            ax.set_xlabel(axis_labels[plot_axes[0]], fontsize=8)
            ax.tick_params(labelsize=7)

    plt.suptitle(
        f"AhmedML Interior — {slice_axis.upper()}-slice (tol={slice_tol})",
        fontsize=12,
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(out_file, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot AhmedML interior slices (fast)")
    parser.add_argument("--output", type=Path, default=Path("output/ahmedml"), help="ETL output directory")
    parser.add_argument("--runs", type=int, default=2, help="Number of runs to plot (rows)")
    parser.add_argument("--fields", nargs="*", default=None, help="Field names to plot (auto-detect if omitted)")
    parser.add_argument("--slice-axis", type=str, default="y", choices=["x", "y", "z"], help="Slice normal axis")
    parser.add_argument("--slice-tol", type=float, default=0.05, help="Slice half-thickness")
    parser.add_argument("--max-pts", type=int, default=50_000, help="Max points per panel")
    parser.add_argument("--out", type=str, default="sample.jpg", help="Output image filename")
    args = parser.parse_args()

    plot_meshes(
        args.output,
        n_runs=args.runs,
        fields=args.fields,
        slice_axis=args.slice_axis,
        slice_tol=args.slice_tol,
        max_pts=args.max_pts,
        out_file=args.out,
    )
