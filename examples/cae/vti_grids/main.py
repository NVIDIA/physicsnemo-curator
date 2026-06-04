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

"""Convert VTK ImageData (``.vti``) structured grids to tensordict memmaps.

Reads ``.vti`` files into dense N-D field tensors (via
:class:`~physicsnemo_curator.domains.mesh.sources.vti.VTISource`) and writes
each as a tensordict memmap ``.grid`` directory (via
:class:`~physicsnemo_curator.domains.mesh.sinks.grid_sidecar.GridSidecarSink`),
mirroring the input directory layout under ``--output``.

Each output preserves the structured-grid layout:

* ``point_data`` — sub-TensorDict, ``batch_size = [Nz, Ny, Nx]``
* ``cell_data``  — sub-TensorDict, ``batch_size = [Cz, Cy, Cx]``
* ``grid``       — ``origin``, ``spacing``, ``dimensions``, ``direction``

Reload an output with ``tensordict.TensorDict.load_memmap(path)``.

Sidecar usage
-------------
To drop each grid **next to** the corresponding mesh outputs
(``.pmsh`` / ``.pdmsh``) for the same samples, point ``--output`` at the
**same output root** your mesh conversion writes to and keep the default
``{relpath}/{stem}`` naming.  Each grid then lands in the same per-sample
directory as that sample's mesh.

Examples
--------
::

    # Quick test on the first 2 files
    python main.py --input /data/grids --output output/grids --limit 2

    # Full run, 8 workers, group-readable output
    python main.py --input /data/grids --output output/grids --workers 8 --group-readable

    # Co-locate grids beside existing mesh outputs (sidecar)
    python main.py --input /data/grids --output output/drivaerml
"""

import argparse
from pathlib import Path

from physicsnemo_curator.domains.mesh.sinks.grid_sidecar import GridSidecarSink
from physicsnemo_curator.domains.mesh.sources.vti import VTISource
from physicsnemo_curator.run import run_pipeline


def main() -> None:
    """Run the VTI -> tensordict-memmap conversion pipeline."""
    parser = argparse.ArgumentParser(description="Convert .vti structured grids to tensordict memmaps")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("input/vti"),
        help="Path to a .vti file or a directory of .vti files (default: input/vti)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/vti"),
        help="Output root for .grid sidecars (default: output/vti)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/*",
        help="Glob pattern for discovering files under --input (default: **/*)",
    )
    parser.add_argument(
        "--naming-template",
        type=str,
        default="{relpath}/{stem}",
        help="Output name template; placeholders {index}, {seq}, {relpath}, {stem} (default: {relpath}/{stem})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N files (default: all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1). Each worker holds a full grid in RAM.",
    )
    parser.add_argument(
        "--fp32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Downcast float64 arrays to float32 (default: True; use --no-fp32 to keep native)",
    )
    parser.add_argument(
        "--group-readable",
        action="store_true",
        help="Make written outputs group-readable (chmod g+r)",
    )
    args = parser.parse_args()

    input_path: Path = args.input.resolve()
    output_dir: Path = args.output.resolve()

    source = VTISource(str(input_path), file_pattern=args.pattern, fp32=args.fp32)

    n_files = len(source)
    print(f"Discovered {n_files} .vti file(s)")
    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    if n_files:
        rel = Path(source.relative_path(0))
        example_out = output_dir / rel.parent / f"{rel.stem}.grid"
        print(f"Example: {rel} -> {example_out}")

    pipeline = source.write(
        GridSidecarSink(
            output_dir=str(output_dir),
            naming_template=args.naming_template,
            group_readable=args.group_readable,
        )
    )

    indices = range(min(args.limit, n_files)) if args.limit is not None else None
    if indices is not None:
        print(f"Limiting to first {len(indices)} file(s)")

    backend = "process_pool" if args.workers > 1 else "sequential"
    results = run_pipeline(pipeline, n_jobs=args.workers, backend=backend, indices=indices)

    print(f"\nConverted {len(results)} file(s)")
    for paths in results:
        for p in paths:
            print(f"  {p}")


if __name__ == "__main__":
    main()
