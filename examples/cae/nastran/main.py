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

"""Curate Nastran models and modal OP2 results into native or VTK meshes."""

import argparse

from physicsnemo_curator.domains.mesh import MeshSink, MeshVTKSink, NastranSource
from physicsnemo_curator.run import run_pipeline


def main() -> None:
    """Run a Nastran deck/OP2 curation pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Deck file or directory of cases")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--format",
        choices=["pmsh", "vtu", "vtk", "vtp"],
        default="pmsh",
        help="Output format; vtp extracts the boundary surface",
    )
    parser.add_argument("--file-pattern", default="**/*", help="Deck discovery glob")
    parser.add_argument("--op2-pattern", default="", help="Relative OP2 template; {stem} is the deck stem")
    parser.add_argument("--encoding", default="latin1", help="Deck encoding, e.g. cp932")
    parser.add_argument("--geometry-only", action="store_true", help="Skip OP2 results")
    parser.add_argument("--punch", action="store_true", help="Bulk data without executive/case control")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    source = NastranSource(
        args.input,
        file_pattern=args.file_pattern,
        op2_pattern=args.op2_pattern,
        encoding=args.encoding,
        read_results=not args.geometry_only,
        punch=args.punch,
    )
    naming = "{relpath}/{stem}_{seq}"
    sink = (
        MeshSink(output_dir=args.output, naming_template=naming)
        if args.format == "pmsh"
        else MeshVTKSink(output_dir=args.output, file_format=args.format, naming_template=naming)
    )
    pipeline = source.write(sink)
    results = run_pipeline(pipeline, backend="process_pool", n_jobs=args.workers, use_tui=False)
    print(f"Curated {len(results)} decks into {sum(len(paths) for paths in results)} meshes")


if __name__ == "__main__":
    main()
