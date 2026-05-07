# AhmedML End-to-End ETL

Process [AhmedML](https://huggingface.co/datasets/neashton/ahmedml) automotive
CFD meshes through a complete Source → Filter → Sink pipeline.

The AhmedML dataset contains 500 geometric variations of the
[Ahmed Car Body](https://en.wikipedia.org/wiki/Ahmed_body) with transient
hybrid RANS-LES CFD simulations (OpenFOAM v2212, ~20 M cells per case).

The pipeline:

1. **MeshStatsFilter** — computes per-field statistics (mean, std,
   skewness, kurtosis) using Welford accumulators that merge across
   parallel workers.
2. **PrecisionFilter** — casts float64 fields to float32.
3. **RandomPermutationFilter** — shuffles point ordering to remove
   spatial bias before training.
4. **MeshSink** — writes each mesh in PhysicsNeMo's native `.pmsh`
   format, grouped into per-run subdirectories.

## Prerequisites

```bash
uv sync --extra mesh --extra loky

# or with pip
pip install physicsnemo-curator[mesh]
pip install huggingface_hub[cli]
```

## Download the Dataset

AhmedML is hosted on HuggingFace at
[neashton/ahmedml](https://huggingface.co/datasets/neashton/ahmedml).
The vtp files are large, so it is recommended to explicitly manage the download with
the HuggingFace CLI rather than relying on streaming while the pipeline is running.

```bash
hf download neashton/ahmedml \
    --repo-type dataset \
    --include "run_1/*" "run_2/*" "run_3/*" \
    --local-dir input/ahmedml
```

Each run directory contains:

```text
run_<i>/
├── ahmed_<i>.stl           # STL geometry
├── boundary_<i>.vtp       # Surface mesh with flow fields (~83 MB)
├── volume_<i>.vtu         # Volumetric field data (~5.6 GB)
├── force_mom_<i>.csv      # Force/moment coefficients (cd, cl)
├── force_mom_varref_<i>.csv  # Variant reference force coefficients
├── geo_parameters_<i>.csv # Geometric parameters (8 columns)
└── slices/                # x/y/z-normal slice planes (VTP)
    ├── x_*.vtp
    ├── y_*.vtp
    └── z_*.vtp
```

## Usage

```bash
# Basic usage (reads from ./input/ahmedml, writes to ./output/ahmedml)
python main.py

# Custom input/output directories
python main.py --input /path/to/ahmedml --output /path/to/output

# Limit to specific number of workers
python main.py --workers 4

# Process in multi mode (DomainMesh + STL)
python main.py --mesh-type multi --mesh-parts domain stl
```

## Output Structure

```text
output/ahmedml/
├── stats.parquet          # Per-field statistics (merged)
├── run_1/
│   └── boundary_1.pmsh/  # Processed surface mesh
├── run_2/
│   └── boundary_2.pmsh/
└── ...
```

## Mesh Types

| Type | Description | Size per Run |
|------|-------------|--------------|
| `boundary` | Surface mesh with flow fields (VTP) | ~83 MB |
| `volume` | Volumetric field data (VTU) | ~5.6 GB |
| `slices` | x/y/z-normal slice planes (VTP) | Variable |
| `multi` | DomainMesh (interior + boundary + STL) | ~5.7 GB |

All modes attach CSV metadata (force coefficients and geometric
parameters) as `global_data` on every yielded mesh.

## See Also

- [AhmedML Dataset](https://huggingface.co/datasets/neashton/ahmedml) — HuggingFace page
- [arXiv:2407.20801](https://arxiv.org/abs/2407.20801) — Dataset paper
- [DrivAerML ETL](../drivaerml_etl/) — Similar pipeline for DrivAerML dataset
- [PhysicsNeMo-Curator Documentation](https://github.com/NVIDIA/physicsnemo-curator) — Framework docs
