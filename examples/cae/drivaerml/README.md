# DrivAerML End-to-End ETL

Process [DrivAerML](https://huggingface.co/datasets/neashton/drivaerml)
automotive CFD meshes through a complete Source → Filter → Sink pipeline.

The pipeline:

1. **MeshStatsFilter** — computes per-field statistics (mean, std,
   skewness, kurtosis) using Welford accumulators that merge across
   parallel workers.
2. **PrecisionFilter** — casts float64 fields to float32.
3. **RandomPermutationFilter** — shuffles point and cell ordering to
   remove spatial bias before training.
4. **MeshSink** — writes each mesh in PhysicsNeMo's native format
   (`.pdmsh` for DomainMesh, `.pmsh` for plain Mesh), grouped into
   per-run subdirectories.

## Prerequisites

```bash
uv sync --extra mesh --extra loky

# or with pip
pip install physicsnemo-curator[mesh,loky]
pip install huggingface_hub[cli]
```

## Download the Dataset

DrivAerML is hosted on HuggingFace at
[neashton/drivaerml](https://huggingface.co/datasets/neashton/drivaerml).
The volume files are large (~50 GB each), so it is recommended to
explicitly manage the download with the HuggingFace CLI rather than
relying on streaming.

> **Warning:** The full dataset is approximately **12 TB** (500 runs,
> ~24 GB each). A single run downloads ~24 GB of volume, surface, and
> geometry data. Ensure you have sufficient disk space before
> downloading.

```bash
# Download specific runs (adjust --include for your needs)
hf download neashton/drivaerml \
    --repo-type dataset \
    --include "run_1/*" "run_2/*" \
    --local-dir input/drivaerml
```

This creates an `input/drivaerml/` directory with per-run subdirectories
containing mesh data, geometry, and metadata files:

```text
input/drivaerml/
├── run_1/
│   ├── volume_1.vtu           # Volume mesh (cell centroids + fields)
│   ├── boundary_1.vtp         # Boundary surface (triangulated)
│   ├── drivaer_1.stl          # Vehicle geometry (multi-part STL)
│   ├── geo_parameters_1.csv   # Geometric parameters
│   ├── geo_ref_1.csv          # Reference geometry
│   ├── force_mom_1.csv        # Force/moment data
│   ├── force_mom_constref_1.csv
│   ├── images/                # Visualization images
│   └── slices/                # Cross-section slices (.vtp)
│       ├── xNormal_*.vtp      # X-normal slice planes
│       ├── yNormal_*.vtp      # Y-normal slice planes
│       └── zNormal_*.vtp      # Z-normal slice planes
├── run_2/
│   └── ...
└── ...
```

## Usage

```bash
# Basic usage (reads from ./input/drivaerml, writes to ./output/drivaerml)
python main.py

# Custom input/output directories
python main.py --input /path/to/drivaerml --output /path/to/output

# Limit to specific number of workers
python main.py --workers 4
```

## Output Structure

```text
output/drivaerml/
├── stats.parquet                        # Per-field statistics (merged)
├── run_1/
│   ├── domain_1.pdmsh/                  # DomainMesh: interior + surface
│   ├── drivaer_1.stl.pmsh/              # STL geometry
│   └── drivaer_1_single_solid.stl.pmsh/ # Merged STL
├── run_2/
│   ├── domain_2.pdmsh/
│   ├── drivaer_2.stl.pmsh/
│   └── drivaer_2_single_solid.stl.pmsh/
└── ...
```
