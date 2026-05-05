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
uv sync --group mesh

# or with pip
pip install physicsnemo-curator[mesh]
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
huggingface-cli download neashton/drivaerml \
    --repo-type dataset \
    --include "run_1/*" "run_2/*" \
    --local-dir input/drivaerml
```

This creates an `input/drivaerml/` directory with per-run subdirectories
containing `.vtu` (volume), `.vtp` (surface), and `.stl` (geometry)
files:

```text
input/drivaerml/
├── run_1/
│   ├── vol_data.vtu          # Volume mesh (cell centroids + fields)
│   ├── surf_data.vtp         # Boundary surface (triangulated)
│   └── drivaer.stl           # Vehicle geometry
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
