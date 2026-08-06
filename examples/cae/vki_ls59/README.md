# VKI-LS59 PLAID ETL

Process the [VKI-LS59](https://huggingface.co/datasets/PLAID-datasets/VKI-LS59) turbine
cascade dataset from the [PLAID library](https://plaid-lib.github.io) through a complete
Source → Filter → Sink pipeline.

The VKI-LS59 dataset contains 839 2-D compressible RANS simulations (Spalart-Allmaras
turbulence model, Broadcast solver) of a transonic turbine cascade at varying inlet angle
and outlet Mach number. Each sample provides:

| Component | Description |
|-----------|-------------|
| `interior` | Volume mesh — 2-D point cloud / QUAD_4 triangulated grid (`Base_2_2`) |
| `boundaries["surface"]` | Blade surface — BAR_2 edge segments (`Base_1_2`) |
| `global_data` | 8 scalar quantities: `angle_in`, `mach_out`, `Pr`, `Q`, `Tr`, `angle_out`, `eth_is`, `power` |

The pipeline:

1. **PlaidSource** — reads from the PLAID HuggingFace disk format; auto-downloads
   from `PLAID-datasets/VKI-LS59` when the local folder is absent.
2. **MeshStatsFilter** — computes per-field statistics (mean, std, skewness, kurtosis)
   across the dataset using Welford accumulators that merge across parallel workers.
3. **MeshSink** — writes each `DomainMesh` in PhysicsNeMo's native `.pmsh` format.

> **Note on 2-D connectivity**: `PlaidSource` handles this 2-D dataset natively.
> QUAD_4 volume elements are automatically split into pairs of triangles and BAR_2
> surface edges are loaded as 2-node line segments. Future work will add support for
> configurable triangulation strategies and boundary identification from the CGNS
> element connectivity.

## Prerequisites

```bash
uv sync --extra mesh --extra loky
```

## Dataset

The dataset is downloaded automatically on first run. To pre-download manually:

```bash
uv run python - <<'EOF'
from plaid.storage import download_from_hub
download_from_hub("PLAID-datasets/VKI-LS59", "./input/vki_ls59")
EOF
```

## Usage

```bash
# Basic — auto-downloads dataset, processes train split
python main.py

# Specify input/output directories
python main.py --input /data/vki_ls59 --output /data/output/vki_ls59

# Process test split
python main.py --split test

# Point-cloud mode (no connectivity)
python main.py --point-cloud

# Limit to first 50 samples with 4 workers
python main.py --n-samples 50 --workers 4

# Verbose logging
python main.py --verbose
```

## Output structure

```text
output/vki_ls59/
├── stats.parquet          # Per-field statistics (merged across workers)
├── sample_0/
│   └── sample_0.pmsh/     # DomainMesh (interior + surface boundary)
├── sample_1/
│   └── sample_1.pmsh/
└── ...
```

## DomainMesh layout

Each `.pmsh` file contains a `DomainMesh` with:

```python
dm.interior            # Mesh — volume, (N_vol, 2) coords, optional triangles
dm.boundaries["surface"]  # Mesh — blade surface, (N_surf, 2) coords, BAR_2 edges
dm.global_data         # TensorDict — 8 scalar simulation parameters
```

## References

- [VKI-LS59 on HuggingFace](https://huggingface.co/datasets/PLAID-datasets/VKI-LS59)
- [PLAID library](https://plaid-lib.github.io)
- [PLAID GitHub](https://github.com/PLAID-lib/plaid)
- [PhysicsNeMo-Curator](https://github.com/NVIDIA/physicsnemo-curator)
