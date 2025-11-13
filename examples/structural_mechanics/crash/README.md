# How to curate data for Crash Simulation models with PhysicsNeMo-Curator

This document describes the Crash Simulation data processing pipeline in PhysicsNeMo Curator.

## Overview

The Crash Simulation ETL pipeline processes LS-DYNA crash simulation data for machine learning training. It:

**Reads:** LS-DYNA simulation data (d3plot binary format)

- d3plot binary files containing mesh and displacement data
- Optional .k files for part thickness information
- Multi-timestep deformation data

**Transforms:** Raw simulation data into ML-optimized format

- Filters wall nodes (nodes with minimal displacement)
- Computes node thickness from part/section definitions
- Builds edge connectivity from mesh
- Remaps and compacts node indices

**Outputs:** Training-ready datasets in VTP or Zarr format

- **VTP format**: Single file per run with all timesteps as displacement fields
- **Zarr format**: Compressed, chunked format optimized for ML training
- Compatible with PhysicsNeMo training workflows

This pipeline handles the complex data engineering required to convert raw LS-DYNA outputs into
datasets suitable for training AI models for crash simulation applications.

## Running the Curator

Example command line to launch Curator for crash simulation data:

```bash
export PYTHONPATH=$PYTHONPATH:examples &&
physicsnemo-curator-etl                                         \
    --config-dir=examples/structural_mechanics/crash/config     \
    --config-name=crash_etl                                     \
    etl.source.input_dir=/data/crash_sims/                      \
    serialization_format=vtp                                    \
    serialization_format.sink.output_dir=/data/crash_processed/ \
    etl.processing.num_processes=4
```

### Configuration Options

The main configuration file is [`crash_etl.yaml`](./config/crash_etl.yaml).

#### Key Parameters

- **`etl.source.input_dir`**: Directory containing run folders with d3plot files
  - Expected structure: `input_dir/Run100/d3plot`, `input_dir/Run100/*.k`

- **`serialization_format`**: The default is VTP, but users can use this CLI flag to toggle between VTP and Zarr.
Please refer to the [serialization_format config files](./config/serialization_format/) to understand the necessary config.

- **`serialization_format.sink.output_dir`**: Directory where processed files will be written

- **`etl.processing.num_processes`**: Number of parallel processes (default: 4)

- **`etl.transformations.crash_transform.wall_threshold`**: Threshold for filtering wall nodes (default: 1.0)
  - Nodes with displacement variation < threshold are considered "wall" nodes and filtered out

#### Output Formats

You can choose between two output formats using the `serialization_format` config group. The default
format is VTP. To switch to Zarr, use `serialization_format=zarr` on the command line.

**Key concept**: The `serialization_format` is a Hydra config group that determines the sink
configuration. To override sink parameters, use the pattern
`serialization_format.sink.<parameter>=<value>`.

##### VTP Output (for visualization and training)

**Command line:**

```bash
physicsnemo-curator-etl \
    --config-dir=examples/structural_mechanics/crash/config \
    --config-name=crash_etl \
    serialization_format=vtp \
    etl.source.input_dir=/data/crash_sims \
    serialization_format.sink.output_dir=/data/crash_processed_vtp \
    serialization_format.sink.overwrite_existing=false
```

**Config:** See [`config/serialization_format/vtp.yaml`](./config/serialization_format/vtp.yaml)

**VTP-specific parameters:**

- `time_step`: Time step between simulation frames in seconds (default: 0.005)
- `overwrite_existing`: Whether to overwrite existing output files (default: true)

Output structure:

```bash
crash_processed_vtp/
├── Run100.vtp
├── Run101.vtp
└── ...
```

Each VTP file contains:

- Reference coordinates at t=0
- Displacement fields for each timestep: `displacement_t0.000`, `displacement_t0.005`, etc.
- Node thickness values

##### Zarr Output (for ML training)

**Command line:**

```bash
physicsnemo-curator-etl \
    --config-dir=examples/structural_mechanics/crash/config \
    --config-name=crash_etl \
    serialization_format=zarr \
    etl.source.input_dir=/data/crash_sims \
    serialization_format.sink.output_dir=/data/crash_processed_zarr \
    serialization_format.sink.compression_level=5 \
    serialization_format.sink.chunk_size_mb=2.0
```

**Config:** See [`config/serialization_format/zarr.yaml`](./config/serialization_format/zarr.yaml)

**Zarr-specific parameters:**

- `compression_level`: Compression level (1-9, higher = more compression, default: 3)
- `compression_method`: Compression codec (default: "zstd")
- `chunk_size_mb`: Target chunk size in MB for automatic chunking (default: 1.0)
  - Smaller values: Better for random access, more metadata overhead
  - Larger values: Better for sequential reads, less metadata overhead
  - Warnings are issued for very small (<0.1 MB) or very large (>100 MB) values
- `overwrite_existing`: Whether to overwrite existing output stores (default: true)

Output structure:

```bash
crash_processed_zarr/
├── Run100.zarr/
│   ├── mesh_pos              # (timesteps, nodes, 3)
│   ├── node_thickness        # (nodes,)
│   ├── edges                 # (num_edges, 2)
│   └── mesh_connectivity_*   # Ragged array format
├── Run101.zarr/
└── ...
```

**When to use each format:**

- **VTP**: Best for visualization (ParaView, PyVista), interactive analysis, and smaller datasets
- **Zarr**: Best for large-scale ML training, cloud storage, and when you need efficient chunked access

### Transformation Pipeline

The ETL pipeline applies the following transformations:

1. **Node Filtering** (`CrashDataTransformation`):
   - Identifies wall nodes (low displacement variation)
   - Filters out wall nodes to reduce dataset size
   - Remaps mesh connectivity to filtered indices
   - Compacts to contiguous node indices

2. **Thickness Computation**:
   - Parses .k files for part/section thickness definitions
   - Maps part thickness to node thickness
   - Averages thickness for nodes connected to multiple elements

3. **Edge Building**:
   - Extracts unique edges from mesh connectivity
   - Creates edge list for graph neural network training

### Advanced Configuration

#### Adjusting Wall Filtering

To change the wall node filtering threshold:

```bash
physicsnemo-curator-etl \
    --config-dir=examples/structural_mechanics/crash/config \
    --config-name=crash_etl \
    etl.source.input_dir=/data/crash_sims \
    serialization_format.sink.output_dir=/data/output \
    etl.transformations.crash_transform.wall_threshold=2.0
```

Higher threshold = more aggressive filtering (more nodes removed)

#### Incremental Processing

To avoid reprocessing existing files:

```bash
physicsnemo-curator-etl \
    --config-dir=examples/structural_mechanics/crash/config \
    --config-name=crash_etl \
    etl.source.input_dir=/data/crash_sims \
    serialization_format.sink.output_dir=/data/output \
    serialization_format.sink.overwrite_existing=false
```

The pipeline will skip runs that already have output files.

## Input Data Requirements

### Directory Structure

The input directory should contain run folders with d3plot files:

```bash
input_dir/
├── Run100/
│   ├── d3plot          # Required: binary mesh/displacement data
│   └── run100.k        # Optional: part thickness definitions
├── Run101/
│   ├── d3plot
│   └── run101.k
└── ...
```

### File Formats

- **d3plot**: LS-DYNA binary format containing:
  - Node coordinates
  - Node displacements over time
  - Shell element connectivity
  - Part IDs for each element

- **.k files**: LS-DYNA keyword format containing:
  - `*PART` definitions linking parts to sections
  - `*SECTION_SHELL` definitions with thickness values

If no .k file is found, node thickness defaults to 0.

## Output Data Format

### VTP Format

Single VTP file per run compatible with PyVista and ParaView:

```python
import pyvista as pv

mesh = pv.read('Run100.vtp')
print(mesh.point_data.keys())
# ['thickness', 'displacement_t0.000', 'displacement_t0.005', ...]

# Get coordinates and displacement at timestep 5
coords = mesh.points
disp_t5 = mesh.point_data['displacement_t0.025']
pos_t5 = coords + disp_t5
```

### Zarr Format

Compressed, chunked arrays optimized for ML training:

```python
import zarr

store = zarr.open('Run100.zarr', mode='r')
print(store.tree())

# Access data
mesh_pos = store['mesh_pos'][:]  # (timesteps, nodes, 3)
thickness = store['node_thickness'][:]  # (nodes,)
edges = store['edges'][:]  # (num_edges, 2)

# Metadata
print(store.attrs['num_timesteps'])
print(store.attrs['num_nodes'])
```

## Model Training

After processing your data with Curator, you can train crash simulation models using PhysicsNeMo. Please refer to [these instructions](physicsnemo-main-repo/examples/structural_mechanics/crash/README.md).

## Extending the Pipeline

The ETL pipeline is designed to be customizable:

1. **Custom Transformations**: Modify `CrashDataTransformation` to add new filtering or feature extraction logic

2. **Additional Output Formats**: Create new DataSource classes for different output formats

3. **Custom Features**: Add new node or element features by extending the data reading logic

See the [PhysicsNeMo-Curator documentation](https://github.com/NVIDIA/physicsnemo-curator) for more details on extending the pipeline.

## Troubleshooting

### Issue: "No d3plot files found"

**Solution**: Ensure your input directory contains subdirectories with d3plot files

### Issue: "No cells left after filtering"

**Solution**: Wall threshold is too low. Try increasing `wall_threshold` parameter.

### Issue: Out of memory

**Solution**: Reduce `num_processes` or process fewer runs at once
