# How to curate data for Crash Simulation models with PhysicsNeMo-Curator

This document describes how to use the Crash Simulation data processing pipeline in PhysicsNeMo Curator.

For detailed information about data formats and the processing pipeline, see the [Crash Data Processing Reference](./Crash_Data_Processing_Reference.md).

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
python run_etl.py                                               \
    --config-dir=examples/structural_mechanics/crash/config     \
    --config-name=crash_etl                                     \
    etl.source.input_dir=/data/crash_sims/                      \
    serialization_format=vtp                                    \
    serialization_format.sink.output_dir=/data/crash_processed/ \
    etl.processing.num_processes=4
```

### Input Data Requirements

Your input directory should contain run folders with d3plot files:

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

For details on input data format, see the
[Input Data Structure](./Crash_Data_Processing_Reference.md#input-data-structure) section of the
reference.

## Configuration Options

The main configuration file is [`crash_etl.yaml`](./config/crash_etl.yaml).

### Key Parameters

- **`etl.source.input_dir`**: Directory containing run folders with d3plot files (required)

- **`serialization_format`**: Output format selection (default: `vtp`)
  - Options: `vtp` or `zarr`
  - See [serialization_format config files](./config/serialization_format/) for format-specific options

- **`serialization_format.sink.output_dir`**: Directory where processed files will be written (required)

- **`etl.processing.num_processes`**: Number of parallel processes (default: 12)

- **`etl.transformations.crash_transform.wall_threshold`**: Threshold for filtering wall nodes
  (default: 1.0)
  - Nodes with displacement variation < threshold are considered "wall" nodes and filtered out
  - Higher values = more aggressive filtering (more nodes removed)

### Output Format Selection

You can choose between two output formats using the `serialization_format` config group.

**Key concept**: The `serialization_format` is a Hydra config group that determines the sink
configuration. To override sink parameters, use the pattern
`serialization_format.sink.<parameter>=<value>`.

#### VTP Output (for visualization and training)

Best for: Visualization (ParaView, PyVista), interactive analysis, and smaller datasets

```bash
python run_etl.py                                           \
    --config-dir=examples/structural_mechanics/crash/config \
    --config-name=crash_etl \
    serialization_format=vtp \
    etl.source.input_dir=/data/crash_sims \
    serialization_format.sink.output_dir=/data/crash_processed_vtp \
    serialization_format.sink.overwrite_existing=false
```

**VTP-specific parameters:**

- `time_step`: Time step between simulation frames in seconds (default: 0.005)
- `overwrite_existing`: Whether to overwrite existing output files (default: true)

See [`config/serialization_format/vtp.yaml`](./config/serialization_format/vtp.yaml) for configuration.

#### Zarr Output (for ML training)

Best for: Large-scale ML training, cloud storage, and efficient chunked access

```bash
python run_etl.py                                           \
    --config-dir=examples/structural_mechanics/crash/config \
    --config-name=crash_etl \
    serialization_format=zarr \
    etl.source.input_dir=/data/crash_sims \
    serialization_format.sink.output_dir=/data/crash_processed_zarr \
    serialization_format.sink.compression_level=5 \
    serialization_format.sink.chunk_size_mb=2.0
```

**Zarr-specific parameters:**

- `compression_level`: Compression level (1-9, higher = more compression, default: 3)
- `compression_method`: Compression codec (default: "zstd")
- `chunk_size_mb`: Target chunk size in MB for automatic chunking (default: 1.0)
  - Smaller values: Better for random access, more metadata overhead
  - Larger values: Better for sequential reads, less metadata overhead
  - Warnings are issued for very small (<0.1 MB) or very large (>100 MB) values
- `overwrite_existing`: Whether to overwrite existing output stores (default: true)

See [`config/serialization_format/zarr.yaml`](./config/serialization_format/zarr.yaml) for configuration.

For detailed output format specifications, see the
[Output Data Structure](./Crash_Data_Processing_Reference.md#output-data-structure) section of
the reference.

## Advanced Configuration

### Adjusting Wall Filtering

To change the wall node filtering threshold:

```bash
python run_etl.py                                           \
    --config-dir=examples/structural_mechanics/crash/config \
    --config-name=crash_etl \
    etl.source.input_dir=/data/crash_sims \
    serialization_format.sink.output_dir=/data/output \
    etl.transformations.crash_transform.wall_threshold=2.0
```

Higher threshold = more aggressive filtering (more nodes removed)

Typical range: 0.5 - 2.0

### Incremental Processing

To avoid reprocessing existing files:

```bash
python run_etl.py                                           \
    --config-dir=examples/structural_mechanics/crash/config \
    --config-name=crash_etl \
    etl.source.input_dir=/data/crash_sims \
    serialization_format.sink.output_dir=/data/output \
    serialization_format.sink.overwrite_existing=false
```

The pipeline will skip runs that already have output files.

### Performance Tuning

**Parallelization:**

```bash
# Process more runs in parallel (adjust based on available memory)
etl.processing.num_processes=16
```

**Zarr Chunk Size:**

```bash
# Optimize for sequential access (larger chunks)
serialization_format.sink.chunk_size_mb=50.0

# Optimize for random access (smaller chunks)
serialization_format.sink.chunk_size_mb=1
```

## Extending the Pipeline

The ETL pipeline is designed to be customizable:

### Custom Transformations

Modify `CrashDataTransformation` to add new filtering or feature extraction logic:

```python
from physicsnemo_curator.etl.base import BaseDataTransformation
from schemas import CrashExtractedDataInMemory

class CustomTransformation(BaseDataTransformation):
    def transform(self, data: CrashExtractedDataInMemory) -> CrashExtractedDataInMemory:
        # Apply custom logic
        # e.g., filter by element quality, add derived features
        return modified_data
```

Add to config:

```yaml
etl:
  transformations:
    crash_transform:
      _target_: data_transformations.CrashDataTransformation
      wall_threshold: 1.0
    custom_transform:
      _target_: my_module.CustomTransformation
      # custom parameters
```

### Additional Output Formats

Create new data source classes for different output formats. See `data_sources.py` for examples
of `CrashVTPDataSource` and `CrashZarrDataSource`.

### Custom Features

Add new node or element features by:

1. Extracting data in `CrashD3PlotDataSource.read_file()`
2. Adding fields to `CrashExtractedDataInMemory` schema
3. Updating transformation logic if needed
4. Adding feature to sink write methods

See the [Field Variables](./Crash_Data_Processing_Reference.md#field-variables) section of the reference for details.

## Model Training

After processing your data with Curator, you can train crash simulation models using PhysicsNeMo:

- [PhysicsNeMo Crash Example](https://github.com/NVIDIA/physicsnemo/blob/main/examples/structural_mechanics/crash/README.md)

The PhysicsNeMo training pipeline provides readers for both VTP and Zarr formats, allowing you to use
either output format directly.

## Troubleshooting

### Issue: "No d3plot files found"

**Solution**: Ensure your input directory contains subdirectories with d3plot files. The pipeline
looks for `input_dir/*/d3plot`.

### Issue: "No cells left after filtering"

**Solution**: Wall threshold is too low, causing all nodes to be filtered. Try increasing
`wall_threshold` parameter (e.g., from 1.0 to 2.0).

### Issue: Out of memory

**Solution**: Reduce `num_processes` to process fewer runs in parallel. Each process loads one
run into memory.

## See Also

- [Crash Data Processing Reference](./Crash_Data_Processing_Reference.md) - Detailed data format
  specifications
- [PhysicsNeMo Crash Example](https://github.com/NVIDIA/physicsnemo/blob/main/examples/structural_mechanics/crash/README.md) -
  Model training guide
- [PhysicsNeMo-Curator Documentation](https://github.com/NVIDIA/physicsnemo-curator) - General ETL
  framework documentation
