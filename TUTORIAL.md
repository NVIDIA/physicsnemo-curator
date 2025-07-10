# PhysicsNeMo-Curator Tutorial

## Overview

This section contains a tutorial for using PhysicsNeMo-Curator to create a dataset.
This tutorial will show how to use the PhysicsNeMo-Curator ETL pipeline to:

1. Extract physics simulation data from a dataset
2. Transform the data into an optimized, AI model training ready format
3. Write the transformed data to disk efficiently

## Create a dataset

PhysicsNeMo-Curator works only with well-defined formats and schemas.
As such, defining that is a necessary first step.
Next, we'll create a custom dataset, in a custom schema, format and storage system.

### Step 1: Define the schema, format, storage system

For this tutorial, we'll create a simple simulation dataset using:

**Format**: HDF5
**Storage**: Local filesystem
**Schema**: This is the structure for each simulation run
(xyz indicates the run number):

```bash
run_xyz.h5
├── /fields/
│   ├── temperature          # Dataset: (N,) float32 - scalar temperature field
│   └── velocity             # Dataset: (N, 3) float32 - 3D velocity vectors
├── /geometry/
│   └── coordinates          # Dataset: (N, 3) float32 - spatial coordinates (x,y,z)
└── /metadata/
    ├── timestamp            # Attribute: string - when simulation was run
    ├── num_points           # Attribute: int - number of data points
    ├── temperature_units    # Attribute: string - "Kelvin"
    ├── velocity_units       # Attribute: string - "m/s"
    └── simulation_params/   # Group containing simulation parameters
        └── total_time      # Attribute: float - total simulation time
```

**Data Description**:

- Each simulation run is one HDF5 file
- `N` represents the number of spatial points in the simulation (varies per case)
- Temperature is a scalar field representing thermal distribution
- Velocity is a 3D vector field representing fluid flow
- Coordinates define the spatial location of each data point
- Metadata includes simulation parameters and units for reproducibility

### Step 2: Generate random data

We'll create a small script to generate 5 simulation runs with random data.
Each file will contain about 1000 data points to keep it lightweight.

First, install the required dependency:

```bash
pip install h5py
```

Create a script called `generate_sample_data.py`:

```python
import h5py
import numpy as np
from datetime import datetime
import os

def generate_simulation_data(run_number, num_points=1000):
    """Generate random simulation data for one run."""

    # Generate random 3D coordinates in a unit cube
    coordinates = np.random.uniform(-1.0, 1.0, size=(num_points, 3)).astype(np.float32)

    # Generate temperature field (scalar, range 250-350 K)
    temperature = np.random.uniform(250.0, 350.0, size=num_points).astype(np.float32)

    # Generate velocity field (3D vectors, range -5 to 5 m/s)
    velocity = np.random.uniform(-5.0, 5.0, size=(num_points, 3)).astype(np.float32)

    return coordinates, temperature, velocity

def create_hdf5_file(run_number, output_dir="tutorial_data"):
    """Create one HDF5 file for a simulation run."""

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate data
    coordinates, temperature, velocity = generate_simulation_data(run_number)
    num_points = len(coordinates)

    # Create HDF5 file
    filename = f"run_{run_number:03d}.h5"
    filepath = os.path.join(output_dir, filename)

    with h5py.File(filepath, 'w') as f:
        # Create groups
        fields_group = f.create_group('fields')
        geometry_group = f.create_group('geometry')
        metadata_group = f.create_group('metadata')
        sim_params_group = metadata_group.create_group('simulation_params')

        # Store field data
        fields_group.create_dataset('temperature', data=temperature)
        fields_group.create_dataset('velocity', data=velocity)

        # Store geometry data
        geometry_group.create_dataset('coordinates', data=coordinates)

        # Store metadata attributes
        metadata_group.attrs['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metadata_group.attrs['num_points'] = num_points
        metadata_group.attrs['temperature_units'] = 'Kelvin'
        metadata_group.attrs['velocity_units'] = 'm/s'

        # Store simulation parameters
        sim_params_group.attrs['total_time'] = np.random.uniform(1.0, 10.0)  # Random simulation time

    print(f"Created {filepath} with {num_points} data points")

def main():
    """Generate sample dataset with 5 simulation runs."""
    print("Generating sample physics simulation dataset...")

    # Generate 5 runs
    for run_num in range(1, 6):
        create_hdf5_file(run_num)

    print(f"\nDataset generation complete!")
    print(f"Created 5 HDF5 files in the 'tutorial_data/' directory")
    print(f"Each file contains ~1000 data points with temperature and velocity fields")

if __name__ == "__main__":
    main()
```

Run the script to generate your sample dataset:

```bash
python generate_sample_data.py
```

This will create a `tutorial_data/` directory with 5 files:

- `run_001.h5`
- `run_002.h5`
- `run_003.h5`
- `run_004.h5`
- `run_005.h5`

Now we're ready to implement the ETL pipeline!

## Implement the ETL pipeline

The PhysicsNeMo-Curator ETL pipeline consists of four main components:

1. **DataSource** - Handles both reading input data AND writing output data
(serves as both source and sink)
2. **DataTransformation** - Transforms data from one format to another
3. **DatasetValidator** - Validates input data structure and content (optional)
4. **ParallelProcessor** - Orchestrates the processing files in parallel

**For this tutorial, our specific pipeline will be:**

1. **H5DataSource** (source) - Reads HDF5 files and extracts raw data
2. **H5ToZarrTransformation** - Converts it to a Zarr-compatible format
3. **ZarrDataSource** (sink) - Writes the transformed data to Zarr stores
4. **TutorialValidator** - Validates our HDF5 input files

The data flow:
`HDF5 files → H5DataSource → H5ToZarrTransformation → ZarrDataSource → Zarr stores`

**Important**:
Notice that we use different DataSource classes for reading and writing -
one specialized for HDF5 input, another for Zarr output.
This shows how you can mix and match different data sources in the same pipeline.
However, notice also that `DataSource` serves dual purposes -
one instance of a class can read your input data, while another
instance of the same class can write your output data.
This design allows the same class to handle both ends of the pipeline.

### Step 1: Implement dataset validation

First, we'll implement validation to ensure our input HDF5 files
meet the required schema and format.
This runs at the beginning of the pipeline to catch issues early.
Create a file called `tutorial_validator.py`:

```python
import h5py
from pathlib import Path
from typing import List

from physicsnemo_curator.etl.dataset_validators import DatasetValidator, ValidationError, ValidationLevel
from physicsnemo_curator.etl.processing_config import ProcessingConfig


class TutorialValidator(DatasetValidator):
    """Validator for HDF5 physics simulation dataset."""

    def __init__(self, cfg: ProcessingConfig, input_dir: str, validation_level: str = "fields"):
        """Initialize the validator.

        Args:
            cfg: Processing configuration
            input_dir: Directory containing HDF5 files to validate
            validation_level: "structure" or "fields"
        """
        super().__init__(cfg)
        self.input_dir = Path(input_dir)
        self.validation_level = ValidationLevel(validation_level)

        # Define our expected schema
        self.required_groups = ['/fields', '/geometry', '/metadata', '/metadata/simulation_params']
        self.required_datasets = {
            '/fields/temperature': {'shape_dims': 1, 'dtype': 'float'},
            '/fields/velocity': {'shape_dims': 2, 'expected_cols': 3, 'dtype': 'float'},
            '/geometry/coordinates': {'shape_dims': 2, 'expected_cols': 3, 'dtype': 'float'}
        }
        self.required_attributes = {
            '/metadata': ['timestamp', 'num_points', 'temperature_units', 'velocity_units'],
            '/metadata/simulation_params': ['total_time']
        }

    def validate(self) -> List[ValidationError]:
        """Validate the entire dataset.

        Returns:
            List of validation errors (empty if validation passes)
        """
        errors = []

        # Check if input directory exists
        if not self.input_dir.exists():
            errors.append(ValidationError(
                path=self.input_dir,
                message=f"Input directory does not exist: {self.input_dir}",
                level=self.validation_level
            ))
            return errors

        # Find all HDF5 files
        h5_files = list(self.input_dir.glob("*.h5"))

        if not h5_files:
            errors.append(ValidationError(
                path=self.input_dir,
                message="No HDF5 files found in input directory",
                level=self.validation_level
            ))
            return errors

        # Validate each file
        for h5_file in h5_files:
            file_errors = self.validate_single_item(h5_file)
            errors.extend(file_errors)

        return errors

    def validate_single_item(self, item: Path) -> List[ValidationError]:
        """Validate a single HDF5 file.

        Args:
            item: Path to HDF5 file to validate

        Returns:
            List of validation errors for this file
        """
        errors = []

        try:
            with h5py.File(item, 'r') as f:
                # Structure validation
                errors.extend(self._validate_structure(f, item))

                # Field validation (if requested and structure is valid)
                if self.validation_level == ValidationLevel.FIELDS and not errors:
                    errors.extend(self._validate_fields(f, item))

        except Exception as e:
            errors.append(ValidationError(
                path=item,
                message=f"Failed to open HDF5 file: {str(e)}",
                level=self.validation_level
            ))

        return errors

    def _validate_structure(self, f: h5py.File, file_path: Path) -> List[ValidationError]:
        """Validate HDF5 file structure."""
        errors = []

        # Check required groups exist
        for group_path in self.required_groups:
            if group_path not in f:
                errors.append(ValidationError(
                    path=file_path,
                    message=f"Missing required group: {group_path}",
                    level=self.validation_level
                ))

        # Check required datasets exist and have correct structure
        for dataset_path, requirements in self.required_datasets.items():
            if dataset_path not in f:
                errors.append(ValidationError(
                    path=file_path,
                    message=f"Missing required dataset: {dataset_path}",
                    level=self.validation_level
                ))
                continue

            dataset = f[dataset_path]

            # Check dimensions
            if len(dataset.shape) != requirements['shape_dims']:
                errors.append(ValidationError(
                    path=file_path,
                    message=f"Dataset {dataset_path} has wrong dimensions: expected {requirements['shape_dims']}D, got {len(dataset.shape)}D",
                    level=self.validation_level
                ))

            # Check column count for 2D arrays
            if 'expected_cols' in requirements and len(dataset.shape) >= 2:
                if dataset.shape[1] != requirements['expected_cols']:
                    expected_cols = requirements['expected_cols']
                    actual_cols = dataset.shape[1]
                    message = f"Dataset {dataset_path} has wrong number of columns: expected {expected_cols}, got {actual_cols}"
                    errors.append(ValidationError(
                        path=file_path,
                        message=message,
                        level=self.validation_level
                    ))

        # Check required attributes exist
        for group_path, attr_list in self.required_attributes.items():
            if group_path in f:
                group = f[group_path]
                for attr_name in attr_list:
                    if attr_name not in group.attrs:
                        errors.append(ValidationError(
                            path=file_path,
                            message=f"Missing required attribute: {group_path}@{attr_name}",
                            level=self.validation_level
                        ))

        return errors

    def _validate_fields(self, f: h5py.File, file_path: Path) -> List[ValidationError]:
        """Validate field data content."""
        errors = []

        # Check that datasets have consistent sizes
        if '/fields/temperature' in f and '/geometry/coordinates' in f:
            temp_size = f['/fields/temperature'].shape[0]
            coord_size = f['/geometry/coordinates'].shape[0]

            if temp_size != coord_size:
                errors.append(ValidationError(
                    path=file_path,
                    message=f"Inconsistent data sizes: temperature has {temp_size} points, coordinates has {coord_size} points",
                    level=self.validation_level
                ))

        # Check for reasonable data ranges
        if '/fields/temperature' in f:
            temp_data = f['/fields/temperature'][:]
            if temp_data.min() < 0 or temp_data.max() > 10000:  # Kelvin range check
                errors.append(ValidationError(
                    path=file_path,
                    message=f"Temperature data out of reasonable range: [{temp_data.min():.1f}, {temp_data.max():.1f}] K",
                    level=self.validation_level
                ))

        return errors
```

**Key Points About This Validator:**

1. **Schema Enforcement**: Validates the exact HDF5 structure we defined (groups, datasets, attributes)

2. **Two-Level Validation**:
   - "structure": Checks file structure and data types
   - "fields": Also validates data content and consistency

3. **Early Error Detection**: Runs before any processing to catch issues immediately

4. **Configurable Depth**: Can run quick structure checks or deep field validation

### Step 2: Implement data source

We'll create a simple DataSource that reads our HDF5 files. Create a file called `h5_data_source.py`:

```python
import h5py
import numpy as np
from pathlib import Path
from typing import Any, Dict, List

from physicsnemo_curator.etl.data_sources import DataSource
from physicsnemo_curator.etl.processing_config import ProcessingConfig


class H5DataSource(DataSource):
    """DataSource for reading HDF5 physics simulation files."""

    def __init__(self, cfg: ProcessingConfig, input_dir: str):
        """Initialize the H5 data source.

        Args:
            cfg: Processing configuration
            input_dir: Directory containing input HDF5 files
        """
        super().__init__(cfg)
        self.input_dir = Path(input_dir)

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory {self.input_dir} does not exist")

    def get_file_list(self) -> List[str]:
        """Get list of HDF5 files to process.

        Returns:
            List of filenames (without extension) to process
        """
        h5_files = list(self.input_dir.glob("*.h5"))
        filenames = [f.stem for f in h5_files]  # Remove .h5 extension

        self.logger.info(f"Found {len(filenames)} HDF5 files to process")
        return sorted(filenames)

    def read_file(self, filename: str) -> Dict[str, Any]:
        """Read one HDF5 file and extract all data.

        Args:
            filename: Base filename (without extension)

        Returns:
            Dictionary containing extracted data and metadata
        """
        filepath = self.input_dir / f"{filename}.h5"
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        self.logger.info(f"Reading {filepath}")

        data = {}

        with h5py.File(filepath, 'r') as f:
            # Read field data
            data['temperature'] = np.array(f['fields/temperature'])
            data['velocity'] = np.array(f['fields/velocity'])

            # Read geometry data
            data['coordinates'] = np.array(f['geometry/coordinates'])

            # Read metadata
            metadata = {}
            for key, value in f['metadata'].attrs.items():
                metadata[key] = value

            data['metadata'] = metadata
            data['filename'] = filename

        self.logger.info(f"Loaded data with {len(data['temperature'])} points")
        return data

    def write(self, data: Dict[str, Any], filename: str) -> None:
        """Not implemented - this DataSource only reads."""
        raise NotImplementedError("H5DataSource only supports reading")

    def should_skip(self, filename: str) -> bool:
        """Never skip files for reading."""
        return False
```

**Key Points:**

1. **Read-only**: This DataSource only implements reading from HDF5 files
2. **Simple extraction**: Reads temperature, velocity, coordinates, and basic metadata

### Step 3: Implement transformations

Now we'll create a transformation that converts our HDF5 data into a format optimized for Zarr storage. Create a file called `h5_to_zarr_transformation.py`:

```python
import numpy as np
from typing import Any, Dict
from numcodecs import Blosc

from physicsnemo_curator.etl.data_transformations import DataTransformation
from physicsnemo_curator.etl.processing_config import ProcessingConfig


class H5ToZarrTransformation(DataTransformation):
    """Transform HDF5 data into Zarr-optimized format."""

    def __init__(self, cfg: ProcessingConfig, chunk_size: int = 500, compression_level: int = 3):
        """Initialize the transformation.

        Args:
            cfg: Processing configuration
            chunk_size: Chunk size for Zarr arrays (number of points per chunk)
            compression_level: Compression level (1-9, higher = more compression)
        """
        super().__init__(cfg)
        self.chunk_size = chunk_size
        self.compression_level = compression_level

        # Set up compression
        self.compressor = Blosc(
            cname='zstd',  # zstd compression algorithm
            clevel=compression_level,
            shuffle=Blosc.SHUFFLE
        )

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform HDF5 data to Zarr-optimized format.

        Args:
            data: Dictionary from H5DataSource.read_file()

        Returns:
            Dictionary with Zarr-optimized arrays and metadata
        """
        self.logger.info(f"Transforming {data['filename']} for Zarr storage")

        # Get the number of points to determine chunking
        num_points = len(data['temperature'])

        # Calculate optimal chunks (don't exceed chunk_size)
        chunk_points = min(self.chunk_size, num_points)

        # Prepare arrays that will be written to Zarr stores
        zarr_data = {
            'temperature': {},
            'velocity': {},
            'coordinates': {},
            'velocity_magnitude': {},
        }

        # Temperature field (1D array)
        zarr_data['temperature'] = {
            'data': data['temperature'].astype(np.float32),
            'chunks': (chunk_points,),
            'compressor': self.compressor,
            'dtype': np.float32
        }

        # Velocity field (2D array: points x 3 components)
        zarr_data['velocity'] = {
            'data': data['velocity'].astype(np.float32),
            'chunks': (chunk_points, 3),
            'compressor': self.compressor,
            'dtype': np.float32
        }

        # Coordinates (2D array: points x 3 dimensions)
        zarr_data['coordinates'] = {
            'data': data['coordinates'].astype(np.float32),
            'chunks': (chunk_points, 3),
            'compressor': self.compressor,
            'dtype': np.float32
        }

        # Add some computed metadata useful for Zarr to existing metadata
        metadata = data['metadata']
        metadata['num_points'] = num_points
        metadata['chunk_size'] = chunk_points
        metadata['compression'] = 'zstd'
        metadata['compression_level'] = self.compression_level

        # Also add some simple derived fields
        # Temperature statistics
        metadata['temperature_min'] = float(np.min(data['temperature']))
        metadata['temperature_max'] = float(np.max(data['temperature']))
        metadata['temperature_mean'] = float(np.mean(data['temperature']))

        # Velocity magnitude
        velocity_magnitude = np.linalg.norm(data['velocity'], axis=1)
        zarr_data['velocity_magnitude'] = {
            'data': velocity_magnitude.astype(np.float32),
            'chunks': (chunk_points,),
            'compressor': self.compressor,
            'dtype': np.float32
        }
        metadata['velocity_max'] = float(np.max(velocity_magnitude))
        zarr_data['metadata'] = metadata

        return zarr_data
```

**Key Points About This Transformation:**

1. **Zarr Optimization**: Prepares data with chunks, compression, and proper dtypes for efficient Zarr storage

2. **Chunking Strategy**: Uses configurable chunk sizes optimized for the data size

3. **Compression**: Uses zstd compression with Blosc for good performance/size balance

4. **Derived data**: Adds derived fields like velocity magnitude and temperature statistics

5. **Metadata**: Adds technical metadata about chunking and compression settings

The output format is specifically designed to be consumed by a `ZarrDataSource` that will create the actual Zarr store structure.

### Step 4: Implement sink

Now we'll create a DataSource that writes to Zarr stores.
Each simulation run will be stored as a separate Zarr store for efficient individual access.
Create a file called `zarr_data_source.py`:

```python
import zarr
import shutil
from pathlib import Path
from typing import Any, Dict, List

from physicsnemo_curator.etl.data_sources import DataSource
from physicsnemo_curator.etl.processing_config import ProcessingConfig


class ZarrDataSource(DataSource):
    """DataSource for writing to Zarr stores."""

    def __init__(self, cfg: ProcessingConfig, output_dir: str):
        """Initialize the Zarr data source.

        Args:
            cfg: Processing configuration
            output_dir: Directory to write Zarr stores
        """
        super().__init__(cfg)
        self.output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_file_list(self) -> List[str]:
        """Not implemented - this DataSource only writes."""
        raise NotImplementedError("ZarrDataSource only supports writing")

    def read_file(self, filename: str) -> Dict[str, Any]:
        """Not implemented - this DataSource only writes."""
        raise NotImplementedError("ZarrDataSource only supports writing")

    def write(self, data: Dict[str, Any], filename: str) -> None:
        """Write transformed data to a Zarr store.

        Args:
            data: Transformed data from H5ToZarrTransformation
            filename: Base filename for the Zarr store
        """
        store_path = self.output_dir / f"{filename}.zarr"

        if store_path.exists():
            self.logger.info(f"Skipping {filename} - Zarr store already exists")
            return

        # Create Zarr store
        self.logger.info(f"Creating Zarr store: {store_path}")
        store = zarr.DirectoryStore(store_path)
        root = zarr.group(store=store)

        # Store metadata as root attributes
        if "metadata" in data:
            for key, value in data["metadata"].items():
                # Convert numpy types to Python types for JSON serialization
                if hasattr(value, "item"):  # numpy scalar
                    value = value.item()
                root.attrs[key] = value
            data.pop("metadata")

        # Write all arrays from the transformation
        for array_name, array_info in data.items():
            root.create_dataset(
                array_name,
                data=array_info["data"],
                chunks=array_info["chunks"],
                compressor=array_info["compressor"],
                dtype=array_info["dtype"]
            )

        # Add some store-level metadata
        root.attrs["zarr_format"] = 2
        root.attrs["created_by"] = "physicsnemo-curator-tutorial"

        # Something weird is happening here.
        # If this error occurs, the stores are created and we move to the next one.
        # If this error does NOT occur, we seem to skip all the remaining files.
        # Debug with Alexey.
        self.logger.info(f"Successfully created Zarr store")

    def should_skip(self, filename: str) -> bool:
        """Check if we should skip writing this store.

        Args:
            filename: Base filename to check

        Returns:
            True if store should be skipped (already exists)
        """
        store_path = self.output_dir / f"{filename}.zarr"
        exists = store_path.exists()

        if exists:
            self.logger.info(f"Skipping {filename} - Zarr store already exists")
            return True

        return False
```

**Key Points About This Sink:**

1. **Individual Stores**: Each simulation run gets its own `.zarr` directory for efficient access

2. **Optimized Storage**: Uses the chunking and compression settings from the transformation

3. **Complete Metadata**: Stores all metadata as Zarr attributes for easy access

4. **Overwrite Control**: Configurable overwrite behavior for reprocessing workflows

5. **Write-Only**: This sink only writes data

### Step 5: Run the ETL pipeline

Now we'll tie everything together with a configuration file and run the complete pipeline. Create a file called `tutorial_config.yaml`:

```yaml
# Tutorial ETL Pipeline Configuration
# This demonstrates the complete H5 -> Zarr processing pipeline

etl:
  # Processing settings
  processing:
    num_processes: 2  # Use 2 processes for this small tutorial dataset
    args: {}

  # Validation (runs first)
  validator:
    _target_: tutorial_validator.TutorialValidator
    _convert_: all
    input_dir: ???  # Will be provided via command line
    validation_level: "fields"  # Full validation including data content

  # Source (reads HDF5 files)
  source:
    _target_: h5_data_source.H5DataSource
    _convert_: all
    input_dir: ???  # Will be provided via command line

  # Transformations (convert to Zarr format)
  transformations:
    h5_to_zarr:
      _target_: h5_to_zarr_transformation.H5ToZarrTransformation
      _convert_: all
      chunk_size: 500
      compression_level: 3

  # Sink (writes Zarr stores)
  sink:
    _target_: zarr_data_source.ZarrDataSource
    _convert_: all
    output_dir: ???  # Will be provided via command line
    overwrite: true
```

**Run the ETL Pipeline:**

Now you can run the complete pipeline using the physicsnemo-curator CLI:

```bash
# IMPORTANT: Make sure your tutorial files are importable (and that the paths are correct)

# Run the ETL pipeline
physicsnemo-curator-etl --config-name tutorial_config \
  etl.validator.input_dir=tutorial_data \
  etl.source.input_dir=tutorial_data \
  etl.sink.output_dir=output_zarr
```

**What Happens During Execution:**

1. **Validation Phase**: The pipeline first validates all HDF5 files in `tutorial_data/`
   - Checks file structure and schema compliance
   - Validates data ranges and consistency
   - Stops execution if any validation errors are found

2. **Processing Phase**: For each validated file:
   - H5DataSource reads the HDF5 file
   - H5ToZarrTransformation converts it to Zarr-optimized format
   - ZarrDataSource writes the result to a `.zarr` store

3. **Parallel Execution**: Uses 2 processes to handle multiple files simultaneously

4. **Output**: Creates individual Zarr stores for each input file

**Expected Output Structure:**

After running, you'll have:

```bash
output_zarr/
├── run_001.zarr/
│   ├── temperature/
│   ├── velocity/
│   ├── coordinates/
│   ├── velocity_magnitude/
│   └── .zattrs (metadata)
├── run_002.zarr/
├── run_003.zarr/
├── run_004.zarr/
└── run_005.zarr/
```

**Verify the Results:**

You can inspect the output using Python:

```python
import zarr

# Open a Zarr store
store = zarr.open("output_zarr/run_001.zarr", mode="r")

print("Arrays in store:", list(store.keys()))
print("Temperature data shape:", store["temperature"].shape)
print("Velocity data shape:", store["velocity"].shape)
print("Metadata:", dict(store.attrs))

# Access the data
temperature = store["temperature"][:]
print(f"Temperature range: {temperature.min():.1f} - {temperature.max():.1f} K")
```

**Key Benefits of This Pipeline:**

1. **Validation**: Ensures data consistency before processing
2. **Format Optimization**: Converts to efficient Zarr format with compression
3. **Parallel Processing**: Handles multiple files simultaneously
4. **Metadata Preservation**: Maintains all original metadata plus derived fields
5. **ML-Ready Output**: Individual stores for efficient training data loading
