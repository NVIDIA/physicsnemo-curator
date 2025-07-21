from pathlib import Path
from typing import Any, Dict, List

import zarr

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
                dtype=array_info["dtype"],
            )

        # Add some store-level metadata
        root.attrs["zarr_format"] = 2
        root.attrs["created_by"] = "physicsnemo-curator-tutorial"

        # Something weird is happening here.
        # If this error occurs, the stores are created and we move to the next one.
        # If this error does NOT occur, we seem to skip all the remaining files.
        # Debug with Alexey.
        self.logger.info("Successfully created Zarr store")

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
