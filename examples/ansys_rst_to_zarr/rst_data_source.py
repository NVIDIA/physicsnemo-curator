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


import pickle
from pathlib import Path
from typing import Any, Dict, List

from physicsnemo_curator.etl.data_sources import DataSource
from physicsnemo_curator.etl.processing_config import ProcessingConfig


class RstDataSource(DataSource):
    """
    DataSource for reading thermal simulation .rst files.

    This implementation reads mock .rst files (pickled Python data) for tutorial purposes.

    ============================================================================
    USING REAL ANSYS .RST FILES:
    ============================================================================
    To adapt this for real Ansys .rst files, you need:

    1. Install Ansys (2021 R1 or later)
    2. Have a valid Ansys license
    3. Install PyDPF-Core: pip install ansys-dpf-core
    4. Replace the read_file() method with PyDPF code:

        from ansys.dpf import core as dpf

        def read_file(self, filename: str) -> Dict[str, Any]:
            filepath = self.input_dir / f"{filename}.rst"
            model = dpf.Model(str(filepath))

            # Extract mesh and temperature data
            mesh = model.metadata.meshed_region
            coords = np.array(mesh.nodes.coordinates_field.data)

            # Extract temperature (for thermal analysis)
            temp_op = model.results.temperature()
            temp_fields = temp_op.outputs.fields_container()
            temperature = np.array(temp_fields[0].data)

            # Extract heat flux if available
            flux_op = model.results.heat_flux()
            flux_fields = flux_op.outputs.fields_container()
            heat_flux = np.array(flux_fields[0].data)

            return {
                "coordinates": coords,
                "temperature": temperature,
                "heat_flux": heat_flux,
                "metadata": {...},
                "filename": filename
            }

    For more info: https://dpf.docs.pyansys.com/
    ============================================================================
    """

    def __init__(self, cfg: ProcessingConfig, input_dir: str):
        super().__init__(cfg)
        self.input_dir = Path(input_dir)

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory {self.input_dir} does not exist")

    def get_file_list(self) -> List[str]:
        """Find all .rst files in the input directory."""
        rst_files = list(self.input_dir.glob("*.rst"))
        filenames = [f.stem for f in rst_files]
        self.logger.info(f"Found {len(filenames)} .rst files to process")
        return sorted(filenames)

    def read_file(self, filename: str) -> Dict[str, Any]:
        """
        Read a mock thermal .rst file.

        For this tutorial, .rst files are pickled Python dictionaries containing:
        - coordinates: (N, 3) array of node positions
        - temperature: (N,) array of temperature values in Kelvin
        - heat_flux: (N, 3) array of heat flux vectors in W/m²
        - metadata: dict with simulation information

        Returns data in a format ready for transformation.
        """
        filepath = self.input_dir / f"{filename}.rst"
        self.logger.info(f"Reading thermal simulation from: {filepath}")

        # Load the pickled thermal data
        with open(filepath, "rb") as f:
            data = pickle.load(f)  # noqa: S301

        # Add filename for tracking
        data["filename"] = filename

        # Log basic info
        self.logger.info(f"  Loaded {data['metadata']['num_nodes']} nodes")
        self.logger.info(
            f"  Temperature range: {data['metadata']['temperature_min']:.1f} - "
            f"{data['metadata']['temperature_max']:.1f} K"
        )
        self.logger.info(f"  Analysis type: {data['metadata']['analysis_type']}")

        return data

    def _get_output_path(self, filename: str) -> Path:
        raise NotImplementedError("RstDataSource only supports reading")

    def _write_impl_temp_file(self, data: Dict[str, Any], output_path: Path) -> None:
        raise NotImplementedError("RstDataSource only supports reading")

    def should_skip(self, filename: str) -> bool:
        return False
