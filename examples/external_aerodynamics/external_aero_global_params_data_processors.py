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

import logging

import numpy as np
from schemas import ExternalAerodynamicsExtractedDataInMemory

logging.basicConfig(
    format="%(asctime)s - Process %(process)d - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def default_global_params_processing_for_external_aerodynamics(
    data: ExternalAerodynamicsExtractedDataInMemory,
    global_parameters: dict,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Default global parameters processing for External Aerodynamics.

    Extracts and flattens global parameter references from config into a 1D numpy array.
    Handles both vector and scalar parameter types.

    Args:
        data: Container with simulation data and metadata
        global_parameters: Dict from config with structure:
            {
                "param_name": {
                    "type": "vector" or "scalar",
                    "reference": value or list
                }
            }

    Returns:
        Updated `data` with global_params_reference set
    """

    # Build dictionaries for types and reference values
    global_params_types = {
        name: params["type"] for name, params in global_parameters.items()
    }

    global_params_reference_dict = {
        name: params["reference"] for name, params in global_parameters.items()
    }

    # Arrange global parameters reference in a list based on the type of the parameter
    global_params_reference_list = []
    for name, param_type in global_params_types.items():
        if param_type == "vector":
            global_params_reference_list.extend(global_params_reference_dict[name])
        elif param_type == "scalar":
            global_params_reference_list.append(global_params_reference_dict[name])
        else:
            raise ValueError(
                f"Global parameter '{name}' has unsupported type '{param_type}'. "
                f"Must be 'vector' or 'scalar'."
            )

    # Convert to numpy array and store in data container
    data.global_params_reference = np.array(
        global_params_reference_list, dtype=np.float32
    )

    return data


def process_global_params(
    data: ExternalAerodynamicsExtractedDataInMemory,
    global_parameters: dict,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Base processor for global parameters - to be overridden for specific datasets.

    This is a placeholder that should be replaced by dataset-specific implementations
    (e.g., process_global_params_hlpw).

    By default, sets global_params_values equal to global_params_reference,
    assuming simulation conditions match reference conditions.

    Args:
        data: Container with simulation data and metadata
        global_parameters: Dict from config with parameter definitions

    Returns:
        Updated `data` with global_params_values set
    """
    # Default behavior: assume simulation values match reference
    data.global_params_values = data.global_params_reference.copy()

    return data


# ============================================================================
# Case-Specific Processors
# ============================================================================
# These functions demonstrate how to extract global_params_values from
# simulation data for specific datasets. Replace process_global_params above
# with these in your config for case-specific processing.


def process_global_params_hlpw(
    data: ExternalAerodynamicsExtractedDataInMemory,
    global_parameters: dict,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Extract global parameters from HLPW simulation data.

    For HLPW, :
    - AoA (Angle of Attack) varies per simulation and can be extracted from filename

    Args:
        data: Container with simulation data and metadata
        global_parameters: Dict from config with parameter definitions

    Returns:
        Updated `data` with global_params_values extracted from simulation
    """

    # Build a dict of extracted values keyed by parameter name
    extracted_values = {}

    # Extract AoA from filename (e.g., "geo_LHC001_AoA_16" -> 16.0)
    filename = data.metadata.filename
    if "AoA_" in filename:
        # Extract string after "AoA_"
        # Example: "geo_LHC001_AoA_16" -> "16"
        # Example: "geo_LHC001_AoA_16_something" -> "16"
        after_aoa = filename.split("AoA_")[1]
        # Take everything up to next underscore or end of string
        aoa_str = after_aoa.split("_")[0] if "_" in after_aoa else after_aoa
        aoa = float(aoa_str)
        extracted_values["AoA"] = aoa
        logger.info(f"Extracted AoA={aoa} from filename: {filename}")
    else:
        raise ValueError(f"AoA pattern not found in filename '{filename}'.")

    # Build the flattened array using the same logic as reference processing
    global_params_values_list = []
    for name, params in global_parameters.items():
        param_type = params["type"]
        if name not in extracted_values:
            raise ValueError(
                f"Global parameter '{name}' was not extracted from simulation data."
            )
        value = extracted_values[name]

        if param_type == "vector":
            global_params_values_list.extend(value)
        elif param_type == "scalar":
            global_params_values_list.append(value)
        else:
            raise ValueError(
                f"Global parameter '{name}' has unsupported type '{param_type}'. "
                f"Must be 'vector' or 'scalar'."
            )

    data.global_params_values = np.array(global_params_values_list, dtype=np.float32)

    return data
