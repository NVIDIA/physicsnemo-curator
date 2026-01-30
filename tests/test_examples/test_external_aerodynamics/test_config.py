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

import os
from pathlib import Path
from unittest.mock import patch

import data_sources
import data_transformations
import dataset_validator
from constants import DatasetKind, ModelType
from hydra import compose, initialize
from hydra.utils import instantiate


def get_config_path() -> Path:
    """Get the path to the config directory."""
    # Hydra requires a relative path.
    return Path("../../../examples/external_aerodynamics/config")


def test_external_aero_etl_drivaerml_config():
    """Test that the external_aero_etl_drivaerml.yaml configuration is valid and has expected values."""
    # Initialize Hydra with the config directory
    with initialize(version_base="1.3", config_path=str(get_config_path())):
        # Compose the configuration
        cfg = compose(
            config_name="external_aero_etl_drivaerml",
            overrides=[
                "etl.source.input_dir=/path/to/input/dataset",
                "etl.sink.output_dir=/path/to/output/directory",
            ],
        )

        # Test common settings
        assert cfg.etl.common.kind == "drivaerml"
        assert cfg.etl.common.model_type == "surface"

        # Test processing settings
        assert cfg.etl.processing.num_processes == 12
        assert cfg.etl.processing.args == {}

        # Test that we can actually create the validator
        with patch.object(
            dataset_validator.ExternalAerodynamicsDatasetValidator,
            "__init__",
            return_value=None,
        ) as m:
            instantiate(cfg.etl.validator)
            assert m.call_count == 1

        assert cfg.etl.validator.validation_level == "fields"

        # Test source settings
        with patch.object(
            data_sources.ExternalAerodynamicsDataSource, "__init__", return_value=None
        ) as m:
            instantiate(cfg.etl.source)
            assert m.call_count == 1
        assert cfg.etl.source.kind == "drivaerml"
        assert cfg.etl.source.model_type == "surface"
        assert cfg.etl.transformations.volume_preprocessing.volume_variables == {
            "UMeanTrim": "vector",
            "pMeanTrim": "scalar",
            "nutMeanTrim": "scalar",
        }
        assert cfg.etl.transformations.surface_preprocessing.surface_variables == {
            "pMeanTrim": "scalar",
            "wallShearStressMeanTrim": "vector",
        }

        # Test transformation settings
        # Test preprocessing transformation
        with patch.object(
            data_transformations.ExternalAerodynamicsVolumeTransformation,
            "__init__",
            return_value=None,
        ) as m:
            instantiate(cfg.etl.transformations.volume_preprocessing)
            assert m.call_count == 1

        # Test zarr transformation
        with patch.object(
            data_transformations.ExternalAerodynamicsSurfaceTransformation,
            "__init__",
            return_value=None,
        ) as m:
            instantiate(cfg.etl.transformations.surface_preprocessing)
            assert m.call_count == 1

        # Test sink settings
        with patch.object(
            data_sources.ExternalAerodynamicsDataSource, "__init__", return_value=None
        ) as m:
            instantiate(cfg.etl.sink)
            assert m.call_count == 1
        assert cfg.etl.sink.kind == "drivaerml"
        assert cfg.etl.sink.overwrite_existing is True


def test_config_validation():
    """Test that the configuration can be validated against the schema."""
    # Initialize Hydra with the config directory
    with initialize(version_base="1.3", config_path=str(get_config_path())):
        # Compose the configuration
        cfg = compose(config_name="external_aero_etl_drivaerml")

        # Test that kind is a valid DatasetKind
        assert cfg.etl.common.kind in [k.value for k in DatasetKind]

        # Test that model_type is a valid ModelType
        assert cfg.etl.common.model_type in [m.value for m in ModelType]

        # Test that validation_level is valid
        assert cfg.etl.validator.validation_level in ["structure", "fields"]


def test_config_paths():
    """Test that the configuration paths are valid."""
    # Initialize Hydra with the config directory
    with initialize(version_base="1.3", config_path=str(get_config_path())):
        # Compose the configuration
        cfg = compose(
            config_name="external_aero_etl_drivaerml",
            overrides=[
                "etl.source.input_dir=/path/to/input/dataset",
                "etl.sink.output_dir=/path/to/output/directory",
            ],
        )

        # Test that input and output directories are specified
        assert cfg.etl.source.input_dir is not None
        assert cfg.etl.sink.output_dir is not None

        # Test that paths are absolute
        assert os.path.isabs(cfg.etl.source.input_dir)
        assert os.path.isabs(cfg.etl.sink.output_dir)


def test_config_override():
    """Test that configuration can be overridden."""
    # Initialize Hydra with the config directory
    with initialize(version_base="1.3", config_path=str(get_config_path())):
        # Compose the configuration with overrides
        cfg = compose(
            config_name="external_aero_etl_drivaerml",
            overrides=[
                "etl.common.kind=drivesim",
                "etl.common.model_type=volume",
                "etl.processing.num_processes=4",
            ],
        )

        # Test that overrides were applied
        assert cfg.etl.common.kind == "drivesim"
        assert cfg.etl.common.model_type == "volume"
        assert cfg.etl.processing.num_processes == 4
