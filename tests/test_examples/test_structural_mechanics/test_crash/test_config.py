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
from hydra import compose, initialize
from hydra.utils import instantiate


def get_config_path() -> Path:
    """Get the path to the config directory."""
    # Hydra requires a relative path.
    return Path("../../../../examples/structural_mechanics/crash/config")


def test_crash_etl_config():
    """Test that the crash_etl.yaml configuration is valid and has expected values."""
    # Initialize Hydra with the config directory
    with initialize(version_base="1.3", config_path=str(get_config_path())):
        # Compose the configuration
        # Override output_dir in the serialization_format config (before interpolation)
        cfg = compose(
            config_name="crash_etl",
            overrides=[
                "etl.source.input_dir=/path/to/input/dataset",
                "serialization_format.sink.output_dir=/path/to/output/directory",
            ],
        )

        # Test processing settings
        assert cfg.etl.processing.num_processes == 12

        # Test source settings
        with patch.object(
            data_sources.CrashD3PlotDataSource, "__init__", return_value=None
        ) as m:
            instantiate(cfg.etl.source)
            assert m.call_count == 1

        assert cfg.etl.source.input_dir == "/path/to/input/dataset"

        # Test transformation settings
        with patch.object(
            data_transformations.CrashDataTransformation,
            "__init__",
            return_value=None,
        ) as m:
            instantiate(cfg.etl.transformations.crash_transform)
            assert m.call_count == 1

        assert cfg.etl.transformations.crash_transform.wall_threshold == 1.0

        # Test sink settings (default is VTP)
        with patch.object(
            data_sources.CrashVTPDataSource, "__init__", return_value=None
        ) as m:
            instantiate(cfg.etl.sink)
            assert m.call_count == 1

        assert cfg.etl.sink.output_dir == "/path/to/output/directory"
        assert cfg.etl.sink.overwrite_existing is True
        assert cfg.etl.sink.time_step == 0.005


def test_config_paths():
    """Test that the configuration paths are valid."""
    # Initialize Hydra with the config directory
    with initialize(version_base="1.3", config_path=str(get_config_path())):
        # Compose the configuration
        cfg = compose(
            config_name="crash_etl",
            overrides=[
                "etl.source.input_dir=/path/to/input/dataset",
                "serialization_format.sink.output_dir=/path/to/output/directory",
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
            config_name="crash_etl",
            overrides=[
                "etl.processing.num_processes=4",
                "etl.transformations.crash_transform.wall_threshold=0.5",
                "serialization_format.sink.overwrite_existing=false",
            ],
        )

        # Test that overrides were applied
        assert cfg.etl.processing.num_processes == 4
        assert cfg.etl.transformations.crash_transform.wall_threshold == 0.5
        assert cfg.etl.sink.overwrite_existing is False


def test_config_raises_errors_when_required_fields_are_not_provided():
    """Test that required fields raise errors when not provided."""
    import pytest
    from omegaconf.errors import MissingMandatoryValue

    # Initialize Hydra with the config directory
    with initialize(version_base="1.3", config_path=str(get_config_path())):
        # Test that missing input_dir raises error
        with pytest.raises(MissingMandatoryValue):
            cfg = compose(
                config_name="crash_etl",
                overrides=["serialization_format.sink.output_dir=/path/to/output"],
            )
            # Try to access the input_dir - should raise an error due to ???
            _ = cfg.etl.source.input_dir

        # Test that missing output_dir raises error
        with pytest.raises(MissingMandatoryValue):
            cfg = compose(
                config_name="crash_etl",
                overrides=["etl.source.input_dir=/path/to/input"],
            )
            # Try to access the output_dir - should raise an error due to ???
            _ = cfg.etl.sink.output_dir


def test_vtp_sink_time_step():
    """Test that VTP sink can be configured with time_step parameter."""
    # Initialize Hydra with the config directory
    with initialize(version_base="1.3", config_path=str(get_config_path())):
        # Compose the configuration with time_step override
        cfg = compose(
            config_name="crash_etl",
            overrides=[
                "etl.source.input_dir=/path/to/input",
                "serialization_format.sink.output_dir=/path/to/output",
                "serialization_format.sink.time_step=0.01",
            ],
        )

        # Test that time_step was set
        assert cfg.etl.sink.time_step == 0.01


def test_serialization_format_switching():
    """Test that serialization_format config group switches work correctly."""
    # Initialize Hydra with the config directory
    with initialize(version_base="1.3", config_path=str(get_config_path())):
        # Test default (VTP)
        cfg_vtp = compose(
            config_name="crash_etl",
            overrides=[
                "etl.source.input_dir=/path/to/input",
                "serialization_format.sink.output_dir=/path/to/output",
            ],
        )
        assert cfg_vtp.etl.sink._target_ == "data_sources.CrashVTPDataSource"
        assert "time_step" in cfg_vtp.etl.sink
        assert "compression_level" not in cfg_vtp.etl.sink

        # Test switching to Zarr
        cfg_zarr = compose(
            config_name="crash_etl",
            overrides=[
                "serialization_format=zarr",
                "etl.source.input_dir=/path/to/input",
                "serialization_format.sink.output_dir=/path/to/output",
            ],
        )
        assert cfg_zarr.etl.sink._target_ == "data_sources.CrashZarrDataSource"
        assert "compression_level" in cfg_zarr.etl.sink
        assert "time_step" not in cfg_zarr.etl.sink


def test_zarr_sink_config():
    """Test that configuration can be modified to use Zarr sink."""
    # Initialize Hydra with the config directory
    with initialize(version_base="1.3", config_path=str(get_config_path())):
        # Compose the configuration with Zarr sink
        # Switch to zarr serialization format via config group
        cfg = compose(
            config_name="crash_etl",
            overrides=[
                "serialization_format=zarr",  # Switch to Zarr config
                "etl.source.input_dir=/path/to/input",
                "serialization_format.sink.output_dir=/path/to/output",
                "serialization_format.sink.compression_level=5",
            ],
        )

        # Test that Zarr sink can be instantiated
        with patch.object(
            data_sources.CrashZarrDataSource, "__init__", return_value=None
        ) as m:
            instantiate(cfg.etl.sink)
            assert m.call_count == 1

        # Verify sink configuration
        assert cfg.etl.sink._target_ == "data_sources.CrashZarrDataSource"
        assert cfg.etl.sink.output_dir == "/path/to/output"
        assert cfg.etl.sink.compression_level == 5
        assert cfg.etl.sink.overwrite_existing is True
        assert cfg.etl.sink.compression_method == "zstd"
