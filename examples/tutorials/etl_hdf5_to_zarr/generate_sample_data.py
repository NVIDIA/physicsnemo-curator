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
from datetime import datetime

import h5py
import numpy as np


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

    with h5py.File(filepath, "w") as f:
        # Create groups
        fields_group = f.create_group("fields")
        geometry_group = f.create_group("geometry")
        metadata_group = f.create_group("metadata")
        sim_params_group = metadata_group.create_group("simulation_params")

        # Store field data
        fields_group.create_dataset("temperature", data=temperature)
        fields_group.create_dataset("velocity", data=velocity)

        # Store geometry data
        geometry_group.create_dataset("coordinates", data=coordinates)

        # Store metadata attributes
        metadata_group.attrs["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata_group.attrs["num_points"] = num_points
        metadata_group.attrs["temperature_units"] = "Kelvin"
        metadata_group.attrs["velocity_units"] = "m/s"

        # Store simulation parameters
        sim_params_group.attrs["total_time"] = np.random.uniform(
            1.0, 10.0
        )  # Random simulation time

    print(f"Created {filepath} with {num_points} data points")


def main():
    """Generate sample dataset with 5 simulation runs."""
    print("Generating sample physics simulation dataset...")

    # Generate 5 runs
    for run_num in range(1, 6):
        create_hdf5_file(run_num)

    print("\nDataset generation complete!")
    print("Created 5 HDF5 files in the 'tutorial_data/' directory")
    print("Each file contains ~1000 data points with temperature and velocity fields")


if __name__ == "__main__":
    main()
