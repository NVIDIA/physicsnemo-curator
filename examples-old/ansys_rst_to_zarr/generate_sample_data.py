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

"""
Generate synthetic thermal simulation data to mimic Ansys .rst files.

This script creates mock thermal analysis data that can be used to demonstrate
the PhysicsNeMo-Curator ETL pipeline without requiring an Ansys license.

The generated files use the .rst extension (like real Ansys result files) but
contain pickled Python data instead of binary Ansys format. This allows the
tutorial to demonstrate the ETL pipeline structure without needing DPF Server.

The generated data includes:
- 3D mesh node coordinates
- Temperature field (scalar)
- Heat flux vectors (3D)
- Simulation metadata

Note: Real Ansys .rst files from thermal simulations would contain similar data
(coordinates, temperature, heat flux) but in Ansys binary format, requiring
PyDPF-Core and DPF Server to read them.
"""

import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np


def generate_3d_mesh(
    nx: int, ny: int, nz: int, domain_size: tuple = (1.0, 1.0, 1.0)
) -> np.ndarray:
    """
    Generate a regular 3D mesh grid.

    Args:
        nx, ny, nz: Number of nodes in x, y, z directions
        domain_size: Physical size of the domain in meters (x, y, z)

    Returns:
        coordinates: (N, 3) array of node coordinates
    """
    x = np.linspace(0, domain_size[0], nx)
    y = np.linspace(0, domain_size[1], ny)
    z = np.linspace(0, domain_size[2], nz)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    coordinates = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    return coordinates


def generate_temperature_field(
    coordinates: np.ndarray, scenario: str = "heat_source"
) -> np.ndarray:
    """
    Generate a temperature field based on different thermal scenarios.

    Args:
        coordinates: (N, 3) array of node coordinates
        scenario: Type of thermal scenario

    Returns:
        temperature: (N,) array of temperature values in Kelvin
    """
    x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]

    if scenario == "heat_source":
        # Heat source at the center
        center = np.array([0.5, 0.5, 0.5])
        distance = np.sqrt(
            (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
        )
        temperature = 300 + 100 * np.exp(-10 * distance**2)

    elif scenario == "linear_gradient":
        # Linear temperature gradient in x-direction
        temperature = 300 + 50 * x

    elif scenario == "hot_cold_sides":
        # Hot on left, cold on right
        temperature = 350 - 100 * x

    elif scenario == "corner_heating":
        # Multiple heat sources at corners
        corners = [np.array([0.1, 0.1, 0.1]), np.array([0.9, 0.9, 0.9])]
        temperature = np.ones(len(coordinates)) * 300
        for corner in corners:
            distance = np.sqrt(
                (x - corner[0]) ** 2 + (y - corner[1]) ** 2 + (z - corner[2]) ** 2
            )
            temperature += 80 * np.exp(-20 * distance**2)

    elif scenario == "periodic_heating":
        # Periodic heating pattern
        temperature = 300 + 50 * np.sin(4 * np.pi * x) * np.cos(4 * np.pi * y)

    else:
        # Default uniform temperature
        temperature = np.ones(len(coordinates)) * 300

    # Add some random noise to make it more realistic
    temperature += np.random.normal(0, 2, len(coordinates))

    return temperature


def generate_heat_flux(
    coordinates: np.ndarray, temperature: np.ndarray, thermal_conductivity: float = 50.0
) -> np.ndarray:
    """
    Generate heat flux vectors based on temperature gradients.

    Uses a simplified approach: flux ∝ -∇T (Fourier's law)

    Args:
        coordinates: (N, 3) array of node coordinates
        temperature: (N,) array of temperatures
        thermal_conductivity: Material thermal conductivity (W/m·K)

    Returns:
        heat_flux: (N, 3) array of heat flux vectors (W/m²)
    """
    # Compute approximate temperature gradients using finite differences
    # This is a simplified approach for mock data

    n_nodes = len(coordinates)
    heat_flux = np.zeros((n_nodes, 3))

    # Simple gradient approximation based on position
    x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]

    # Approximate gradients (this is mock data, so we use a simple approach)
    # In reality, this would require proper finite element calculations
    dT_dx = np.gradient(temperature.reshape(-1), x.reshape(-1))[0] if len(x) > 1 else 0
    dT_dy = np.gradient(temperature.reshape(-1), y.reshape(-1))[0] if len(y) > 1 else 0
    dT_dz = np.gradient(temperature.reshape(-1), z.reshape(-1))[0] if len(z) > 1 else 0

    # Fourier's law: q = -k * ∇T
    heat_flux[:, 0] = -thermal_conductivity * dT_dx
    heat_flux[:, 1] = -thermal_conductivity * dT_dy
    heat_flux[:, 2] = -thermal_conductivity * dT_dz

    # Add some randomness to make it more realistic
    heat_flux += np.random.normal(0, 50, heat_flux.shape)

    return heat_flux


def generate_thermal_simulation(
    simulation_name: str,
    mesh_resolution: tuple = (10, 10, 10),
    scenario: str = "heat_source",
) -> Dict[str, Any]:
    """
    Generate a complete thermal simulation dataset.

    Args:
        simulation_name: Name identifier for this simulation
        mesh_resolution: Number of nodes in (x, y, z)
        scenario: Thermal scenario type

    Returns:
        data: Dictionary containing all simulation data
    """
    # Generate mesh
    coordinates = generate_3d_mesh(*mesh_resolution)
    n_nodes = len(coordinates)
    n_elements = (
        (mesh_resolution[0] - 1) * (mesh_resolution[1] - 1) * (mesh_resolution[2] - 1)
    )

    # Generate temperature field
    temperature = generate_temperature_field(coordinates, scenario)

    # Generate heat flux
    heat_flux = generate_heat_flux(coordinates, temperature)

    # Create metadata
    metadata = {
        "simulation_name": simulation_name,
        "num_nodes": n_nodes,
        "num_elements": max(n_elements, 1),  # Ensure at least 1
        "mesh_resolution": mesh_resolution,
        "scenario": scenario,
        "units": "SI",
        "temperature_units": "Kelvin",
        "heat_flux_units": "W/m^2",
        "coordinate_units": "meters",
        "time_step": 1,
        "analysis_type": "Steady-State Thermal",
        "solver": "Mock Thermal Solver v1.0",
        "temperature_min": float(temperature.min()),
        "temperature_max": float(temperature.max()),
        "temperature_mean": float(temperature.mean()),
    }

    data = {
        "coordinates": coordinates,
        "temperature": temperature,
        "heat_flux": heat_flux,
        "metadata": metadata,
    }

    return data


def save_simulation(data: Dict[str, Any], output_path: Path) -> None:
    """
    Save simulation data to disk.

    Args:
        data: Simulation data dictionary
        output_path: Path to save the data
    """
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved simulation to: {output_path}")


def main():
    """Generate 5 different thermal simulation datasets."""

    # Create output directory
    output_dir = Path("mock_thermal_data")
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Generating Mock Thermal Simulation Data")
    print("=" * 60)

    # Define 5 different simulation scenarios
    simulations = [
        {
            "name": "thermal_sim_001",
            "resolution": (12, 12, 12),
            "scenario": "heat_source",
            "description": "Central heat source with radial heat dissipation",
        },
        {
            "name": "thermal_sim_002",
            "resolution": (15, 10, 10),
            "scenario": "linear_gradient",
            "description": "Linear temperature gradient",
        },
        {
            "name": "thermal_sim_003",
            "resolution": (10, 10, 15),
            "scenario": "hot_cold_sides",
            "description": "Hot and cold boundary conditions",
        },
        {
            "name": "thermal_sim_004",
            "resolution": (14, 14, 8),
            "scenario": "corner_heating",
            "description": "Multiple localized heat sources",
        },
        {
            "name": "thermal_sim_005",
            "resolution": (16, 16, 10),
            "scenario": "periodic_heating",
            "description": "Periodic heating pattern",
        },
    ]

    # Generate each simulation
    for sim_config in simulations:
        print(f"\nGenerating: {sim_config['name']}")
        print(f"  Description: {sim_config['description']}")
        print(f"  Resolution: {sim_config['resolution']}")

        data = generate_thermal_simulation(
            simulation_name=sim_config["name"],
            mesh_resolution=sim_config["resolution"],
            scenario=sim_config["scenario"],
        )

        # Save to disk with .rst extension (mimicking real Ansys format)
        output_path = output_dir / f"{sim_config['name']}.rst"
        save_simulation(data, output_path)

        print(f"  Nodes: {data['metadata']['num_nodes']}")
        print(
            f"  Temperature range: {data['metadata']['temperature_min']:.1f} - "
            f"{data['metadata']['temperature_max']:.1f} K"
        )

    print("\n" + "=" * 60)
    print("Data generation complete!")
    print(
        f"Generated {len(simulations)} thermal simulation files (.rst) in: {output_dir}"
    )
    print("=" * 60)
    print("\nThese mock .rst files can be processed by the ETL pipeline.")
    print("To use real Ansys .rst files, you would need:")
    print("  - Ansys installation (2021 R1+)")
    print("  - Valid Ansys license")
    print("  - PyDPF-Core configured with DPF Server")


if __name__ == "__main__":
    main()
