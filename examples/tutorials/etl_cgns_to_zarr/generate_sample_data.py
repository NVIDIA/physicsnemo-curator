# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

import numpy as np
import pyvista as pv


def generate_mesh_with_fields(run_number, resolution=20):
    """Generate a mesh with simulation fields for one run.
    
    Args:
        run_number: Run identifier to add variation
        resolution: Mesh resolution (higher = more points)
    
    Returns:
        pyvista mesh with point data fields
    """
    # Create a sphere mesh with some variation based on run number
    radius = 1.0 + 0.1 * np.sin(run_number)
    mesh = pv.Sphere(radius=radius, theta_resolution=resolution, phi_resolution=resolution)
    
    # Get points for field generation
    points = mesh.points
    num_points = mesh.n_points
    
    # Generate temperature field based on distance from origin
    distances = np.linalg.norm(points, axis=1)
    temperature = 250.0 + 100.0 * (distances / distances.max()) + np.random.normal(0, 5, num_points)
    mesh.point_data["Temperature"] = temperature.astype(np.float32)
    
    # Generate pressure field with some variation
    pressure = 101325.0 + 1000.0 * np.sin(points[:, 2] * np.pi) + np.random.normal(0, 100, num_points)
    mesh.point_data["Pressure"] = pressure.astype(np.float32)
    
    # Generate velocity field (3D vectors)
    # Create a rotational flow pattern
    velocity = np.zeros((num_points, 3), dtype=np.float32)
    velocity[:, 0] = -points[:, 1] * 2.0 + np.random.normal(0, 0.5, num_points)
    velocity[:, 1] = points[:, 0] * 2.0 + np.random.normal(0, 0.5, num_points)
    velocity[:, 2] = points[:, 2] * 0.5 + np.random.normal(0, 0.3, num_points)
    mesh.point_data["Velocity"] = velocity
    
    # Generate density field
    density = 1.2 + 0.2 * np.sin(distances * np.pi * 2) + np.random.normal(0, 0.05, num_points)
    mesh.point_data["Density"] = density.astype(np.float32)
    
    # Generate vorticity magnitude (scalar derived from position)
    vorticity = 2.0 * np.ones(num_points) + 0.5 * np.cos(distances * 3) + np.random.normal(0, 0.1, num_points)
    mesh.point_data["Vorticity"] = vorticity.astype(np.float32)
    
    return mesh


def create_cgns_file(run_number, output_dir="tutorial_data", resolution=20):
    """Create one CGNS file for a simulation run.
    
    Args:
        run_number: Run identifier
        output_dir: Directory to save the file
        resolution: Mesh resolution
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate mesh with fields
    mesh = generate_mesh_with_fields(run_number, resolution)
    num_points = mesh.n_points
    num_cells = mesh.n_cells
    
    # Create CGNS filename
    filename = f"run_{run_number:03d}.cgns"
    filepath = os.path.join(output_dir, filename)
    
    # Save as CGNS file
    # Note: PyVista writes CGNS files through VTK's CGNS writer
    mesh.save(filepath)
    
    # Print summary of generated data
    print(f"Created {filepath}:")
    print(f"  - {num_points} points")
    print(f"  - {num_cells} cells")
    print(f"  - Fields: {', '.join(mesh.point_data.keys())}")


def main():
    """Generate sample dataset with 5 simulation runs."""
    print("Generating sample CGNS physics simulation dataset...")
    print()
    
    # Generate 5 runs with varying resolution for diversity
    resolutions = [15, 20, 18, 22, 20]
    
    for run_num in range(1, 6):
        create_cgns_file(run_num, resolution=resolutions[run_num - 1])
    
    print("\nDataset generation complete!")
    print("Created 5 CGNS files in the 'tutorial_data/' directory")
    print("Each file contains a sphere mesh with temperature, pressure, velocity, density, and vorticity fields")
    print("\nYou can now run the ETL pipeline with:")
    print("  python run_etl.py etl.source.input_dir=tutorial_data")



if __name__ == "__main__":
    main()
