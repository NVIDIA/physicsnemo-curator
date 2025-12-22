import os
import numpy as np
import pyvista as pv
from vtk import vtkEnSightWriter

def generate_mesh_with_fields(run_number, resolution=20):
    """Generate a mesh with simulation fields for one run."""
    radius = 1.0 + 0.1 * np.sin(run_number)
    mesh = pv.Sphere(radius=radius, theta_resolution=resolution, phi_resolution=resolution)

    points = mesh.points
    num_points = mesh.n_points

    distances = np.linalg.norm(points, axis=1)
    temperature = 250.0 + 100.0 * (distances / distances.max()) + np.random.normal(0, 5, num_points)
    mesh.point_data["Temperature"] = temperature.astype(np.float32)

    pressure = 101325.0 + 1000.0 * np.sin(points[:, 2] * np.pi) + np.random.normal(0, 100, num_points)
    mesh.point_data["Pressure"] = pressure.astype(np.float32)

    velocity = np.zeros((num_points, 3), dtype=np.float32)
    velocity[:, 0] = -points[:, 1] * 2.0 + np.random.normal(0, 0.5, num_points)
    velocity[:, 1] = points[:, 0] * 2.0 + np.random.normal(0, 0.5, num_points)
    velocity[:, 2] = points[:, 2] * 0.5 + np.random.normal(0, 0.3, num_points)
    mesh.point_data["Velocity"] = velocity

    density = 1.2 + 0.2 * np.sin(distances * np.pi * 2) + np.random.normal(0, 0.05, num_points)
    mesh.point_data["Density"] = density.astype(np.float32)

    vorticity = 2.0 * np.ones(num_points) + 0.5 * np.cos(distances * 3) + np.random.normal(0, 0.1, num_points)
    mesh.point_data["Vorticity"] = vorticity.astype(np.float32)

    return mesh

def create_ensight_case_file(run_number, output_dir="tutorial_data", resolution=20):
    """Create one EnSight Gold case file for a simulation run."""
    os.makedirs(output_dir, exist_ok=True)

    mesh = generate_mesh_with_fields(run_number, resolution)
    num_points = mesh.n_points
    num_cells = mesh.n_cells

    # Create base filename
    base_name = f"run_{run_number:03d}"
    case_file = os.path.join(output_dir, f"{base_name}.case")

    # PyVista VTK EnSightWriter requires vtk.vtkUnstructuredGrid
    # Wrap the mesh as unstructured grid if necessary
    ug = pv.wrap(mesh)

    # Write EnSight Gold binary files
    writer = vtkEnSightWriter()
    writer.SetFileName(case_file)
    writer.SetInputData(ug)
    writer.SetCaseFileName(case_file)
    writer.WriteAllVariablesOn()
    writer.Update()

    print(f"Created {case_file}:")
    print(f"  - {num_points} points")
    print(f"  - {num_cells} cells")
    print(f"  - Fields: {', '.join(mesh.point_data.keys())}")

def main():
    print("Generating sample EnSight Gold physics simulation dataset...\n")
    resolutions = [15, 20, 18, 22, 20]

    for run_num in range(1, 6):
        create_ensight_case_file(run_num, resolution=resolutions[run_num - 1])

    print("\nDataset generation complete!")
    print("Created 5 EnSight Gold .case files in the 'tutorial_data/' directory")
    print("Each file contains a sphere mesh with Temperature, Pressure, Velocity, Density, and Vorticity fields")
    print("\nYou can now run the ETL pipeline with your EnSightDataSource")

if __name__ == "__main__":
    main()
