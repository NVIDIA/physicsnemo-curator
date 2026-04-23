# Drop Test (Solid Element) Data Processing

ETL pipeline for drop-test simulations with **solid elements** (tet / hex / wedge / pyramid, plus surface tris). Reads per-timestep VTK files produced by OpenRadioss `anim_to_vtk`, filters rigid-wall nodes, and writes either a VTU (UnstructuredGrid) per run or a Zarr store per run. The VTU output is directly consumed by the `physicsnemo/examples/structural_mechanics/drop_test` recipe.

## Input

VTK files produced by OpenRadioss `anim_to_vtk` — one file per timestep per run. The default glob is `Cell_Phone_DropA*.vtk` (override via `etl.source.vtk_glob` for other simulations):

```
input_dir/
├── run0001/
│   ├── Cell_Phone_DropA001.vtk
│   ├── Cell_Phone_DropA002.vtk
│   └── ...
├── run0002/
│   └── ...
└── ...
```

Each run directory is treated as a separate sample; files within it are sorted and stacked into a `(T, N, 3)` position trajectory.

## Running

From this directory:

```bash
python run_etl.py                                                \
    --config-name=drop_test_etl                                  \
    etl.source.input_dir=/path/to/openradioss_runs/              \
    serialization_format=vtu                                     \
    serialization_format.sink.output_dir=/path/to/drop_test/vtu/ \
    etl.processing.num_processes=4
```

Swap `serialization_format=vtu` for `serialization_format=zarr` to emit Zarr stores instead. To process data with a different file-naming scheme, override the glob: `etl.source.vtk_glob="MyDropSim*.vtk"`.

## Processing

1. **Source** (`data_sources.DropTestVTKDataSource`): reads per-timestep VTKs, stacks points into `pos_raw: (T, N, 3)`, extracts mesh connectivity, and pulls any available per-node or per-element fields — velocity, acceleration, temperature, residual forces, stress tensor (Voigt 6-vec), element effective plastic strain, element strain / plastic-strain tensors. Nodal Von Mises is derived from the nodal Voigt tensor when present; if only element-solid stress is available, it is reduced across through-thickness layers and Von Mises is computed on the element.
2. **Transform** (`data_transformations.DropTestDataTransformation`): filters rigid-wall nodes (nodes whose per-timestep displacement variation falls below `wall_threshold`, default `1e-5`), remaps the surviving node indices through the mesh connectivity, drops cells that reference removed nodes, and rebuilds the edge set from the filtered connectivity. All optional per-node / per-element fields are filtered alongside the positions so every field stays aligned.
3. **Sink** (VTU or Zarr): writes one output per run into `output_dir/` with `run_id` set from the source folder name. `overwrite_existing: true` forces re-emission; the default skips runs whose output already exists.

## Output — VTU

`UnstructuredGrid` per run. Supported cell types from the source connectivity:

| Cell type  | VTK code      | Node count |
|------------|---------------|-----------:|
| Triangle   | `TRIANGLE`    | 3          |
| Tetra      | `TETRA`       | 4          |
| Pyramid    | `PYRAMID`     | 5          |
| Wedge      | `WEDGE`       | 6          |
| Hexahedron | `HEXAHEDRON`  | 8          |

7-node cells are skipped (no standard VTK type; would need tet decomposition). Triangle vertex order is reversed when `sink.flip_triangle_normals: true` (default) to match VTK normal conventions; tets are reoriented to positive volume automatically.

Point coordinates are set to the reference frame (`t=0`) position; subsequent frames are stored as displacement fields.

**Time-invariant point data:**

| Field       | Shape / dtype | Notes                                  |
|-------------|---------------|----------------------------------------|
| `thickness` | `[N] float32` | Zeros for solid elements (kept for pipeline compatibility with shell recipes) |

**Per-timestep fields** — emitted for every `t ∈ {0, 1, ..., T-1}`, indexed as a 4-digit zero-padded string (e.g. `t0000`, `t0001`, ...). Optional fields appear only when the corresponding array is present in the source:

| Field                                       | Location   | Shape / dtype    | Emitted when                       |
|---------------------------------------------|------------|------------------|------------------------------------|
| `displacement_tNNNN`                        | point_data | `[N, 3] float32` | always                             |
| `velocity_tNNNN`                            | point_data | `[N, 3] float32` | nodal velocity present             |
| `acceleration_tNNNN`                        | point_data | `[N, 3] float32` | nodal acceleration present         |
| `temperature_tNNNN`                         | point_data | `[N] float32`    | nodal temperature present          |
| `residual_forces_tNNNN`                     | point_data | `[N, 3] float32` | nodal residual forces present      |
| `Von_Mises_tNNNN`                           | point_data | `[N] float32`    | nodal Voigt stress present         |
| `stress_voigt_tNNNN`                        | point_data | `[N, 6] float32` | nodal Voigt stress present         |
| `cell_effective_plastic_strain_tNNNN`       | cell_data  | `[E] float32`    | element EPS present                |

Downstream note: the PhysicsNeMo drop_test recipe matches fields via regex. Its reader auto-detects the `tNNNN` indexed form produced here.

## Output — Zarr

One Zarr store per run (`<run_id>.zarr/`), compressed with `zstd` at level 3, chunked to ~`chunk_size_mb` MB per chunk (default `1.0`).

**Always-present arrays:**

| Array       | Shape / dtype       | Notes                                |
|-------------|---------------------|--------------------------------------|
| `mesh_pos`  | `[T, N, 3] float32` | Absolute positions per timestep      |
| `thickness` | `[N] float32`       | Zeros for solids                     |
| `edges`     | `[E, 2] int64`      | Unique edges from filtered connectivity |

**Optional arrays** — written only when the source provides them:

| Array                               | Shape / dtype          |
|-------------------------------------|------------------------|
| `node_velocity`                     | `[T, N, 3] float32`    |
| `node_acceleration`                 | `[T, N, 3] float32`    |
| `node_temperature`                  | `[T, N] float32`       |
| `node_residual_forces`              | `[T, N, 3] float32`    |
| `node_stress_voigt`                 | `[T, N, 6] float32`    |
| `node_stress_vm`                    | `[T, N] float32`       |
| `element_effective_plastic_strain`  | `[T, E] float32`       |
| `element_strain_voigt`              | `[T, E, 6] float32`    |
| `element_plastic_strain_voigt`      | `[T, E, 6] float32`    |

**Store attributes:** `filename`, `num_timesteps`, `num_nodes`, `num_edges`, `thickness_min`, `thickness_max`, `thickness_mean`.

## Configuration reference

All knobs are set via Hydra overrides on the command line or by editing the files under `config/`:

| Override                                       | Default                       | Meaning                                                     |
|------------------------------------------------|-------------------------------|-------------------------------------------------------------|
| `etl.source.input_dir`                         | *required*                    | Parent directory containing per-run folders                  |
| `etl.source.vtk_glob`                          | `"Cell_Phone_DropA*.vtk"`     | Per-timestep VTK filename pattern inside each run folder     |
| `etl.transformations.drop_test_transform.wall_threshold` | `1e-5`              | Displacement-variation cutoff for rigid-wall-node filtering  |
| `etl.processing.num_processes`                 | `12`                          | Parallel worker count                                        |
| `serialization_format`                         | `vtu`                         | Sink format: `vtu` or `zarr`                                 |
| `serialization_format.sink.output_dir`         | *required*                    | Output directory                                             |
| `serialization_format.sink.overwrite_existing` | `true`                        | Skip vs. overwrite existing outputs                          |
| `sink.time_step` (VTU)                         | `0.001`                       | Physical timestep size (metadata only; frame naming uses indices) |
| `sink.flip_triangle_normals` (VTU)             | `true`                        | Reverse triangle vertex order for VTK normal convention      |
| `sink.compression_method` (Zarr)               | `"zstd"`                      | Compressor                                                   |
| `sink.compression_level` (Zarr)                | `3`                           | Compressor level                                             |
| `sink.chunk_size_mb` (Zarr)                    | `1.0`                         | Target chunk size                                            |

## Downstream consumers

The VTU output of this ETL is consumed directly by `physicsnemo/examples/structural_mechanics/drop_test`. That recipe expects:

- `mesh.points` at reference coordinates
- `displacement_tNNNN` point vectors to reconstruct per-timestep positions
- `Von_Mises_tNNNN` point scalars as the stress target (when training with `dynamic_targets: [Von_Mises]`)

No post-processing of the ETL output is needed to feed the recipe.
