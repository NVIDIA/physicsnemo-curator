# Nastran modal curation

Read CTETRA4/10 models with PSOLID/MAT1 materials and paired real modal OP2
results. Each case becomes a PhysicsNeMo mesh, saved natively or exported to
VTK, with node-aligned
mode shapes, rotations, frequencies, parent element/property/material IDs,
and SPC/MPC/RBE2 definitions. No Nastran installation or license is needed
to read existing files.

Install with Python 3.12:

```bash
pip install 'physicsnemo-curator[mesh,nastran]'
```

Arrange one deck and OP2 per case directory:

```text
cases/
  case_01/
    Input_model.deck
    Output_model.op2
  case_02/
    model.bdf
    model.op2
```

Run:

```bash
python examples/cae/nastran/main.py --input ./cases --output ./curated
# Full volume mesh for ParaView (recommended VTK export):
python examples/cae/nastran/main.py --input ./cases --output ./vtu --format vtu
# Legacy VTK volume mesh:
python examples/cae/nastran/main.py --input ./cases --output ./vtk --format vtk
# Boundary surface only (omits interior nodes and volume cells):
python examples/cae/nastran/main.py --input ./cases --output ./surfaces --format vtp
# Geometry only, without OP2:
python examples/cae/nastran/main.py --input ./cases --output ./geometry --geometry-only
```

The source prefers a same-stem OP2, otherwise requires a unique sibling
OP2. Use `--op2-pattern 'results_{stem}.op2'` for explicit filename pairing,
`--encoding cp932` for Japanese comments, and `--workers 4` to process
independent decks in parallel. `--punch` accepts bulk-data-only decks;
`.pch` files automatically use this mode.

Outputs mirror the input case directories. Each OP2 eigenvector table
(usually one subcase) yields a mesh with all of its modes. Reload with:

```python
from physicsnemo.mesh import Mesh

mesh = Mesh.load("./curated/case_01/Input_model_0.pmsh")
print(mesh.point_data["mode_shape"].shape)  # (nodes, modes, 3)
print(mesh.global_data["frequencies"])
```

The default `pmsh` format retains the native TensorDict structure for
training. `vtu` and legacy `vtk` retain the complete volume for visualization
and interchange. `vtp` exports its boundary surface, carrying the boundary
subset of nodal/cell results and original IDs. Interior nodal and volumetric
results are absent from VTP.

VTK exports expose each mode as separate vectors named `mode_shape_m0001`,
`mode_rotation_m0001`, etc., using the original mode numbers. In ParaView,
select a `mode_shape_m...` array in **Warp By Vector** and choose a scale
factor to inspect that mode. These arrays are modes, not time steps.

Frequencies, mode numbers, subcase selections, and nested constraint tables
are stored in VTK **Field Data** (e.g. `constraints/mpc_terms`). Integer IDs
and float64 precision are retained. `physicsnemo_schema` records original
tensor paths, shapes and types, including empty arrays that VTK cannot
store directly. VTP constraint metadata still refers to the full model's
original node IDs. VTK consumers do not automatically enforce these
constraints or reconstruct the native TensorDict structure.

CTETRA10 is subdivided into eight linear tetrahedra, preserving all midside
nodes and repeating parent cell fields. This approximates curved quadratic
elements with piecewise linear geometry. Solver units and normalization are
preserved. Other element/material families, nonrectangular output frames,
complex modes, stress/strain, energies, and transient analysis are not
supported. See the mesh domain documentation for the full field schema.

No customer samples are bundled. Tests generate small BDF and binary OP2
fixtures locally without a solver.
