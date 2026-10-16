# Nastran modal curation

Read CTETRA4/10 models with PSOLID/MAT1 materials and paired real modal OP2
results. Each case becomes a native PhysicsNeMo mesh with node-aligned
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

CTETRA10 is subdivided into eight linear tetrahedra, preserving all midside
nodes and repeating parent cell fields. This approximates curved quadratic
elements with piecewise linear geometry. Solver units and normalization are
preserved. Other element/material families, nonrectangular output frames,
complex modes, stress/strain, energies, and transient analysis are not
supported. See the mesh domain documentation for the full field schema.

No customer samples are bundled. Tests generate small BDF and binary OP2
fixtures locally without a solver.
