# Nastran curation

Convert Nastran decks and paired OP2 results to native PhysicsNeMo meshes,
VTU, legacy VTK, or VTP. No Nastran solver or license is needed to read
existing files. Install the reader explicitly (validated with Python 3.12):

```bash
pip install 'physicsnemo-curator[mesh]' 'pyNastran>=1.4.1,<1.5'
```

Arrange one deck and OP2 per case directory, then run:

```bash
python examples/cae/nastran/main.py --input ./cases --output ./curated
# Original cells, including quadratic elements, for ParaView:
python examples/cae/nastran/main.py --input ./cases --output ./vtu --format vtu
python examples/cae/nastran/main.py --input ./cases --output ./vtk --format vtk
# Boundary surface with the full global result/source archive:
python examples/cae/nastran/main.py --input ./cases --output ./surfaces --format vtp
# Geometry without OP2:
python examples/cae/nastran/main.py --input ./cases --output ./geometry --geometry-only
```

The source prefers a same-stem OP2, otherwise requires a unique sibling
OP2. Use `--op2-pattern 'results_{stem}.op2'` for explicit pairing,
`--encoding cp932` for Japanese comments, and `--workers 4` to process
independent decks in parallel. `--punch` accepts bulk-data-only decks;
`.pch` files automatically use this mode. If INCLUDE files also have deck
extensions, pass the main deck directly or use a discovery glob such as
`--file-pattern '**/Input*.deck'`. Outputs mirror case directories.
Each subcase and topological dimension yields a separate mesh.

All OP2 table families exposed by pyNastran are discovered by default.
Standalone tables, nested PSD families, and sparse monitor tables are
included. Their numeric arrays, IDs, time/frequency/mode axes, component headers,
and other attributes are retained under `global_data["nastran", "results"]`.
This includes complex, transient, energy, and layered stress/strain tables.
Unambiguous GRID and element tables also become named mesh fields;
sparse fields contain NaN and a `__valid` mask. Layered/integration-point
results remain raw, without guessed averaging or coordinate transforms.
Superelement/adaptive IDs require a dedicated association adapter and also
remain raw.
Use `--result-patterns 'displacements,strain_energy.*'` to select families;
the catalog records exclusions. Matrix names use `matrices.<name>`.

```python
from physicsnemo.mesh import Mesh
from physicsnemo_curator.domains.mesh.sources.nastran import (
    decode_nastran_json,
    restore_nastran_sources,
)

mesh = Mesh.load("./curated/case_01/Input_model_0.pmsh")
print(decode_nastran_json(mesh.global_data["nastran", "catalog"]))
print(decode_nastran_json(mesh.global_data["nastran", "projection"]))
# Restore original deck, INCLUDE files, and OP2 into a new directory:
restore_nastran_sources(mesh, "./recovered_inputs")
```

Original file bytes and SHA-256 checksums are embedded by default, so
backend-unsupported content remains recoverable. This increases output
size, particularly when one input yields several meshes. Use
`--no-source-files` to omit these bytes; uninterpreted content then cannot
be recovered. Parsed deck cards (including materials, properties, loads,
constraints, parameters, and control text) remain in `deck_metadata`.
This archive does not make every Nastran card an interpreted mesh feature.

Native `.pmsh` uses a piecewise linear simplicial projection and retains
original connectivity/types alongside it. CTETRA10 becomes eight linear
tetrahedra. Parent cell fields repeat on children: do not sum child strain
energies. Original element values are in `nastran/element_data`. VTU/VTK
export original cells and parent fields by default. Use
`--projected-geometry` to export the native linear cells, including any
transformed cell features. Original-cell export rejects removed, duplicated,
or remeshed native cells; point/cell permutations remain valid. Supported geometry includes common rods/beams,
triangles/quads, tetrahedra, hexahedra, wedges, and pyramids. Unsupported
topology fails explicitly.

Complete real modes also expose `mode_shape`, `mode_rotation`, frequencies,
and original mode numbers. VTK splits modal aliases into vectors such as
`mode_shape_m0001`; select one in ParaView's **Warp By Vector**. Complex
fields use separate real/imaginary arrays. XML ASCII export rejects
negative infinity; use binary output (the default) or legacy VTK ASCII
to retain those values exactly. Solver units and normalization
are unchanged. Recognized vectors in rectangular output frames are rotated
to basic axes; cylindrical/spherical results stay raw with a report.

Nested tensors are stored in VTK Field Data with a `physicsnemo_schema`
recording paths, shapes, dtypes, and empty arrays. Restore their structure:

```python
from physicsnemo_curator.domains.mesh.sinks.mesh_vtk import read_vtk_data

fields = read_vtk_data("./vtu/case_01/Input_model_0.vtu")
restore_nastran_sources(fields["global_data"], "./recovered_from_vtk")
```

VTP point/cell arrays represent the boundary only. Its global archive still
contains the full original result tables and embedded inputs; consumers
must read Field Data to access them. VTK does not enforce constraints or
interpret the Nastran archive automatically.

pyNastran 1.4.1 requires NumPy 1.x. Keep that compatibility requirement in
the environment explicitly used for this reader. From a source checkout,
`uv run --extra mesh --with 'pyNastran>=1.4.1,<1.5' python ...` provides the
reader for one invocation. Curator's default dependencies, Python
requirement, and shared lockfile remain unchanged.

No customer samples are bundled. Tests generate BDF and binary OP2 fixtures
without a solver. See the mesh domain documentation for field paths and
interpretation limits.
