# Nastran curation

Convert Nastran decks and paired OP2 results to native PhysicsNeMo meshes,
VTU, legacy VTK, or VTP. Install the optional reader in the environment
used for this example:

```bash
pip install 'physicsnemo-curator[mesh]' 'pyNastran>=1.4.1,<1.5'
```

From a source checkout, provide the reader for one invocation:

```bash
uv run --extra mesh --with 'pyNastran>=1.4.1,<1.5' \
  python examples/cae/nastran/main.py --input ./cases --output ./curated
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

The source prefers a same-stem OP2, otherwise requires a unique sibling.
Use `--op2-pattern 'results_{stem}.op2'` for explicit pairing,
`--encoding cp932` for Japanese comments, and `--workers 4` for parallel
cases. `--punch` accepts bulk-data-only decks; `.pch` selects this
automatically. When INCLUDE files also have deck extensions, pass the main
deck directly or use `--file-pattern '**/Input*.deck'`.

Outputs mirror case directories, with one mesh per subcase and topological
dimension. OP2 fields are discovered dynamically. Original tables, parsed
cards, connectivity, and source bytes are preserved in global data.
`--result-patterns` selects result families; `--no-source-files` omits input
bytes to reduce storage at the cost of exact source recovery.

Native `.pmsh` uses linear simplices and repeats parent cell fields on
children; do not sum repeated child strain energies. VTU/VTK preserves
original cells by default. Use `--projected-geometry` after modifying native
cells or cell fields. VTP displays the boundary subset and retains the full
global archive. Complete real modes provide vectors such as
`mode_shape_m0001` for ParaView's **Warp By Vector**.

See the [mesh domain guide](../../../docs/domains/mesh.md#nastransource)
for supported topologies, field layouts, coordinate conventions, raw-result
limitations, and source recovery. No solver or bundled customer data is
required; tests generate synthetic BDF/OP2 fixtures.
