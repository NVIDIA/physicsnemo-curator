# Datasets

PhysicsNeMo Curator provides built-in sources for several curated scientific
datasets.  Each source handles file discovery, caching, and format conversion
transparently — see the linked API pages for full parameter descriptions,
usage examples, and reference links.

## Mesh Sources

Sources that yield {py:class}`~physicsnemo.mesh.Mesh` objects from CFD
benchmark datasets hosted on [HuggingFace Hub](https://huggingface.co).
Requires the `mesh` dependency group (`uv sync --group mesh`).

```{list-table}
:widths: 30 70

* - {py:class}`~physicsnemo_curator.domains.mesh.sources.vtk.VTKSource`
  - Generic VTK file source
* - {py:class}`~physicsnemo_curator.domains.mesh.sources.drivaerml.DrivAerMLSource`
  - DrivAerML dataset source
* - {py:class}`~physicsnemo_curator.domains.mesh.sources.ahmedml.AhmedMLSource`
  - AhmedML dataset source
* - {py:class}`~physicsnemo_curator.domains.mesh.sources.windsorml.WindsorMLSource`
  - WindsorML dataset source
* - {py:class}`~physicsnemo_curator.domains.mesh.sources.windtunnel.WindTunnelSource`
  - Wind Tunnel dataset source
* - {py:class}`~physicsnemo_curator.domains.mesh.sources.ns_cylinder.NavierStokesCylinderSource`
  - Navier-Stokes Cylinder dataset source
```

## DataArray Sources

Sources that yield {py:class}`xarray.DataArray` objects from reanalysis
and climate datasets.  Requires the `da` dependency group
(`uv sync --group da`).

```{list-table}
:widths: 30 70

* - {py:class}`~physicsnemo_curator.domains.da.sources.era5.ERA5Source`
  - ERA5 reanalysis dataset source
```

## AtomicData Sources

Sources that yield {py:class}`~nvalchemi.data.AtomicData` objects from
atomic/molecular simulation datasets.  Requires the `atm` dependency
group (`uv sync --group atm`).

```{list-table}
:widths: 30 70

* - {py:class}`~physicsnemo_curator.domains.atm.sources.aselmdb.ASELMDBSource`
  - ASE LMDB atomic data source (OMol25, OPoly26)
```
