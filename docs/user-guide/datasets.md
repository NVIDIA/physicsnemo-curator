# Datasets

PhysicsNeMo Curator provides built-in sources for several curated scientific
datasets.  Each source handles file discovery, caching, and format conversion
transparently — see the linked API pages for full parameter descriptions,
usage examples, and reference links.

## Mesh Sources

Sources that yield {py:class}`~physicsnemo.mesh.Mesh` objects from CFD
benchmark datasets hosted on [HuggingFace Hub](https://huggingface.co).
Requires the `mesh` dependency group (`uv sync --group mesh`).

```{eval-rst}
.. autosummary::
   :nosignatures:

   physicsnemo_curator.mesh.sources.vtk.VTKSource
   physicsnemo_curator.mesh.sources.drivaerml.DrivAerMLSource
   physicsnemo_curator.mesh.sources.ahmedml.AhmedMLSource
   physicsnemo_curator.mesh.sources.windsorml.WindsorMLSource
   physicsnemo_curator.mesh.sources.windtunnel.WindTunnelSource
   physicsnemo_curator.mesh.sources.ns_cylinder.NavierStokesCylinderSource
```

## DataArray Sources

Sources that yield {py:class}`xarray.DataArray` objects from reanalysis
and climate datasets.  Requires the `da` dependency group
(`uv sync --group da`).

```{eval-rst}
.. autosummary::
   :nosignatures:

   physicsnemo_curator.da.sources.era5.ERA5Source
```
