<!---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
-->

# Domains

PhysicsNeMo Curator ships domain-specific pipeline components for different
scientific data types.  Each domain provides its own sources, filters, and sinks
that communicate through a single data structure.

| Domain | Data type | Dependency group |
|--------|-----------|------------------|
| **Mesh** | `physicsnemo.mesh.Mesh` (tensorclass) | `mesh` |
| **DataArray** | `xarray.DataArray` | `da` |
| **Atomic** | `nvalchemi.data.AtomicData` | `atm` |

## Mesh

```{toctree}
:maxdepth: 2

mesh
/autoapi/physicsnemo_curator/mesh/index
```

## DataArray

```{toctree}
:maxdepth: 2

da
/autoapi/physicsnemo_curator/da/index
```

## Atomic

```{toctree}
:maxdepth: 2

atm
/autoapi/physicsnemo_curator/atm/index
```
