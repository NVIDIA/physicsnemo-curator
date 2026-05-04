# Creating a Custom Sink

This example shows how to implement and register a custom Sink. We create an `HDF5Sink` that writes
xarray DataArray fields to HDF5 files — one file per source index, with each variable stored as a
separate dataset.

## Prerequisites

```bash
pip install physicsnemo-curator[da] h5py
```

## Usage

```bash
python main.py
```
