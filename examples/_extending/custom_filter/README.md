# Creating a Custom Filter

This example shows how to implement and register a custom Filter. We create a `LogTransformFilter`
that applies a `log1p` transform to a chosen variable in an xarray DataArray — a common
preprocessing step for ERA5 total precipitation, which has a highly skewed distribution.

## Prerequisites

```bash
pip install physicsnemo-curator[da]
```

## Usage

```bash
python main.py
```
