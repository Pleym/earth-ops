"""Quality control: Scene Classification (SCL) masking, product statistics and status."""
from __future__ import annotations

import math

import xarray as xr

# SCL classes rejected for NDVI: 0 no data, 1 saturated/defective, 3 cloud shadow,
# 8 cloud medium probability, 9 cloud high probability, 10 thin cirrus
INVALID_SCL = (0, 1, 3, 8, 9, 10)
CLOUD_SCL = (8, 9, 10)
MIN_VALID_RATIO = 0.5
NOMINAL, DEGRADED = "NOMINAL", "DEGRADED"


def apply_mask(ndvi: xr.DataArray, scl: xr.DataArray) -> xr.DataArray:
    """Keep only clear pixels with a physically possible NDVI in [-1, 1]."""
    return ndvi.where(~scl.isin(INVALID_SCL) & (abs(ndvi) <= 1))


def quality_stats(ndvi_masked: xr.DataArray, scl: xr.DataArray) -> xr.Dataset:
    """Product statistics. Lazy when inputs are Dask-backed (computed with the product)."""
    return xr.Dataset({
        "valid_ratio": ndvi_masked.notnull().mean(),
        "cloud_ratio": scl.isin(CLOUD_SCL).mean(),
        "ndvi_mean": ndvi_masked.mean(),
        "ndvi_std": ndvi_masked.std(),
        "ndvi_min": ndvi_masked.min(),
        "ndvi_max": ndvi_masked.max(),
    })


def as_dict(stats: xr.Dataset) -> dict[str, float | None]:
    values = {name: float(value) for name, value in stats.data_vars.items()}
    return {k: None if math.isnan(v) else round(v, 6) for k, v in values.items()}


def assess(stats: dict, min_valid_ratio: float = MIN_VALID_RATIO) -> str:
    ratio = stats.get("valid_ratio")
    return NOMINAL if ratio is not None and ratio >= min_valid_ratio else DEGRADED
