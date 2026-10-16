"""NDVI processing with Xarray (labelled rasters) and optional Dask block processing."""
from __future__ import annotations

import threading
from collections.abc import Sequence
from math import ceil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
import rioxarray  # noqa: E402
import xarray as xr  # noqa: E402

from earth_ops.ingest import GTIFF_OPTIONS  # noqa: E402
from earth_ops.quality import INVALID_SCL, apply_mask  # noqa: E402
from earth_ops.validate import load_metadata  # noqa: E402

NDVI_SCALE = 10_000  # NDVI stored as int16 = round(NDVI * 10000), like Copernicus Land products
NDVI_NODATA = -32768
# One L2A quantisation step. With the -0.1 offset, float32 sums are never exactly 0, and they
# can be negative on dark pixels: red + nir below one step means "no signal", not a valid NDVI.
MIN_REFLECTANCE_SUM = 1e-4
BAND_FILES = {"red": "B04.tif", "nir": "B08.tif", "scl": "SCL.tif"}
QUICKLOOK_MAX_PX = 1000
NDVI_CMAP = plt.get_cmap("RdYlGn").with_extremes(bad="white")
NDVI_VMIN, NDVI_VMAX = -0.5, 1.0


def open_product(product_dir: Path, chunks: int | None = None) -> xr.Dataset:
    """B04/B08/SCL as one Dataset on (y, x). chunks=None: in memory; int: lazy Dask blocks."""
    bands = {
        name: rioxarray.open_rasterio(Path(product_dir) / file, chunks=chunks, lock=False)
        .squeeze("band", drop=True)
        for name, file in BAND_FILES.items()
    }
    return xr.Dataset(bands)


def reflectance(dn: xr.DataArray, scale: float, offset: float) -> xr.DataArray:
    """Digital number -> surface reflectance; DN 0 is the Sentinel-2 nodata value."""
    dn = dn.astype(np.float32)
    return dn.where(dn != 0) * np.float32(scale) + np.float32(offset)


def compute_ndvi(red_dn: xr.DataArray, nir_dn: xr.DataArray, scale: float,
                 offset: float) -> xr.DataArray:
    """NDVI = (NIR - RED) / (NIR + RED), NaN where undefined. Works on (y, x) or (time, y, x)."""
    red, nir = reflectance(red_dn, scale, offset), reflectance(nir_dn, scale, offset)
    total = nir + red
    return ((nir - red) / total.where(total >= MIN_REFLECTANCE_SUM)).rename("ndvi")


def masked_ndvi_array(red_dn: np.ndarray, nir_dn: np.ndarray, scl: np.ndarray, scale: float,
                      offset: float) -> np.ndarray:
    """NumPy twin of compute_ndvi + quality.apply_mask, on whole in-memory arrays."""
    scale, offset = np.float32(scale), np.float32(offset)
    red = np.where(red_dn == 0, np.nan, red_dn.astype(np.float32) * scale + offset)
    nir = np.where(nir_dn == 0, np.nan, nir_dn.astype(np.float32) * scale + offset)
    total = nir + red
    with np.errstate(invalid="ignore"):
        ndvi = (nir - red) / np.where(total >= MIN_REFLECTANCE_SUM, total, np.nan)
        valid = ~np.isin(scl, INVALID_SCL) & (np.abs(ndvi) <= 1)
    return np.where(valid, ndvi, np.nan)


def open_ndvi_series(product_dirs: Sequence[Path], chunks: int | None = None) -> xr.DataArray:
    """Masked NDVI of several products stacked on a sorted `time` dimension: (time, y, x)."""
    layers = []
    for product_dir in product_dirs:
        meta = load_metadata(product_dir)  # scale/offset may differ between processing baselines
        ds = open_product(product_dir, chunks)
        ndvi = apply_mask(compute_ndvi(ds.red, ds.nir, meta["scale"], meta["offset"]), ds.scl)
        time = pd.to_datetime(meta["datetime"], utc=True).tz_convert(None)
        layers.append(ndvi.expand_dims(time=[time]))
    # join="exact": products on different grids must fail loudly, never be silently realigned
    return xr.concat(layers, dim="time", join="exact").sortby("time")


def write_ndvi(ndvi: xr.DataArray, path: Path) -> Any:
    """Write the int16 NDVI GeoTIFF. Dask input: returns a Delayed that streams blocks to disk."""
    scaled = (ndvi * NDVI_SCALE).round().fillna(NDVI_NODATA).astype(np.int16)
    if scaled.chunks:  # Dask blocks aligned on GeoTIFF tiles: no tile is rewritten (file bloat)
        tile = GTIFF_OPTIONS["blockxsize"]
        scaled = scaled.chunk({dim: -(-max(sizes) // tile) * tile
                               for dim, sizes in zip(scaled.dims, scaled.chunks, strict=True)})
    return scaled.rio.write_nodata(NDVI_NODATA).rio.to_raster(
        path, **GTIFF_OPTIONS, lock=threading.Lock(), compute=False)


def plot_ndvi_map(ndvi: np.ndarray, bounds: Any, png: Path, subtitle: str) -> None:
    """NDVI map, projected coordinates (m) on the axes; bounds = (left, bottom, right, top)."""
    left, bottom, right, top = bounds
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(ndvi, cmap=NDVI_CMAP, vmin=NDVI_VMIN, vmax=NDVI_VMAX,
                      extent=(left, right, bottom, top), interpolation="nearest")
    fig.colorbar(image, ax=ax, extend="min", shrink=0.9)
    ax.set_title(f"NDVI - Normalized Difference Vegetation Index\n{subtitle}", fontsize=11)
    ax.set_xlabel("X coordinate (m)")
    ax.set_ylabel("Y coordinate (m)")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_quicklook(ndvi_tif: Path, png: Path, subtitle: str) -> None:
    """Browse image (<= 1000 px), read decimated from the product itself."""
    with rasterio.open(ndvi_tif) as src:
        factor = max(1, ceil(max(src.shape) / QUICKLOOK_MAX_PX))
        shape = (ceil(src.height / factor), ceil(src.width / factor))
        ndvi = src.read(1, out_shape=shape, masked=True) / NDVI_SCALE
        bounds = src.bounds
    plot_ndvi_map(ndvi, bounds, png, subtitle)


def plot_series(stats: pd.DataFrame, png: Path, min_valid_ratio: float) -> None:
    """Mean NDVI per date; cloudy dates (valid_ratio below threshold) drawn hollow."""
    clear = stats["valid_ratio"] >= min_valid_ratio
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(stats.index[clear], stats["ndvi_mean"][clear], "o-", color="tab:green",
            label=f"valid pixels >= {min_valid_ratio:.0%}")
    if (~clear).any():
        ax.plot(stats.index[~clear], stats["ndvi_mean"][~clear], "o", mfc="none", color="grey",
                label="too cloudy (not trusted)")
    ax.set_ylabel("mean NDVI")
    ax.set_title("NDVI time series")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.savefig(png, dpi=120, bbox_inches="tight")
    plt.close(fig)
