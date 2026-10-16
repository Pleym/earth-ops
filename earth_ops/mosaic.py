"""National NDVI mosaic: Sentinel-2 L2A over a whole country, at coarse resolution.

For each MGRS tile and each month, the clearest scene is read from the overviews of the
Cloud Optimized GeoTIFFs (hundreds of kB per band instead of ~200 MB), warped on the fly to
Lambert-93 and merged as a maximum value composite: the per-pixel maximum NDVI, the classic
MODIS/AVHRR synthesis (clouds and haze lower NDVI, so the maximum favours clear views).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine, array_bounds, from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

from earth_ops.ingest import COLLECTION, GDAL_ENV, STAC_URL, _href, _scale_offset, write_band
from earth_ops.ndvi import NDVI_NODATA, NDVI_SCALE, masked_ndvi_array, plot_ndvi_map

log = logging.getLogger(__name__)

FRANCE_BBOX = (-5.2, 41.3, 9.6, 51.1)  # metropolitan France, lon/lat
MOSAIC_CRS = "EPSG:2154"  # Lambert-93, the French official projection
WORKERS = 8  # network-bound: threads overlap HTTP latency
SCL_WATER = 6  # masked on the national map so that coastlines and lakes stand out


def best_scenes(bbox: tuple[float, ...], start: str, end: str, max_cloud: float) -> list:
    """The clearest, most complete scene of every MGRS tile for every month of the period.

    Stratifying by month guarantees each tile a spring date when one exists: picking the N
    clearest scenes overall gave some tiles July (post-harvest) dates only, hence visible seams.
    """
    from pystac_client import Client  # network-only dependency, imported on demand

    by_slot = defaultdict(list)
    for item in Client.open(STAC_URL).search(
        collections=[COLLECTION], bbox=bbox, datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    ).items():
        by_slot[item.properties["grid:code"], item.datetime.month].append(item)

    def badness(item: Any) -> float:
        props = item.properties
        return props["eo:cloud_cover"] + props.get("s2:nodata_pixel_percentage", 0)

    return [min(items, key=badness) for items in by_slot.values()]


def mosaic_grid(bbox: tuple[float, ...], resolution: float) -> tuple[Affine, tuple[int, int]]:
    left, bottom, right, top = transform_bounds("EPSG:4326", MOSAIC_CRS, *bbox)
    shape = (ceil((top - bottom) / resolution), ceil((right - left) / resolution))
    return from_origin(left, top, resolution, resolution), shape


def _open_coarse(href: str, resolution: float) -> Any:
    """Open the coarsest overview still finer than `resolution`: kB of HTTP reads, not MB."""
    with rasterio.open(href) as src:
        native, factors = src.res[0], src.overviews(1)
    usable = [level for level, f in enumerate(factors) if native * f <= resolution]
    return rasterio.open(href, overview_level=usable[-1]) if usable else rasterio.open(href)


def scene_ndvi(item: Any, transform: Affine, shape: tuple[int, int]) -> tuple | None:
    """Masked NDVI of one scene on its window of the mosaic grid (None if rejected)."""
    bands, window = {}, None
    with rasterio.Env(**GDAL_ENV):
        for key, resampling in (("red", Resampling.average), ("nir", Resampling.average),
                                ("scl", Resampling.nearest)):
            with _open_coarse(_href(item, key), transform.a) as src, WarpedVRT(
                src, crs=MOSAIC_CRS, transform=transform, width=shape[1], height=shape[0],
                resampling=resampling,
            ) as vrt:
                if window is None:
                    tile = transform_bounds(src.crs, MOSAIC_CRS, *src.bounds)
                    window = (from_bounds(*tile, transform=transform).round_offsets()
                              .round_lengths().intersection(Window(0, 0, shape[1], shape[0])))
                bands[key] = vrt.read(1, window=window)
    scale, offset = _scale_offset(item)
    red_dn = bands["red"][bands["red"] > 0]
    if red_dn.size and np.median(red_dn) * scale + offset < 0:  # same check as REQ-VAL-08
        log.warning("%s rejected: scale/offset inconsistent with the data", item.id)
        return None
    ndvi = masked_ndvi_array(bands["red"], bands["nir"], bands["scl"], scale, offset)
    return np.where(bands["scl"] == SCL_WATER, np.nan, ndvi), window


def build_mosaic(items: list, transform: Affine, shape: tuple[int, int]) -> np.ndarray:
    """Maximum value composite of all scenes; NaN where no clear observation exists."""
    mosaic = np.full(shape, np.nan, np.float32)
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = {pool.submit(scene_ndvi, item, transform, shape): item for item in items}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
            except (rasterio.errors.RasterioError, OSError) as exc:
                log.error("%s skipped: %s", futures[future].id, exc)  # one scene never stops all
                continue
            if result is not None:
                ndvi, window = result
                rows, cols = window.toslices()
                mosaic[rows, cols] = np.fmax(mosaic[rows, cols], ndvi)
            if done % 20 == 0:
                log.info("%d / %d scenes merged", done, len(futures))
    return mosaic


def run_mosaic(bbox: tuple[float, ...], start: str, end: str, max_cloud: float,
               resolution: float, out_dir: Path) -> Path:
    items = best_scenes(bbox, start, end, max_cloud)
    log.info("%d scenes selected over %d tiles", len(items),
             len({i.properties["grid:code"] for i in items}))
    transform, shape = mosaic_grid(bbox, resolution)
    mosaic = build_mosaic(items, transform, shape)
    log.info("mosaic %dx%d px, %.0f %% of the grid observed", shape[1], shape[0],
             100 * np.isfinite(mosaic).mean())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scaled = np.nan_to_num(np.round(mosaic * NDVI_SCALE), nan=NDVI_NODATA).astype(np.int16)
    write_band(out / "ndvi_mosaic.tif", scaled, MOSAIC_CRS, transform, NDVI_NODATA)
    plot_ndvi_map(mosaic, array_bounds(*shape, transform), out / "ndvi_mosaic.png",
                  f"Sentinel-2 L2A {start} / {end}, max composite, {resolution:.0f} m")
    return out / "ndvi_mosaic.tif"
