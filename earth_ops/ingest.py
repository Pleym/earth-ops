"""Ingestion: fetch Sentinel-2 L2A bands from a public STAC catalogue into the local archive.

Only the area of interest is downloaded (windowed HTTP reads on Cloud Optimized GeoTIFFs),
and SCL (20 m) is resampled onto the 10 m grid of B04/B08 so every band of a product shares
one grid.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds
from rasterio.windows import bounds as window_bounds

log = logging.getLogger(__name__)

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
ASSETS = {"B04": "red", "B08": "nir", "SCL": "scl"}  # local band name -> STAC asset key
DEFAULT_SCALE, DEFAULT_OFFSET = 1e-4, 0.0  # products processed before baseline 04.00 (2022)
GDAL_ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "GDAL_HTTP_MAX_RETRY": "3",
            "GDAL_HTTP_RETRY_DELAY": "2"}
GTIFF_OPTIONS = {"driver": "GTiff", "tiled": True, "blockxsize": 256, "blockysize": 256,
                 "compress": "deflate", "predictor": 2}


def write_band(path: Path, data: np.ndarray, crs: Any, transform: Any, nodata: float) -> None:
    profile = {**GTIFF_OPTIONS, "height": data.shape[0], "width": data.shape[1], "count": 1,
               "dtype": data.dtype, "crs": crs, "transform": transform, "nodata": nodata}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def search(bbox: tuple[float, float, float, float], start: str, end: str,
           max_cloud: float = 40.0, limit: int = 10) -> list:
    """Up to `limit` items of one MGRS tile, one per date, evenly spread over the period."""
    from pystac_client import Client  # network-only dependency, imported on demand

    items = list(Client.open(STAC_URL).search(
        collections=[COLLECTION], bbox=bbox, datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}}, sortby="properties.datetime",
    ).items())
    if not items:
        return []
    # ponytail: keep the most frequent tile so every product shares one grid
    tile = Counter(_tile(i) for i in items).most_common(1)[0][0]
    dated = list({i.datetime.date(): i for i in items if _tile(i) == tile}.values())
    picks = np.unique(np.linspace(0, len(dated) - 1, min(limit, len(dated))).round().astype(int))
    return [dated[i] for i in picks]


def _tile(item: Any) -> str:
    return item.properties["grid:code"]


def _href(item: Any, key: str) -> str:
    """Asset URL from the catalogue (remote data): HTTPS only, no local or /vsi paths."""
    href = item.assets[key].href
    if not href.startswith("https://"):
        raise OSError(f"refusing non-HTTPS asset {href!r}")
    return href


def _scale_offset(item: Any) -> tuple[float, float]:
    band = (item.assets["red"].extra_fields.get("raster:bands") or [{}])[0]
    scale = float(band.get("scale", DEFAULT_SCALE))
    offset = float(band.get("offset", DEFAULT_OFFSET))
    # Earth Search removes the baseline-04.00 offset from the DNs of these items but still
    # advertises it in raster:bands. Found in operations: median red reflectance was < 0.
    if item.properties.get("earthsearch:boa_offset_applied"):
        offset = 0.0
    return scale, offset


def ingest_item(item: Any, bbox: tuple[float, float, float, float], archive: Path) -> Path:
    if not re.fullmatch(r"\w[\w.-]*", item.id):  # the id comes from the network: no "../"
        raise OSError(f"unsafe product id {item.id!r}")
    out = Path(archive) / item.id
    if (out / "metadata.json").exists():
        log.info("%s already archived, skipped", item.id)
        return out
    out.mkdir(parents=True, exist_ok=True)
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(_href(item, "red")) as ref:
            aoi = transform_bounds("EPSG:4326", ref.crs, *bbox)
            win = from_bounds(*aoi, transform=ref.transform).round_offsets().round_lengths()
            win = win.intersection(Window(0, 0, ref.width, ref.height))
            crs, transform = ref.crs, ref.window_transform(win)
            exact_bounds, shape = window_bounds(win, ref.transform), (win.height, win.width)
        for band, key in ASSETS.items():
            with rasterio.open(_href(item, key)) as src:
                data = src.read(1, window=from_bounds(*exact_bounds, transform=src.transform),
                                out_shape=shape, resampling=Resampling.nearest)
            write_band(out / f"{band}.tif", data, crs, transform, nodata=0)
    scale, offset = _scale_offset(item)
    meta = {
        "product_id": item.id, "datetime": item.datetime.isoformat(), "mgrs_tile": _tile(item),
        "cloud_cover": item.properties.get("eo:cloud_cover"),
        "processing_baseline": item.properties.get("s2:processing_baseline"),
        "scale": scale, "offset": offset, "bbox": list(bbox), "source": STAC_URL,
        "boa_offset_applied": item.properties.get("earthsearch:boa_offset_applied"),
    }
    # metadata.json is written last, atomically: it is the "product complete" marker
    partial = out / "metadata.json.partial"
    partial.write_text(json.dumps(meta, indent=2))
    partial.replace(out / "metadata.json")
    log.info("%s ingested (%dx%d px, cloud %.0f %%)", item.id, shape[1], shape[0],
             meta["cloud_cover"] or 0)
    return out


def ingest(bbox: tuple[float, float, float, float], start: str, end: str, max_cloud: float,
           limit: int, archive: Path) -> list[Path]:
    items = search(bbox, start, end, max_cloud, limit)
    log.info("%d product(s) found in %s", len(items), STAC_URL)
    products = []
    for item in items:
        try:
            products.append(ingest_item(item, bbox, archive))
        except (rasterio.errors.RasterioError, OSError, KeyError) as exc:
            # one bad product must not stop the batch
            log.error("%s ingestion failed: %s", item.id, exc)
    return products
