"""Input validation: a product is rejected before processing if it is incomplete or inconsistent."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError

BANDS = {"B04": "uint16", "B08": "uint16", "SCL": "uint8"}
REQUIRED_METADATA = ("product_id", "datetime", "scale", "offset")
MAX_NODATA_RATIO = 0.99
# A negative median surface reflectance is physically impossible: scale/offset don't match the DNs
MIN_MEDIAN_REFLECTANCE = 0.0
DECIMATION = 8  # radiometric checks run on a 1/8 decimated read (cheap on large tiles)


class ValidationError(Exception):
    """Product rejected; `code` is a stable identifier used in KPIs and operations."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def load_metadata(product_dir: Path) -> dict:
    try:
        meta = json.loads((Path(product_dir) / "metadata.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("E_METADATA", f"unreadable metadata.json ({exc})") from exc
    if not isinstance(meta, dict):
        raise ValidationError("E_METADATA", "metadata.json must hold a JSON object")
    missing = [k for k in REQUIRED_METADATA if k not in meta]
    if missing:
        raise ValidationError("E_METADATA", f"missing keys {missing}")
    if not all(isinstance(meta[k], (int, float)) and math.isfinite(meta[k])
               for k in ("scale", "offset")):
        raise ValidationError("E_METADATA", "scale and offset must be finite numbers")
    try:
        datetime.fromisoformat(str(meta["datetime"]))
    except ValueError as exc:
        raise ValidationError("E_METADATA", f"invalid datetime {meta['datetime']!r}") from exc
    return meta


def validate_product(product_dir: Path) -> dict:
    """Run every input check; return the metadata (+ measured nodata ratio) if all pass."""
    product_dir = Path(product_dir)
    expected = [f"{b}.tif" for b in BANDS] + ["metadata.json"]
    missing = [f for f in expected if not (product_dir / f).is_file()]
    if missing:
        raise ValidationError("E_MISSING_FILE", f"missing {missing}")
    meta = load_metadata(product_dir)

    reference = None
    for band, dtype in BANDS.items():
        try:
            with rasterio.open(product_dir / f"{band}.tif") as src:
                grid, found = (src.crs, src.transform, src.shape), (src.count, src.dtypes[0])
        except RasterioIOError as exc:
            raise ValidationError("E_CORRUPT", f"{band}: {exc}") from exc
        if found != (1, dtype):
            raise ValidationError("E_DTYPE", f"{band}: {found[0]} band(s) of {found[1]}, "
                                             f"expected 1 of {dtype}")
        reference = grid if reference is None else reference
        if grid != reference:
            raise ValidationError("E_GEOMETRY",
                                  f"{band} grid (CRS/transform/shape) differs from B04")

    crs = reference[0]
    if crs is None or not crs.is_projected:
        raise ValidationError("E_CRS", f"expected a projected CRS (UTM), got {crs}")
    try:
        valid_dn, nodata_ratio = _sample_valid(product_dir / "B04.tif")
    except RasterioIOError as exc:  # header fine, compressed blocks broken
        raise ValidationError("E_CORRUPT", f"B04: {exc}") from exc
    if nodata_ratio > MAX_NODATA_RATIO:
        raise ValidationError("E_EMPTY", f"{nodata_ratio:.1%} of B04 is nodata")
    median_red = float(np.median(valid_dn)) * meta["scale"] + meta["offset"]
    if median_red < MIN_MEDIAN_REFLECTANCE:
        raise ValidationError("E_RADIOMETRY", f"median B04 reflectance {median_red:.3f} < 0: "
                                              "scale/offset inconsistent with the data")
    return {**meta, "nodata_ratio": round(nodata_ratio, 4)}


def _sample_valid(path: Path) -> tuple[np.ndarray, float]:
    """Valid DNs of a decimated read, and the nodata ratio."""
    with rasterio.open(path) as src:
        shape = (max(src.height // DECIMATION, 1), max(src.width // DECIMATION, 1))
        sample = src.read(1, out_shape=shape)
        nodata = 0 if src.nodata is None else src.nodata
    valid = sample[sample != nodata]
    return valid, 1 - valid.size / sample.size
