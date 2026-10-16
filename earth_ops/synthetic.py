"""Product simulator: Sentinel-2 L2A-like products for tests, demos and benchmarks.

A ground segment is never tested only on live data: simulated products with known content
make every check reproducible (and allow fault injection).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rasterio.transform import from_origin

from earth_ops.ingest import write_band

CRS = "EPSG:32631"
ORIGIN = (360000.0, 4840000.0)  # UTM 31N, Toulouse area
PIXEL_M = 10.0
SCALE, OFFSET = 1e-4, -0.1  # L2A processing baseline >= 04.00
SCL_VEGETATION, SCL_CLOUD_HIGH = 4, 9
FIELD_PX = 32  # side of one synthetic agricultural field
NOISE = 0.003  # reflectance noise (1 sigma)


def to_dn(reflectance: np.ndarray) -> np.ndarray:
    """Reflectance -> L2A digital number (DN 0 is the nodata value)."""
    dn = np.round((np.asarray(reflectance, np.float32) - OFFSET) / SCALE)
    return np.clip(dn, 0, 65535).astype(np.uint16)


def write_product(archive: Path, product_id: str, red: np.ndarray, nir: np.ndarray,
                  scl: np.ndarray, date: str = "2025-06-01", cloud_cover: float = 0.0) -> Path:
    out = Path(archive) / product_id
    out.mkdir(parents=True, exist_ok=True)
    transform = from_origin(*ORIGIN, PIXEL_M, PIXEL_M)
    for band, data in (("B04", red), ("B08", nir), ("SCL", scl)):
        write_band(out / f"{band}.tif", data, CRS, transform, nodata=0)
    meta = {"product_id": product_id, "datetime": f"{date}T10:30:00Z", "mgrs_tile": "31TCJ",
            "cloud_cover": cloud_cover, "processing_baseline": "05.11", "scale": SCALE,
            "offset": OFFSET, "source": "simulator"}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    return out


def make_product(archive: Path, size: int = 256, seed: int = 0, cloud_fraction: float = 0.1,
                 date: str = "2025-06-01") -> Path:
    """Fields of random vigour + noise; clouds cover the top `cloud_fraction` of the rows."""
    rng = np.random.default_rng(seed)
    n = -(-size // FIELD_PX)
    vigour = rng.uniform(0, 1, (n, n)).astype(np.float32)
    vigour = vigour.repeat(FIELD_PX, 0).repeat(FIELD_PX, 1)[:size, :size]
    red = 0.12 - 0.08 * vigour + NOISE * rng.standard_normal((size, size), np.float32)
    nir = 0.15 + 0.35 * vigour + NOISE * rng.standard_normal((size, size), np.float32)
    scl = np.full((size, size), SCL_VEGETATION, np.uint8)
    scl[: round(cloud_fraction * size)] = SCL_CLOUD_HIGH
    product_id = f"SIM_{date.replace('-', '')}_{seed:03d}"
    return write_product(archive, product_id, to_dn(red), to_dn(nir), scl, date,
                         round(cloud_fraction * 100, 1))
