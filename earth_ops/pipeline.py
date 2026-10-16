"""Orchestration: validate -> process (NDVI + QC in one Dask pass) -> quicklook, with KPIs."""
from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import dask
import xarray as xr

from earth_ops import quality
from earth_ops.monitoring import DEFAULT_KPI_FILE, RunMonitor
from earth_ops.ndvi import (compute_ndvi, open_ndvi_series, open_product, plot_series,
                              save_quicklook, write_ndvi)
from earth_ops.validate import ValidationError, validate_product

log = logging.getLogger(__name__)
REJECTED, FAILED = "REJECTED", "FAILED"
COMPLETED = (quality.NOMINAL, quality.DEGRADED)
SERIES_CHUNKS = 1024


def run_product(product_dir: Path, out_dir: Path, kpi_file: Path = DEFAULT_KPI_FILE,
                chunks: int | None = None,
                min_valid_ratio: float = quality.MIN_VALID_RATIO) -> dict:
    """Process one product end to end. Never raises: the outcome is the returned KPI record."""
    product_dir = Path(product_dir)
    out = Path(out_dir) / product_dir.name
    mon = RunMonitor(product_dir.name, kpi_file)
    try:
        with mon.stage("validate"):
            meta = validate_product(product_dir)
        with mon.stage("process"):
            stats, pixels = _process(product_dir, meta, out, chunks)
        with mon.stage("quicklook"):
            save_quicklook(out / "NDVI.tif", out / "quicklook.png",
                           f"NDVI {meta['product_id']}\n{meta['datetime'][:10]}")
    except ValidationError as exc:
        return mon.record(REJECTED, error_code=exc.code, error=str(exc))
    except Exception as exc:  # any fault becomes a FAILED record instead of stopping the batch
        log.exception("[%s] unexpected processing error", mon.run_id)
        return mon.record(FAILED, error_code="E_PROCESSING", error=f"{type(exc).__name__}: {exc}")
    elapsed = sum(mon.stages_s.values())
    return mon.record(quality.assess(stats, min_valid_ratio), chunks=chunks, pixels=pixels,
                      throughput_mpx_s=round(pixels / elapsed / 1e6, 3), **stats)


def _process(product_dir: Path, meta: dict, out: Path, chunks: int | None) -> tuple[dict, int]:
    ds = open_product(product_dir, chunks)
    ndvi = quality.apply_mask(compute_ndvi(ds.red, ds.nir, meta["scale"], meta["offset"]), ds.scl)
    out.mkdir(parents=True, exist_ok=True)
    (stats,) = _deliver(out / "NDVI.tif", ndvi, quality.quality_stats(ndvi, ds.scl))
    return quality.as_dict(stats), int(ndvi.size)


def _deliver(path: Path, ndvi: xr.DataArray, *extra: object) -> list:
    """Write `ndvi` to `path` atomically; `extra` is computed in the same Dask pass.

    One graph: input blocks are read once and feed both the writer and the statistics.
    A delivered product is always complete, and a failure leaves no partial file behind.
    """
    partial = path.with_suffix(".partial.tif")
    try:
        _, *results = dask.compute(write_ndvi(ndvi, partial), *extra)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(path)
    return results


def run_series(product_dirs: Sequence[Path], out_dir: Path, chunks: int | None = SERIES_CHUNKS,
               min_valid_ratio: float = quality.MIN_VALID_RATIO) -> Path:
    """Time series of validated products: per-date stats (CSV + plot) and a median composite."""
    valid = [p for p in map(Path, product_dirs) if _is_valid(p)]
    if not valid:
        raise ValueError("no valid product for the time series")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cube = open_ndvi_series(valid, chunks)
    per_date = xr.Dataset({"ndvi_mean": cube.mean(["y", "x"]),
                           "valid_ratio": cube.notnull().mean(["y", "x"])})
    composite_tif = out / "ndvi_median_composite.tif"
    composite = cube.chunk({"time": -1}).median("time")  # clouds are NaN, so the median skips them
    (stats,) = _deliver(composite_tif, composite, per_date)
    table = stats.drop_vars("spatial_ref", errors="ignore").to_dataframe()
    table.to_csv(out / "ndvi_timeseries.csv", float_format="%.4f")
    plot_series(table, out / "ndvi_timeseries.png", min_valid_ratio)
    save_quicklook(composite_tif, out / "ndvi_median_composite.png",
                   f"Median NDVI composite ({len(valid)} dates)")
    return out / "ndvi_timeseries.csv"


def _is_valid(product_dir: Path) -> bool:
    try:
        validate_product(product_dir)
    except ValidationError as exc:
        log.warning("%s excluded from the series: %s", product_dir.name, exc)
        return False
    return True
