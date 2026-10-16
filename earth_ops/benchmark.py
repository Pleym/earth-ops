"""Benchmark: classic Rasterio + NumPy vs Xarray + Dask blocks, on the same product.

Each configuration runs in a fresh subprocess so that its peak memory (RSS) is measured in
isolation. Both implementations are independent, so comparing their outputs is also a
numerical cross-check of the NDVI chain.
"""
from __future__ import annotations

import json
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

from earth_ops import quality
from earth_ops.ingest import GTIFF_OPTIONS
from earth_ops.ndvi import (NDVI_NODATA, NDVI_SCALE, compute_ndvi, masked_ndvi_array,
                              open_product, plt, write_ndvi)
from earth_ops.validate import load_metadata

DEFAULT_CHUNKS = (256, 512, 1024, 2048)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RSS_TO_BYTES = 1 if sys.platform == "darwin" else 1024  # ru_maxrss: bytes on macOS, KiB on Linux


def ndvi_numpy(product_dir: Path, out_path: Path) -> None:
    """Approach A: whole bands in RAM, NumPy arithmetic, single write."""
    meta = load_metadata(product_dir)
    with rasterio.open(product_dir / "B04.tif") as src:
        red_dn, profile = src.read(1), src.profile
    with rasterio.open(product_dir / "B08.tif") as src:
        nir_dn = src.read(1)
    with rasterio.open(product_dir / "SCL.tif") as src:
        scl = src.read(1)
    ndvi = masked_ndvi_array(red_dn, nir_dn, scl, meta["scale"], meta["offset"])
    out = np.nan_to_num(np.round(ndvi * NDVI_SCALE), nan=NDVI_NODATA).astype(np.int16)
    with rasterio.open(out_path, "w", **{**profile, **GTIFF_OPTIONS, "dtype": "int16",
                                         "nodata": NDVI_NODATA}) as dst:
        dst.write(out, 1)


def ndvi_dask(product_dir: Path, out_path: Path, chunk: int) -> None:
    """Approach B: lazy Xarray + Dask blocks, streamed to disk block by block."""
    meta = load_metadata(product_dir)
    ds = open_product(product_dir, chunks=chunk)
    ndvi = quality.apply_mask(compute_ndvi(ds.red, ds.nir, meta["scale"], meta["offset"]), ds.scl)
    write_ndvi(ndvi, out_path).compute()


def measure(method: str, chunk: int, product_dir: Path, out_path: Path) -> dict:
    """Worker side: run one configuration and report wall time and peak RSS."""
    t0 = time.perf_counter()
    if method == "numpy":
        ndvi_numpy(product_dir, out_path)
    elif method == "dask":
        ndvi_dask(product_dir, out_path, chunk)
    seconds = time.perf_counter() - t0  # method "baseline" only measures the interpreter + libs
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * RSS_TO_BYTES / 2**20
    return {"seconds": seconds, "peak_rss_mb": peak}


def _spawn(method: str, chunk: int | None, product_dir: Path, out_path: Path) -> dict:
    cmd = [sys.executable, "-m", "earth_ops.benchmark", method, str(chunk or 0),
           str(Path(product_dir).resolve()), str(Path(out_path).resolve())]
    done = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    if done.returncode != 0:
        raise RuntimeError(f"benchmark worker {method}/{chunk} failed:\n{done.stderr[-2000:]}")
    return json.loads(done.stdout.splitlines()[-1])


def compare(reference: Path, candidate: Path) -> tuple[float, int]:
    """Max |NDVI difference| on pixels valid in both, and number of mask disagreements."""
    with rasterio.open(reference) as a, rasterio.open(candidate) as b:
        x, y = a.read(1), b.read(1)
    nodata_x, nodata_y = x == NDVI_NODATA, y == NDVI_NODATA
    both = ~nodata_x & ~nodata_y
    diff = np.abs(x[both].astype(np.int32) - y[both]).max(initial=0) / NDVI_SCALE
    return float(diff), int(np.count_nonzero(nodata_x != nodata_y))


def run_benchmark(product_dir: Path, workdir: Path, chunks: tuple[int, ...] = DEFAULT_CHUNKS,
                  repeats: int = 1) -> list[dict]:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(Path(product_dir) / "B04.tif") as src:
        pixels = src.width * src.height
    reference = workdir / "ndvi_numpy.tif"
    outputs = {c: workdir / f"ndvi_dask_{c}.tif" for c in chunks}
    try:
        baseline = _spawn("baseline", None, product_dir, reference)["peak_rss_mb"]
        results = []
        for method, chunk in [("numpy", None)] + [("dask", c) for c in chunks]:
            out = reference if method == "numpy" else outputs[chunk]
            runs = [_spawn(method, chunk, product_dir, out) for _ in range(repeats)]
            seconds = statistics.median(r["seconds"] for r in runs)
            diff, mismatch = compare(reference, out)
            if out != reference:
                out.unlink()  # disk budget: keep only the reference output
            results.append({
                "method": method, "chunk": chunk, "seconds": round(seconds, 3),
                "peak_rss_mb": round(max(r["peak_rss_mb"] for r in runs), 1),
                "baseline_rss_mb": round(baseline, 1),
                "mpx_per_s": round(pixels / seconds / 1e6, 2),
                "max_abs_diff": diff, "mask_mismatch": mismatch, "pixels": pixels,
            })
    finally:  # disk budget, even when a worker fails
        for path in [reference, *outputs.values()]:
            path.unlink(missing_ok=True)
    return results


def format_table(results: list[dict]) -> str:
    lines = ["| method | chunk | time (s) | peak RSS (MB) | throughput (Mpx/s) "
             "| max abs dNDVI | mask mismatch |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['method']} | {r['chunk'] or '-'} | {r['seconds']:.2f} "
              f"| {r['peak_rss_mb']:.0f} | {r['mpx_per_s']:.1f} | {r['max_abs_diff']:.4f} "
              f"| {r['mask_mismatch']} |"
              for r in results]
    return "\n".join(lines)


def plot_results(results: list[dict], png: Path) -> None:
    labels = ["NumPy\n(whole scene)" if r["method"] == "numpy" else f"Dask\n{r['chunk']} px"
              for r in results]
    colors = ["tab:grey" if r["method"] == "numpy" else "tab:blue" for r in results]
    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(11, 4))
    ax_t.bar(labels, [r["seconds"] for r in results], color=colors)
    ax_t.set_ylabel("wall time (s)")
    ax_m.bar(labels, [r["peak_rss_mb"] for r in results], color=colors)
    ax_m.axhline(results[0]["baseline_rss_mb"], ls="--", color="k", lw=1,
                 label="Python + libraries")
    ax_m.set_ylabel("peak RSS (MB)")
    ax_m.legend()
    side = int(results[0]["pixels"] ** 0.5)
    fig.suptitle(f"NDVI on {side}x{side} px: read -> NDVI -> QC mask -> GeoTIFF")
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)


if __name__ == "__main__":  # worker entry point used by _spawn
    _method, _chunk, _product, _out = sys.argv[1:5]
    print(json.dumps(measure(_method, int(_chunk), Path(_product), Path(_out))))
