"""Command line interface. Usage: python -m earth_ops <command> --help"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from earth_ops import anomaly_detection, benchmark, ingest, mosaic, pipeline, quality
from earth_ops.monitoring import DEFAULT_KPI_FILE, load_runs, setup_logging
from earth_ops.synthetic import make_product

log = logging.getLogger("earth_ops")
DATA = Path("data")
SIMULATION_START = date(2025, 3, 1)
SIMULATION_STEP_DAYS = 10
SIMULATION_MAX_CLOUD = 0.6


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{text} must be a positive integer")
    return value


def ratio(text: str) -> float:
    value = float(text)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError(f"{text} must be between 0 and 1")
    return value


def iso_date(text: str) -> str:
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text} is not a YYYY-MM-DD date") from exc


def bbox_ok(bbox: list[float]) -> bool:
    west, south, east, north = bbox
    if -180 <= west < east <= 180 and -90 <= south < north <= 90:
        return True
    log.error("invalid bbox %s (expected W S E N in degrees)", bbox)
    return False


def cmd_ingest(args: argparse.Namespace) -> int:
    if not bbox_ok(args.bbox):
        return 2
    products = ingest.ingest(tuple(args.bbox), args.start, args.end, args.max_cloud,
                             args.limit, args.archive)
    return 0 if products else 1


def cmd_mosaic(args: argparse.Namespace) -> int:
    if not bbox_ok(args.bbox):
        return 2
    tif = mosaic.run_mosaic(tuple(args.bbox), args.start, args.end, args.max_cloud,
                            args.resolution, args.out)
    log.info("mosaic written to %s", tif.parent)
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(0)
    for i in range(args.count):
        day = SIMULATION_START + timedelta(days=SIMULATION_STEP_DAYS * i)
        make_product(args.out, args.size, seed=i, date=day.isoformat(),
                     cloud_fraction=float(rng.uniform(0, SIMULATION_MAX_CLOUD)))
    log.info("%d simulated product(s) written to %s", args.count, args.out)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    records = [pipeline.run_product(p, args.out, args.kpi, args.chunks, args.min_valid)
               for p in args.products]
    bad = [r for r in records if r["status"] not in pipeline.COMPLETED]
    log.info("%d run(s): %d completed, %d rejected or failed", len(records),
             len(records) - len(bad), len(bad))
    return 1 if bad else 0


def cmd_series(args: argparse.Namespace) -> int:
    csv = pipeline.run_series(args.products, args.out, args.chunks, args.min_valid)
    log.info("time series written to %s", csv.parent)
    return 0


def cmd_anomalies(args: argparse.Namespace) -> int:
    try:
        flagged = anomaly_detection.detect_anomalies(load_runs(args.kpi))
    except ValueError as exc:
        log.error("%s", exc)
        return 1
    print(anomaly_detection.format_report(flagged))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    product = args.product or make_product(args.out / "input", args.size, cloud_fraction=0.2)
    try:
        results = benchmark.run_benchmark(product, args.out, tuple(args.chunks), args.repeats)
    finally:
        if args.product is None:  # disk budget: only the simulated product this run created
            shutil.rmtree(product)
    table = benchmark.format_table(results)
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    benchmark.plot_results(results, args.out / "benchmark.png")
    print(table)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="earth_ops", description="Earth-ops: mini Sentinel-2 ground segment")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="download B04/B08/SCL over an AOI (public STAC, no account)")
    p.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("W", "S", "E", "N"))
    p.add_argument("--start", type=iso_date, required=True)
    p.add_argument("--end", type=iso_date, required=True)
    p.add_argument("--max-cloud", type=float, default=40.0, help="max scene cloud cover in %%")
    p.add_argument("--limit", type=positive_int, default=10)
    p.add_argument("--archive", type=Path, default=DATA / "products")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("mosaic", help="national NDVI mosaic (default: metropolitan France)")
    p.add_argument("--bbox", nargs=4, type=float, default=list(mosaic.FRANCE_BBOX),
                   metavar=("W", "S", "E", "N"))
    p.add_argument("--start", type=iso_date, default="2025-05-01")
    p.add_argument("--end", type=iso_date, default="2025-07-31")
    p.add_argument("--max-cloud", type=float, default=60.0,
                   help="max scene cloud cover in %% (clouds are masked per pixel with SCL)")
    p.add_argument("--resolution", type=positive_int, default=500, help="pixel size in m")
    p.add_argument("--out", type=Path, default=DATA / "mosaic")
    p.set_defaults(func=cmd_mosaic)

    p = sub.add_parser("simulate", help="write simulated products (tests, demos)")
    p.add_argument("--out", type=Path, default=DATA / "products")
    p.add_argument("--count", type=positive_int, default=10)
    p.add_argument("--size", type=positive_int, default=512)
    p.set_defaults(func=cmd_simulate)

    for name, func, help_text in (("run", cmd_run, "process products (one KPI record each)"),
                                  ("series", cmd_series, "time series + median composite")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("products", nargs="+", type=Path)
        p.add_argument("--out", type=Path, default=DATA / ("output" if name == "run" else "series"))
        p.add_argument("--chunks", type=positive_int,
                       default=None if name == "run" else pipeline.SERIES_CHUNKS,
                       help="Dask block size in px (default for run: whole scene in memory)")
        p.add_argument("--min-valid", type=ratio, default=quality.MIN_VALID_RATIO)
        if name == "run":
            p.add_argument("--kpi", type=Path, default=DEFAULT_KPI_FILE)
        p.set_defaults(func=func)

    p = sub.add_parser("anomalies", help="Isolation Forest on the KPI history")
    p.add_argument("--kpi", type=Path, default=DEFAULT_KPI_FILE)
    p.set_defaults(func=cmd_anomalies)

    p = sub.add_parser("benchmark", help="NumPy vs Dask: time, peak memory, exactness")
    p.add_argument("--product", type=Path, help="product to use (default: simulated)")
    p.add_argument("--size", type=positive_int, default=5490, help="simulated scene side in px")
    p.add_argument("--chunks", type=positive_int, nargs="+", default=list(benchmark.DEFAULT_CHUNKS))
    p.add_argument("--repeats", type=positive_int, default=3)
    p.add_argument("--out", type=Path, default=DATA / "benchmark")
    p.set_defaults(func=cmd_benchmark)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
