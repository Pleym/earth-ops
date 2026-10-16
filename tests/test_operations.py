"""Orchestration, monitoring, anomaly detection and CLI (REQ-OPS, REQ-MON, REQ-ANO)."""
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio

from earth_ops import pipeline
from earth_ops.__main__ import main
from earth_ops.anomaly_detection import detect_anomalies
from earth_ops.monitoring import RunMonitor, load_runs
from earth_ops.synthetic import make_product


@pytest.fixture
def kpi(tmp_path):
    return tmp_path / "kpi" / "runs.jsonl"


def read_ndvi(out_dir, product):
    with rasterio.open(out_dir / product.name / "NDVI.tif") as src:
        return src.read(1), src.profile


# --- REQ-OPS: orchestration and error handling ----------------------------------


def test_req_ops_01_nominal_run_produces_ndvi_product_and_quicklook(product, tmp_path, kpi):
    rec = pipeline.run_product(product, tmp_path / "out", kpi)
    assert rec["status"] == "NOMINAL"
    assert (tmp_path / "out" / product.name / "quicklook.png").exists()
    _, profile = read_ndvi(tmp_path / "out", product)
    assert profile["dtype"] == "int16" and profile["nodata"] == -32768
    assert profile["crs"].to_epsg() == 32631


def test_req_ops_02_invalid_product_is_rejected_without_crash(product, tmp_path, kpi):
    (product / "SCL.tif").unlink()
    rec = pipeline.run_product(product, tmp_path / "out", kpi)
    assert (rec["status"], rec["error_code"]) == ("REJECTED", "E_MISSING_FILE")
    assert not (tmp_path / "out" / product.name / "NDVI.tif").exists()


def test_req_ops_03_unexpected_error_is_reported_as_failed(product, tmp_path, kpi, monkeypatch):
    def boom(*_args, **_kwargs):
        raise MemoryError("simulated")

    monkeypatch.setattr(pipeline, "compute_ndvi", boom)
    rec = pipeline.run_product(product, tmp_path / "out", kpi)
    assert (rec["status"], rec["error_code"]) == ("FAILED", "E_PROCESSING")
    assert "simulated" in rec["error"]


def test_req_ops_04_cloudy_product_is_degraded(tmp_path, kpi):
    p = make_product(tmp_path / "p", size=64, cloud_fraction=0.9)
    rec = pipeline.run_product(p, tmp_path / "out", kpi, min_valid_ratio=0.5)
    assert rec["status"] == "DEGRADED"
    assert rec["valid_ratio"] == pytest.approx(0.1, abs=0.02)


def test_req_ops_05_chunked_and_in_memory_runs_write_identical_products(product, tmp_path, kpi):
    pipeline.run_product(product, tmp_path / "a", kpi)
    pipeline.run_product(product, tmp_path / "b", kpi, chunks=32)  # not a multiple of 256
    a, _ = read_ndvi(tmp_path / "a", product)
    b, _ = read_ndvi(tmp_path / "b", product)
    np.testing.assert_array_equal(a, b)
    size = (tmp_path / "a" / product.name / "NDVI.tif").stat().st_size
    assert (tmp_path / "b" / product.name / "NDVI.tif").stat().st_size <= 1.2 * size  # no bloat


def test_req_ops_05_failed_write_leaves_no_partial_product(product, tmp_path, kpi, monkeypatch):
    def boom(*_args, **_kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(pipeline.dask, "compute", boom)
    rec = pipeline.run_product(product, tmp_path / "out", kpi)
    assert rec["status"] == "FAILED"
    assert not list((tmp_path / "out" / product.name).glob("*.tif"))


# --- REQ-MON: monitoring ------------------------------------------------------


def test_req_mon_01_every_run_is_recorded_with_stage_durations(product, tmp_path, kpi):
    pipeline.run_product(product, tmp_path / "out", kpi)
    (product / "B04.tif").unlink()
    pipeline.run_product(product, tmp_path / "out", kpi)
    runs = load_runs(kpi)
    assert [r["status"] for r in runs] == ["NOMINAL", "REJECTED"]
    assert {"validate", "process", "quicklook"} <= runs[0]["stages_s"].keys()
    assert runs[0]["total_s"] > 0 and runs[0]["throughput_mpx_s"] > 0


def test_req_mon_02_stage_duration_is_recorded_even_on_error(kpi):
    mon = RunMonitor("P", kpi)
    with pytest.raises(RuntimeError), mon.stage("broken"):
        raise RuntimeError
    assert "broken" in mon.stages_s


def test_load_runs_on_missing_file_returns_empty(tmp_path):
    assert load_runs(tmp_path / "none.jsonl") == []


# --- REQ-ANO: anomaly detection -----------------------------------------------


def _runs(n=30, seed=0):
    rng = np.random.default_rng(seed)
    return [
        {"run_id": str(i), "product_id": f"P{i}", "status": "NOMINAL",
         "total_s": rng.normal(2.0, 0.1), "valid_ratio": rng.uniform(0.7, 0.9),
         "ndvi_mean": rng.normal(0.5, 0.03)}
        for i in range(n)
    ]


def test_req_ano_01_slow_run_is_flagged_as_most_anomalous():
    slow = {"run_id": "slow", "product_id": "X", "status": "NOMINAL",
            "total_s": 20.0, "valid_ratio": 0.8, "ndvi_mean": 0.5}
    flagged = detect_anomalies(_runs() + [slow], contamination=0.05)
    worst = max(flagged, key=lambda r: r["anomaly_score"])
    assert worst["run_id"] == "slow" and worst["is_anomaly"]


def test_req_ano_02_failed_runs_are_excluded_and_input_untouched():
    failed = {"run_id": "f", "status": "FAILED", "total_s": 0.1,
              "valid_ratio": None, "ndvi_mean": None}
    runs = _runs() + [failed]
    snapshot = json.dumps(runs)
    flagged = detect_anomalies(runs)
    assert "f" not in {r["run_id"] for r in flagged}
    assert json.dumps(runs) == snapshot


def test_req_ano_03_too_few_runs_raise():
    with pytest.raises(ValueError):
        detect_anomalies(_runs(n=3))


# --- CLI end to end -------------------------------------------------------------


def test_cli_end_to_end(tmp_path):
    prod, out, kpi = tmp_path / "products", tmp_path / "out", tmp_path / "runs.jsonl"
    assert main(["simulate", "--out", str(prod), "--count", "8", "--size", "48"]) == 0
    products = sorted(str(p) for p in prod.iterdir())
    assert main(["run", *products, "--out", str(out), "--kpi", str(kpi), "--chunks", "16"]) == 0
    assert main(["anomalies", "--kpi", str(kpi)]) == 0
    assert main(["series", *products, "--out", str(out / "series")]) == 0
    assert (out / "series" / "ndvi_timeseries.png").exists()
    assert (out / "series" / "ndvi_timeseries.csv").read_text().count("\n") == 9
    (Path(products[0]) / "B04.tif").unlink()
    assert main(["run", products[0], "--out", str(out), "--kpi", str(kpi)]) == 1
