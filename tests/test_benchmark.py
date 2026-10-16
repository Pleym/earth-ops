"""Performance benchmark (REQ-PERF)."""
from earth_ops.benchmark import format_table, plot_results, run_benchmark


def test_req_perf_01_benchmark_reports_time_memory_and_exactness(product, tmp_path):
    work = tmp_path / "bench"
    results = run_benchmark(product, work, chunks=(32,))
    assert [(r["method"], r["chunk"]) for r in results] == [("numpy", None), ("dask", 32)]
    for r in results:
        assert r["seconds"] > 0 and r["peak_rss_mb"] > 0 and r["mpx_per_s"] > 0
        assert r["max_abs_diff"] <= 1e-4 and r["mask_mismatch"] == 0
    assert not list(work.glob("ndvi_dask_*.tif"))  # outputs are cleaned up (disk budget)
    assert "| numpy" in format_table(results)
    plot_results(results, work / "bench.png")
    assert (work / "bench.png").stat().st_size > 0
